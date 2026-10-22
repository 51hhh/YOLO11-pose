# 完整图像帧生命周期

## 生命周期边界

一帧的物理起点是一次 NX GPIO 上升沿，逻辑身份是相机 payload 内的 `FrameCounter`，不是 USB 到达顺序。其逻辑终点是结果回调完成；若启用 v2.1 采集，还必须等录制器复制所需图像和元数据后，三缓冲槽才可以安全复用。

新主线通过 `ManagedCameraSource`、`TriggerController` 和 `runManagedStereoTrial` 固化进程级顺序：相机抓取成功并回报 started 后才能启动 GPIO，任何正常或异常退出都先停止 GPIO 再停止相机。当前旧 NX 管线仍承载实际 CUDA/VPI 帧槽和回调，以下资源细节描述该兼容实现。

```text
触发计划
  -> GPIO 写高及边沿时间戳
  -> 左右曝光
  -> 左右 payload 独立到达
  -> FrameSpecInfo 解析、FrameCounter 对齐、水印清除
  -> FrameSlot 写入
  -> Stage 0 前处理与校正
  -> Stage 1 左右 YOLO
  -> Stage 2 关联、ROI 深度、融合
  -> ResultCallback / FrameCallback
  -> 可选录制器复制
  -> slot reset 并复用
```

## 逐阶段所有权

| 阶段 | 主要所有者 | 写入内容 | 退出条件 |
|---|---|---|---|
| 触发 | PWM 触发线程 | 触发序号、计划/实际边沿时刻、写入不确定度 | GPIO 写入完成 |
| 取流与配对 | grab 线程、海康适配器 | 左右 payload、设备时间、到达时间、帧号、触发计数 | 两侧 `FrameCounter` 相同或明确失败 |
| FrameSlot 装载 | pipeline/grab 协作 | raw Bayer 和完整 `FrameMetadata` | raw VPI host lock 解除 |
| Stage 0 | pipeline 线程 + CUDA/VPI | 校正 BGR、校正 Gray | VPI stream 同步成功 |
| Stage 1 | 两条 TensorRT CUDA stream | 左右候选框及置信度 | detect event 完成并 collect |
| Stage 2 | pipeline 线程或异步 ROI worker | 配对尝试、所有深度候选、直接 3D、融合状态 | 同步完成或异步结果在截止线前返回 |
| 发布 | pipeline 线程 | ROS2 直接观测、结果回调 | 回调返回 |
| 采集 | callback + writer 线程 | 帧表、候选表、配对表、可选 PGM | callback 内完成复制；磁盘写入异步完成 |

## FrameSlot 三缓冲

`FrameSlot` 是图像资源和一帧逻辑状态的聚合容器，固定使用三个槽位：

```text
slot_index = frame_id % 3
```

初始化时一次性创建 VPI 图像和 CUDA event，但不缓存解锁后的 CUDA 指针。VPI 只保证 `vpiImageLockData()` 返回的地址在本次 lock 到匹配 unlock 之间有效，因此每个 Stage 仅在真实使用区间取得短期 CUDA view。之后 `reset()` 只清理帧号、候选、结果和状态，不反复分配大图像。每个槽包含：

- 原始 `rawL/rawR`；
- 去 Bayer 的临时 BGR；
- 校正后的左右 BGR 和 Gray；
- 左右检测候选、全部配对尝试和 3D 结果；
- `evtRectDone`、左右 `evtDetectDone` 和 `evtStereoDone`；
- 与左右 detect event 配对的 `*_event_recorded` 状态；event 随槽位复用，只有本帧成功 record 后才允许 wait，不能把上一轮 event 当成本帧完成；
- 触发、设备、USB 到达和配对残差元数据。

原始 VPI 图像严格按 profile 的宽高预分配。海康相机打开时必须成功回读左右两侧的 `Width/Height/OffsetX/OffsetY`，并与 profile 的完整画面、零偏移契约一致；不能读取时不会退回配置值继续运行。每次 SDK payload 写入槽位前还会检查实际宽高和目标 pitch。任一项变化时先归还 SDK buffer，再让该帧明确失败，绝不把未知尺寸数据复制进既有 VPI 缓冲。

槽位复用前有两个保护：

1. 普通 CUDA/VPI 工作必须已被对应 stream/event 同步，并释放该帧全部 VPI CUDA view；
2. 实时异步 ROI 若正在从该槽做 D2D/D2H 快照，`waitAsyncRoiSlotSnapshotDone()` 必须先完成，才能 unlock 和 `reset()`。

`reset()` 故意不清理 lock 状态或 CUDA 地址：锁只能由知道相关 stream 是否已经结束的调度层释放。这样遗漏 unlock 会被下一次 lock 识别为 stale-frame 错误，而不会被重置操作掩盖。

取流失败的槽不会继续沿用上一次留在 raw buffer 中的像素。raw VPI 锁定/解锁、Stage 0 stream 同步、TNR、remap 或格式转换任一失败都会设置 `grab_failed`；该帧跳过 YOLO 和深度计算，并沿失败回调进入采集的 `fail_trial` 契约。采集回调深拷贝校正 Gray8 后也必须成功解锁 VPI 图像，否则计为 `copy_failure` 并使 trial 失败。

## 实时 `nx-rdk` 生命周期

实时主线使用 `ROI_ONLY` 循环，逻辑上交错四个阶段：

1. Phase A 请求下一帧抓取，grab 线程可在 pipeline 线程工作时等待 USB；
2. Phase B 对上一轮已经校正的帧提交左右 YOLO；
3. Phase C 等更早一帧的 YOLO，提交异步 ROI Stage 2；
4. Phase D 等当前 grab 完成并提交 Bayer、校正和灰度转换。

Phase A 一旦发出抓帧请求，本轮 Phase C 的失败、主动 stale drop 或空结果都只能结束当前融合块，不能直接 `continue` 外层循环。Phase D 是每一轮的必达清理点：它必须收割当前异步抓帧并推进 `next_grab_frame`，否则下一轮可能覆盖尚未完成的请求或把 payload 写进错误槽位。

异步抓帧是严格 single-flight 握手。`stop()` 只停止接受新请求，不把已经接受的请求伪装成“已完成”：grab 线程仍会完成当前 raw 写入、解除左右 host lock 并发布真实结果，`waitGrab()` 才能返回。若停止发生在新请求被接受前，该请求被拒绝且不会生成一个没有 payload 的虚假帧。这样 Stage 0、槽位排空和相机写线程不会同时访问同一 raw image。

相机写入前不仅检查 VPI lock 状态，还验证左右 host plane 的地址、宽高和 pitch 至少覆盖配置的 Bayer 帧；lock 成功但 view 不完整仍是抓帧失败，不能调用 MVS SDK 写入未知容量的缓冲区。

显示、显式启用的实时诊断和 v2.1 采集需要把 VPI 图像复制到 CPU。它们统一经 `vpi_image_copy` 执行同步复制：VPI read lock 必须覆盖整个 OpenCV/CUDA copy，且无论尺寸校验、分配、拷贝还是解锁失败，输出 `cv::Mat` 都被清空并向调用方返回失败。诊断支路不能缓存 VPI 指针，也不能把半张图当成有效帧。

当前投影中 C++ SOT tracker 为关闭，因此每帧都是 YOLO 检测帧。实时 profile 允许：

- 当更新帧的 YOLO 已就绪时丢弃过期 ROI 任务；
- pending ROI 只保留较新的任务；
- Stage 2 超过约 11.1 ms 截止线后丢弃旧结果；
- 三缓冲槽已被复用时，只发布由保存元数据构造的结果回调，不再调用含图像引用的帧回调。

最后一点防止回调读取已经属于后续帧的图像。异步结果的滤波/融合和对外回调仍回到 pipeline 主线程执行，worker 不直接修改对外状态。

## `free-flight` 采集生命周期

真实采集通过 `volleyball capture` 生成临时旧管线 YAML，并强制：

- `async_roi_stage2=false`；
- `drop_stale_roi_frames=false`；
- queue capacity 256、overflow policy `block`；
- v2.1 recorder 开启；
- 原始产品为左右校正 Gray8 PGM；
- 任一帧号、触发、复制或写入错误使整次 trial 失败。

因此采集路径在同一 `FrameSlot` 上同步完成 Stage 2 和 `FrameCallback`。录制 callback 在返回前把必须保存的灰度图复制到自己的队列对象；磁盘 writer 之后处理独立副本，不持有 VPIImage 引用。

## 回调有效期

`ResultCallback` 接收值语义元数据和结果引用，用于 ROS2、预测器或轻量观测消费。`FrameCallbackData` 中的 VPIImage 句柄和 vector 引用只在同步回调期间有效：

- 需要跨线程保存时，必须在回调内复制；
- 显示线程只能接收复制后的 `cv::Mat` 和值类型结果；
- 录制器不能把 VPIImage 句柄直接放入 writer 队列；
- 异步 ROI 的槽位已经复用时，禁止补发 FrameCallback。

## 失败帧与排空

- `grab_failed` 是为兼容旧回调保留的“本帧不可发布为有效观测”总门控：FrameCounter 无法对齐、抓取/Stage 0、TensorRT/CUDA 同步或 VPI Stereo 失败都会设置它；不能按字段名把它仅解释为相机抓取失败；
- TensorRT `enqueue` 失败时仍可能已有预处理、推理或 D2H 工作进入 stream，失败路径必须先排空对应 stream；
- detect event 记录失败时禁止等待槽位中旧 event，改为直接同步真正的源 stream，并把本帧交给 fail-trial；
- 任一图像 CUDA view 必须携带 `locked_frame_id`，消费者同时校验 lock、帧号、地址和 pitch；解锁后立即清空地址，禁止跨帧复用；
- Stage 2 需要 host Gray 时，左右 VPI image 必须同时 lock/unlock 成功；基础设施失败不能退化为“没有纹理候选”；
- 当前首选所依赖的 P1 GPU batch 若 H2D、kernel、D2H 或 collect 同步失败，必须沿 `processing_failed` 进入帧失败；空 vector 不得被解释成普通候选无效并继续发布 bbox/circle 回退；
- 全帧 Stage 3 使用显式成功状态区分“计算成功但没有有效视差”和 Coordinate3D CUDA/图像锁失败，后一种必须使帧失败；
- ROI 失败帧可能同时经过提交和融合阶段，`roi_callbacks_published` 保证 Result/Frame 回调至多一次；
- 异步 ROI 快照 event 失败、handoff 不完整或 worker 异常会随 `AsyncRoiResult.processing_failed` 回到主线程；buffer 先归还，结果不进入 HybridDepth，并以 `grab_failed` 发布，而不是伪装成 `predict_only`；
- 采集模式仍会触发帧回调，让 fail-trial 录制器立即停止，而不是静默制造序号缺口；
- 停止时先让 pipeline 排空已提交的工作，再关闭 ROI/P2 worker 和 grab 线程，最后同步所有 stream 并停止 PWM/相机；
- 停止发生在 USB 等待期间时，已接受的抓帧必须完成握手；`running=false` 本身绝不能唤醒 `waitGrab()` 并复用上一帧成功状态；
- 若 owner 在线程真正执行首帧前请求停止，主循环直接退出且不额外抓取一帧；若首帧已经开始，则仍按已接受工作完成并排空；
- `running=false` 只停止新调度，不等于资源已经清理；独立的 cleanup token 保证即使 worker 先异常退出，owner 线程仍会且只会执行一次 join、stream drain、VPI view 解锁及相机/触发停止；
- pipeline、grab、异步 ROI、P2 诊断、采集写线程和可选调试落盘线程都有 `noexcept` 入口边界。用户回调或正式 worker 的未捕获异常会标记 pipeline/capture failure 并触发受控退出；非关键调试落盘失败则停用或丢弃对应 artifact，不得让 `std::thread` 调用 `std::terminate`；
- pipeline 主循环若在仍为 `RUNNING` 时意外 `return/break`，线程入口会将其升级为 failure，避免进程主循环误以为设备仍在运行而永久等待；
- failure 对当前 `Pipeline` 实例是终态；完成清理后也不能原地重启，必须销毁并重新初始化全部 CUDA/VPI/TensorRT/相机资源；正常主动停止则仍允许再次启动；
- 若最终 CUDA/VPI stream drain 失败，停止路径不会提前解除导出的 VPI CUDA view；这些锁保留到资源销毁，同时整次运行标记失败；
- 实时模式的主动丢弃必须有性能计数，采集模式的任何丢弃都不允许发布完整 trial。

## 代码核对点

- 容器和回调时效：`src/pipeline/frame_slot.h`、`pipeline_callbacks.h`
- 启停顺序：`src/pipeline/pipeline_lifecycle.cpp`
- 运行/失败/清理状态：`src/pipeline/pipeline_run_state.h`
- 四阶段循环：`src/pipeline/pipeline_loops.cpp`
- 异步槽位快照：`src/pipeline/pipeline_async_roi.cpp`
- 采集深拷贝、队列和收尾：`src/capture/capture_v21_stage_recorder.cpp`
- 可在主机验证的 staging CSV 合同：`adapters/capture_v21/`

