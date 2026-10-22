# 完整图像帧生命周期

## 生命周期边界

一帧的物理起点是一次 NX GPIO 上升沿，逻辑身份是相机 payload 内的 `FrameCounter`，不是 USB 到达顺序。其逻辑终点是结果回调完成；若启用 v2.1 采集，还必须等录制器复制所需图像和元数据后，三缓冲槽才可以安全复用。

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

初始化时一次性创建 VPI 图像和 CUDA event，并缓存 VPI 图像对应的 CUDA 指针。之后 `reset()` 只清理帧号、候选、结果和状态，不反复分配大图像。每个槽包含：

- 原始 `rawL/rawR`；
- 去 Bayer 的临时 BGR；
- 校正后的左右 BGR 和 Gray；
- 左右检测候选、全部配对尝试和 3D 结果；
- `evtRectDone`、左右 `evtDetectDone` 和 `evtStereoDone`；
- 与左右 detect event 配对的 `*_event_recorded` 状态；event 随槽位复用，只有本帧成功 record 后才允许 wait，不能把上一轮 event 当成本帧完成；
- 触发、设备、USB 到达和配对残差元数据。

原始 VPI 图像严格按 profile 的宽高预分配。海康相机打开时必须成功回读左右两侧的 `Width/Height/OffsetX/OffsetY`，并与 profile 的完整画面、零偏移契约一致；不能读取时不会退回配置值继续运行。每次 SDK payload 写入槽位前还会检查实际宽高和目标 pitch。任一项变化时先归还 SDK buffer，再让该帧明确失败，绝不把未知尺寸数据复制进既有 VPI 缓冲。

槽位复用前有两个保护：

1. 普通 CUDA/VPI 工作必须已被对应 stream/event 同步；
2. 实时异步 ROI 若正在从该槽做 D2D/D2H 快照，`waitAsyncRoiSlotSnapshotDone()` 必须先完成。

取流失败的槽不会继续沿用上一次留在 raw buffer 中的像素。raw VPI 锁定/解锁、Stage 0 stream 同步、TNR、remap 或格式转换任一失败都会设置 `grab_failed`；该帧跳过 YOLO 和深度计算，并沿失败回调进入采集的 `fail_trial` 契约。采集回调深拷贝校正 Gray8 后也必须成功解锁 VPI 图像，否则计为 `copy_failure` 并使 trial 失败。

## 实时 `nx-rdk` 生命周期

实时主线使用 `ROI_ONLY` 循环，逻辑上交错四个阶段：

1. Phase A 请求下一帧抓取，grab 线程可在 pipeline 线程工作时等待 USB；
2. Phase B 对上一轮已经校正的帧提交左右 YOLO；
3. Phase C 等更早一帧的 YOLO，提交异步 ROI Stage 2；
4. Phase D 等当前 grab 完成并提交 Bayer、校正和灰度转换。

Phase A 一旦发出抓帧请求，本轮 Phase C 的失败、主动 stale drop 或空结果都只能结束当前融合块，不能直接 `continue` 外层循环。Phase D 是每一轮的必达清理点：它必须收割当前异步抓帧并推进 `next_grab_frame`，否则下一轮可能覆盖尚未完成的请求或把 payload 写进错误槽位。

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
- ROI 失败帧可能同时经过提交和融合阶段，`roi_callbacks_published` 保证 Result/Frame 回调至多一次；
- 采集模式仍会触发帧回调，让 fail-trial 录制器立即停止，而不是静默制造序号缺口；
- 停止时先让 pipeline 排空已提交的工作，再关闭 ROI/P2 worker 和 grab 线程，最后同步所有 stream 并停止 PWM/相机；
- 实时模式的主动丢弃必须有性能计数，采集模式的任何丢弃都不允许发布完整 trial。

## 代码核对点

- 容器和回调时效：`src/pipeline/frame_slot.h`、`pipeline_callbacks.h`
- 启停顺序：`src/pipeline/pipeline_lifecycle.cpp`
- 四阶段循环：`src/pipeline/pipeline_loops.cpp`
- 异步槽位快照：`src/pipeline/pipeline_async_roi.cpp`
- 采集深拷贝：`src/utils/capture_v21_stage_recorder.cpp`

