# 流水线观测、性能与验收

## 软件可见性

当前代码通过 `globalPerf()`、日志和 NVTX 暴露多个阶段。至少应持续观察：

旧 C++ logger 当前输出带单调时钟、线程 ID 和级别的文本行，并非 JSON。主线因此显式配置 `json_logs=false`；`metrics_interval_frames` 才是统计日志周期的配置真相，并被投影为旧管线 `performance.log_interval`。

| 类别 | 关键指标 |
|---|---|
| 取流 | `Stage0_WaitGrab`、FrameCounter 跳变、左右到达时间差 |
| 前处理 | `Stage0_Process`、`Stage1_WaitRect`、`Stage0_VpiCudaLock/Unlock`、`Rectified_VpiCudaLock/Unlock`、Bayer stream 错误 |
| YOLO | `Stage1_DetectSubmit`、`Stage2_WaitYOLOComplete`、左右提交标志 |
| ROI | `Stage2_DualYoloMatch`、各候选 attempted/valid/selected |
| 异步队列 | pending depth、free buffers、worker busy、replace/drop/no-buffer |
| 截止线 | `Stage2_AsyncRoiOverDeadline`、stale pending/ready/result drop |
| 回调 | `Stage2_ResultCB`、`Stage2_FrameCB` |
| 输出 | FPS、3D 有效率、深度来源、置信度、pair reject reason |

只看平均 FPS 不足以验收。需要同时看端到端年龄、p50/p95/p99、连续丢帧长度和每个候选的有效率。

## Nsight Systems 建议标注

已有 Stage 级 NVTX range，但目标 NX 验收时应确认时间线上能区分：

```text
GPIO/Grab wait
BayerToBGR
VPI remap + convert
YOLO-L preprocess/infer/decode/D2H
YOLO-R preprocess/infer/decode/D2H
pair gate
GPU geometry batch
center/multi-point NCC
async snapshot/worker
fusion/callback
```

重点检查：

- `cudaStreamBGR` 同步和 `vpiStreamSync` 是否成为主瓶颈；
- 左右 YOLO 是否真正并行，还是受功耗/engine/context 串行化；
- ROI worker 是否与下一帧 Stage 0/1 重叠；
- 是否存在意外的整帧 D2H；
- BGR 快照在当前关闭彩色/神经特征时是否确实没有发生；
- callback 或磁盘写入是否回压 pipeline 线程。

## 实时 `nx-rdk` 验收

### 正确性

- 最终 legacy YAML 与预期 profile 指纹一同归档；
- 左右 FrameCounter 始终按身份配对，USB 到达顺序不参与配对；
- YOLO 坐标已经逆 letterbox 回到校正图分辨率；
- 每个发布结果可追溯到左右候选和 pair attempt；
- `stereo_depth_source` 与真正被选择的候选一致；
- 槽位复用后的异步结果不触发 FrameCallback。

### 实时性

- 90 Hz 输入下报告 end-to-end p50/p95/p99，而不是只报循环 FPS；
- 统计 stale、no-buffer、pending-replace 和 over-deadline 的比率；
- 若允许丢帧，给出最大连续丢帧长度及对 RDK 轨迹预测的影响；
- ROS2 消息使用原曝光/触发时间语义，而非发布时刻冒充采样时刻。

建议初始门限不是直接写死为“零丢帧”，而是先用真实投掷数据测量，再根据 RDK 能接受的最大观测间隔确定。配置和报告必须显式保留丢弃原因。

## `free-flight` 采集验收

采集模式的门限更严格：

- `async_roi_stage2=false`、`drop_stale_roi_frames=false`；
- `FrameCounter`、trigger sequence 和 frame id 连续；
- 左右计数器一致，时间字段不倒退；
- callback 前完成图像深拷贝；
- 所有外部 CUDA 指针仅在 VPI lock 区间使用，unlock 后地址被清空；槽位复用时不存在旧 `locked_frame_id`；
- raw lock/unlock、VPI submit/sync 或 CUDA 前处理失败均形成失败帧，不能复用旧像素伪造成功帧；
- TensorRT enqueue、CUDA event record/wait/sync 和 full-frame VPI Stereo 任一错误均形成失败帧；event 未成功记录时不得查询或等待旧槽位 event；
- 注入 Stage 1 失败时，已入队 GPU 工作完成排空、失败回调恰好一次，v2.1 recorder 以 `fail_trial` 结束而不是写出正常空检测；
- 注入 ROI host image lock/unlock 与全帧 Coordinate3D CUDA 失败时，帧状态为失败；而合法“没有有效视差”仍是成功空结果，两者不可混淆；
- ROI Phase C 的失败/stale-drop 分支之后 Phase D 仍执行，异步 grab 请求没有遗留；
- 在 USB 抓帧等待中注入停止：已接受请求完成并解锁后 Stage 0 才可继续；不得读取上一帧 `grab_done_ok`，不得产生无 payload 的额外 frame；
- 注入 VPI host/CUDA view 非法尺寸、CUDA D2H 失败或 unlock 失败：显示/诊断/基线/v2.1 复制均返回失败并清空目标图；实时诊断的 `dropped` 计数增加，正式采集不得写出部分图像；
- 注入 raw/ROI host view 的空地址、过小宽高或 pitch，以及同步 ROI 核心异常：不得调用相机写入无效 raw 缓冲，不得进入 HybridDepth predict-only；所有已取得的 VPI lock 必须释放，帧以 `grab_failed` 发布；
- writer 队列使用 block，不丢旧帧；
- `status.json.complete=true`；
- captured、written 数相等，drop/copy/write failure 全为 0；
- 数据集校验、文件哈希和确定性 replay 全部通过。

## 像素与深度回归

建议建立固定小型回放集，覆盖远近距离、快慢球、模糊、边缘入画和单侧漏检。每次 CUDA/VPI 或 ROI 算法修改比较：

- 校正 BGR/Gray 的逐像素差和边缘差；
- 左右候选数、框坐标和置信度；
- pair gate 通过/拒绝原因；
- 每种 disparity 候选及支持数；
- 最终 3D 和 covariance；
- 性能分位数与 D2H 流量。

GPU 浮点/算法版本可能导致微小差异，回归应按字段设置容差，不能对所有中间值盲目做字节相等；帧身份、配对原因和数据表结构则应严格相等。

## 当前仍需目标硬件确认

以下内容不能由开发机静态 review 证明：

- MVS、VPI、TensorRT 和当前 engine 在 Orin NX 上的实际启用路径；
- VPI remap 真正运行的 backend 和资源占用；
- 双路 TensorRT 并发与功耗降频行为；
- 90 Hz 曝光、GPIO 边沿和 FrameSpecInfo 之间的固定延迟/抖动；
- async ROI 在真实球速与检测密度下的截止线分布；
- TNR 路径的 Bayer 语义和视觉效果；当前关闭，不属于验收通过项。
- 本次开发环境和已授权 A800 主机均无 VPI/JetPack，因而 CUDA/VPI/TensorRT 目标源的最终编译与上述故障注入仍必须在 Orin NX 完成。
- VPI 3/4 的平面地址适配已用最小头桩通过严格编译和偏移测试，但这不是实际 VPI SDK ABI/API 构建证据；目标机报告必须记录 `NV_VPI_VERSION_STRING`、CMake configure 和完整链接结果。

## 执行命令

开发机静态与契约检查：

```bash
PYTHONPATH=python python3 -m volleyball_vision config validate --profile nx-rdk
PYTHONPATH=python python3 -m volleyball_vision config render-legacy --profile nx-rdk
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 tools/check_repository.py
```

目标 NX 在启动前执行：

```bash
volleyball doctor --profile nx-rdk
volleyball doctor --profile nx-rdk --json
volleyball run --profile nx-rdk --dry-run
```

`doctor` 覆盖配置、SDK 与动态链接证据，不声称逐帧算子已经执行；四级启用判定见
[运行依赖与实际启用判定](07-runtime-enablement.md)。

实际采集完成后执行：

```bash
volleyball dataset validate data/captures
volleyball replay --profile replay --trial data/captures/trials/<trial-id>
```

完整硬件门禁仍以 [仓库验收指南](../acceptance.md) 为准。

