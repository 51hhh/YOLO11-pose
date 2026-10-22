# 架构与数据流

## 目标

系统只把可追溯的传感器事实交给后续滤波、轨迹预测和 RL。原始层不保存“看起来更平滑”的状态；所有滤波、参数辨识和影子仿真都是可删除、可重算的派生结果。

```text
NX GPIO 上升沿 ────────┐
                       ├─ 左/右同步曝光
海康 FrameSpecInfo ────┤  FrameCounter 对齐
左右 USB 独立队列 ─────┘  分别记录到达时刻
          │
          ▼
校正图像 → 左右全部检测候选 → 全部关联尝试 → 直接三角化 3D
          │                                      │
          ├──────── v2.1 原始 trial ─────────────┤
          │                                      ▼
          └────────────────────── 确定性回放 / 影子仿真 / RL 数据构造
```

## 分层

### Core

`libs/core` 只定义事实类型及其不变量：相机帧、2D 候选、关联尝试、直接 3D 观测、时钟、坐标系和标定。它不认识 MVS、OpenCV、CUDA、TensorRT、ROS2 或磁盘格式。

关键类型刻意分开：

- `TriangulatedObservation` 是直接测量；
- `ValidatedObservation` 是质量判定，不修改原测量；
- `TrackEstimate` 是滤波状态；
- `LandingPrediction` 是预测结果。

因此滤波位置不能冒充 `trajectory.csv` 中的测量。

### Ports

`ports` 定义相机、检测器、trial 写入器和发布器的最小接口。用例依赖接口，不依赖某一型号相机或推理后端。

### Algorithms

`algorithms/stereo` 只完成左右候选关联、校正双目的三角化和协方差传播。输入输出均为 core 类型，可以在无相机、无 GPU 的主机上测试。

### Services

`StereoObservationService` 负责一次双目观测用例：读取配对帧、调用两侧检测器、保存全部候选、枚举并保存全部配对尝试、三角化通过门控的配对，然后写入或发布直接观测。任何必需写入失败都会使本次 trial 失败。

### Adapters 与 Apps

`adapters` 实现实际硬件和外部系统；`apps` 只组装依赖并暴露命令。现有 NX/CUDA 代码集中在 `adapters/legacy_nx`，并由根工程 ExternalProject 隔离构建。

## 三条独立时间链

### 曝光链

两台相机在抓取线程启动后进入 `TriggerMode=On`、`TriggerSource=Line0`、`TriggerActivation=RisingEdge`。NX 对 GPIO 写高，左右相机同时开始一次曝光。记录：

- 触发序号；
- 计划的单调时钟时间；
- 写高前后夹逼得到的单调/实时时钟中点；
- 写操作不确定度；
- 可配置的相机输入到曝光延迟及其不确定度。

软件时间戳是 GPIO 写操作的估计，不等于已经由电气仪器证明的传感器曝光时刻。真实固定延迟与抖动必须通过示波器或相机 I/O 实测后填入配置。

### 帧身份链

相机在图像首部嵌入 FrameSpecInfo。SDK 解析 `Timestamp`、`Framecounter` 和 `ExtTriggerCount` 后，代码以 `FrameCounter` 为双目身份键：

- 相同计数器才构成左右帧对；
- 一侧落后时只推进落后一侧的 `OneByOne` 队列；
- 超过有界对齐次数仍不一致则拒绝；
- 计数器按无符号 32 位语义允许自然回绕；
- 采集模式要求连续，否则整次 trial 失败。

`Timestamp` 用于设备时钟映射和相位残差，`ExtTriggerCount` 只用于诊断，二者都不替代帧身份。

### 传输链

左右 USB 传输和 SDK 队列彼此独立。每侧 `GetImageBuffer` 返回时分别记录：

- `CLOCK_MONOTONIC` 到达时刻；
- `CLOCK_REALTIME` 到达时刻；
- SDK 提供的原始主机时间；
- 回调完成时间和后续处理时间。

这些字段允许计算曝光到到达延迟、左右延迟差、乱序率和长尾分位数，但绝不参与帧配对。

## 实时模式与采集模式

实时模式允许有界低延迟队列和 `drop_oldest`，但每个发布观测仍必须拥有明确的来源帧与时间语义。采集模式使用大容量阻塞队列和 `fail_trial`：不能为追求实时性丢帧，也不能在缺少原图、候选、配对或时序字段时发布一个“部分成功”的 trial。

## NX 与 RDK 边界

NX 负责相机、校正、检测、双目关联和直接 3D 观测。RDK 负责最终稳健滤波、未来轨迹和控制。NX 内部可为关联使用临时滤波器，但不得把滤波位置伪装成直接测量发布；`publish_filtered_as_measurement` 在 schema 中固定为 `false`。

NX 内部 CUDA/VPI 图像流、双路 YOLO、shifted-IoU 配对、灰度 ROI 深度候选以及一帧资源所有权的详细说明见 [NX 图像处理流水线](pipeline/README.md)。该文档同时区分最终 profile、旧配置投影和运行时自动降级，避免从历史 YAML 推断当前启用状态。

