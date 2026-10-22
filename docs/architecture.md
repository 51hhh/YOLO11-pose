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

`libs/logging` 只提供适配器共用的轻量日志接口，不保存流水线状态，也不参与数据契约。它独立于 `legacy_nx`，避免已提取的适配器反向依赖旧目录。

关键类型刻意分开：

- `TriangulatedObservation` 是直接测量；
- `ValidatedObservation` 是质量判定，不修改原测量；
- `TrackEstimate` 是滤波状态；
- `LandingPrediction` 是预测结果。

因此滤波位置不能冒充 `trajectory.csv` 中的测量。

### Ports

`ports` 定义相机、双目预处理、检测器、trial 写入器和发布器的最小接口。用例依赖接口，不依赖某一型号相机或推理后端。

原始 `StereoFrame` 与 `RectifiedStereoFrame` 是不同类型。前者携带 Bayer 图像和不可修改的采集事实；后者只用 `FrameReference` 引用来源，并携带同一校正模型下的 BGR8 detector 输入与 Gray8 几何输入。`Detector` 只接受后者，因此不能隐式把 Bayer 当成 BGR，也不能在检测器内部悄悄采用另一套校正坐标。

### Algorithms

`algorithms/stereo` 只完成左右候选关联、校正双目的三角化和协方差传播。输入输出均为 core 类型，可以在无相机、无 GPU 的主机上测试。

### Services

`StereoObservationService` 负责一次双目观测用例：先校验并保存原始配对帧，再调用 `StereoPreprocessor` 生成与来源帧强绑定的校正 BGR/Gray 图，之后调用两侧检测器、保存全部候选、枚举并保存全部配对尝试、三角化通过门控的配对，然后写入或发布直接观测。预处理输出若交换帧身份、改变尺寸、缺少图像或使用错误像素格式，整次 trial 会明确失败；原始 Bayer 事实仍留在未发布 staging 中供排障。任何必需写入失败同样会使本次 trial 失败。

`StereoPipelineRunner` 负责一个 trial 的平台无关生命周期。它从 `CameraSource` 逐项读取配对帧，把帧交给 `StereoObservationService`，并区分正常流结束、可容忍的连续超时、超时越限、数据源错误和源协议错误。只有 `end_of_stream` 能触发 `finalize`；其他终止路径统一 `abort`，同一 runner 不能运行第二次。相机线程、SDK 句柄、GPU 缓冲和 detector 资源仍属于 adapter，runner 不接管硬件所有权，也不是实机可执行入口。

`accepted` 表示该配对最终成功生成一条直接 3D 观测，不只是通过了 2D 几何门控。同一曝光时刻只有最终选中的配对可以保持 `accepted`；其他通过 2D 门控但未选中的组合记录为 `not_selected`，三角化或 3D 质量门控失败分别记录为 `triangulation_failure` 或 `quality_gate`。数据校验还要求每个 accepted 配对与唯一轨迹样本的帧引用、3D、协方差和误差数值一致，因此不会出现“接受配对存在但对应 3D 缺失或被替换”的原始数据。

检测器输出还必须声明与输入完全相同的相机、角色和帧号，候选 ID 在单帧内唯一，候选的 pipeline ID 与批次一致。候选传入的 `selected` 值不受信任；服务先全部清零，只在直接 3D 成功后标记最终左右候选，防止 detector 内部临时选择污染原始契约。

进入检测前，左右帧还必须具有相同帧号、不同相机 ID、非负且处于同一映射时钟域的曝光中心时间；已标记 dropped 的帧不能继续处理。左右曝光中心允许存在有界残差，关联层以 `max_capture_delta_ns` 门控，并把实测 `delta_t_lr_ns` 原样写入配对证据；超限项记录为 `temporal_mismatch`，而不是要求两个软件估计值逐纳秒相等。图像缓冲的宽高、像素格式、stride 和实际容量必须覆盖 metadata 声明的 ROI，避免错误 adapter 把尺寸漂移或短缓冲传入 GPU。

### Adapters 与运行入口

`adapters` 实现实际硬件、存储格式和外部系统。主线 `MainlineStageTrialWriter`、GPIO 触发和海康双目相机分别位于 `adapters/capture_v21`、`adapters/gpio` 与 `adapters/hik_mvs`。写入器按帧、候选、配对和直接观测四类事实分别落盘，只有四张表与全部原图都关闭后才发布完成状态；Python finalizer 再独立校验并原子发布冻结的 v2.1 trial。海康公开头不暴露 MVS SDK；真实实现仅在找到固定 SDK 头文件和动态库时构建，主机通过共用 SDK 桩验证生命周期、帧水印和同步拒绝路径。

`HikStereoCameraSource` 是海康 SDK 到平台无关 `CameraSource` 的唯一桥。它不直接依赖某个 GPIO 类，而是通过只读查询回调取得与 `FrameCounter` 相同序号的上升沿事实；每帧同时保留左右设备时间、NX 单调/实时时钟到达时间、曝光开始、由实际曝光计算的曝光中心和映射不确定度。图像先复制进有界的 adapter 自有缓冲，若下游仍持有全部缓冲则显式失败，不能覆盖尚在处理的旧帧。相机必须先进入抓取状态，调用方再启动 GPIO；停止、异常和 GPIO 生命周期最终由尚待建立的应用层统一编排。

现有 CUDA/VPI/TensorRT/ROS2 代码仍集中在 `adapters/legacy_nx`。正式磁盘 `TrialWriter` 的主线 staging 写入与 v2.1 finalizer 已迁出，平台无关服务也已显式建立 `StereoPreprocessor` 契约，但真实 CUDA/VPI 预处理、TensorRT detector 尚未实现该新端口，更未组装成新的原生应用入口。当前唯一用户入口仍是 Python 包导出的 `volleyball` CLI，旧 NX 工程由 CLI 生成配置并由根工程 ExternalProject 隔离构建。仓库不保留没有可执行实现的空应用层目录。

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

