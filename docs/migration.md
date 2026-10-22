# 旧 NX 管线迁移说明

## 当前状态

`adapters/legacy_nx` 保留成熟的 VPI 校正、ROI 深度和当前实机 ROS2 编排，并复用已提取到 `adapters/hik_mvs` 与 `adapters/gpio` 的相机和 GPIO 实现。标定读取、Bayer→BGR 和目标到源浮点映射/灰度输出 CUDA 基础算子已进入 `adapters/cuda_preprocess`；旧管线已经直接调用其中的 Bayer 算子，但尚未切换到新的完整 `StereoPreprocessor`。TensorRT engine/context/buffer 生命周期、静态 I/O 验证、单输出布局与类别契约、检测值类型、单输出/DFL CPU 解码、NMS、检测输入 CUDA 预处理和 GPU DFL 解码 kernel 已经迁入 `adapters/tensorrt`；该目录也已实现主线 `Detector` 端口和检测事实映射，旧管线只以兼容类型别名持有 runtime。直接观测到 `NxBallObservation` 的严格映射、rclcpp 发布器和进程源纪元持久化已进入 `adapters/ros2`，并由 NX 目标强制编译；旧 publisher 也复用该源纪元实现，不再在兼容目录维护副本。重构不会一次性重写这些高风险硬件代码，而是先冻结外部契约，再逐步拆分其内部边界。

旧的轨迹 CSV、baseline clip 录制器和一次性特征匹配导出入口已经删除；它们既不满足 v2.1 原始事实契约，也无法由正式 profile 到达。真实图像、候选、配对和直接 3D 只允许通过 `volleyball capture` 驱动的 v2.1 staging 写入，低频排障截图则由显式诊断配置控制。

旧 `HybridDepth` 可以为内部关联使用 bbox 单目深度、双目深度和 Kalman 状态，但其输出中的 `raw_*` 只能原样传递当前帧已验证的直接双目三角化测量。纯单目、单/双目融合、偏差校正和预测结果均不得设置 `raw_observation_valid`，因此 NX 的 ROS2 publisher 不会把派生状态冒充为直接测量。

主线目前已有 `CameraSource → StereoPipelineRunner → StereoObservationService → StereoPreprocessor/Detector/DirectObservationEstimator → TrialWriter/ObservationPublisher` 的平台无关编排，并用脚本相机源覆盖结束、超时、错误和中止语义。外层 `runManagedStereoTrial` 已证明“相机先进入抓取、GPIO 后启动；退出时 GPIO 先停止”的单次 trial 顺序。`StereoPreprocessor` 的输出与原始帧身份强绑定，必须同时提供同尺寸的校正 BGR8 和 Gray8；`Detector` 不再接受原始 Bayer。2D 关联中心视差与最终深度测量证据已经分离，中心三角化只保留为显式测试/回放基线，服务不会把它隐式当成 `roi_multi_point`。`HikStereoCameraSource` 已把海康双目帧接入该端口：它按 `FrameCounter` 查询 GPIO 上升沿、分别保留左右设备时钟和主机到达时钟、按每帧实际曝光计算曝光中心，并以有界自有缓冲保证下游持有图像时不会被覆盖。`MainlineStageTrialWriter` 已按四类事实原子写入 v6 staging，其中直接观测与实际深度方法、精化视差、离散度和支持点数不可拆分；独立 finalizer 会重新验证计数、时序、时钟映射、方法是否来自冻结配置、关系和原图后发布 v2.1。上述平台无关边界均有主机契约测试。

这仍不等于新的实机入口已经完成。主线 `TrialWriter`、TensorRT/CUDA `Detector`、ROS2 `ObservationPublisher` 与平台无关启停编排已完成，但 CUDA/VPI `StereoPreprocessor`、ROI/CUDA `DirectObservationEstimator`、配置装配和原生进程入口尚未接入。新 ROS2 目标目前只作为 `stereo_pipeline` 的构建依赖接受真实接口编译，旧可执行文件仍调用旧 publisher；只有原生主线进程装配完成后才会切换运行时。当前实机仍由旧管线编排，不能仅凭主机 SDK 桩测试宣称新入口可采集或达到实时性能。服务层已用两个常驻工作线程并发调用左右同步 detector，并以主机屏障测试证明调用重叠；独立 CUDA stream 是否在目标机形成实际 GPU 重叠仍需 Nsight 证据。

根工程通过 `nx-legacy-release` 预设将旧工程作为 ExternalProject 编译。这样可以复用实机能力，同时防止旧工程的全局 CUDA、VPI、MVS、OpenCV 和 ROS2 依赖进入 `libs/core`、`ports`、`algorithms` 或 `services`。

## 配置所有权

`configs/` 是唯一主线配置源。启动时将合并后的配置投影为旧管线 YAML：

```text
分层主线配置 → schema/语义校验 → 配置 SHA-256
             → legacy YAML 生成 → 旧 NX 可执行文件
```

旧目录中的生产 YAML 现阶段仍是投影所需的基础模板，但不得直接作为新的部署入口。深度候选由主线 `candidate_methods` 完整覆盖旧 `depth_modes`；`legacy_activation_policy.py` 还会统一覆盖 DLA、双路检测、全帧视差、神经特征、异步 ROI、P2 诊断和显示等启用开关。永远关闭且没有正式模型资产的旧图像 SOT、未验证 Bayer 语义的 TNR 分支、没有生产端的 ROS2 全帧深度诊断，以及违背 NX/RDK 边界的 NX 落点预测与控制目标发布，均已直接删除。正式 P0/P1 使用的 bbox 视差一致性、亚像素几何门限、fallback 搜索范围和圆拟合采样上限也已提升到主线 schema；其余已禁用实验候选的数值仍需在启用前决定删除或提升，不能直接修改模板投入正式 profile。

## 迁移阶段

1. **冻结契约（已完成）**：v2.1 数据、强类型核心、配置和 CI 已建立；
2. **包裹旧实现（已完成）**：主线 CLI 负责 doctor、配置生成、采集 staging 和最终发布；
3. **提取算法与用例编排（已完成主机门禁）**：纯关联/三角化已进入 `algorithms/stereo`，平台无关 runner 已覆盖 trial 的正常与故障终态；仍需逐项用录制数据对比旧实现；
4. **提取适配器（进行中）**：GPIO 触发、海康相机及其 `CameraSource` 桥、主线 v6 `TrialWriter` 与 v2.1 finalizer 已迁入独立边界；标定读取和通用 CUDA 图像基础算子已进入 `adapters/cuda_preprocess`，但完整预处理端口尚未装配；TensorRT engine/context/buffer、I/O 契约、CPU 后处理、检测输入 CUDA 预处理、GPU DFL kernel、主线 `Detector` 端口桥及检测事实映射也已从旧目录迁出；`adapters/ros2` 会逐项验证帧、候选、配对、时间、坐标系、协方差和深度方法后发布直接观测，不再二次跟踪或合成协方差；CUDA/VPI `StereoPreprocessor`、ROI/CUDA `DirectObservationEstimator` 的真实实现仍待完成；
5. **替换可执行入口（已建立生命周期骨架）**：相机与触发的安全启停已进入平台无关服务；待预处理和估计适配器完成后建立配置装配及原生进程入口，再把运行时切到新 detector/writer/publisher，旧管线只保留回归基准；
6. **删除兼容层（未开始）**：只有在同一批实机数据、性能和输出契约均通过时执行。

## 兼容性门禁

每次迁移必须固定同一组录制输入并比较：

- 帧与候选数量；
- 接受/拒绝配对及理由；
- 3D 差异和协方差；
- 端到端延迟 p50/p95/p99；
- GPU/CPU 占用与队列水位；
- 输出 schema、配置指纹和原始文件哈希。

数值更接近旧实现并不自动代表更正确；标定靶、静止球、已知距离和独立参考轨迹是绝对精度判据。

## 不允许的迁移方式

- 新核心层直接包含 MVS、CUDA、VPI、TensorRT、OpenCV 或 ROS2 头文件；
- 为赶进度从主线反向依赖旧目录内部全局状态；
- 同时修改算法、数据 schema 和硬件时序而没有可重放基线；
- 直接编辑生成的 legacy YAML 后把结果当作可复现实验；
- 删除拒绝候选或用滤波轨迹覆盖直接观测。

