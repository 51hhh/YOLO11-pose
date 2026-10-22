# 排球双目感知与真实数据闭环

本仓库用于把海康双目相机的同步图像转换为可审计的 3D 排球观测，并把真实自由飞行数据送入回放、影子仿真和后续 RL 随机域辨识。目标平台是 Jetson Orin NX；RDK 只消费直接观测并负责最终状态估计、轨迹预测和控制。

当前主线已经完成以下软件契约：

- 分层配置、配置来源追踪和稳定 SHA-256 指纹；
- GPIO 上升沿触发、海康 FrameSpecInfo 帧身份和 USB 到达时刻的独立记录；
- 不依赖 USB 到达顺序的双目配对；
- 相机先抓取、GPIO 后启动以及 GPIO 先停止的单次 trial 生命周期；
- 原始 Bayer 采集事实与校正 BGR/Gray 图像的强类型预处理边界；
- 全候选、全配对尝试、3D 观测和处理时序的 v2.1 原子数据包；
- 真实 trial 的哈希校验、确定性回放和影子轨迹误差评估；
- 平台无关的 C++ 核心、算法、服务边界和 CPU CI。

真实 NX、相机线缆和示波器上的验收仍必须在目标硬件上执行，不能由主机测试代替。详细门禁见 [验收指南](docs/acceptance.md)。

## 最重要的同步语义

| 事实 | 权威来源 | 用途 |
|---|---|---|
| 左右曝光属于哪一次触发 | NX GPIO 上升沿序号与时间戳 | 曝光时间、轨迹时间轴 |
| 收到的 payload 是哪一帧 | 海康 FrameSpecInfo `FrameCounter` | 双目配对、检出漏帧 |
| payload 何时到达主机 | 每侧 `GetImageBuffer` 返回时的单调/实时时钟 | USB/SDK 排队与延迟诊断 |

USB 传输可能让左右图像乱序到达。代码因此使用随图像 payload 到达的 `FrameCounter` 配对，而不使用回调先后或 USB 到达时间配对。`ExtTriggerCount` 会保存，但已在现有相机上观察到 0/+2 步进，只作为诊断字段。图像首部水印必须先由 SDK 解析，再清除对应像素行，之后才能进入去 Bayer、YOLO 和图像保存。

## 快速开始

```bash
cd YOLO11-pose
python3 -m pip install -e '.[test]'

volleyball config validate --profile nx-rdk
volleyball config validate --profile free-flight
volleyball config validate --profile replay
volleyball config pipeline-report --profile nx-rdk

PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s tests -v
cmake --preset host-core-debug
cmake --build --preset host-core-debug --parallel 2
ctest --preset host-core-debug
python3 tools/check_repository.py
```

## 目录边界

| 路径 | 职责 |
|---|---|
| `libs/core/` | 强类型领域事实、时钟和坐标语义；不得依赖硬件 SDK |
| `libs/logging/` | 跨适配器复用的轻量日志接口；不拥有业务状态 |
| `ports/` | 相机、双目预处理、检测器、直接观测估计器、写入器和发布器接口 |
| `algorithms/stereo/` | 双目关联、中心三角化基线和协方差传播；不包含硬件实现 |
| `services/` | 将端口与算法编排为观测用例 |
| `adapters/` | 硬件、推理、存储和中间件适配器 |
| `adapters/capture_v21/` | 与硬件无关的主线 v6 staging 原子写入适配器；最终格式仍是 v2.1 |
| `adapters/gpio/` | GPIO 周期触发及上升沿多时钟审计；主机默认使用无 libgpiod 构建 |
| `adapters/hik_mvs/` | 海康双目相机生命周期、FrameSpecInfo、帧对齐及主线 `CameraSource` 桥 |
| `adapters/tensorrt/` | TensorRT 检测适配器边界；当前先集中无 SDK 的 engine I/O 契约 |
| `adapters/legacy_nx/` | CUDA/VPI、TensorRT 与 ROS2 的 NX 兼容管线 |
| `python/volleyball_vision/` | 配置、v4/v6 staging 校验与 v2.1 发布、回放和离线评估 |
| `configs/` | 默认值、硬件、标定、算法、模式和 profile 分层配置 |
| `tests/contract/`、`tests/unit/` | 数据包契约与 Python 组件测试 |
| `tests/support/` | `unit` 与 `contract` 共用的最小有效 trial 夹具；测试模块之间不互相导入 |
| `tests/cpp/mainline/` | 平台无关的领域、算法和服务测试 |
| `tests/cpp/adapters/` | 独立硬件适配器的 SDK 桩测试 |
| `tests/cpp/legacy_nx/` | 尚未迁出的 NX 兼容代码及 CUDA/VPI 桩测试；`tools/` 镜像实机工具目录 |
| `tests/cpp/fakes/` | 多测试套件共用的最小第三方 SDK 桩头文件 |
| `tools/` | 仅保留仓库维护门禁；不提供第二套运行入口，也不放一次性实验脚本 |

旧 NX 实现仍承载 CUDA、TensorRT 和 ROS2 的实机代码，并复用已提取的海康 MVS、GPIO 与 v2.1 适配器，但不再定义顶层配置或数据契约。迁移边界见 [迁移说明](docs/migration.md)。

## 配置与运行

三个正式 profile：

- `nx-rdk`：NX 实时双目感知，向 RDK 发布直接观测，不隐式录制；
- `free-flight`：阻塞式、无丢帧自由飞行采集，任何契约破坏使整次 trial 失败；
- `replay`：读取已发布的 v2.1 trial，不启动相机或模型。

```bash
# 查看合并结果、某个值的来源和两个 profile 的差异
volleyball config render --profile free-flight
volleyball config explain --profile free-flight cameras.sync.frame_identity_source
volleyball config diff nx-rdk free-flight

# 只读环境检查；未固定模型哈希、缺少标定或可执行文件时不允许实机启动
volleyball doctor --profile nx-rdk

# 生成旧 NX 适配器配置并查看启动命令，不启动硬件
volleyball run --profile nx-rdk --dry-run
```

NX 构建使用根工程预设，旧工程在隔离的 ExternalProject 中编译，避免 CUDA/SDK 全局依赖污染核心层：

```bash
cmake --preset nx-legacy-release
cmake --build --preset nx-legacy-release --parallel
volleyball run --profile nx-rdk

# 调试时显示检测与深度叠加窗口
volleyball run --profile nx-rdk --visualize
```

## 真实采集

先为每次实验准备条件 JSON；未知值使用 `null` 或省略，禁止伪造精度。球 ID、压力、温湿度、风、相机姿态等条件属于 trial 元数据，不应塞进轨迹 CSV。

可先在无 NX 硬件的开发机预演最终采集 YAML；该 staging 会被永久标为不可发布：

```bash
volleyball capture \
  --profile free-flight \
  --trial dry-run-001 --session planning --ball v200w-01 \
  --duration 3.0 --dataset-root /tmp/volleyball-capture-plan --dry-run
```

正式采集必须去掉 `--dry-run`，并通过模型、NX、MVS、GPIO、CUDA/VPI/TensorRT 等 doctor 门禁：

```bash
volleyball capture \
  --profile free-flight \
  --trial 2026-10-22-v200w-001 \
  --session free-flight-a \
  --ball v200w-01 \
  --duration 3.0 \
  --conditions conditions.json

volleyball dataset validate data/captures
```

采集先写入同文件系统的隐藏 staging 目录，完成帧数、触发、配对、时钟、表格、原始图像和 SHA-256 检查后才原子发布。当前 CLI 实机入口仍由旧 NX 适配器产生 v4 staging；新 C++ 端口链产生逐事实 v6 staging，并额外保存每条 3D 的实际深度方法和精化视差证据。二者共享同一最终化命令并发布为同一个冻结的 v2.1 trial，版本不会被混写。崩溃后的完整 staging 可以显式恢复：

```bash
volleyball capture-finalize \
  --profile free-flight \
  --stage /path/to/.partial-trial \
  --dataset-root data/captures
```

## 回放与影子仿真

```bash
volleyball replay \
  --profile replay \
  --trial data/captures/trials/2026-10-22-v200w-001

volleyball replay \
  --profile replay \
  --trial data/captures/trials/2026-10-22-v200w-001 \
  --shadow shadow_trajectory.csv \
  --json
```

原始观测不可被滤波结果覆盖；标定、参数辨识、轨迹平滑和影子仿真都写入派生目录，并记录输入哈希与配置指纹。这样才能分别估计相机观测噪声与排球动力学残差，而不是把二者混成一个随机域。

## 文档

- [架构与数据流](docs/architecture.md)
- [CUDA/VPI、YOLO、ROI 与完整帧生命周期](docs/pipeline/README.md)
- [实机与标定辅助工具](docs/hardware-tools.md)
- [配置系统](docs/configuration.md)
- [采集契约](docs/capture.md)
- [旧 NX 管线迁移](docs/migration.md)
- [验收标准](docs/acceptance.md)
- [架构决策记录](docs/adr/README.md)
- [贡献与提交规范](CONTRIBUTING.md)
