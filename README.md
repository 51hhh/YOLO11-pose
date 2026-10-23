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
python3 -m pip install -e .

volleyball config validate --profile nx-rdk
volleyball config validate --profile free-flight
volleyball config validate --profile replay
volleyball config render-mainline --profile nx-rdk

PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s tests -v
cmake --preset host-core-debug
cmake --build --preset host-core-debug --parallel 2
ctest --preset host-core-debug
python3 tools/check_repository.py
```

离线 NX 若已经预装 `setuptools>=68`、PyYAML 和 jsonschema，可用
`python3 -m pip install --no-build-isolation -e .`，避免 PEP 517 隔离环境尝试
联网下载构建依赖。该参数不会安装缺失的运行依赖，不能用来绕过环境准备。

## 目录边界

| 路径 | 职责 |
|---|---|
| `libs/core/` | 强类型领域事实、时钟和坐标语义；不得依赖硬件 SDK |
| `libs/logging/` | 跨适配器复用的轻量日志接口；不拥有业务状态 |
| `ports/` | 相机、双目预处理、检测器、直接观测估计器、写入器和发布器接口 |
| `algorithms/stereo/` | 双目关联、中心三角化基线和协方差传播；不包含硬件实现 |
| `services/` | 将端口与算法编排为观测用例 |
| `apps/nx_mainline/` | 原生 NX 进程装配、配置解析、信号处理和退出状态 |
| `adapters/` | 硬件、推理、存储和中间件适配器 |
| `adapters/capture_stage/` | 与硬件无关的主线 v6 staging 原子写入适配器；最终格式仍是 v2.1 |
| `adapters/gpio/` | GPIO 周期触发及上升沿多时钟审计；主机默认使用无 libgpiod 构建 |
| `adapters/hik_mvs/` | 海康双目相机生命周期、FrameSpecInfo、帧对齐及主线 `CameraSource` 桥 |
| `adapters/cuda_preprocess/` | 双目标定映射、Bayer 转换及 CUDA 校正 BGR/Gray |
| `adapters/cuda_depth/` | CUDA 圆/边缘/bbox 球心代理、显式回退选择及主线 `DirectObservationEstimator`；纹理 ROI 算子仅供研究 |
| `adapters/tensorrt/` | TensorRT 检测适配器边界；集中 engine/context/buffer 生命周期、检测契约、主线 `Detector` 端口、事实映射及按 CUDA 构建的输入与 DFL kernel |
| `adapters/ros2/` | 直接 3D 观测到 ROS2 消息的严格映射、发布及源纪元持久化；不执行跟踪、预测或坐标系改名 |
| `python/volleyball_vision/` | 配置、v6 staging 校验与 v2.1 发布、回放和离线评估 |
| `configs/` | 默认值、硬件、标定、算法、模式和 profile 分层配置 |
| `tests/contract/`、`tests/unit/` | 数据包契约与 Python 组件测试 |
| `tests/support/` | `unit` 与 `contract` 共用的最小有效 trial 夹具；测试模块之间不互相导入 |
| `tests/cpp/mainline/` | 平台无关的领域、算法和服务测试 |
| `tests/cpp/adapters/` | 独立硬件适配器的行为与契约测试；只消费替身，不拥有替身实现 |
| `tests/cpp/hardware_tools/` | 标定和硬件验收程序的无硬件单元测试；目录与 `tools/hardware/` 对应 |
| `tests/cpp/fakes/` | 测试专属的第三方 SDK 桩、适配器运行时替身与主线端口替身；不进入生产构建图 |
| `tools/hardware/` | 帧元数据探针、标定图采集和离线双目标定三个受测辅助程序 |
| `tools/check_repository.py` | 仓库结构、依赖方向、文档导航和构建接线门禁 |

部署入口只有 `apps/nx_mainline`；配置、构建和采集均不再维护第二套兼容流水线。

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

# 生成原生主线最小配置并查看启动命令，不启动硬件
volleyball run --profile nx-rdk --dry-run
```

NX 构建使用根工程的原生主线预设：

```bash
cmake --preset nx-mainline-release
cmake --build --preset nx-mainline-release --parallel
volleyball run --profile nx-rdk
```

主线不内置 GUI；实时检查通过 ROS2、结构化运行统计和录制数据完成。

## 真实采集

先为每次实验准备条件 JSON；未知值使用 `null` 或省略，禁止伪造精度。球 ID、压力、温湿度、风、相机姿态等条件属于 trial 元数据，不应塞进轨迹 CSV。

可先在无 NX 硬件的开发机预演最终采集 YAML；该 staging 会被永久标为不可发布：

```bash
volleyball capture \
  --profile free-flight \
  --trial dry-run-001 --session planning --ball v200w-01 \
  --duration 3.0 --dataset-root /tmp/volleyball-capture-plan --dry-run
```

正式采集必须去掉 `--dry-run`，使用没有 tracked/untracked 改动的干净 Git 工作树，
并通过模型、NX、MVS、GPIO、CUDA、TensorRT 和 ROS2 等 doctor 门禁。先复制
[`rig-manifest.example.json`](configs/hardware/rig-manifest.example.json) 到实验目录，
填入两台相机、镜头和软件版本，导出相机参数文件并固定 SHA-256，最后将状态改为
`verified`：

```bash
volleyball capture \
  --profile free-flight \
  --trial 2026-10-22-v200w-001 \
  --session free-flight-a \
  --ball v200w-01 \
  --duration 3.0 \
  --conditions conditions.json \
  --hardware-manifest /path/to/rig-manifest.json

volleyball dataset validate data/captures
```

采集先写入同文件系统的隐藏 staging 目录。原生 C++ 链产生逐事实 v6 staging，
保存原始 Bayer、全部检测/配对尝试，以及每条 3D 的实际方法和最终 UVD 输入证据；
Python 同时固定 Git 状态、原生可执行文件、模型、标定和硬件清单哈希；原生进程会把
清单中的相机型号与固件版本同 MVS 枚举结果逐侧核对。清单、两个相机参数导出文件
与版本来源随后作为原始审计资产归档。在帧数、触发、时钟、
引用闭包和 SHA-256 全部通过后才原子发布冻结的 v2.1 trial。崩溃后的完整 staging
可以显式恢复：

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

原始观测不可被滤波结果覆盖；标定、参数辨识、轨迹平滑和影子仿真都写入派生目录，并记录输入哈希、配置指纹和代码状态。工作树不干净时，回放 revision 会明确带 `-dirty`，不会冒充对应的干净 commit。这样才能分别估计相机观测噪声与排球动力学残差，而不是把二者混成一个随机域。

## 文档

- [架构与数据流](docs/architecture.md)
- [原生 NX 帧生命周期、CUDA、YOLO 与几何深度](docs/pipeline.md)
- [实机与标定辅助工具](docs/hardware-tools.md)
- [配置系统](docs/configuration.md)
- [采集契约](docs/capture.md)
- [验收标准](docs/acceptance.md)
- [架构决策记录](docs/adr/README.md)
- [贡献与提交规范](CONTRIBUTING.md)
