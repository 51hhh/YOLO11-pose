# 运行依赖与实际启用判定

“配置中选择”“主机已安装”“二进制已链接”和“这一帧实际执行”是四种不同证据。
任何一层都不能替代后一层。尤其是看到 CUDA/VPI 头文件或动态库，不代表
VPI remap 已经在 CUDA backend 上提交，也不代表左右 TensorRT 真正并发。

## 四级证据

| 级别 | 回答的问题 | 当前证据 |
|---|---|---|
| L1 配置意图 | profile 想启用什么 | `pipeline_backend_config`、最终 `pipeline.yaml`、配置指纹 |
| L2 SDK 能力 | 主机能否编译/加载所需框架 | `tensorrt_sdk`、`vpi_sdk`、`opencv_sdk`、MVS/CUDA/ROS2 检查 |
| L3 二进制能力 | 当前 `stereo_pipeline` 是否链接所需框架且无缺库 | `legacy_linkage` 对 `ldd` 的检查 |
| L4 实际执行 | 初始化是否成功、算子在哪个 backend 上执行、是否并发 | 启动日志、逐阶段计数、NVTX/Nsight、真实帧输出 |

`volleyball doctor` 只完成 L1–L3。L4 必须在 Orin NX 上启动真实管线后验收，
不能由开发机静态检查替代。

可用下面的只读报告集中查看 L1。报告同时列出“资源会分配”和“当前 profile 会
提交逐帧工作”，避免仅凭 stream 名或候选字段判断启用状态：

```bash
volleyball config pipeline-report --profile nx-rdk
volleyball config pipeline-report --profile nx-rdk --format json
```

## `doctor` 检查范围

对 `nx-rdk` 和 `free-flight`：

- `pipeline_backend_config`：打印相机、加速器、校正、检测、三角化和发布配置；
- `hik_mvs_runtime`：检查 MVS 开发头与 `libMvCameraControl`；
- `cuda_runtime`：检查 CUDA 根目录和 GPU 设备节点；
- `tensorrt_sdk`：检查 `NvInfer.h`、`libnvinfer` 和 `libnvinfer_plugin`；
- `vpi_sdk`：发现 `/opt/nvidia/vpi2`、`vpi3`、`vpi4` 等目录中的头文件和 `libnvvpi`；这只证明 SDK 文件齐全，不等于当前源代码已用该具体版本完整编译；
- `opencv_sdk`：检查旧工程 CMake 强制要求的 core、imgproc、features2d、
  calib3d、highgui、imgcodecs、cudastereo、cudaimgproc、cudafilters；
- `legacy_linkage`：拒绝任何 `not found`，并确认当前二进制确实链接 MVS、
  VPI、OpenCV、CUDA runtime、TensorRT，以及 `nx-rdk` 所需的 ROS2；
- 模型文件与 SHA-256、标定语义、GPIO 字符设备和 ROS2 环境仍分别检查。

`replay` 不启动硬件管线，因此跳过上述硬件 SDK 与链接门禁。

```bash
volleyball doctor --profile nx-rdk
volleyball doctor --profile nx-rdk --json
volleyball doctor --profile free-flight
```

## 各组件的 L4 验收证据

| 组件 | 初始化成功证据 | 逐帧执行证据 | 不能据此断言 |
|---|---|---|---|
| MVS | 两相机序列号匹配，TriggerMode/Source/Activation 回读正确 | FrameCounter、ExtTriggerCount 连续且左右同号 | USB 回调先后就是曝光先后 |
| CUDA | stream/event/kernel 创建成功 | Bayer kernel、ROI kernel 的 NVTX/CUDA timeline | 仅有 `/dev/nvhost-gpu` 就已执行 GPU 算子 |
| VPI | stream、image、remap payload 创建成功，日志打印 rectifier backend | Nsight 中 remap/convert，且无 VPI 错误；像素回归通过 | 变量名 `vpiStreamPVA` 表示当前使用 PVA |
| TensorRT | engine 反序列化、双 detector context/buffer 创建成功 | 左右 preprocess/infer/decode event 与有效候选计数 | engine 文件存在就兼容当前 TensorRT/GPU |
| OpenCV | 必需模块可加载 | 仅对实际选中的 CPU/CUDA OpenCV 分支记录调用与候选计数 | 链接 OpenCV 表示主线 ROI 都走 OpenCV |
| ROS2 | node、publisher 和消息类型创建成功 | topic 频率、时间戳语义和 source epoch 正确 | 环境已 source 就代表消息已经发布 |

当前主线的 VPI remap/Gray 转换选择 CUDA backend；双路 YOLO 使用两个
TensorRT detector 和各自 CUDA stream；ROI 候选集合由最终 profile 投影确定。
这些代码路径事实仍需用 L4 证据确认目标机没有初始化失败、自动降级或被配置关闭。

源码通过 `vpi_plane_access.h` 适配 VPI 2/3 的 `plane.data` 与 VPI 4 的
`plane.pBase + offsetBytes`，并由仓库审计禁止绕过该边界。这个适配只解决
pitch-linear 地址布局差异；`doctor` 的 L2 成功不能代替用目标机实际 VPI 版本
完成 CMake configure、编译、链接和短帧运行。

## 推荐的目标机验收顺序

1. 运行 `doctor --json`，保存输出和 profile 指纹；任何 L1–L3 error 都不启动采集。
2. 运行 `volleyball run --profile nx-rdk --dry-run`，归档最终旧管线 YAML。
3. 启动短时真实管线，保存完整初始化日志，核对 backend、engine/context、相机回读。
4. 用固定双目回放帧做像素、候选框、pair、深度来源和 3D 回归。
5. 用 Nsight Systems 录制至少一个稳定窗口，确认 CUDA/VPI/TensorRT 时间线和双路并发。
6. 再进入 90 Hz 长时间实拍，统计分位延迟、队列水位、丢弃原因、温度和降频。

只有第 3–6 步可以把状态从“可用/已链接”提升为“实际启用且性能达标”。

## 参考

- [NVIDIA VPI 安装说明](https://docs.nvidia.com/vpi/4.0/installation.html)：不同版本安装在 `/opt/nvidia/vpiN`，包含头文件和库目录。
- [NVIDIA VPI Stereo Disparity C API](https://docs.nvidia.com/vpi/group__VPI__StereoDisparityEstimator.html)：创建/提交参数初始化和 disparity submit 返回 `VPIStatus`。
- [NVIDIA VPI Rescale C API](https://docs.nvidia.com/vpi/group__VPI__Rescale.html)：half-resolution 链的每次 rescale submit 都返回 `VPIStatus`，必须逐项检查。
- [NVIDIA VPI Image C API](https://docs.nvidia.com/vpi/group__VPI__Image.html)：外部 buffer 只在 image lock 期间有效；相容的多重 lock 必须逐次匹配 unlock。
- [NVIDIA VPI Basic Concepts](https://docs.nvidia.com/vpi/basic_concepts.html)：CUDA 互操作应在 lock 期间使用导出的 pitch-linear view；跨 backend 包装外部 allocation 不保证零拷贝。
- [NVIDIA TensorRT 安装说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)：C++ 部署需要 TensorRT 动态库；仅安装 Python wheel 不能替代当前 C++ 管线依赖。
