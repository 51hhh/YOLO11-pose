# Jetson NX 实机适配器

本目录隔离海康 MVS、GPIO、CUDA/VPI、TensorRT 和 ROS2 等平台依赖。它负责把硬件帧转换为主线定义的观测与采集记录，但不拥有顶层配置、数据契约或用户工作流。

日常运行统一使用仓库根目录安装的 `volleyball` 命令。不要直接复制或修改本目录的 YAML 作为实验配置；`config/base_pipeline.yaml` 只是主线配置投影到旧 C++ 字段时使用的兼容模板。

## 构建

在 Jetson Orin NX 上从仓库根目录执行：

```bash
cmake --preset nx-legacy-release
cmake --build --preset nx-legacy-release --parallel
```

构建产物位于 `adapters/legacy_nx/build/`。主机 CPU 预设不会配置本目录，因此 CI 不需要 CUDA、VPI、TensorRT、海康 SDK 或 ROS2。

依赖由 JetPack/系统包提供；海康 SDK 路径通过根工程的 `VOLLEYBALL_MVS_SDK_ROOT` 指定，可选 libSGM 路径通过 `VOLLEYBALL_LIBSGM_ROOT` 指定。正式 NX 预设要求 MVS 头文件和动态库同时存在，缺少任一项会直接停止配置。仓库不内嵌 SDK 或第三方源码。

## 保留的可执行程序

| 程序 | 用途 | 使用边界 |
|---|---|---|
| `stereo_pipeline` | 双目采集、YOLO、ROI 深度、记录与发布 | 由根目录 `volleyball run/capture` 生成配置并启动 |
| `capture_chessboard` | 用同一硬触发链采集双目标定图 | 仅用于标定，不生成正式 trial |
| `stereo_calibrate` | 从已采集棋盘图计算双目标定 | 输出经复核后放入 `configs/calibration/` |
| `hik_frame_metadata_probe` | 检查 FrameCounter、触发计数和到达顺序 | 正式采集前的硬件验收工具 |

具体入口与验收顺序见[实机辅助工具](../../docs/hardware-tools.md)和[验收标准](../../docs/acceptance.md)。

## 代码边界

```text
config/       旧 C++ 字段的唯一兼容模板
src/capture/  海康取流、帧身份与 GPIO 触发
src/detect/   TensorRT 检测
src/rectify/  VPI 校正
src/stereo/   ROI、特征与深度候选
src/fusion/   兼容性融合和本地诊断预测
src/pipeline/ 实时流水线编排
src/ros/      ROS2 发布适配
src/utils/    v2.1 staging 记录与底层工具
tests/        仅本适配器内部的 C++ 测试
tools/        必须直接接触相机 SDK 的诊断程序
```

平台无关的时间、坐标、配对和数据语义应放在根目录 `libs/`、`algorithms/`、`services/` 或 `python/volleyball_vision/`，不得继续堆入本目录。一次性实验脚本、生成结果和历史报告不进入仓库。

## 权威文档

- [架构与边界](../../docs/architecture.md)
- [图像帧生命周期](../../docs/pipeline/README.md)
- [配置系统](../../docs/configuration.md)
- [采集契约](../../docs/capture.md)
- [旧实现迁移边界](../../docs/migration.md)
