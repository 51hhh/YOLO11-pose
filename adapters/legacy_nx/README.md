# Jetson NX 实机适配器

本目录隔离 CUDA/VPI、TensorRT 和 ROS2 等尚未迁出的平台依赖，并复用根目录 `adapters/hik_mvs`、`adapters/gpio` 与 `adapters/capture_v21`。它负责兼容编排硬件帧、观测与采集记录，但不拥有顶层配置、数据契约或用户工作流。

日常运行统一使用仓库根目录安装的 `volleyball` 命令。不要直接复制或修改本目录的 YAML 作为实验配置；`config/base_pipeline.yaml` 只是主线配置投影到旧 C++ 字段时使用的兼容模板，其自身默认关闭发布、预测和写盘。

## 构建

在 Jetson Orin NX 上从仓库根目录执行：

```bash
cmake --preset nx-legacy-release
cmake --build --preset nx-legacy-release --parallel
```

构建产物统一位于 `build/nx-legacy-superbuild/legacy_nx/`。主机 CPU 预设不会配置本目录，因此 CI 不需要 CUDA、VPI、TensorRT、海康 SDK 或 ROS2。

依赖由 JetPack/系统包提供；海康 SDK 路径通过根工程的 `VOLLEYBALL_MVS_SDK_ROOT` 指定，可选 libSGM 路径通过 `VOLLEYBALL_LIBSGM_ROOT` 指定。正式 NX 预设要求 MVS 头文件和动态库同时存在，缺少任一项会直接停止配置。仓库不内嵌 SDK 或第三方源码。

## 保留的可执行程序

| 程序 | 用途 | 使用边界 |
|---|---|---|
| `stereo_pipeline` | 双目采集、YOLO、ROI 深度、记录与发布 | 由根目录 `volleyball run/capture` 生成配置并启动 |
| `capture_chessboard` | 用同一硬触发链采集双目标定图 | 仅用于标定，不生成正式 trial |
| `stereo_calibrate` | 从已采集棋盘图计算双目标定 | 输出经复核后放入 `configs/calibration/` |
| `hik_frame_metadata_probe` | 检查 FrameCounter、触发计数和到达顺序 | 正式采集前的硬件验收工具 |

具体入口与验收顺序见[实机与标定辅助工具](../../docs/hardware-tools.md)和[验收标准](../../docs/acceptance.md)。

## 代码边界

```text
config/       旧 C++ 字段的唯一兼容模板
src/calibration/ 运行时双目标定加载
src/capture/  异步抓帧状态与 v2.1 staging 录制
src/detect/   TensorRT 检测
src/rectify/  VPI 校正
src/stereo/   ROI、特征与深度候选
src/fusion/   兼容性融合和本地诊断预测
src/pipeline/ 实时流水线编排
src/ros/      ROS2 发布适配
src/diagnostics/ 可选实时调试 artifact 写出
src/ui/       本地显示与点击测距叠加
src/utils/    VPI 图像访问和性能计数等底层工具
tools/        按可执行程序分目录的标定与硬件诊断入口及 common 文件提交边界
```

海康双目相机、GPIO 触发与公共日志分别归 `adapters/hik_mvs/`、`adapters/gpio/` 与 `libs/logging/` 所有；旧工程只复用这些目标，不再维护私有副本。

独立硬件 adapter 的 SDK 桩测试位于 `tests/cpp/adapters/`；尚未迁出的兼容逻辑与 CUDA/VPI 桩位于 `tests/cpp/legacy_nx/`。两者都由根目录主机预设执行，本目录不再保留第二套测试入口。

平台无关的时间、坐标、配对和数据语义应放在根目录 `libs/`、`algorithms/`、`services/` 或 `python/volleyball_vision/`，不得继续堆入本目录。一次性实验脚本、生成结果和历史报告不进入仓库。

## 权威文档

- [架构与边界](../../docs/architecture.md)
- [图像帧生命周期](../../docs/pipeline/README.md)
- [配置系统](../../docs/configuration.md)
- [采集契约](../../docs/capture.md)
- [旧实现迁移边界](../../docs/migration.md)
