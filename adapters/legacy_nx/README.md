# Jetson NX 兼容适配器

本目录临时保留尚未迁出的 CUDA/VPI、ROI 深度和 ROS2 实机实现。它由根工程
以隔离的 `ExternalProject` 构建，只负责兼容编排，不拥有顶层配置、数据契约、
海康/GPIO/TensorRT 公共适配器或用户工作流。

## 使用边界

- 日常运行只使用根目录的 `volleyball run` 和 `volleyball capture`；
- `config/base_pipeline.yaml` 只是主线配置投影所需的字段模板，不是部署配置；
- 构建从仓库根目录执行 `cmake --preset nx-legacy-release` 和
  `cmake --build --preset nx-legacy-release --parallel`；
- 标定与硬件验收程序的源码位于 `tools/hardware/`，本目录只提供其 NX 构建环境；
- 新的平台无关代码必须进入 `libs/`、`ports/`、`algorithms/` 或 `services/`，
  不得反向依赖本目录。

当前保留源码按职责分为 `capture/`、`rectify/`、`stereo/`、`fusion/`、
`pipeline/`、`ros/`、`diagnostics/`、`ui/` 和 `utils/`。每迁出一项能力，必须
先完成同批录制数据的输出与性能回归，再删除对应兼容实现。

详细说明统一维护在：

- [迁移状态与删除门禁](../../docs/migration.md)
- [完整图像流水线](../../docs/pipeline/README.md)
- [硬件与标定工具](../../docs/hardware-tools.md)
- [验收标准](../../docs/acceptance.md)
