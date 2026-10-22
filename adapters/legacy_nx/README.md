# Legacy NX Adapter

此目录记录旧 NX/CUDA 工程的适配边界。实际源代码暂时位于 `NX_volleyball/stereo_3d_pipeline`，由根 `CMakeLists.txt` 作为 ExternalProject 构建；主线 Python 负责配置投影、启动、采集 staging 与 v2.1 发布。

新代码不得从 core、ports、algorithms 或 services 反向依赖旧工程。迁移完成并通过同输入回归后，旧实现才可移动或删除。

