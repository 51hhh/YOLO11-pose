# 模型清单

代码和配置只引用 `model_id`，不保存 `/home/nvidia/...` 等机器绝对路径。`manifest.yaml` 把 ID 映射到项目相对 artifact，并记录输入、精度、目标平台和 SHA-256。

模型文件本身不入库。用于真实数据采集前，`sha256` 必须从实机上的最终 TensorRT engine 计算后固定，同时记录 TensorRT/CUDA/JetPack 版本。

