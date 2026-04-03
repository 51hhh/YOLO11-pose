# 问题速查：PyTorch JetPack 环境恢复

> 环境: JetPack 6.2 · torch 2.5.0a0+872d972e41.nv24.08 · CUDA 12.6

## 问题
编译 torchvision 或 pip install 操作可能破坏 NVIDIA 预编译 PyTorch wheel，
导致 `torch.cuda.is_available()` 返回 `False`。

## 症状
```python
>>> import torch
>>> torch.cuda.is_available()
False  # 应为 True
>>> torch.__version__
'2.x.x+cpu'  # 丢失 CUDA 标识
```

## 修复步骤

### 1. 确认 CUDA Toolkit 正常
```bash
nvcc --version        # 应显示 12.6
nvidia-smi            # 应显示 GPU
```

### 2. 卸载损坏的 torch
```bash
pip uninstall torch torchvision torchaudio -y
pip cache purge
```

### 3. 重装 NVIDIA JetPack wheel
```bash
# 从 NVIDIA 官方下载（需代理）
wget -e use_proxy=yes -e http_proxy=http://192.168.31.30:10000 \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

pip install torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
```

### 4. 验证
```python
import torch
print(torch.__version__)        # 2.5.0a0+872d972e41.nv24.08
print(torch.cuda.is_available()) # True
print(torch.cuda.get_device_name())  # Orin
```

## 预防
- **不要** `pip install torch` — 会覆盖为 CPU 版本
- **不要** 从源码编译 torchvision — 可能破坏 torch CUDA 符号
- torchvision 使用 `pip install torchvision --no-deps` 安装预编译版本
