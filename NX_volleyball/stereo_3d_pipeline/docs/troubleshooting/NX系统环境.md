# 问题速查：NX 系统环境配置

> 硬件: Jetson Orin NX 16GB · JetPack 6.2 · Ubuntu 22.04

## 关键软件版本
| 组件 | 版本 | 路径 |
|---|---|---|
| CUDA | 12.6 | `/usr/local/cuda-12.6` |
| TensorRT | 10.3.0 | `/usr/lib/aarch64-linux-gnu` |
| VPI | 3.2.4 | `/opt/nvidia/vpi3` |
| OpenCV | 4.10 (with CUDA) | 系统安装 |
| PyTorch | 2.5.0a0 nv24.08 | pip |
| 海康 MVS SDK | 3.x | `/opt/MVS` |

## 功耗模式
```bash
# 查看当前模式
sudo nvpmodel -q
# 设置最大性能 (MAXN)
sudo nvpmodel -m 0
sudo jetson_clocks
```

## GPIO / PWM
```bash
# 安装 libgpiod（用户态 GPIO）
sudo apt install libgpiod-dev gpiod
# 查看可用 GPIO
gpioinfo
# DTS overlay 释放 GPIO（如有占用）
# 见 scripts/release-j16-30-gpio.dts
```

## 编译环境
```bash
sudo apt install cmake build-essential libyaml-cpp-dev
# VPI 头文件
sudo apt install libnvvpi3-dev
# ONNX Runtime (可选)
# 从 https://github.com/microsoft/onnxruntime 下载 aarch64 release
```

## 文件同步（Windows→NX）
```powershell
# PowerShell (Windows 端)
scp -r NX_volleyball nvidia@192.168.31.56:/home/nvidia/
# 或使用 scripts/sync_to_nx.ps1
```

## tegrastats 监控
```bash
sudo tegrastats --interval 200 --logfile perf.log
# 关键字段: GR3D(GPU%), RAM, VDD_IN(功耗)
```
