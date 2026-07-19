# YOLO26 TensorRT 导出指南 (Jetson Orin NX + TRT 10.3)

## 环境信息

| 项目 | 值 |
|------|-----|
| 设备 | Jetson Orin NX Super 16GB |
| JetPack | 6.x (CUDA 12.6, TensorRT 10.3.0) |
| Python | 3.10.12 |
| ultralytics | 8.4.33 |
| torch | 2.10.0 (JetPack 6 兼容版) |
| 模型 | yolo26n.pt (nc=1, volleyball) |

## 当前生产配置

- **Engine**: `yolo26_gpu_fp16.engine` (8.1 MB)
- **精度**: FP16 (GPU-only, 无量化)
- **性能**: 224 qps / 4.44ms GPU compute / 管线稳定 ~100 FPS
- **输出格式**: (1, 5, 8400) → 4 box + 1 class_score, 8400 anchors

## 导出流程 (已验证可用)

现有 engine 通过以下方式生成：

```bash
# 1. ONNX 导出 (需设置 head.end2end=False 获取 raw 输出)
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.model.model[-1].end2end = False
model.export(format='onnx', imgsz=640, simplify=True, opset=17)
"

# 2. trtexec 构建 FP16 engine
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo26_sim.onnx \
  --saveEngine=yolo26_gpu_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096MiB
```

## ⚠️ TRT 10.3 兼容性问题

**ultralytics 8.4.33 的 `model.export(format='engine')` 无法直接在 TRT 10.3 上使用。**

### 原因

ultralytics 8.4.33 改变了 attention 子图拓扑：
- **可用版本** (yolo26_sim.onnx): PE Conv 在 attention 计算完成**之后**
- **新导出版本**: PE Conv 插入到 Q×K^T 和 Softmax **之间**

TRT 10.3 无法为新拓扑找到可用 tactic，报错：
```
Could not find any implementation for node
{ForeignNode[/model.10/m/m.0/attn/Split_16.../model.10/m/m.0/attn/Add]}
```

此外 ultralytics 内部调用 TRT builder 生成的 engine 包含需要 LLVM JIT 的层，
加载时触发 `LLVM ERROR: out of memory`。

### 解决方案

保留现有 `yolo26_sim.onnx`（早期兼容版本导出），使用 `trtexec` 构建 engine。
待升级 JetPack 7 / TRT 10.7+ 后可重新尝试标准 ultralytics 导出。

## 性能测试数据 (2026-05-27)

### Pipeline 时序

| Stage | Avg(ms) | Min(ms) | Max(ms) |
|-------|---------|---------|---------|
| Stage0_Process (Grab+TNR+Rect) | 4.98 | 0.08 | 10.85 |
| Stage1_DetectSubmit (YOLO infer) | 4.50 | 2.50 | 105.74 |
| Stage1_WaitRect | 0.02 | 0.00 | 0.55 |
| Stage2_WaitDetect | 0.01 | 0.00 | 0.23 |
| Stage2_WaitYOLOComplete | 0.38 | 0.00 | 7.53 |
| Stage2_ROIMatchFuse | 0.10 | 0.04 | 0.89 |
| Stage0_WaitGrab | 0.24 | 0.00 | 11.20 |

### GPU 利用率 (tegrastats)

| 时段 | GR3D_FREQ | 功耗 (VDD_IN) | GPU温度 |
|------|-----------|--------------|---------|
| 初始化 | 22% | 7W | 56°C |
| 稳态运行 | 78-91% | 18-19W | 61°C |

### 瓶颈分析

1. **Stage1_DetectSubmit (4.50ms avg)**: YOLO26 FP16 推理，占主导
2. **Stage0_Process (4.98ms avg)**: 图像采集+TNR+校正，与检测并行不阻塞
3. **管线吞吐**: 两个瓶颈阶段并行执行，100FPS 触发率下无积压
4. **GPU 利用率 78-91%**: 合理范围，推理+VPI预处理共享GPU
5. **Max spike (105.74ms)**: 极少数帧的调度抖动，不影响平均帧率

## 文件清单 (NX: /home/nvidia/NX_volleyball/model/)

| 文件 | 大小 | 用途 |
|------|------|------|
| yolo26_gpu_fp16.engine | 8.1M | **生产用** TRT FP16 engine |
| yolo26_sim.onnx | 9.3M | TRT 兼容 ONNX (重建 engine 用) |
| yolo26n.pt | 5.2M | PyTorch 源模型 |

## 依赖安装备注

NX 无外网，需离线安装：
```bash
# torch 2.10.0 (本地下载后 scp)
pip3 install /tmp/torch-2.10.0-cp310-cp310-linux_aarch64.whl
pip3 install /tmp/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl

# cuDSS (torch 2.10 运行时依赖)
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/libcudss0-cuda-12_0.7.1.4-1_arm64.deb

# 设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/libcudss/12:$LD_LIBRARY_PATH
```
