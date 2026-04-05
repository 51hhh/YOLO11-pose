# Jetson Orin NX 性能检测工具指南

## 概述

Jetson Orin NX 上可用于 TensorRT / CUDA / DLA 性能分析的核心工具：

| 工具 | 用途 | 安装状态 |
|------|------|----------|
| `nsys` (Nsight Systems) | GPU/DLA/CPU timeline profiling | 自带 (2024.5.4) |
| `trtexec` | TensorRT engine benchmark & layer profiling | 自带 (TRT 10.3) |
| `tegrastats` | 实时 GPU/DLA/CPU/内存/温度监控 | 自带 |
| `jtop` | 交互式系统资源监控 | pip install jetson-stats |
| `nvpmodel` | 功耗/性能模式切换 | 自带 |
| `jetson_clocks` | 锁定最高频率 | 自带 |

> **注意**：`nvvp` (Visual Profiler) 在 JetPack 6.x 已废弃，由 `nsys` 替代。

---

## 1. nsys (Nsight Systems)

### 1.1 基本用法

```bash
# 版本确认
nsys --version
# NVIDIA Nsight Systems version 2024.5.4.34-...

# 基础 profiling (应用级)
nsys profile -o my_report ./my_application

# 指定采样选项
nsys profile \
    -t cuda,nvtx,osrt \        # 追踪 CUDA, NVTX markers, OS Runtime
    --duration=10 \             # 采集 10 秒
    --stats=true \              # 生成统计摘要
    -o profile_output \
    ./my_application
```

### 1.2 TensorRT + DLA Profiling

```bash
# 对 trtexec DLA engine 进行 profiling
nsys profile \
    --stats=true \
    -o dla_profile \
    /usr/src/tensorrt/bin/trtexec \
        --loadEngine=model/yolo26_dla0_int8_640.engine \
        --iterations=200 \
        --warmUp=2000 \
        --dumpProfile
```

### 1.3 关键输出解读

```
# nsys stats 输出示例:
CUDA API Statistics:
  Time(%)  Total Time  Calls  Average  Name
   45.2%   234.5ms     200    1.17ms   cudaStreamSynchronize
   22.1%   114.8ms     200    0.57ms   cudaLaunchKernel
   ...

CUDA Kernel Statistics:
  Time(%)  Total Time  Instances  Average  Name
   28.0%    45.6ms     200        0.23ms   ForeignNode[2]   ← DLA subgraph
   ...
```

### 1.4 报告查看

```bash
# 命令行统计
nsys stats my_report.nsys-rep

# 导出为 JSON (便于脚本处理)
nsys export --type=json --output=report.json my_report.nsys-rep

# GUI 查看 (在 Windows/Linux PC 上安装 Nsight Systems GUI)
# 下载: https://developer.nvidia.com/nsight-systems
# 从 NX 拷贝 .nsys-rep 文件到 PC 打开
scp nvidia@192.168.31.56:/path/to/report.nsys-rep .
```

---

## 2. trtexec

### 2.1 Engine 构建

```bash
TRTEXEC=/usr/src/tensorrt/bin/trtexec

# GPU FP16
$TRTEXEC --onnx=model.onnx --saveEngine=model_gpu_fp16.engine \
    --fp16 --memPoolSize=workspace:4096MiB

# GPU INT8 (需要校准数据或 PTQ)
$TRTEXEC --onnx=model.onnx --saveEngine=model_gpu_int8.engine \
    --int8 --fp16 --memPoolSize=workspace:4096MiB

# DLA INT8 (DLA core 0, 允许 GPU fallback)
$TRTEXEC --onnx=model.onnx --saveEngine=model_dla0_int8.engine \
    --useDLACore=0 --allowGPUFallback --int8 --fp16 \
    --memPoolSize=workspace:4096MiB

# DLA-GPU Hybrid (指定层设备)
$TRTEXEC --onnx=model.onnx --saveEngine=model_hybrid.engine \
    --useDLACore=0 --allowGPUFallback --int8 --fp16 \
    --memPoolSize=workspace:4096MiB \
    --layerDeviceTypes="/model.10/m/m.0/attn/MatMul:GPU,/model.10/m/m.0/attn/Softmax:GPU"
```

### 2.2 Engine Benchmark

```bash
# 基本推理测试
$TRTEXEC --loadEngine=model.engine --iterations=500 --warmUp=3000

# 带 layer profiling 的详细测试
$TRTEXEC --loadEngine=model.engine --iterations=200 \
    --dumpProfile --exportProfile=profile.json

# 关键输出指标:
#   Throughput: xxx qps          ← 吞吐量
#   GPU Compute Time: mean=x.xxms  ← 单帧 GPU 计算时间
#   Total Host Walltime: x.xxms    ← 总延迟(含 H2D/D2H)
```

### 2.3 Layer 信息导出

```bash
# 导出层级信息 (JSON)
$TRTEXEC --onnx=model.onnx --useDLACore=0 --allowGPUFallback --int8 --fp16 \
    --memPoolSize=workspace:4096MiB \
    --dumpLayerInfo --exportLayerInfo=layers.json --skipInference

# 导出层级 timing profile
$TRTEXEC --loadEngine=model.engine --iterations=100 \
    --dumpProfile --exportProfile=timing.json
```

### 2.4 DLA-GPU Hybrid 层控制

```bash
# --layerDeviceTypes 语法: "layerName:GPU" 或 "layerName:DLA"
# 多层用逗号分隔

# 示例: 将注意力层强制到 GPU
$TRTEXEC --onnx=model.onnx --useDLACore=0 --allowGPUFallback --int8 --fp16 \
    --layerDeviceTypes="/model.10/m/m.0/attn/qkv/conv/Conv:GPU,\
/model.10/m/m.0/attn/Split:GPU,\
/model.10/m/m.0/attn/Transpose:GPU,\
/model.10/m/m.0/attn/MatMul:GPU,\
/model.10/m/m.0/attn/Softmax:GPU" \
    --saveEngine=hybrid.engine
```

---

## 3. tegrastats

### 3.1 基本用法

```bash
# 实时监控 (1 秒刷新)
tegrastats

# 自定义刷新间隔 (毫秒)
tegrastats --interval 500

# 输出到文件
tegrastats --interval 1000 --logfile /tmp/tegra_log.txt &
```

### 3.2 输出字段说明

```
RAM 6543/15823MB   ← 内存使用/总量
GR3D_FREQ 76%     ← GPU 利用率
NVDLA0_FREQ 100%  ← DLA0 利用率
NVDLA1_FREQ 85%   ← DLA1 利用率
CPU [20%@2201]     ← CPU 使用率@频率(MHz)
tj: 52C            ← 芯片结温(Thermal Junction)
VDD_CPU_GPU_CV 4500mW  ← CPU/GPU/CV 功耗
```

---

## 4. nvpmodel & jetson_clocks

### 4.1 性能模式

```bash
# 查看当前模式
nvpmodel -q

# 设置为最高性能 (MAXN_SUPER for Orin NX)
sudo nvpmodel -m 0

# 可用模式查看
nvpmodel -p --verbose
```

### 4.2 频率锁定

```bash
# 锁定所有时钟到最高频率 (benchmark 必须)
sudo jetson_clocks

# 查看当前频率状态
sudo jetson_clocks --show

# 恢复默认 (dynamic scaling)
sudo jetson_clocks --restore
```

---

## 5. VPI Profiling

### 5.1 VPI Python 基准测试

```python
import vpi, numpy as np, time

W, H = 1280, 720
src = vpi.asimage(np.random.randint(0, 255, (H, W), dtype=np.uint8))
warp = vpi.WarpMap(vpi.WarpGrid((W, H)))

for backend in [vpi.Backend.CUDA, vpi.Backend.VIC]:
    # warmup
    for _ in range(30):
        with backend:
            out = src.remap(warp)
        out.cpu()
    # benchmark
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        with backend:
            out = src.remap(warp)
        out.cpu()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"{backend}: avg={sum(times)/len(times):.3f}ms min={times[0]:.3f}ms")
```

### 5.2 VPI 支持的后端

| 后端 | 硬件 | 说明 |
|------|------|------|
| `VPI_BACKEND_CUDA` | GPU CUDA Cores | 通用计算，延迟最低 |
| `VPI_BACKEND_PVA` | PVA (Programmable Vision Accelerator) | 图像处理专用，功耗低 |
| `VPI_BACKEND_VIC` | VIC (Video Image Compositor) | 视频处理专用，带宽优化 |
| `VPI_BACKEND_NVENC` | NVENC (Video Encoder) | 编码专用 |
| `VPI_BACKEND_CPU` | CPU | 最慢，调试用 |

> **Remap 操作支持**: CUDA, VIC, CPU（不支持 PVA）

### 5.3 VPI 在 Pipeline 中的硬件分配

当前 stereo_3d_pipeline 中的 VPI 调用：

| 操作 | 后端 | 硬件 | 延迟 |
|------|------|------|------|
| Remap (双目校正) L+R | CUDA | GPU | ~2.8ms (dual) |
| ConvertImageFormat (NV12→Gray) | CUDA | GPU | ~0.1ms |
| TemporalNoiseReduction | CUDA | GPU | ~0.5ms (如启用) |

---

## 6. 常用 Benchmark 流程

### 完整性能测试流程

```bash
# 1. 设置最高性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 2. 预热系统 (运行 5 秒空闲)
sleep 5

# 3. TRT Engine benchmark
/usr/src/tensorrt/bin/trtexec --loadEngine=model.engine \
    --iterations=500 --warmUp=3000 --avgRuns=10

# 4. Pipeline 全链路测试
cd /home/nvidia/NX_volleyball/stereo_3d_pipeline
timeout 15 ./build/stereo_pipeline -c config/pipeline_triple.yaml

# 5. nsys 全链路 profiling
nsys profile --stats=true -o pipeline_profile \
    timeout 10 ./build/stereo_pipeline -c config/pipeline_triple.yaml

# 6. 同时监控系统状态
tegrastats --interval 200 --logfile /tmp/bench_tegra.log &
```

### Benchmark 注意事项

1. **始终锁定频率**：`jetson_clocks` 必须在测试前执行，否则 DVFS 导致结果不稳定
2. **充分预热**：TRT engine 首次推理较慢（JIT 优化），至少 warmUp 2-3 秒
3. **温度影响**：长时间运行会导致 thermal throttling，关注 `tj` 温度
4. **DLA 独立计时**：DLA 延迟不反映在 `GPU Compute Time` 中，需要用 `nsys` 或 `--dumpProfile` 查看
5. **内存带宽竞争**：DLA 和 GPU 共享 LPDDR5 带宽，同时使用会互相影响
