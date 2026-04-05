# DLA-GPU 混合推理方案评估

> 测试日期：2026-04-04 | 平台：Jetson Orin NX Super 16GB | JetPack 6.2 | TensorRT 10.3  
> 模型：yolo26.onnx (YOLO11n-pose variant, 640x640, Pre-NMS, 402 ONNX nodes)

## 1. 背景与目标

### 1.1 核心需求

| 指标 | 要求 | 说明 |
|------|------|------|
| 单帧延迟 | **< 10ms** | 完整 pipeline：抓取 → 校正 → 检测 → 融合 |
| 帧率 | 100 Hz | 与 PWM 相机触发同步 |
| GPU 负载 | **尽可能低** | GPU 需保留算力给后续功能（立体匹配、轨迹预测等） |
| 热控 | 长时间稳定运行 | GPU 重载导致发热，影响系统稳定性 |

### 1.2 问题分析

纯 DLA 推理 yolo26 模型时，两个 attention 模块（`model.10` 和 `model.22`）在 DLA 上耗时极高，
nsys profiling 显示：

```
ForeignNode[2] (model.10 attention body): 4.56ms  28.0%  ← 瓶颈
ForeignNode[4] (model.22 attention body): 4.59ms  28.2%  ← 瓶颈
两者合计: 9.15ms = 56% of total 16.26ms
```

**核心思路**：将 DLA 执行效率低的 attention 层强制分配到 GPU，保留 backbone/neck 在 DLA 上，
实现 DLA-GPU 混合推理，降低总延迟同时减少 GPU 负载。

---

## 2. 模型结构分析

### 2.1 ONNX 模型结构 (402 nodes)

```
model.0-model.9   : Backbone (Conv + C3k2 blocks)      [DLA 兼容]
model.10           : C2PSA block (含 PSA attention)      [DLA 兼容但慢]
model.11-model.21  : Neck (SPPF + Upsample + C3k2)     [DLA 兼容]
model.22           : C2PSA block (含 PSA attention)      [DLA 兼容但慢]
model.23           : Detection Head (Conv + Concat + decode) [DLA 兼容]
```

### 2.2 Attention 节点 (36 个 ONNX nodes)

```
model.10 attention: /model.10/m/m.0/attn/* (19 nodes)
  - qkv Conv → Reshape → Split → Transpose → MatMul → Softmax
  - → Transpose → MatMul → Reshape → pe Conv → Add → proj Conv

model.22 attention: /model.22/m.0/m.0.1/attn/* (17 nodes)
  - 结构相同，处理更大 feature map
```

DLA 为何慢：PSA attention 内的 Softmax + 大矩阵 Reshape + Transpose 操作在 DLA 上
需要多次 reformatting（DLA↔DRAM 数据搬运），每次搬运约 0.3-0.5ms。

---

## 3. Benchmark 数据（实测）

### 3.1 TRT Engine 推理延迟 (trtexec, 500 iterations)

| 配置 | GPU Compute Mean | Min | Max | Throughput | GPU 负载 |
|------|-----------------|-----|-----|------------|----------|
| **GPU INT8 640** | **2.929ms** | 2.864ms | 4.170ms | 339 fps | 100% |
| GPU FP16 640 | 3.566ms | 3.438ms | 4.644ms | 280 fps | 100% |
| DLA0 INT8 640 | 15.666ms | 15.600ms | 15.992ms | 62 fps | ~5% |
| DLA0 FP16 640 | 27.669ms | 27.573ms | 28.303ms | 36 fps | ~5% |
| **Hybrid A INT8** | **4.863ms** | 4.802ms | 4.987ms | 192 fps | ~50% |
| Hybrid B INT8 | 5.671ms | 5.613ms | 5.815ms | 171 fps | ~30% |

### 3.2 Hybrid 方案说明

| 方案 | 策略 | GPU 层 | DLA 层 |
|------|------|--------|--------|
| **Hybrid A** | attention → GPU, 其余 → DLA | 36 attn nodes | backbone + neck + head |
| **Hybrid B** | attention + head → GPU, backbone+neck → DLA | 36 attn + 99 head nodes | backbone + neck |
| Hybrid C | backbone → DLA, 其余 → GPU | backbone (113 nodes) | 289 non-backbone | 构建超时 |

### 3.3 VPI Remap 延迟 (1280x720, 200 iterations)

| 后端 | 单路 avg | 双路(L+R) avg | 双路 min | 双路 p99 | 硬件 |
|------|----------|--------------|----------|----------|------|
| **CUDA** | 1.620ms | **2.844ms** | 1.891ms | 8.823ms | GPU CUDA Cores |
| VIC | 3.492ms | 6.649ms | 5.796ms | 8.373ms | VIC 专用硬件 |

> **结论**：VPI Remap 在 CUDA 后端最快（CUDA 比 VIC 快 2.3×），使用 GPU 执行。  
> VIC 虽然不占 GPU，但绝对延迟更高，不适合 10ms 延迟目标。  
> **Remap 不支持 PVA 后端**。

### 3.4 VPI 硬件说明

| VPI 操作 | 支持的后端 | 当前 Pipeline 使用 | 硬件 |
|----------|-----------|-------------------|------|
| Remap (双目校正) | **CUDA**, VIC, CPU | CUDA | GPU |
| ConvertImageFormat | CUDA, VIC, CPU | CUDA | GPU |
| TemporalNoiseReduction | CUDA | CUDA | GPU |
| StereoDisparity | CUDA, PVA+NVENC | CUDA | GPU |

> PVA (Programmable Vision Accelerator) 是 Jetson 独立硬件加速器，
> 但 VPI 的 Remap 操作不支持 PVA。当前 pipeline 所有 VPI 操作均在 GPU 上执行。

---

## 4. 完整 Pipeline 延迟估算

### 4.1 单帧延迟分解

```
┌─────────────────┬──────────┬──────────┬─────────────────────────────┐
│ Stage           │ Hardware │ Time(ms) │ Notes                       │
├─────────────────┼──────────┼──────────┼─────────────────────────────┤
│ Camera Grab     │ USB3 DMA │ ~0.50    │ Hikvision MV-CA016 BayerRG8 │
│ VPI Remap (×2)  │ GPU CUDA │ ~2.84    │ 1440×1080 → 1280×720 双路   │
│ TRT Preprocess  │ GPU      │ ~1.00    │ Letterbox resize 720→640    │
│ TRT Inference   │ varies   │ varies   │ ← 关键变量                   │
│ TRT Postprocess │ GPU      │ ~0.30    │ Decode boxes + threshold    │
│ ROI Stereo+Fuse │ GPU      │ ~0.10    │ Multi-point stereo matching │
├─────────────────┼──────────┼──────────┼─────────────────────────────┤
│ Fixed Overhead  │          │ ~4.74    │ 不含 inference               │
└─────────────────┴──────────┴──────────┴─────────────────────────────┘
```

### 4.2 各方案总延迟

| 方案 | Inference(ms) | Total(ms) | < 10ms? | GPU 负载估计 |
|------|---------------|-----------|---------|-------------|
| **GPU INT8** | 2.93 | **7.67** | **YES** | ~100% (最大负载) |
| GPU FP16 | 3.57 | 8.31 | YES | ~100% |
| **Hybrid A** | 4.86 | **9.60** | **YES** (临界) | ~50% |
| Hybrid B | 5.67 | 10.41 | **NO** | ~30% |
| DLA0 INT8 | 15.67 | 20.41 | NO | ~5% |
| DLA0 FP16 | 27.67 | 32.41 | NO | ~5% |

### 4.3 GPU 时间占用分析

在 100Hz pipeline 中，每帧 10ms 时间窗口内的 GPU 占用：

| 方案 | GPU 总占用 | GPU 空闲 | 可用于其他任务 |
|------|-----------|----------|--------------|
| GPU INT8 | ~7.17ms | ~2.83ms (28%) | 立体匹配受限 |
| **Hybrid A** | ~5.04ms | **~4.96ms (50%)** | **充足** |
| DLA INT8 | ~1.40ms | ~8.60ms (86%) | 极充裕 |

> Hybrid A 将 GPU 占用从 7.17ms 降到 5.04ms，释放约 2ms GPU 时间给其他任务。

---

## 5. 方案对比与建议

### 5.1 方案 A: GPU INT8（最低延迟）

```
优点: ✅ 最低延迟 7.67ms, 远低于 10ms
      ✅ 实现最简单, 无需 DLA
      ✅ 延迟方差最小 (2.86-4.17ms)
缺点: ❌ GPU 100% 用于检测
      ❌ 高负载发热严重
      ❌ 无法为后续功能预留 GPU 资源
```

### 5.2 方案 B: Hybrid A（推荐 - DLA backbone + GPU attention）

```
优点: ✅ 总延迟 9.60ms, 刚好满足 10ms
      ✅ GPU 负载降至 ~50%, 释放 ~5ms GPU 时间
      ✅ DLA 处理 backbone 减少 GPU 发热
      ✅ 单引擎文件, pipeline 无需大改
缺点: ⚠️ 延迟裕量仅 0.4ms, 偶尔可能超 10ms
      ⚠️ DLA↔GPU 数据传输增加额外开销
      ⚠️ 需要 --layerDeviceTypes 构建混合 engine
```

### 5.3 方案 C: Dual DLA + GPU 三路轮换（高吞吐）

```
优点: ✅ 吞吐量可达 ~81fps (实测)
      ✅ GPU 负载仅 ~33% (每 3 帧处理 1 帧)
缺点: ❌ 单帧最大延迟 = DLA 延迟 = 16ms, 不满足 10ms
      ❌ 适合吞吐优先场景, 不适合实时轨迹预测
```

### 5.4 推荐方案

**短期（立即可用）**: GPU INT8 — 7.67ms 单帧延迟，满足 10ms 且有 2.3ms 裕量

**中期（推荐）**: Hybrid A — 编译混合 engine，9.6ms 单帧延迟，GPU 负载降 50%

**优化路径**:
1. 先用 GPU INT8 保证功能正确
2. 构建 Hybrid A engine (`yolo26_hybrid_a_int8.engine` 已在 NX 上生成)
3. Pipeline 中直接替换 engine 文件即可（Hybrid engine 对 Pipeline 透明）
4. 后续可探索预处理优化（将 Letterbox 移至 DLA/VIC）进一步降低 GPU 占用

---

## 6. Hybrid Engine 构建方法

### 6.1 构建命令

```bash
# 获取 attention 层名称
python3 -c "
import onnx
m = onnx.load('yolo26.onnx')
attn = [n.name for n in m.graph.node if 'attn' in n.name.lower()]
print(','.join([f'{n}:GPU' for n in attn]))
"

# 构建 Hybrid A engine (attention → GPU, rest → DLA)
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolo26.onnx \
    --useDLACore=0 --allowGPUFallback \
    --int8 --fp16 \
    --memPoolSize=workspace:4096MiB \
    --layerDeviceTypes="\
/model.10/m/m.0/attn/qkv/conv/Conv:GPU,\
/model.10/m/m.0/attn/Constant:GPU,\
/model.10/m/m.0/attn/Reshape:GPU,\
/model.10/m/m.0/attn/Split:GPU,\
/model.10/m/m.0/attn/Transpose:GPU,\
/model.10/m/m.0/attn/MatMul:GPU,\
/model.10/m/m.0/attn/Constant_1:GPU,\
/model.10/m/m.0/attn/Mul:GPU,\
/model.10/m/m.0/attn/Softmax:GPU,\
/model.10/m/m.0/attn/Transpose_1:GPU,\
/model.10/m/m.0/attn/MatMul_1:GPU,\
/model.10/m/m.0/attn/Constant_2:GPU,\
/model.10/m/m.0/attn/Constant_3:GPU,\
/model.10/m/m.0/attn/Reshape_1:GPU,\
/model.10/m/m.0/attn/Reshape_2:GPU,\
/model.10/m/m.0/attn/pe/conv/Conv:GPU,\
/model.10/m/m.0/attn/Add:GPU,\
/model.10/m/m.0/attn/proj/conv/Conv:GPU,\
/model.22/m.0/m.0.1/attn/qkv/conv/Conv:GPU,\
/model.22/m.0/m.0.1/attn/Constant:GPU,\
/model.22/m.0/m.0.1/attn/Reshape:GPU,\
/model.22/m.0/m.0.1/attn/Split:GPU,\
/model.22/m.0/m.0.1/attn/Transpose:GPU,\
/model.22/m.0/m.0.1/attn/MatMul:GPU,\
/model.22/m.0/m.0.1/attn/Constant_1:GPU,\
/model.22/m.0/m.0.1/attn/Mul:GPU,\
/model.22/m.0/m.0.1/attn/Softmax:GPU,\
/model.22/m.0/m.0.1/attn/Transpose_1:GPU,\
/model.22/m.0/m.0.1/attn/MatMul_1:GPU,\
/model.22/m.0/m.0.1/attn/Constant_2:GPU,\
/model.22/m.0/m.0.1/attn/Constant_3:GPU,\
/model.22/m.0/m.0.1/attn/Reshape_1:GPU,\
/model.22/m.0/m.0.1/attn/Reshape_2:GPU,\
/model.22/m.0/m.0.1/attn/pe/conv/Conv:GPU,\
/model.22/m.0/m.0.1/attn/Add:GPU,\
/model.22/m.0/m.0.1/attn/proj/conv/Conv:GPU" \
    --saveEngine=yolo26_hybrid_a_int8.engine
```

### 6.2 Pipeline 集成

Hybrid engine 对 Pipeline 完全透明——它仍然是单个 `.engine` 文件。
只需在 YAML 配置中指定该 engine 路径即可：

```yaml
detector:
  engine_path: "/home/nvidia/NX_volleyball/model/yolo26_hybrid_a_int8.engine"
  input_size: 640
  use_dla: false       # hybrid engine 内部已包含 DLA 调度
  confidence_threshold: 0.5
```

> **注意**: hybrid engine 内部由 TRT runtime 自动调度 DLA 和 GPU，
> 无需 Pipeline 代码处理 DLA stream。设置 `use_dla: false` 即可。

---

## 7. 数据来源

所有数据通过以下工具在 NX 上采集：

- **nsys 2024.5.4**: DLA layer timing 分析
- **trtexec**: Engine benchmark (500 iterations, 3s warmup)
- **VPI Python 3.2.4**: Remap backend 对比 (300 iterations)
- **Pipeline perf**: 实际 pipeline 运行统计 (15s 运行)

测试环境：MAXN_SUPER 模式，`jetson_clocks` 锁频，温度 45-55°C。

原始数据: `/home/nvidia/NX_volleyball/benchmark_results.json`
