# DLA-GPU 混合推理优化指南

## Jetson Orin NX Super 16GB — 排球检测

### 目录
1. [架构概述](#架构概述)
2. [DLA 限制与能力矩阵](#dla-限制与能力矩阵)
3. [格式转换优化](#格式转换优化)
4. [双 DLA 交替流水线](#双dla交替流水线)
5. [INT8 校准优化](#int8-校准优化)
6. [全组合测试矩阵](#全组合测试矩阵)
7. [推荐配置](#推荐配置)

---

## 架构概述

### 硬件资源 (MAXN_SUPER 模式)
| 资源 | 规格 |
|------|------|
| GPU | Ampere, 8 SM, 1024 CUDA Cores |
| DLA | 2 × DLA 2.0 Cores |
| 内存 | 16GB LPDDR5 统一内存 |
| TensorRT | 10.3.0 |
| JetPack | 6.2 (L4T R36.4.7) |

### 检测流水线架构
```
Stage0: 抓取+VPI校正 (PVA)
  │
  ├─→ Stage1: TRT检测 (DLA0/DLA1/GPU)  → CUDA Stream 独立
  │
  ├─→ Stage2: VPI立体匹配 (GPU)         → CUDA Stream 独立
  │
  └─→ Stage3: 融合+3D坐标 (GPU)         → 等待 Stage1+2 完成
```

---

## DLA 限制与能力矩阵

### Orin DLA 2.0 支持的算子

| 算子 | DLA支持 | 备注 |
|------|---------|------|
| Conv2d | ✅ | 核心计算 |
| BatchNorm | ✅ | 通常融合进Conv |
| ReLU/SiLU/Sigmoid | ✅ | 激活函数 |
| MaxPool/AvgPool | ✅ | |
| Concat | ⚠️ | **仅channel轴**, 所有输入需相同空间维度 |
| Resize | ⚠️ | nearest: int scale [1,32]; bilinear: [1,4] |
| Softmax | ✅ | axis dim ≤ 1024 (DFL的dim=16 安全) |
| Add/Mul | ✅ | 逐元素 |
| **MatMul** | ❌ | **不支持 — 注意力机制的核心** |
| Reshape | ❌ | 动态shape不支持 |
| Sub (broadcast) | ❌ | 广播减法 |
| Gather/Shape | ❌ | 动态索引 |

### 模型 DLA 安全分析

| 模型 | model.0-9 | model.10 | model.11+ |
|------|-----------|----------|-----------|
| yolov11n_dla | ✅ Conv/Pool/Sigmoid | ❌ **MatMul** (PSA注意力) | 混合 |
| yolo26 | ✅ Conv/Pool/Sigmoid | ❌ **MatMul** (注意力) | model.22也有MatMul |
| yolov10n | ✅ Conv/Pool/Sigmoid | ❌ **MatMul** (注意力) | 混合 |

**结论**: 所有3个模型的 model.10 均包含 MatMul (注意力机制)，DLA 无法执行。
backbone (model.0-9) 是 DLA 安全的。

---

## 格式转换优化

### DLA 原生数据格式
- **FP16 模式**: `kCHW16` (通道按16分组)
- **INT8 模式**: `kCHW32` (通道按32分组)
- **GPU 标准格式**: `NCHW` (线性排列)

### CopyNode 开销
当数据在 DLA↔GPU 之间传输时，TensorRT 自动插入 `CopyNode` 进行格式转换：

```
GPU(NCHW) → CopyNode(→kCHW16) → DLA层 → CopyNode(→NCHW) → GPU层
```

每次格式转换约增加 **0.1-0.3ms** 延迟。

### 优化策略

#### 1. 减少 DLA↔GPU 切换次数
使用 `--allowGPUFallback` 时，TRT 会自动尝试将连续的 DLA 不兼容层合并为一个 GPU 段，
而不是每遇到不兼容层就切换一次。

#### 2. 显式 I/O 格式提示
```bash
# 指定输入格式为 DLA 原生格式，避免输入端转换
trtexec --onnx=model.onnx \
    --fp16 --useDLACore=0 --allowGPUFallback \
    --inputIOFormats=fp16:chw16 \
    --outputIOFormats=fp16:chw
```

**注意**: 这要求上游代码也使用对应格式提供数据。

#### 3. DLA SRAM 调优
```bash
# Orin DLA SRAM 大小（需要实测可用值）
trtexec --memPoolSize=dlaSRAM:524288  # 512KB
```
> ⚠️ 在 NX 上 `dlaSRAM:1048576` (1MB) 导致错误，需使用更小值。

#### 4. 层设备指定 (高级)
```bash
# 精确控制每层运行位置
trtexec --layerDeviceTypes=/model.0:DLA,/model.1:DLA,...,/model.10:GPU
```

### 实测格式转换影响
| 配置 | 额外延迟 | 原因 |
|------|----------|------|
| GPU-only INT8 | 0ms (基线) | 无DLA切换 |
| DLA0 FP16 + GPU fallback | +1-2ms | backbone→attention 切换 |
| DLA0 INT8 + GPU fallback | +0.5-1ms | INT8 切换开销较小 |

---

## 双 DLA 交替流水线

### 设计原理
Orin NX 有 2 个独立 DLA 核心，可以并行工作：

```
时间线：
DLA0: ──[Frame N backbone]────────[Frame N+2 backbone]──
DLA1: ────[Frame N+1 backbone]────────[Frame N+3 backbone]──
GPU:  ──[N head]──[N+1 head]──[N+2 head]──[N+3 head]──
```

### 现有实现状态
管道已实现完整的双 DLA 支持：

```cpp
// pipeline.h - 三检测器实例
std::unique_ptr<TRTDetector> detector_;    // DLA0
std::unique_ptr<TRTDetector> detector1_;   // DLA1
std::unique_ptr<TRTDetector> detector2_;   // GPU (三后端模式)

// 帧分配策略
TRTDetector* getDetector(int frame_id) const {
    if (triple_backend) return frame_id % 3 选择;
    if (dual_dla)       return frame_id & 1 ? detector1_ : detector_;
    return detector_;
}
```

### Bug 修复记录 (已完成)
1. **stage3_fuse collect 使用错误检测器**
   - 修复前: `detector_->collect()` (始终用DLA0)
   - 修复后: `getDetector(slot.frame_id)->collect()`

2. **evtDetectDone 记录在错误的流上**
   - 修复前: `cudaEventRecord(evtDetectDone, cudaStreamDLA)` (始终DLA0流)
   - 修复后: `cudaEventRecord(evtDetectDone, getDLAStream(slot.frame_id))`

3. **evtRectDone 等待在错误的流上**
   - 修复前: `cudaStreamWaitEvent(cudaStreamDLA, evtRectDone, 0)` (始终DLA0流)
   - 修复后: `cudaStreamWaitEvent(getDLAStream(slot.frame_id), evtRectDone, 0)`

### 配置方式
```yaml
# pipeline.yaml
detect:
  engine_file: "model/v11dla_dla0_fp16.engine"     # DLA0 引擎
  engine_file_dla1: "model/v11dla_dla1_fp16.engine" # DLA1 引擎
  dual_dla: true
  use_dla: true
  dla_core: 0
```

### 性能预期
| 模式 | 单帧延迟 | 吞吐量 | GPU占用 |
|------|----------|--------|---------|
| GPU-only INT8 | 3.2ms | 312 FPS | 100% |
| 单DLA + GPU | 8-22ms | 45-125 FPS | ~40% |
| 双DLA 交替 + GPU | 8-22ms | 90-200 FPS | ~30% |

> 双 DLA 不降低单帧延迟，但提高吞吐量并降低 GPU 占用。

---

## INT8 校准优化

### 校准算法对比
| 算法 | 特点 | 适用场景 |
|------|------|----------|
| **EntropyCalibrator2** | 基于KL散度最小化 | 通用，推荐默认 |
| **MinMaxCalibrator** | 基于激活值范围 | 数值范围稳定的模型 |

### 校准实践
1. **使用真实数据**: 500 张排球场景图像 (calib_500/)
2. **充足样本量**: 建议 ≥ 200 张
3. **多样性**: 不同角度、光照、遮挡
4. **模型专属**: 每个 ONNX 模型单独校准

### 校准缓存
```
model/
├── yolov11n_dla_calib_entropy2.cache  # Entropy2 校准
├── yolov11n_dla_calib_minmax.cache    # MinMax 校准
├── yolov11n_attn_calib_entropy2.cache
├── yolov11n_attn_calib_minmax.cache
├── yolo26_calib_entropy2.cache
└── yolo26_calib_minmax.cache
```

### DLA INT8 精度问题
**已知问题**: DLA INT8 精度显著低于 GPU INT8。
- GPU INT8: 74% 检测率 (split A: 12%)
- DLA FP16: 96% 检测率

**根因**: DLA 的 INT8 量化可能是 per-tensor 而非 per-channel，
精度损失在 backbone 特征图积累后传播到检测头。

**缓解措施**:
1. DLA 层使用 FP16，GPU fallback 层使用 INT8 (混合精度)
2. 优化校准数据集和算法
3. 关键层（如第一个 Conv 和输出头）强制 FP16

---

## 全组合测试矩阵

### 测试维度
- **模型**: yolov11n_dla, yolov11n_attn, yolo26, yolov10n
- **精度**: FP16, INT8 (entropy2/minmax/legacy)
- **设备**: GPU-only, DLA0+GPU, DLA1+GPU
- **指标**: 检测率(%), 平均得分, 延迟(ms), GPU占用(%)

### 测试结果
> 见 `model/matrix_results/accuracy_report.json` 和 `build_report.txt`

| # | 引擎 | 延迟(ms) | 检测率 | 评分 | 状态 |
|---|------|----------|--------|------|------|
| | _待测试数据填充_ | | | | |

---

## 推荐配置

### 优先级1: 最高精度 (检测率 > 90%)
```
DLA0 FP16 + GPU fallback: ~22ms (太慢)
GPU FP16 + 后处理优化
```

### 优先级2: 延迟 < 7.5ms
```
GPU INT8 (entropy2): ~3.2ms
```

### 优先级3: GPU 占用最低
```
双DLA FP16 交替 + GPU INT8 head
```

### 最佳平衡 (待测试验证)
```
DLA0 FP16 backbone + GPU INT8 head (混合精度)
目标: 检测率>85%, 延迟<7.5ms, GPU省30%+
```

---

*文档版本: v1.0*  
*最后更新: 2026-04-05*  
*平台: Jetson Orin NX Super 16GB, JetPack 6.2, TRT 10.3.0*
