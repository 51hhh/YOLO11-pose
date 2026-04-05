# DLA-GPU 混合推理模式指南

## 1. 概述

Jetson Orin NX 配备 2 个 DLA (Deep Learning Accelerator) 核心，独立于 GPU 运行。
将部分算子卸载到 DLA 可以：
- **节省 GPU 算力**（释放 GPU 给立体视差、后处理等任务）
- **提高总体吞吐量**（DLA 和 GPU 可并行执行不同任务）

## 2. 三种混合模式

### 2.1 GPU-Only（基线）
```
所有算子 → GPU
```
- 延迟最低（2.73ms）
- GPU 利用率 100%
- 无 DLA 使用

### 2.2 DLA-Only（全 DLA + GPU Fallback）
```
默认算子 → DLA
不兼容算子 → GPU Fallback
```
- 延迟最高（6.89ms）
- GPU 节省最多（75.5%）
- DLA↔GPU 频繁切换带来 overhead
- **超过 10ms 预算** ❌

### 2.3 DLA-GPU Split（推荐 Split A 方案）
```
backbone (model.0-9, 41% 节点) → DLA0
PSA + neck + head (model.10-23) → GPU
```
- 延迟适中（4.27ms，校准后）
- GPU 节省 33.2%
- 仅 1 次 DLA→GPU 切换（model.9 → model.10）
- **满足 10ms 预算** ✅

## 3. 实现步骤

### 3.1 模型导出（DLA 友好）

使用 `export_yolov11.py` 导出 DLA 友好的 ONNX 模型：

```bash
# 方式1: 完整 DLA 优化（Attention 分解 + 原始 Detect 输出，6 个张量）
python3 scripts/export_yolov11.py --pt model/yolov11n.pt --dla

# 方式2: 仅 Attention 分解（保留标准 Detect 输出 [1,5,8400]）
python3 scripts/export_yolov11.py --pt model/yolov11n.pt --attn-only
```

**关键**: `--dla` 模式将 Softmax 分解为 `max + exp + div`，消除 DLA 不兼容算子。

### 3.2 INT8 校准

使用真实场景图片生成校准缓存：

```bash
python3 scripts/generate_calib_cache.py \
    --onnx model/yolov11n_dla.onnx \
    --images calibration_images \
    --cache model/yolov11n_dla_calib.cache \
    --max-images 200
```

### 3.3 引擎构建（Split A）

**核心命令**：使用 `--layerDeviceTypes` 指定每层设备：

```bash
# Step 1: 生成 GPU 节点列表（model.10+ 放 GPU）
python3 -c "
import onnx
m = onnx.load('model/yolov11n_dla.onnx')
gpu_nodes = []
for n in m.graph.node:
    for p in n.name.split('/'):
        if p.startswith('model.'):
            idx = int(p.split('.')[1])
            if idx >= 10:
                gpu_nodes.append(n.name)
            break
# 保存为 trtexec 参数格式
with open('split_a.txt', 'w') as f:
    f.write(','.join([n+':GPU' for n in gpu_nodes]))
"

# Step 2: 构建 Split A 引擎
SPLIT_A=$(cat split_a.txt)
trtexec --onnx=model/yolov11n_dla.onnx \
    --int8 --fp16 \
    --calib=model/yolov11n_dla_calib.cache \
    --useDLACore=0 --allowGPUFallback \
    --layerDeviceTypes=$SPLIT_A \
    --memPoolSize=workspace:4096MiB \
    --saveEngine=model/v11dla_splitA_calib.engine
```

### 3.4 `--layerDeviceTypes` 参数详解

```
--layerDeviceTypes=<node1>:GPU,<node2>:GPU,...
```

- **默认设备**: 由 `--useDLACore=0` 决定（DLA Core 0）
- **覆盖层**: 用 `nodeName:GPU` 强制指定到 GPU
- **节点名称**: 对应 ONNX 中的 node name（可用 `onnx.load()` 查看）
- **粒度**: 每个 ONNX 节点独立控制

### 3.5 Split 分割点选择原则

| 分割策略 | DLA 层 | GPU 层 | 延迟 | GPU 节省 |
|---|---|---|---|---|
| Split A (推荐) | model.0-9 (backbone) | model.10-23 | 4.27ms | 33.2% |
| Split B | model.0-16 (含 neck) | model.17-23 | 8.51ms | 51.9% |
| Split C | model.0-22 (仅 head GPU) | model.23 | 7.91ms | 57.2% |
| Split D | model.0-10 | model.11-23 | 9.02ms | ~35% |

**选择原则**:
- DLA 擅长: Conv, ReLU, Pooling, BatchNorm, Sigmoid, Mul
- DLA 不擅长: Softmax, GatherElements, Resize (Upsample)
- 每次 DLA↔GPU 切换都有 ~0.2ms overhead
- Split A 仅 1 次切换，且 backbone 全是 DLA-native 运算

## 4. 输出格式差异

### 标准模型 (yolov11n.onnx)
```
1 个输出: [1, 5, 8400]  (cx, cy, w, h, conf)
后处理: 直接提取 bbox + NMS
```

### DLA 优化模型 (yolov11n_dla.onnx)
```
6 个输出 (per-scale cls + bbox):
  output:  [1, 80, 80, 1]   cls stride=8
  969:     [1, 80, 80, 64]  bbox stride=8
  987:     [1, 40, 40, 1]   cls stride=16
  997:     [1, 40, 40, 64]  bbox stride=16
  1015:    [1, 20, 20, 1]   cls stride=32
  1025:    [1, 20, 20, 64]  bbox stride=32

后处理:
  1. sigmoid(cls) → 置信度
  2. DFL decode: softmax(bbox[4×16]) × [0..15] → 4 个偏移量
  3. dist2bbox: anchor ± offset × stride → (cx, cy, w, h)
  4. NMS
```

**DLA 模型比标准模型快 ~1.3ms**，因为 DFL Softmax + anchor decode + concat 被移到了 CPU 后处理中。

## 5. 性能数据

### 引擎基准 (Orin NX Super, MAXN_SUPER, trtexec 300 iters)

| 配置 | Mean | P99 | GPU 内核时间 | GPU 节省 |
|---|---|---|---|---|
| v11dla GPU INT8 (校准) | 2.73ms | 3.96ms | 84.7ms/iter | 0% |
| **v11dla Split A (校准)** | **4.27ms** | **5.59ms** | **56.5ms/iter** | **33.2%** |
| v11dla DLA0 Full | 6.89ms | 10.59ms | 20.8ms/iter | 75.5% |

### 全流水线预算 (10ms 目标)

| 阶段 | 延迟 |
|---|---|
| 相机抓取 | 0.5ms |
| VPI CUDA 双目畸变校正 | 2.84ms |
| **检测 (Split A)** | **4.27ms** |
| 预处理 + 后处理 | 1.3ms |
| ROI 深度计算 | 0.1ms |
| **总计** | **9.01ms** ✅ |

## 6. 常见问题

**Q: DLA 和 GPU 能同时运行不同任务吗？**
A: 可以。DLA 有独立的执行引擎。当 Split A 引擎推理时，DLA 处理 backbone 的同时 GPU 可处理其他 CUDA 任务（如 VPI 立体匹配）。但 TRT enqueueV3 是序列化的——同一个 engine 内部 DLA 和 GPU 层是顺序执行。

**Q: 两个 DLA 核心可以同时用吗？**
A: 可以。每个 DLA 核心独立，可分别运行不同引擎或不同推理请求。但同一个引擎只能绑定一个 DLA 核心。

**Q: INT8 校准对 DLA 有影响吗？**
A: DLA 只支持 INT8 和 FP16。校准缓存中的 scale factor 对 DLA 层同样生效。

**Q: 如何监控 DLA 利用率？**
A: 使用 `jtop` 或 `tegrastats` 查看 DLA 使用率。nsys 可采集详细内核级 profile。

## 7. 文件清单

```
model/
  yolov11n_dla.onnx          # DLA 友好 ONNX（Softmax 分解，6 输出）
  yolov11n_dla_calib.cache    # INT8 校准缓存（200 张真实图片）
  v11dla_splitA_calib.engine  # 校准后 Split A 引擎（推荐部署）
  v11dla_gpu_int8_calib.engine # 校准后 GPU-only 引擎（备选）
scripts/
  export_yolov11.py           # ONNX 导出（支持 --dla / --attn-only）
  generate_calib_cache.py     # INT8 校准缓存生成
  build_v11dla_calib.sh       # 校准引擎构建脚本
```
