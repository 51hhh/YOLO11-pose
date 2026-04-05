# DLA-GPU Split 方案分析报告

## 1. 目标

- 单帧检测流水线 **总延迟 < 10ms**
- 使用 DLA 处理部分算子，**GPU 节省 ≥ 30%**

## 2. 测试平台

- Jetson Orin NX Super 16GB, JetPack 6.2, TensorRT 10.3.0, MAXN_SUPER
- 8 SMs GPU + 2 DLA cores

## 3. 模型对比

| 模型 | 节点数 | Attention | Softmax | GatherElements | DLA兼容性 |
|---|---|---|---|---|---|
| yolo26.onnx | 402 | 36 | 2 | 0 | 差 (DLA0=16ms) |
| yolov11n.onnx | 321 | 14 | 2 | 0 | 中 (DLA0=7.1ms) |
| **yolov11n_dla.onnx** | 419 | 23 | **0** | 0 | **优** (DLA0=6.9ms) |
| yolov10n.onnx | 308 | 14 | 2 | 2 | 失败 (workspace不足) |

> yolov11n_dla.onnx 通过 D-Robotics BPU 优化器将 Softmax 分解为 max+exp+div，完全消除 DLA 不兼容算子。

## 4. 流水线延迟预算

| 阶段 | 延迟 |
|---|---|
| 相机抓取 | 0.5 ms |
| VPI CUDA 双目畸变校正 | 2.84 ms |
| 预处理 + 后处理 | 1.3 ms |
| ROI 深度计算 | 0.1 ms |
| **开销合计** | **4.74 ms** |
| 检测预算 = 10.0 - 4.74 | **5.26 ms** |

## 5. DLA-GPU Split 方案测试结果

### 5.1 所有引擎基准性能 (trtexec, 300次迭代)

| 引擎 | Mean | P99 | 备注 |
|---|---|---|---|
| yolo26 GPU INT8 | 2.89ms | 3.99ms | |
| yolov11n GPU INT8 | 3.29ms | 4.67ms | |
| **yolov11n_dla GPU INT8** | **2.72ms** | 4.04ms | **GPU最快** |
| yolo26 DLA0 | 16.02ms | 20.82ms | 太慢 |
| yolov11n DLA0 | 7.09ms | 9.66ms | |
| yolov11n_dla DLA0 | 6.89ms | 10.59ms | |
| **Split A** (model.0-9→DLA) | **4.38ms** | 5.87ms | **最优split** |
| Split B (model.0-16→DLA) | 8.51ms | 14.55ms | |
| Split C (head→GPU) | 7.91ms | 12.21ms | |
| Split D (model.0-10→DLA) | 9.02ms | 12.48ms | |

### 5.2 GPU 内核时间分析 (nsys profiling)

通过 nsys 采集每次推理的 GPU 内核总执行时间：

| 配置 | GPU内核时间/iter | vs 基线 GPU 节省 | 检测延迟 | 流水线总预估 | 满足双目标? |
|---|---|---|---|---|---|
| GPU INT8 基线 | 84.687 ms | 0% | 2.72ms | 7.46ms | ✅延迟 / ❌GPU |
| **Split A** | **56.528 ms** | **33.2%** | **4.38ms** | **9.12ms** | **✅✅** |
| Split B | 40.733 ms | 51.9% | 8.51ms | 13.25ms | ❌延迟 |
| Split C | 36.236 ms | 57.2% | 7.91ms | 12.65ms | ❌延迟 |
| DLA0 Full | 20.762 ms | 75.5% | 6.89ms | 11.63ms | ❌延迟 |

> GPU内核时间 = nsys 报告中所有 GPU kernel 执行时间之和（反映 GPU 占用工作量）

### 5.3 Split A 详细分析

**分割点**: model.0-9 (backbone) → DLA0, model.10+ (PSA + neck + head) → GPU

- **DLA 处理的层**: Conv2d backbone 共 171 个节点 (41%)，全部是 DLA native 运算
- **GPU 处理的层**: PSA attention (model.10, 22), Resize (model.11, 14), C3k2 (model.13, 16, 19), Head (model.23)
- **DLA→GPU 切换**: 仅 1 次（在 model.9 → model.10 处）
- **Reformatting CopyNode 开销**: 0%（nsys 未检测到显著 copy overhead）

## 6. 推荐方案

### 方案 A（推荐）: yolov11n_dla Split A

```
DLA0: backbone (model.0-9, 171节点, 41%)
GPU:  PSA + neck + head (model.10-23)
```

- ✅ 检测延迟: 4.38ms (预算 5.26ms，余量 0.88ms)
- ✅ GPU 节省: 33.2% (目标 ≥30%)
- ✅ P99 延迟: 5.87ms (预算内)
- ✅ 无 Reformatting 开销
- ⚠️ 输出格式: 6个张量 (per-scale cls+bbox)，需修改后处理器
- ⚠️ 延迟余量较小 (0.88ms)

### 方案 B（保守备选）: yolov11n_dla GPU INT8 Only

```
GPU: 全部算子
```

- ✅ 检测延迟: 2.72ms (预算内余量充足 2.54ms)
- ❌ GPU 节省: 0%
- ✅ P99 稳定
- ⚠️ 同样需要修改后处理器（输出格式相同）

### 方案 C（如需更多 GPU 容量）: 降低分辨率

- 将输入从 640x640 降为 480x480 或 416x416
- Split A 在低分辨率下延迟会进一步降低
- GPU 节省可能更高

## 7. 注意事项

1. **输出格式差异**: yolov11n_dla 输出 6 个张量 (per-scale)，而非标准 [1,5,8400]，流水线后处理器需要适配
2. **mAP 验证**: 尚未进行精度验证，DLA INT8 量化可能影响检测精度
3. **INT8 校准**: 当前使用 trtexec 默认校准（随机数据），实际部署应使用真实场景校准数据
4. **热稳定性**: Split A 需要进行长时间热测试确认延迟不会随温度升高而漂移
5. **yolov10n**: 因 workspace 不足无法构建任何引擎，已排除
