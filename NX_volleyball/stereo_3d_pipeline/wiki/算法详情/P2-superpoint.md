# P2 SuperPoint

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

神经特征匹配。SuperPoint fixed TensorRT extractor 输出固定数量的 keypoints、descriptors 和 scores；实时 C++ 后处理做左右 descriptor mutual-NN 和几何 gate。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | rectified gray ROI crop |
| TensorRT engine | 需要 `superpoint_extractor_*` |

## 当前实现

| 路径 | 状态 |
|---|---|
| batch=1 extractor + CPU mutual-NN | 已有回退路径，但实时效果不准入 |
| batch=2 extractor + GPU mutual-NN/gate | 已实现，`ok_gpu_b2` |
| split LightGlue matcher | 仅保留 schema 入口；当前未作为通过项 |
| fused SuperPoint+LightGlue | 未落地 engine |

batch=2 路径:

```text
left/right rectified gray GPU
  -> cropResizeGPU 到 batch 0/1
  -> TensorRT SuperPoint fixed extractor 一次 enqueue
  -> GPU descriptor mutual-NN + score/margin/y/disparity gate
  -> 回传通过 gate 的点对
  -> median/MAD + spatial gate 输出 z_roi_neural_feature 或 sidecar row
```

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_roi_neural_feature` | 神经特征候选深度；sidecar 时不回写主 CSV 字段 |
| `disparity_roi_neural_feature` | 神经特征鲁棒视差 |
| sidecar `mode=neural_feature` | P2 diagnostic lane 输出 |

## 实测结论

NX 上 `superpoint_extractor_160_top64_b2.engine` 的 `trtexec` 单模型 mean GPU compute 为 `1.743ms`。在实时管线里和双路 YOLO 共享 GPU 后，P2 diagnostic stage 为 `11.5ms` 左右，因此不能每帧进入主路径。

| 配置 | FPS | 有效/行 | median/MAD | 结论 |
|---|---:|---:|---:|---|
| b2 stride=1 strict | `98.4` | `110/992` | `3.4810/0.0008m` | 外层 stddev gate 过严，且拖低主 FPS |
| b2 stride=2 strict | `98.1` | `194/916` | `3.4050/0.0009m` | 仍不适合默认 |
| b2 stride=2 relaxed stddev=6px | `99.6` | `920/920` | `3.4403/0.0050m` | 可作为低频训练候选 sidecar |

代表样张:

| 样张 | 图 |
|---|---|
| SuperPoint b2 relaxed | <img src="../assets/superpoint_b2_20260705/frame_000002_00_neural_feature_ok_gpu_b2.png" width="320"> |
| SuperPoint b2 relaxed | <img src="../assets/superpoint_b2_20260705/frame_000010_00_neural_feature_ok_gpu_b2.png" width="320"> |

## 当前判断

SuperPoint 点对质量有潜力，适合用 `p2_diagnostic_stride=2` 和宽松外层 stddev gate 记录训练候选。它不应进入 P0/P1 默认主字段，也不应替换几何主深度；后续若要每帧 100fps，需要 fused matcher 或进一步减少与 YOLO 的 GPU 竞争。
