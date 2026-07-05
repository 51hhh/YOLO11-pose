# P2 GFTT/LK

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

关键点/光流方法。OpenCV CUDA GFTT/Harris 检点，SparsePyrLK 将左 ROI 点跟踪到右 ROI。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | rectified gray GPU snapshot |
| host gray / BGR | 不需要 |

## 左右视差

```text
left ROI -> GFTT points
right ROI shifted by initial_disp
SparsePyrLK tracks left points to right ROI
each match disparity = left_x - right_x
robust gate / median / MAD
z = fb / robust disparity
```

## 输出字段

| 字段 | 含义 |
|---|---|
| sidecar `mode=opencv_cuda_gftt_lk` | P2 diagnostic CSV mode |
| `z_roi_opencv_cuda_gftt_lk` | dataset.py 合并后的训练候选名 |

## 当前结论

warmup 和 workspace 复用已消除首次冷启动尖峰。后续又验证过批量 flush、stream event wait、pinned HostMem 等调度修复，但每帧 diagnostic-only 仍为 `94-95fps`，并会把 `Stage1_DetectSubmit` p95 从 baseline 约 `5.7ms` 抬到约 `7.5ms`。当前不准入每帧 100fps，只能低频或单独 A/B。
