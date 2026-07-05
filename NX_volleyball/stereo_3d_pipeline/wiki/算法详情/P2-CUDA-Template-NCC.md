# P2 CUDA Template/NCC

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

模板/NCC 单点匹配。它不是关键点方法，也不是 dense stereo。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | rectified gray GPU |
| host gray / BGR | 不需要 |

## 左右视差

```text
left_x = left YOLO bbox center
left patch = gray patch around left_x,left_y
right search = predicted x = left_x - initial_disp, small epipolar window
right_x = NCC peak center
disparity = left_x - right_x
z = fb / disparity
```

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_roi_cuda_template_match` | CUDA Template/NCC 深度 |
| `disparity_roi_cuda_template_match` | peak 视差 |
| `roi_cuda_template_match_support` | 单点模板 support，通常为 `1` |
| `roi_cuda_template_match_confidence` | peak score |

## 实现位置

| 项 | 位置 |
|---|---|
| 入口 | `matchCudaTemplateDisparityGPU()` |
| 自研 kernel | `computeCudaTemplateCcoeffNormedScoreMap()` |
| peak reduce | `findCudaTemplateScorePeak()` |
| OpenCV baseline | `STEREO_CUDA_TEMPLATE_BACKEND=opencv` |
| profile | `Stage2_CudaTemplateNccMatch` |

## 当前结论

自研 CUDA Template/NCC isolated A/B: `100.1fps`、`1372/1374` 有效、algo `avg/p95/p99/max=0.30/1.06/1.20/4.89ms`。它满足单算法 isolated 准入，但仍是 P2 单点模板候选，默认关闭，尚未证明和 XFeat/P0/P1 联合长测稳定。
