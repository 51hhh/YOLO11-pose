# P2 XFeat

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

神经特征匹配。TensorRT extractor 输出特征、keypoint logits 和 heatmap，后处理得到左右点对。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | rectified gray ROI crop；XFeat 模型内部转灰度 |
| TensorRT engine | 需要 |

## 左右视差

```text
left ROI crop -> XFeat features/keypoints/heatmap
right ROI crop -> XFeat features/keypoints/heatmap
GPU postprocess selects keypoints, samples descriptors, mutual matches
geometry gate filters matches
disparities = left_x - right_x
z = fb / robust disparity
```

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_roi_neural_feature` | 神经特征候选深度 |
| `disparity_roi_neural_feature` | 神经特征鲁棒视差 |
| sidecar `mode=neural_feature` | diagnostic lane 输出 |

## 实现位置

| 项 | 位置 |
|---|---|
| matcher | `neural_feature_matcher_xfeat.cpp` |
| GPU postprocess | `neural_feature_xfeat_gpu_postprocess.cu` |
| config | `neural_feature_matching` |

## 当前结论

XFeat 160_b2/top64 + GPU postprocess + diagnostic sidecar 曾测得主 CSV `100.10fps`，但这是 sidecar-only 口径。主 CSV inline 路径和默认联合组合仍未通过稳定 100fps 准入。XFeat 可作为 diagnostic sidecar 候选，不应直接视为默认主线。
