# P0 ROI edge centroid

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

几何测距。利用左右 ROI 内边缘/梯度质心估计中心线，不做特征点描述子匹配。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | rectified gray ROI |
| host gray / BGR | 不需要 |

## 左右视差

左右 ROI 分别计算边缘/梯度质心:

```text
left_x = left_roi_edge_centroid_x
right_x = right_roi_edge_centroid_x
disparity = left_x - right_x
z = fb / disparity
```

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_roi_edge_centroid` | ROI 边缘/梯度质心视差深度 |
| `disparity_roi_edge_centroid` | 对应视差 |

## 当前结论

这是稳定 P0 几何候选，适合作为训练输入。它不是纹理匹配方法，因此不能直接发现左右表面纹理是否错配。
