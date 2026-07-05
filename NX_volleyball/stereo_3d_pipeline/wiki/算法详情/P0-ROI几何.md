# P0 ROI 几何

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

几何测距。利用 ROI 内边缘、径向响应或成对边缘估计球心/中心线，不做特征点描述子匹配。

## 方法

| 字段 | 几何来源 |
|---|---|
| `z_roi_edge_centroid` | ROI 边缘/梯度质心 |
| `z_roi_radial_center` | ROI 径向响应中心 |
| `z_roi_edge_pair_center` | 左右边缘成对中心 |

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | rectified gray ROI |
| host gray / BGR | 不需要 |

## 左右视差

每个方法分别在左右 ROI 内得到一个几何中心:

```text
left_x = left_roi_geometry_center_x
right_x = right_roi_geometry_center_x
disparity = left_x - right_x
z = fb / disparity
```

## 当前结论

这些字段是稳定几何候选，适合作为 P0 训练输入。它们不是纹理匹配方法，因此不能直接发现左右表面纹理是否错配。
