# P0 ROI edge pair center

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

几何测距。利用 ROI 内左右边缘成对中心估计球心/中心线，不做特征点描述子匹配。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | rectified gray ROI |
| host gray / BGR | 不需要 |

## 左右视差

```text
left_x = left_roi_edge_pair_center_x
right_x = right_roi_edge_pair_center_x
disparity = left_x - right_x
z = fb / disparity
```

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_roi_edge_pair_center` | ROI 成对边缘中心视差深度 |
| `disparity_roi_edge_pair_center` | 对应视差 |

## 当前结论

这是稳定 P0 几何候选。静态数据中 MAD 很低，但它仍是几何中心估计，不等价于真实表面纹理匹配。
