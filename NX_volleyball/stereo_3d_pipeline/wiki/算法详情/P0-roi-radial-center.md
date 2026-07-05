# P0 ROI radial center

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

几何测距。利用 ROI 内径向响应估计球心/中心线，不做特征点描述子匹配。

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
left_x = left_roi_radial_center_x
right_x = right_roi_radial_center_x
disparity = left_x - right_x
z = fb / disparity
```

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_roi_radial_center` | ROI 径向响应中心视差深度 |
| `disparity_roi_radial_center` | 对应视差 |

## 当前结论

这是稳定 P0 几何候选，静态测试深度与 P0 主几何路线接近。它依赖左右 YOLO direct pair 正确，不负责单侧 fallback。
