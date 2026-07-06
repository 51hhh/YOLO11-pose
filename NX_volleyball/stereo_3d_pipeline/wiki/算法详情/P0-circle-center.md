# P0 circle center

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

几何测距。主字段是 `z_circle_center`，基于 YOLO ROI 内的圆/球心拟合结果，不做左右纹理匹配。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | 左右 ROI gray，圆/边缘拟合 |
| host gray / BGR | 默认不需要 host gray |

## 左右视差

`z_circle_center`:

```text
left_x = left_circle.center_x
right_x = right_circle.center_x
disparity = left_x - right_x
z = fb / disparity
```

`z_circle_left_edge` / `z_circle_right_edge` 是旧字段，当前不再计算。排球是球面，
左右轮廓在两相机里通常不是同一个物理表面点，不能作为球心深度候选。
实时圆匹配的诊断点使用圆心、上点、下点，三者共享圆心 x 视差。
在线 `z_circle_center` 也要求圆心和上下轮廓轴一致，圆形约束仍保留。

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_circle_center` | 圆/球心中心视差深度 |
| `z_circle_left_edge` | 兼容旧 CSV 列，当前保持 `-1` |
| `z_circle_right_edge` | 兼容旧 CSV 列，当前保持 `-1` |

## 当前结论

`z_circle_center` 是当前主观测优先项之一，适合静态和低纹理排球场景。它仍依赖左右 YOLO direct pair 正确，不能处理单侧缺失。
