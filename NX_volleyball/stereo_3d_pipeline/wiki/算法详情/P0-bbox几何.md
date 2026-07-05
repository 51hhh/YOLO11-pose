# P0 bbox 几何

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

几何测距。它不是模板、关键点或神经特征方法。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | 左右 YOLO bbox，不需要额外图像匹配 |
| host gray / BGR | 不需要 |

## 左右视差

`z_bbox_center` 使用左右 bbox 中心:

```text
left_x = left_bbox.cx
right_x = right_bbox.cx
disparity = left_x - right_x
z = fb / disparity
```

`z_bbox_left_edge` 和 `z_bbox_right_edge` 分别使用 bbox 左右边缘的 x 差。

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_bbox_center` | bbox 中心视差深度 |
| `z_bbox_left_edge` | bbox 左边缘视差深度 |
| `z_bbox_right_edge` | bbox 右边缘视差深度 |
| `disparity_bbox_center` 等 | 对应视差 |

这些字段写入主 CSV，可作为训练候选。不要把 legacy `z_stereo` 当作训练候选。

## 实现位置

| 项 | 位置 |
|---|---|
| direct pair gate | `pipeline_dual_yolo_match.cpp` |
| 字段结构 | `object3d_types.h` |
| recorder | `trajectory_recorder*` |

## 当前结论

这是最轻量的 P0 几何后备路线，速度稳定，覆盖率高。缺点是 bbox 受检测框宽度和中心抖动影响，不能修正球表面纹理错配。
