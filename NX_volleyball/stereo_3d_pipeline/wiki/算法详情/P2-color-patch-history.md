# P2 color patch 历史方法

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

BGR 颜色/边缘 patch 匹配。它使用排球彩色纹理或边缘响应，不是关键点描述子。

## 方法

| 字段 | 含义 |
|---|---|
| `z_roi_iou_region_color_patch` | 彩色区域 IoU + patch |
| `z_roi_patch_iou_color_edge` | 彩色边缘 IoU + patch |

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | rectified BGR GPU |

## 左右视差

```text
left ROI color / color-edge samples
right epipolar window around initial_disp
best color/edge patch response -> right_x
disparity = left_x - right_x
z = fb / disparity
```

## 当前结论

部分 isolated 数据曾达到 100fps 和高有效率，但 artifact 显示存在系统性 stripe/edge 错配。因此这些字段已退出默认配置，只保留历史对照和后续算法设计参考。
