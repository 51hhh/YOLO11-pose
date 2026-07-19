# P2 Hough circle refinement

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

圆检测 refinement。它更适合修正 fallback 或检测中心跳变，不是通用纹理匹配。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 可用 direct pair 或 fallback 场景 |
| 输入图像 | gray ROI / edge map |
| fallback 极线 | 可作为后续 fallback refinement |

## 左右视差

```text
left ROI -> refined circle center
right ROI -> refined circle center
disparity = left_center_x - right_center_x
```

## 当前结论

已有少量 diagnostic artifact，但有效率不足，不进入默认录制。更合理的方向是作为 fallback 命中后的中心/半径 refinement，而不是独立主深度。
