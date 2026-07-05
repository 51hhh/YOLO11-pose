# P2 ring/edge profile

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

边界/profile 匹配。它不是通用关键点，也不是 dense stereo。

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
sample ring/edge/radial profiles around ball ROI
search right epipolar window around initial_disp
best response -> right_x
disparities -> gate / robust aggregate
```

## 当前结论

kernel 很轻，适合排球边界先验，但当前有效率不足，仍需看采样、score 和 reject reason。默认关闭。
