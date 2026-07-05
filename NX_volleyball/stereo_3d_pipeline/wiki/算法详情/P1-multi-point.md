# P1 multi-point

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

亚像素/多点 patch 匹配。它不是通用关键点方法，而是在 YOLO direct pair 和初始视差附近采样多个 ROI 点。

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
left sample points in ball ROI
right search window around initial_disp
each point -> best right_x
disparities -> robust median/MAD/gate
z = fb / median_disparity
```

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_roi_multi_point` | 多点亚像素候选深度 |
| `disparity_roi_multi_point` | 多点鲁棒视差 |
| support/std/confidence | 多点有效数量、离散度和置信度 |

## 当前结论

覆盖率和速度好，已作为 P1 训练候选。但静态测试相对 P0 几何有约数厘米系统差，需要真实距离标定验证。
