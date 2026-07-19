# P1 ROI center patch

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

中心模板/patch 匹配。它只围绕球心或 ROI 中心做小范围匹配，不是多点关键点方法。

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
left_x = left ROI center / circle center
right_x = right peak from small patch search
disparity = left_x - right_x
z = fb / disparity
```

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_roi_center_patch` | 中心 patch 候选深度 |
| `disparity_roi_center_patch` | 中心 patch 视差 |

## 当前结论

可作为多维度训练输入，但覆盖率偏低，不适合作为默认主路径。
