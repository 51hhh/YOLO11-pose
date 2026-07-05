# Fallback epipolar 极线

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

单侧漏检退化保护。它不是正常双 YOLO direct pair，也不是 shifted IoU 匹配。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 不需要完整左右 direct pair |
| 单侧检测 | 需要至少一侧命中 |
| fallback 极线 | 是 |
| 输入图像 | 另一目极线附近 ROI |
| host gray / BGR | 视 fallback 后端而定 |

## 左右视差

```text
known side center -> project epipolar row
search missing side along bounded epipolar window
right_or_left_center from fallback hit
disparity = left_x - right_x
z = fb / disparity
```

## 输出字段

| 字段 | 含义 |
|---|---|
| `z_fallback_epipolar` | fallback 极线候选深度 |
| `disparity_fallback_epipolar` | 后续应独立记录的 fallback 视差 |
| `stereo_match_source` | 区分 direct / fallback L2R / fallback R2L |

## 当前结论

只能作为退化保护和训练标记，不应和正常 direct pair 深度混作同一分布。fallback 命中后应降低在线深度更新权重。
