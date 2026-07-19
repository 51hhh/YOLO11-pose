# P2 关键点描述子历史方法

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

关键点/描述子匹配。包括 OpenCV CUDA ORB、VPI ORB、CPU ORB/BRISK/AKAZE/SIFT、CUDA-SIFT 占位。

## 依赖

| 项 | 值 |
|---|---|
| YOLO direct pair | 需要 |
| shifted IoU / pair gate | 需要 |
| fallback 极线 | 不使用 |
| 输入图像 | gray ROI |
| host gray | CPU 方法需要；实时测试中会污染 profile |

## 左右视差

```text
left ROI keypoints/descriptors
right ROI keypoints/descriptors
descriptor or BF matching
geometry gate / MAD / RANSAC
disparity = left_x - right_x
```

## 当前结论

这些方法当前不属于默认 100fps 路线。OpenCV CUDA ORB/VPI ORB 有少量样张，但有效率和尾延迟不准入。CPU ORB/BRISK/AKAZE/SIFT 只允许离线或 debug，不代表实时 GPU 结论。CUDA-SIFT 保留概念占位，未作为当前依赖引入。
