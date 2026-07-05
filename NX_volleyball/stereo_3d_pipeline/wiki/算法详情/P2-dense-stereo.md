# P2 dense stereo

最后核对: 2026-07-05

返回: [深度算法分类总览](../深度算法分类总览.md)

## 类型

小 ROI dense disparity。包括 OpenCV CUDA StereoBM/SGM、VPI Stereo 和 Fixstars libSGM。

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
left/right ROI -> dense disparity patch
valid pixels inside ball mask / ROI
robust median disparity
z = fb / median_disparity
```

## 方法

| 方法 | 当前结论 |
|---|---|
| OpenCV CUDA StereoBM | 有效候选不足，默认关闭 |
| OpenCV CUDA StereoSGM | 有少量有效但长尾/覆盖率不准入 |
| VPI Stereo | 可输出 confidence patch，但默认退出 |
| Fixstars libSGM | 依赖第三方库，当前只作实验参考 |

## 当前问题

dense ROI 方法的主要问题是小 ROI/小 disparity 下有效率、confidence 和长尾不稳定。若后续继续，应优先裁剪搜索范围并减少中间 disparity patch 下载。
