# GFTT-LK 诊断 Sidecar

最后核对: 2026-07-04

本页记录 OpenCV CUDA GFTT/Harris + SparsePyrLK 深度候选。它当前不是主 CSV 字段，不回写 `Object3D`，只通过 `*.p2_diagnostic.csv` 低频记录，再由训练数据读取器按 `frame_id + mode=opencv_cuda_gftt_lk` 合并为 `z_roi_opencv_cuda_gftt_lk`。

## 当前结论

GFTT/LK 可以作为低频 diagnostic sidecar 保留，但不能视为高质量关键点深度源。它能输出真实 LK 点对 artifact，且在关闭 XFeat 后可维持 `99-100fps`；但当前关键点质量门槛不足，缺少反向 LK、一致性 error 和空间分布质量约束。

| 项 | 当前状态 |
|---|---|
| 记录位置 | `*.p2_diagnostic.csv` |
| 合并字段 | `z_roi_opencv_cuda_gftt_lk` |
| 默认频率 | `p2_diagnostic_stride: 10` |
| 是否回写主 CSV | 否 |
| 是否参与 `z_stereo/obj.z` | 否 |
| 当前准入 | 可作为 sidecar 观察项，不作为主深度 |

## 左右视差怎么来

流程:

1. 从当前 direct pair 的左右 YOLO bbox 构造 shifted ROI。
2. 在左 ROI 上用 OpenCV CUDA GFTT/Harris 检点。
3. 用 OpenCV CUDA SparsePyrLK 从左 ROI 跟踪到右 ROI。
4. 将 ROI 内点坐标映射回 rectified 原图坐标。
5. 对每个点计算 `disparity = left_x - right_x`。
6. 用 y 残差、初始视差差值、overlap ellipse、sphere radius、MAD/stddev 聚合有效视差。

当前 artifact 图中的连线来自真实 `debug_matches`，不是用 anchor 均值伪造。

## 配置入口

```yaml
detector:
  dual_yolo:
    depth_modes:
      roi_cuda_gftt_lk: true

performance:
  p2_feature_job_scaffold_enabled: true
  p2_diagnostic_lane_decision_enabled: true
  p2_diagnostic_stride: 10
  p2_diagnostic_max_in_flight: 1
  p2_diagnostic_results_enabled: true
```

由 `--recording-out xxx.csv` 自动派生:

```text
xxx.p2_diagnostic.csv
```

主要实现入口:

- `src/pipeline/pipeline_feature_jobs.cpp`
- `src/pipeline/pipeline_async_roi.cpp`
- `src/stereo/roi_feature_match_experimental.cpp`
- `trajectory_fusion/dataset.py`

## 已知表现

当前组合实测:

- `XFeat + GFTT/LK sidecar`: ROI 输出 `93-95fps`，不准入。
- `no_xfeat_keep_gftt`: 排除启动后回到 `99-100fps`。
- sidecar `opencv_cuda_gftt_lk`: `114/251` 有效，median/MAD `3.4091/0.0004m`。

代表 artifact:

```text
wiki/assets/p2_current_combo_20260704/opencv_cuda_gftt_lk.png
```

## 关键质量问题

当前 GFTT/LK 不是高质量关键点深度源，原因:

- 只做左到右 LK，没有 right-to-left 反查。
- `feature_reverse_check_px` 已有配置，但当前 GFTT/LK 路径没有真正执行反向一致性。
- OpenCV LK 的 `err_gpu` 已生成但没有用于质量过滤。
- 支撑点数量下限低，代表图只有 `support=4/11`。
- confidence 主要来自视差一致性、support ratio 和 mean score，不包含点位空间分布质量。
- 对排球条纹，少量点落在同一条边缘上也可能产生很小 MAD，但不代表深度真实可靠。

## 后续改进准入

进入更高优先级前，需要补齐:

- right-to-left LK 反查。
- LK error 阈值。
- 点位空间分布约束，例如至少覆盖球面多个象限或最小 baseline spread。
- best frame / bad frame artifact 分组，不只看单张有效图。
- `attempted/support/reject_reason` 统计。
- 动态排球片段帧间跳变分析。

在这些完成前，GFTT/LK 只能保留为 sidecar 诊断和训练候选，不应进入在线主深度。
