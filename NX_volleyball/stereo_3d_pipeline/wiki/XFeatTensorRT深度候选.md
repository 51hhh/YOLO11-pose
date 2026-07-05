# XFeat TensorRT 深度候选

最后核对: 2026-07-05

本页记录 XFeat TensorRT 神经特征深度候选。该路径已经拆出独立字段 `z_roi_neural_xfeat`，通过 sidecar 输出 `mode=neural_xfeat` 的真实左右匹配结果。它现在归入 P1 sidecar 训练候选，但不参与 legacy `z_stereo/obj.z`。

## 当前结论

XFeat 目前不能作为在线主深度方案。主要原因不是模型推理慢，而是 inline 主路径会把 TensorRT、GPU postprocess、D2H 和同步放进 `Stage2_AsyncRoiWorker` deadline；关键点质量也还不足。当前使用 sidecar 承载，把它作为 P1 训练候选记录。

- 每帧 inline 主路径开销导致 100fps 不准入。
- 有效率和稳定性不足。
- 当前关键点质量门槛太低，`support=4/4` 就可产出有效深度。
- 现在已有 best-vs-second margin、空间象限/覆盖 gate 和几何 gate，但动态质量仍需继续验证。

| 项 | 当前状态 |
|---|---|
| 训练字段 | `z_roi_neural_xfeat` |
| Diagnostic sidecar | `*.p2_diagnostic.csv` 中 `mode=neural_xfeat` |
| 默认配置 | `neural_feature_matching.enabled: false`，`neural_feature_matching_xfeat.enabled: true`，`p2_realtime_lane_decision_enabled: false` |
| 当前用途 | P1 sidecar 训练候选、artifact 诊断、后续质量优化 |
| 是否参与 `z_stereo/obj.z` | 否 |
| 当前准入 | 不准入在线主深度；准入 P1 sidecar 训练候选 |

## 左右视差怎么来

流程:

1. 从 direct pair 的左右 YOLO bbox 裁剪正方形 ROI。
2. 使用 TensorRT XFeat extractor 提取左右 ROI keypoint / descriptor / heatmap；batch=2 engine 可一次提交左右 ROI。
3. 对左右 descriptor 做 dot-product mutual best matching，并可在 CUDA 中完成 topK、descriptor sampling、mutual match 和少量 D2H。
4. 将 ROI 内关键点坐标映射回 rectified 原图坐标。
5. 用 `y` 残差、初始视差差值和最终 disparity inlier gate 过滤。
6. 对剩余匹配点视差求均值并输出深度:

```text
disparity = mean(left_x - right_x)
z = focal * baseline / disparity
```

## 配置入口

```yaml
neural_feature_matching_xfeat:
  enabled: true
  backend: "xfeat"
  extractor_engine_path: "/home/nvidia/NX_volleyball/stereo_3d_pipeline/models/neural/xfeat_extractor_128.engine"
  roi_size: 128
  top_k: 32
  descriptor_dim: 64
  min_matches: 4
  max_y_error_px: 2.0
  max_disp_delta_px: 6.0
  final_disp_gate_px: 3.0
  min_score: 0.05
performance:
  p2_realtime_lane_decision_enabled: false
  p2_diagnostic_lane_decision_enabled: true
```

若测试 sidecar-only 路径，使用 batch=2 engine 和 GPU postprocess:

```yaml
neural_feature_matching_xfeat:
  enabled: true
  extractor_engine_path: "/home/nvidia/NX_volleyball/stereo_3d_pipeline/models/neural/xfeat_extractor_160_b2.engine"
  roi_size: 160
  top_k: 64
  gpu_postprocess: true
  match_margin: 0.0
performance:
  p2_feature_job_scaffold_enabled: true
  p2_realtime_lane_decision_enabled: false
  p2_diagnostic_lane_decision_enabled: true
  p2_diagnostic_stride: 1
  p2_diagnostic_results_enabled: true
```

不要同时打开 XFeat realtime lane 和 diagnostic lane。当前实现为避免共用同一个 TensorRT execution context 并发，`p2_realtime_lane_decision_enabled=true` 时 neural diagnostic 会标记 `skipped_realtime_lane_enabled`。

主要实现入口:

- `src/stereo/neural_feature_matcher.cpp`
- `src/stereo/neural_feature_matcher_xfeat.cpp`
- `src/stereo/neural_feature_xfeat_gpu_postprocess.cu`
- `src/pipeline/pipeline_async_roi.cpp`
- `src/stereo/neural_feature_matcher_direct.cpp`
- `src/pipeline/pipeline_dual_yolo_match.cpp`
- `src/stereo/depth_candidate_builder_feature.cpp`

## 已知表现

当前组合和 A/B 结果:

| 测试 | 结果 | 判断 |
|---|---|---|
| `XFeat 160_b2/top64 GPU postprocess diagnostic sidecar stride=1` | 主 CSV `3374` 行，`100.10fps`；主 `Stage2_AsyncRoiWorker avg/p95/max=4.34/5.08/7.47ms`；neural sidecar `1469/3373` 有效，median/MAD `3.4492/0.0015m`，`algo_ms median/p95/max=3.68/6.12/9.37ms`，`over_deadline=0` | 调度路径可保持主线 100fps，但仍只是 diagnostic，不回写主深度 |
| `P0/P1+NCC+XFeat+SuperPoint 联合 run` | `/home/nvidia/trajectory_dataset/p0p1_xfeat_superpoint_ncc_review_20260705_103601`: 主线 `99.95fps`；`mode=neural_xfeat` sidecar `185/317` 有效，median/MAD `3.4627/0.0085m`，algo `avg/p95/max=5.70/6.66/10.13ms` | 已归入 P1 sidecar 训练候选；有效率仍低 |
| `XFeat 128/top32 + GFTT/LK sidecar` | ROI 输出约 `94.4fps`，worker max `15.41ms` | 不满足 100fps |
| `xfeat_no_gftt` | 约 `98.30fps`，仍不到 100fps | XFeat 自身是主瓶颈 |
| 历史 targeted | `317/579` 有效，`93.2fps` | 有效率改善但不准入 |
| 当前组合主 CSV | `52/2508` 有效，median/MAD `3.4287/0.0024m` | 覆盖率不足 |

代表 artifact:

```text
wiki/assets/p2_current_combo_20260704/xfeat_neural_feature.png
wiki/assets/xfeat_sidecar_20260705/xfeat_ok_gpu_b2.png
wiki/assets/xfeat_sidecar_20260705/xfeat_geometry_reject.png
```

artifact 图证明 XFeat 可以输出真实左右 neural keypoint match overlay；`geometry_reject` 图也会保留点对，便于分析被 gate 拒绝的错配。

## 关键质量问题

当前 XFeat 不是高质量关键点深度源，原因:

- `min_matches=4` 太低，少量重复纹理点即可通过。
- best-vs-second margin 已接入，但阈值仍需实测。
- `min_score=0.05` 仍偏宽松。
- 最终 inlier gate 主要看视差一致性，重复条纹可能稳定错配。
- confidence 混合 support、descriptor score 和 stddev，但没有空间分布质量。
- 已有象限/空间覆盖 gate，但动态排球仍需验证是否足够。
- 没有跨帧一致性或动态跳变约束。

## 后续改进准入

XFeat 若要重新进入采集候选，建议先按低频/条件触发重做:

- 提高 `min_matches` 到至少 `8-12`。
- 增加 ratio/margin test。
- 记录 best/second score、reject reason 和点位分布。
- 增加空间覆盖约束，避免所有点集中在同一条边缘或同一块纹理。
- 尝试 fused matcher 或 LightGlue，减少 CPU 后处理和误匹配。
- 只在 P0/P1 质量下降或 fallback 时触发，避免每帧拖慢主线。
- 用动态排球片段评估帧间跳变，而不是只看静态 median/MAD。

在这些完成前，`z_roi_neural_xfeat` 不进入在线主深度，也不参与 legacy `z_stereo/obj.z`；它只作为 P1 sidecar 训练候选。
