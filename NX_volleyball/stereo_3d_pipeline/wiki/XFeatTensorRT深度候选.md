# XFeat TensorRT 深度候选

最后核对: 2026-07-04

本页记录 XFeat TensorRT 神经特征深度候选。该路径已经接入实时管线和主 CSV 字段 `z_roi_neural_feature`，也能输出真实左右匹配 artifact；但当前默认关闭，不属于正式 100fps 采集集合。

## 当前结论

XFeat 目前不能作为正式深度测量方案。主要原因不是“无法运行”，而是:

- 每帧主路径开销导致 100fps 不准入。
- 有效率和稳定性不足。
- 当前关键点质量门槛太低，`support=4/4` 就可产出有效深度。
- 只有 mutual best + 几何 gate，没有 ratio/margin、空间分布和强反查质量约束。

| 项 | 当前状态 |
|---|---|
| 主 CSV 字段 | `z_roi_neural_feature` |
| 默认配置 | `neural_feature_matching.enabled: false`，`p2_realtime_lane_decision_enabled: false` |
| 当前用途 | 单独 A/B、artifact 诊断、后续低频/条件触发候选 |
| 是否参与 `z_stereo/obj.z` | 否 |
| 当前准入 | 不准入正式 100fps |

## 左右视差怎么来

流程:

1. 从 direct pair 的左右 YOLO bbox 裁剪正方形 ROI。
2. 使用 TensorRT XFeat extractor 分别提取左右 ROI keypoint / descriptor / heatmap。
3. 对左右 descriptor 做 dot-product mutual best matching。
4. 将 ROI 内关键点坐标映射回 rectified 原图坐标。
5. 用 `y` 残差、初始视差差值和最终 disparity inlier gate 过滤。
6. 对剩余匹配点视差求均值并输出深度:

```text
disparity = mean(left_x - right_x)
z = focal * baseline / disparity
```

## 配置入口

```yaml
neural_feature_matching:
  enabled: false
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
  p2_realtime_lane_decision_enabled: false  # XFeat A/B 时必须显式改为 true
```

主要实现入口:

- `src/stereo/neural_feature_matcher.cpp`
- `src/stereo/neural_feature_matcher_xfeat.cpp`
- `src/stereo/neural_feature_matcher_direct.cpp`
- `src/pipeline/pipeline_dual_yolo_match.cpp`
- `src/stereo/depth_candidate_builder_feature.cpp`

## 已知表现

当前组合和 A/B 结果:

| 测试 | 结果 | 判断 |
|---|---|---|
| `XFeat 128/top32 + GFTT/LK sidecar` | ROI 输出约 `94.4fps`，worker max `15.41ms` | 不满足 100fps |
| `xfeat_no_gftt` | 约 `98.30fps`，仍不到 100fps | XFeat 自身是主瓶颈 |
| 历史 targeted | `317/579` 有效，`93.2fps` | 有效率改善但不准入 |
| 当前组合主 CSV | `52/2508` 有效，median/MAD `3.4287/0.0024m` | 覆盖率不足 |

代表 artifact:

```text
wiki/assets/p2_current_combo_20260704/xfeat_neural_feature.png
```

这张图证明 XFeat 可以输出真实左右 neural keypoint match overlay，但不证明关键点质量足以做稳定深度。

## 关键质量问题

当前 XFeat 不是高质量关键点深度源，原因:

- `min_matches=4` 太低，少量重复纹理点即可通过。
- 只做 mutual best，没有 Lowe ratio 或 best-vs-second-best margin。
- `min_score=0.05` 太宽松。
- 最终 inlier gate 主要看视差一致性，重复条纹可能稳定错配。
- confidence 混合 support、descriptor score 和 stddev，但没有空间分布质量。
- 没有要求点覆盖球面多个区域。
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

在这些完成前，`z_roi_neural_feature` 不进入正式 100fps 默认采集，也不参与 legacy `z_stereo/obj.z`。
