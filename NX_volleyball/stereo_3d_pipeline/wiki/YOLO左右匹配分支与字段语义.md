# YOLO左右匹配分支与字段语义

最后核对: 2026-07-04

本页记录实时管线中左右 YOLO 检测框进入 ROI 深度候选前后的分支状态。它回答两个容易混淆的问题:

- 正常左右 YOLO pair 的 `z_circle_center` 和单侧漏检 `fallback_epipolar` 都会使用圆心视差，但二者不是同一种观测。
- 训练数据统计时必须同时看 `stereo_match_source`、`left_circle_source/right_circle_source` 和原始 `z_*` 字段，不能只看 `stereo_depth_source` 或 `z_stereo`。

相关代码入口:

| 代码 | 作用 |
|---|---|
| `src/pipeline/pipeline_roi_stage2.cpp::runRoiStage2Core()` | Stage2 入口、predict-only 回退、无效 raw 过滤 |
| `src/pipeline/pipeline_dual_yolo_match.cpp::matchDualYoloDetections()` | 左右 YOLO pair、单侧 fallback、统计字段 |
| `src/pipeline/pipeline_dual_yolo_match.cpp` 内 `build_object` | 将一对左右 circle/feature 结果转成 `Object3D` 和 `z_*` |
| `src/stereo/depth_match_contract.cpp::evaluateStereoRoiPair()` | 直接左右 bbox pair gate |
| `src/stereo/roi_geometry_circle_cpu.cpp::searchCircleOnEpipolarCPU()` | 单侧漏检时的有界极线圆搜索 |
| `src/stereo/depth_candidate_builder.cpp` | 深度候选 method/source 映射和 legacy 选择顺序 |
| `src/utils/trajectory_recorder_writer.cpp` | target CSV 字段写入 |
| `src/utils/trajectory_recorder_summary.cpp` | frame sidecar 统计字段 |

## 枚举字段

`stereo_match_source` 表示左右观测来源:

| 值 | 含义 | 说明 |
|---:|---|---|
| `0` | 无 raw stereo 观测 | predict-only 或无有效结果 |
| `1` | 直接左右 YOLO pair | 左右都存在检测框，并通过 bbox pair gate |
| `2` | 左到右 fallback | 左目有检测框，右目未直接配对，向右目极线窗口搜索 |
| `3` | 右到左 fallback | 右目有检测框，左目未直接配对，向左目极线窗口搜索 |

`left_circle_source/right_circle_source` 表示 circle 如何得到:

| 值 | 名称 | 说明 |
|---:|---|---|
| `0` | none | 无 circle |
| `1` | bbox proxy | ROI 圆拟合失败后用 bbox 生成近似圆 |
| `2` | ROI fit | 在真实 YOLO bbox ROI 内拟合圆，正常 `z_circle_center` 只接受这种来源 |
| `3` | epipolar search | 在另一目 predicted center 附近做有界极线圆搜索 |
| `4` | template search | 在另一目极线窗口做模板搜索 |
| `5` | feature proxy | 极线/模板都找不到时，用预测位置构造代理 circle，再尝试 fallback feature |

`stereo_depth_source=1` 不能单独区分正常圆心和极线 fallback。`circle_center`、`fallback_epipolar`、circle edge 都映射到 source `1`；必须结合 `stereo_match_source` 和 circle source 判断语义。

## 主管线路径矩阵

当前主管线配置以 P0 + P1 基线为准，GFTT/LK 走 diagnostic sidecar，XFeat 是未准入测试态。字段落点按左右检测分支拆开看:

| 路径 | `stereo_match_source` | 运行条件 | 主 trajectory CSV 字段 | diagnostic sidecar 字段 | 不会产生的字段 |
|---|---:|---|---|---|---|
| 直接左右 YOLO pair | `1` | 左右都有 YOLO 框，且通过类别、正视差、极线 y、尺寸比、shifted IoU gate | P0: `z_bbox_center`, `z_circle_center`, `z_roi_edge_centroid`, `z_roi_radial_center`, `z_roi_edge_pair_center`; P1: `z_roi_multi_point`, `z_roi_center_patch`, `z_roi_patch_iou_color_edge`, `z_roi_iou_region_color_patch`, `z_roi_cuda_stereo_sgm` | `mode=vpi_template_match` -> `z_roi_vpi_template_match`; `mode=vpi_orb` -> `z_roi_vpi_orb` | `z_fallback_epipolar`, `z_fallback_template`, `z_fallback_feature_points` |
| 左 YOLO 单侧 fallback | `2` | 左框未被 direct pair 占用，右侧 direct pair 缺失或未通过 gate，host gray 可用 | `z_fallback_epipolar`, `z_fallback`; 若开启 CPU fallback 才可能有 `z_fallback_template`, `z_fallback_feature_points` | 当前不把 VPI/颜色/SGM direct pair 字段回写到 fallback 行 | `z_bbox_center`, `z_circle_center`, `z_roi_*` direct pair 字段 |
| 右 YOLO 单侧 fallback | `3` | 右框未被 direct pair 占用，左侧 direct pair 缺失或未通过 gate，host gray 可用 | 同左到右 fallback；左侧可能是 `left_proxy` | 当前不把 VPI/颜色/SGM direct pair 字段回写到 fallback 行 | `z_bbox_center`, `z_circle_center`, `z_roi_*` direct pair 字段 |
| 无有效左右观测 | `0` | 无检测、过期丢弃或全部 gate/reject | raw_mode 下目标 CSV 不写 raw 行；非 raw 模式可能只有滤波预测 | 无 | 所有 raw `z_*` |

注意: `z_roi_iou_region_color_patch`、`z_roi_patch_iou_color_edge`、`z_roi_cuda_stereo_sgm` 都是直接左右 YOLO pair 候选。它们依赖真实 `left_det/right_det/initial_disparity`，不能复用到单侧 fallback 行；单侧搜索要使用 `z_fallback_*` 字段单独评估。

## 直接左右 YOLO Pair

直接 pair 只发生在左右都有检测框时。每个左框和右框先进入 `evaluateStereoRoiPair()`，硬 gate 顺序为:

1. 类别一致。
2. bbox 宽高有效。
3. `left.cx - right.cx` 为正视差。
4. 视差不超过 `stereo.max_disparity`。
5. 校正后中心 y 残差不超过自适应 `epipolar_y_tolerance`。
6. 左右 bbox 宽高比例不超过 `max_size_ratio`。
7. 右 bbox 按中心视差平移到左目坐标后，`shifted_iou >= min_shifted_iou`。

通过 gate 的全部 pair 会加入 `direct_pairs`，再叠加 bbox 物理视差一致性惩罚，按 score 全局排序后一对一占用左右检测框。这一步不是简单按左框顺序贪心，目的是避免早出现的假框抢占唯一真框。

直接 pair 成功后调用 `build_object(... match_source=1, right_det=&right, pair_info=&best_pair, ...)`。这时字段语义如下:

| 字段/候选 | 条件 | 结果 |
|---|---|---|
| `stereo_match_source` | 固定 | `1` |
| `pair_*` | 直接 pair 成功 | 有效，记录 bbox pair gate 质量 |
| `z_bbox_center` / `z_yolo_bbox_pair` | bbox 深度开启 | 有效，来自左右 YOLO bbox 中心视差 |
| `z_circle_center` / `disparity_circle_center` | 左右 circle source 都是 `2` 且深度范围有效 | 有效，正常 ROI 圆心/球心拟合 |
| `z_circle_center` | 任一侧 circle source 为 `1` bbox proxy | 无效，即使 `circle_disparity` 数值存在也不作为正常圆心候选 |
| `z_roi_edge_centroid`, `z_roi_radial_center`, `z_roi_edge_pair_center` | 对应 depth mode 开启并通过 y/depth gate | 有效，属于直接左右 YOLO pair 的 P0 几何候选 |
| `z_roi_multi_point`, `z_roi_center_patch` | 对应 mode 开启并通过自身 gate | 写入原始字段，但当前不参与 legacy `z_stereo/obj.z` 选择 |
| `z_roi_patch_iou_color_edge`, `z_roi_iou_region_color_patch` | BGR GPU snapshot 可用，且对应 mode 开启 | 字段保留；2026-07-04 artifact 显示错配，当前默认关闭，不参与 legacy `z_stereo/obj.z` |
| `z_roi_cuda_stereo_sgm` | gray GPU snapshot 和 CUDA stream 可用，且对应 mode 开启 | 字段保留；联合运行长尾/覆盖率不准入，当前默认关闭 |
| `z_roi_neural_feature` | TensorRT engine 可用，`neural_feature_matching.enabled=true`，direct pair 成功 | 写入主 CSV；当前 XFeat 128/top32 已接入但组合实测只有 `93-95fps`，正式 100fps 采集前应关闭或低频化 |
| `z_roi_opencv_cuda_gftt_lk` | P2 diagnostic lane 命中 stride，且 gray GPU snapshot 已复制到 sidecar buffer | 写入 `*.p2_diagnostic.csv`，由 `trajectory_fusion/dataset.py` 合并为训练候选；不回写主 `Object3D` |
| `z_roi_vpi_template_match`, `z_roi_vpi_orb` | VPI diagnostic mode 开启且命中 stride | 字段保留；当前默认去 VPI，以避免联合运行尾延迟 |
| `z_fallback*` | 直接 pair | 无效 |

主管线中 `p2_realtime_lane_decision_enabled=true` 时，当前 diagnostic sidecar 只保留 OpenCV CUDA GFTT/LK；SGM、VPI Template/ORB 和 color/color-edge 已退出默认配置。P2 isolated diagnostic-only 测试把 realtime lane 关闭，仍可单独跑 OpenCV CUDA、VPI、libSGM 等其他后端。

## 单侧 Fallback

fallback 只在开启对应配置且有 host gray 图时运行:

```yaml
detector:
  dual_yolo:
    fallback_epipolar_search: true
    depth_modes:
      epipolar_fallback: true
      fallback_template: false
      fallback_feature_points: false
```

单侧 fallback 使用的是校正 gray 图上的 CPU 极线搜索。它不会调用左右 YOLO direct pair 的 BGR color patch、CUDA SGM、XFeat 或 GFTT/LK diagnostic 候选，也不会把 `z_circle_center` 伪装成正常 ROI 圆心字段。

fallback 的先验视差来自两处:

1. `HybridDepthEstimator` 的已有轨迹深度。
2. 如果没有可用轨迹深度，则用 bbox 尺寸和已知球直径估计。

左到右 fallback:

```text
left circle
  -> expected_disp
  -> predicted_right_cx = left_circle.cx - expected_disp
  -> searchCircleOnEpipolarCPU(right image, predicted_right_cx, left_circle.cy)
  -> build_object(match_source=2)
```

右到左 fallback:

```text
right circle
  -> expected_disp
  -> predicted_left_cx = right_circle.cx + expected_disp
  -> searchCircleOnEpipolarCPU(left image, predicted_left_cx, right_circle.cy)
  -> build_object(match_source=3)
```

`searchCircleOnEpipolarCPU()` 不是整条极线全局搜索。它只在 predicted center 附近开一个有限 ROI，并要求搜索结果满足:

| 限制 | 当前逻辑 |
|---|---|
| 期望半径 | `max(4, source_circle.radius)` |
| 横向窗口 | 不超过 `fallback_max_width_px`，默认 `220px` |
| 中心横向偏移 | 不超过 `fallback_search_margin_px` 和窗口半宽共同约束，默认 margin `48px` |
| y 窗口 | `source radius + epipolar_y_tolerance + 2px` |
| 半径比例 | `0.45x` 到 `1.70x` |
| 中心偏移 | `abs(cx - predicted_cx)` 和 `abs(cy - predicted_cy)` 必须过门限 |

极线搜索成功后确实会得到圆心视差:

```text
circle_disparity = left_circle.cx - right_circle.cx
z_circle_raw = focal * baseline / circle_disparity
```

但它不会写成正常 `z_circle_center`。正常 `z_circle_center` 要求左右 source 都是 `ROI fit=2`；极线搜索的另一侧 source 是 `3`，因此它被记为 `fallback_epipolar`。

## Fallback 字段落点

| fallback 类型 | match source | circle source 组合 | 深度字段 | 视差字段 | 说明 |
|---|---:|---|---|---|---|
| 左到右 epipolar | `2` | left `2/1`, right `3` | `z_fallback_epipolar`, `z_fallback` | `disparity_fallback_epipolar` | `z_circle_center=-1`; 候选 method 是 `fallback_epipolar` |
| 右到左 epipolar | `3` | left `3`, right `2/1` | `z_fallback_epipolar`, `z_fallback` | `disparity_fallback_epipolar` | `z_circle_center=-1`; 候选 method 是 `fallback_epipolar` |
| 左到右 template | `2` | left `2/1`, right `4` | `z_fallback`, `z_fallback_template` | `disparity_fallback_template` | 100fps 默认关闭 |
| 右到左 template | `3` | left `4`, right `2/1` | `z_fallback`, `z_fallback_template` | `disparity_fallback_template` | 100fps 默认关闭 |
| fallback feature | `2/3` | 搜索侧可能是 `5` feature proxy | `z_fallback_feature_points`, `z_fallback` | `disparity_fallback_feature_points` | 只有 feature 匹配本身有效时才输出 |

`z_fallback` 是兼容聚合字段，优先级为:

```text
z_fallback = z_fallback_feature_points if valid
             else z_circle_raw if epipolar/template fallback valid
             else -1
```

因此分析 fallback 质量时不能只看 `z_fallback`。需要同时看:

- `stereo_match_source`
- `left_circle_source`
- `right_circle_source`
- `z_fallback_template`
- `z_fallback_feature_points`
- `z_fallback_epipolar`
- `disparity_fallback_epipolar`
- `disparity_fallback_template`
- `disparity_fallback_feature_points`

旧 CSV 可能没有显式 `z_fallback_epipolar` 和 `disparity_fallback_epipolar` 字段。读取旧数据时，用 `z_fallback` 加 `stereo_match_source in (2,3)` 且任一 circle source 为 `3` 过滤。

## 全部分支情况

| 输入/状态 | 分支 | 输出行为 | 训练字段口径 |
|---|---|---|---|
| 左右都无检测 | predict-only | `output.detections` 为空；由 hybrid depth 预测或无结果 | 无 raw stereo；raw_mode target CSV 不写目标行 |
| 左右都有检测，至少一对通过 gate 且 build 成功 | 直接左右 YOLO pair | `stereo_match_source=1`; 一对一占用左右框 | 读取 P0/P1 原始字段；`pair_*` 可用于学习 pair 质量 |
| 左右都有检测，但某些左框未成功配对 | 左到右 fallback | 对未匹配左框尝试右目极线/模板/feature fallback | `stereo_match_source=2`; fallback 字段与正常 `z_circle_center` 分开 |
| 左右都有检测，但某些右框未被占用 | 右到左 fallback | 对未占用且未被左 fallback 标记的右框尝试左目搜索 | `stereo_match_source=3`; 若左侧已有近邻检测则填该左检测，否则不新增独立右目结果 |
| 左有检测，右无检测 | 左到右 fallback | 成功则输出左检测对应 raw stereo；失败则无 raw 观测 | 大多数直接 P0/P1 字段无效，只可能有 fallback |
| 左无检测，右有检测 | 右到左 fallback | 成功则创建 `left_proxy` 并 push 输出；失败则 predict-only | `stereo_match_source=3`; 左 bbox 是 proxy，不是左 YOLO 实检 |
| 直接 pair 通过但 circle ROI fit 失败 | bbox proxy 或其他直接候选 | 若 bbox/edge/radial 等候选可用，仍可输出 | `z_circle_center=-1`; 不要把 bbox proxy circle 当正常圆心 |
| circle 视差非正、超 max 或深度越界 | build reject | 若没有其他有效候选则 `build_object()` 返回 false | 不写 raw；统计看 reject 计数 |
| fallback 开关打开但 host gray 不可用 | fallback 不运行 | 单侧缺失通常无法产生 raw stereo | 统计看 `image_lock_fail` 或 frame sidecar raw/stereo 数 |
| `fallback_to_roi_match=false` 且 semantic 输出无效 | 无效项过滤 | 只保留 `z>0 && confidence>min_confidence` 的 raw stereo | 当前 P0/P1 采集默认应避免把无效检测写入训练 |
| `fallback_to_roi_match=true` | legacy ROI texture 兜底 | semantic 无效时可能被旧 ROI texture 结果替换 | 当前 100fps 主配置不应依赖此路径 |

## Legacy `z_stereo` 选择

`buildDepthCandidateObservations()` 会把所有候选放进一个列表，再由 `selectLegacyDepthOutputCandidate()` 选择第一个可用于 legacy 输出的候选。当前 legacy allow-list 允许 P0 几何/bbox 和退化 fallback，不允许 P1/实验候选抢占 `z_stereo/obj.z`。

候选构造顺序中，`fallback_epipolar` 和正常 `circle_center` 共用同一个槽位:

```text
fallback_feature_points
fallback_template
fallback_epipolar 或 circle_center
...
roi_radial_center
roi_edge_pair_center
roi_edge_centroid
bbox_center
```

注意:

- `stereo_depth_source=1` 可能表示正常 `circle_center`，也可能表示 `fallback_epipolar`。
- 训练可靠性模型必须读取原始候选字段，例如 `z_circle_center`、`z_roi_radial_center`、`z_roi_edge_pair_center`、`z_bbox_center`、`z_roi_multi_point`、`z_roi_center_patch`。
- `z_stereo` / `obj.z` 只是在线兼容输出和 baseline，不能作为候选方法标签。

## 统计方法

目标级 CSV 用于统计每个输出目标的来源和候选有效率。推荐先统计这些组合:

```python
import pandas as pd

df = pd.read_csv("trajectory.csv")
raw = df[df["raw_observation_valid"] == 1].copy()

case_counts = raw.groupby([
    "stereo_match_source",
    "left_circle_source",
    "right_circle_source",
    "stereo_depth_source",
]).size().reset_index(name="count")

field_valid = {
    col: int((raw[col] > 0).sum())
    for col in [
        "z_bbox_center",
        "z_circle_center",
        "z_roi_edge_centroid",
        "z_roi_radial_center",
        "z_roi_edge_pair_center",
        "z_roi_multi_point",
        "z_roi_center_patch",
        "z_roi_patch_iou_color_edge",
        "z_roi_iou_region_color_patch",
        "z_roi_cuda_stereo_sgm",
        "z_fallback",
        "z_fallback_epipolar",
        "z_fallback_template",
        "z_fallback_feature_points",
    ]
    if col in raw.columns
}

epipolar_fallback = raw[
    raw["stereo_match_source"].isin([2, 3]) &
    ((raw["left_circle_source"] == 3) | (raw["right_circle_source"] == 3)) &
    (raw.get("z_fallback_epipolar", raw["z_fallback"]) > 0)
]
```

frame sidecar 用于统计每帧是否退化:

| 字段 | 说明 |
|---|---|
| `result_count` | 当帧 callback 里结果数量，可能包含预测结果 |
| `raw_count` | `raw_observation_valid=1` 的结果数量 |
| `stereo_count` | `z_stereo>0` 的结果数量 |
| `direct_pair_count` | `stereo_match_source=1` 数量 |
| `fallback_l2r_count` | `stereo_match_source=2` 数量 |
| `fallback_r2l_count` | `stereo_match_source=3` 数量 |
| `pair_positive_count` | 直接 pair 中正视差且未超过 max disparity 的数量 |
| `pair_shifted_iou_min/mean` | 直接 pair 的 shifted IoU 质量 |
| `pair_score_mean` | 直接 pair 全局排序 score 均值 |
| `pair_bbox_prior_penalty_mean` | bbox 物理视差一致性惩罚均值 |

运行时 `[DualYOLO]` 日志可补充统计调度内拒绝原因:

| 日志字段 | 含义 |
|---|---|
| `left/right` | 当帧左右检测数量 |
| `matches/valid` | match 成功数量和有效 stereo 数量 |
| `missL/missR` | 左/右单侧缺失计数 |
| `fb/fallback_attempted/fail` | fallback 成功、尝试和失败 |
| `l2r/r2l` | fallback 方向统计 |
| `noCand` | 左框没有任何可行直接 pair candidate |
| `cls/badBox/d<=0/dMax/epi/size/iou` | 直接 pair gate 拒绝原因 |
| `circle` | circle fit 失败计数 |
| `depth` | 候选深度最终被拒绝计数 |
| `lock` | CPU host 图像不可用计数 |

统计训练数据时，建议同时保留三张表:

1. target CSV 的 `stereo_match_source + circle_source + stereo_depth_source` 组合计数。
2. target CSV 中每个原始 `z_*` 字段的有效率、median、MAD。
3. frame sidecar 的 direct/fallback/raw/stereo 每帧计数，用来发现 YOLO 单侧漏检和 fallback 退化比例。

## 设计结论

- 正常 P0 圆心观测是 `stereo_match_source=1` 且左右 circle source 都为 `2` 的 `z_circle_center`。
- 极线搜索得到的是圆心视差数值，但语义是 `fallback_epipolar`，字段落在 `z_fallback_epipolar` 和兼容汇总 `z_fallback`，不是 `z_circle_center`。
- `stereo_depth_source=1` 是兼容 source id，不足以区分正常圆心和 fallback epipolar。
- 训练可靠性模型或做质量报表时优先使用 `z_fallback_epipolar` 和 `disparity_fallback_epipolar`；旧 CSV 再用 source 组合过滤。
- 后续功能记录见 [深度后续TODO](深度后续TODO.md)。
