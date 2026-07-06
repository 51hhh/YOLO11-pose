# 轨迹可靠性融合实验

这个目录用于验证“多种深度测量 + 历史序列 + 物理约束”的后处理/半在线融合方案。它不替代当前实时双目链路，第一阶段只做离线实验，确认收益后再把轻量部分接回 `HybridDepthEstimator`。

## 结论

最终模型不应该是一个直接生成真实轨迹的黑盒生成器，也不应该只输出一个“深度是否可信”的二分类结果。推荐结构是：

```text
多路观测/质量特征/历史序列
        |
MeasurementReliabilityNet
        |-- 每种测距方法的 log_sigma
        |-- 每种测距方法的 bias
        |-- 每种测距方法的 outlier_prob
        |-- 可选 common_jitter
        v
鲁棒 EKF/UKF/RTS 物理滤波器
        |
最终 x/y/z/vx/vy/vz 轨迹和预测
```

神经网络负责学“什么时候信谁”，滤波器负责输出物理一致的轨迹。这样比 GAN 或端到端轨迹生成更适合当前场景：没有真实轨迹标签，且在线系统必须可解释、可限幅、可回退。

## 为什么不以 GAN 为主

- 没有真实轨迹标签时，GAN 的判别器很容易只学习“看起来平滑”，不能保证绝对深度正确。
- 对 100 FPS 在线闭环来说，GAN/扩散类生成器的延迟、稳定性和失效解释都不好控制。
- 当前噪声主要来自 bbox 抖动、视差误匹配、远距视差量化、左右漏检和同步异常，更适合建模成观测噪声、偏差和离群概率。

GAN/扩散可以作为离线数据增强或论文对照组，不建议作为主路线。

## 参考方向

已检索并纳入设计判断的方向：

- KalmanNet, `arXiv:2107.10043`: 用 RNN 辅助 Kalman，在部分模型已知时学习滤波增益。可作为后续强 baseline，但第一版不直接替代 EKF。
- Unsupervised Learned Kalman Filtering, `arXiv:2110.09005`: 用观测预测误差做无监督训练，适合没有真值轨迹的条件。
- DANSE, `arXiv:2306.03897`: 无监督数据驱动非线性状态估计，表达能力强，但工程复杂度高。
- Deep Kalman Filters, `arXiv:1511.05121` 和 DVBF, `arXiv:1605.06432`: 变分状态空间模型，适合研究对照，不适合作为第一版实时方案。
- Multi-sensor Student's t Filter, `arXiv:2204.11098`: 多传感器重尾噪声/离群鲁棒融合，直接支持当前多测距值融合。
- OOSTraj, `arXiv:2404.02227`: 无监督 noisy trajectory denoising，说明“无真值轨迹 + 轨迹去噪”方向可行，但场景和约束不同。
- `jonasTorz/physical_spline`: 物理样条轨迹拟合思路，可作为离线平滑对照。
- `Lukas-Kozel/KalmanNet-for-state-estimation`, `DanieleGammelli/DeepKalmanFilter`, `gregorsemmler/pytorch-dvbf`: 可复现项目参考。

## 损失设计

没有真实轨迹标签时，训练目标要避免“平滑但错深度”的退化。推荐组合：

1. 观测似然损失
   对每个有效测距方法 `m`，网络输出 `sigma_m`、`bias_m`、`outlier_m`。使用 Student-t 或 Huber NLL：

   ```text
   r_m = z_m - bias_m - z_state
   loss_obs = StudentT_NLL(r_m, sigma_m) * (1 - outlier_prob_m)
   ```

2. 留一法一致性
   随机遮住一种测距方法，用其他方法和历史预测当前状态，再检查被遮住观测的创新分布是否合理。这个比简单平均更能学到互补关系。

3. 物理约束
   对状态序列施加排球飞行约束：

   ```text
   x'' ~= 0
   z'' ~= 0
   y'' ~= g_camera
   jerk 不应长期过大
   ```

   如果相机坐标未严格对齐重力，先把 `g_camera` 作为可配置参数或小范围可学习偏置。

4. 不确定度校准
   归一化创新 `r/sigma` 应接近单位尺度，避免网络把所有 `sigma` 放大逃避误差。

5. 离群稀疏正则
   `outlier_prob` 允许处理错匹配和 YOLO 抖动，但不能每帧都判离群。

## 评估方式

必须同时评估准确性、稳定性、物理一致性和实时可用性：

- 静止球：`z` 标准差、峰峰值、短窗漂移、不同测距方法之间的偏差。
- 动态球：速度/加速度/jerk RMS，落点预测稳定性，深度跳变率。
- 留一法：遮住 mono/stereo/subpixel/circle 后的重建误差。
- 创新校准：`abs(residual) / sigma` 的分布是否接近预期。
- 异常场景：单目漏检、右目漏检、fallback、同步 delta 异常、远距离小视差。
- 在线成本：特征构建 + 网络推理 + EKF update 必须小于单帧预算，目标 < 0.3 ms。

## 文件

- `schema.md`: 未来采集多路观测时需要记录的字段。
- `dataset.py`: 读取旧 CSV 和未来扩展 CSV 的基础工具。
- `robust_smoother.py`: 候选深度驱动的鲁棒物理平滑 baseline；默认不使用 legacy `z_stereo/z` 或静态 `known_z` 作为观测，会先按 mono/bbox/circle/ROI/fallback 等相关方法组聚合再更新，避免同源候选被当作独立观测重复收缩协方差。`--rts` 会在前向滤波后追加离线 Rauch-Tung-Striebel 后向平滑，只用于全序列离线对照/伪真值，不代表实时因果路径。需要标签辅助平滑时显式加 `--use-static-known-z`。
- `depth_polyfit_smoother.py`: 对多路候选深度做离线 batch 多项式拟合，输出 `depth_polyfit_smooth` 上限 baseline；默认不使用 legacy `z/z_stereo`，只用于静态/动态伪真值候选和过平滑对照，不进实时路径。
- `models.py`: 可靠性网络骨架，只输出噪声/偏差/离群概率，不直接输出最终轨迹。
- `losses.py`: 自监督观测似然、物理约束和正则项。
- `train_reliability.py`: 可靠性训练脚手架，支持 `metadata.yaml` 中的 `known_z`, `known_z_min/max`, `static` 弱标签，并支持 `--leave-one-weight` 对有效测距方法做留一法一致性训练。
- `evaluate_reliability_checkpoint.py`: 将 `train_reliability.py` 产出的 checkpoint 应用到 recorder CSV，写出 `smooth_z` 形式的 learned consensus 供 `evaluate_fusion.py` 对比。`smooth_sigma_z` 使用融合权重推导的 consensus sigma；未训练的 `common_log_sigma` 只保留为 `reliability_common_sigma` 诊断列。
- `evaluate_reliability_smoother.py`: 将 ReliabilityNet 输出的 `bias/sigma/outlier_prob` 转成每路候选观测的偏置修正和方差，再交给 `robust_smoother` 物理平滑；这是当前推荐的模型接入评估入口。默认不把 `known_z` 作为 smoother update，避免评估标签泄漏。`--rts` 只用于离线全序列对照。JSON `method_summary` 会记录每种方法的 `valid/top_count/mean_sigma/mean_bias/mean_inlier_prob/mean_corrected_z`，用于发现模型是否过度信任某一路或把固定偏置学错。
- `evaluate_fusion.py`: 对原始/平滑输出做稳定性、known-distance bias/MAD、P0 median、pair gate 质量和动态运动学残差评估；`gravity_y` 可在 metadata 中提供，缺省按 0 处理。
- `analyze_candidate_consistency.py`: 对候选深度做一致性诊断，按 `known_z` / P0 median / 指定方法统计每路候选残差和两两共现偏置矩阵，用于训练前判断哪些方法需要模型学习 bias 或 outlier。
- `fit_method_calibration.py`: 从 `train` split 的 known-distance 静态片段拟合每种测距方法的 median bias 和 sigma，生成非神经 calibration JSON。
- `evaluate_calibrated_smoother.py`: 应用 `fit_method_calibration.py` 输出的每方法 bias/sigma，构造校准后候选观测，再交给 `robust_smoother` 物理平滑；`--rts` 输出校准观测的离线全序列对照。
- `build_dataset_manifest.py`: 从采集目录生成 `dataset_manifest.yaml/json`，自动跳过 `.frames.csv`、`.p2_diagnostic.csv` 和 suite/sweep 派生 CSV，并按 `known_z` 片段生成 train/val/test split；多距离静态数据可加 `--stratify-known-z`，按每个已知距离桶独立切分 heldout。
- `validate_dataset_manifest.py`: 对多段 `dataset_manifest.yaml` 做训练前预检，汇总 split、known_z 覆盖、FPS、帧号缺口、水印异常和 P0/P1 字段命中率；用于采集完成后先判断数据集是否值得进入训练/sweep。
- `run_evaluation_suite.py`: 对单 CSV、多 CSV 或 manifest 批量生成 `check_dataset`、候选一致性诊断、raw eval、robust smoother baseline、`robust_rts_smooth` 和 `depth_polyfit_smooth` 离线 baseline；传 `--calibration` 时额外生成 `calibrated_smoother` / `calibrated_rts_smoother`，传 `--checkpoint` 时额外生成 direct ReliabilityNet、`reliability_smoother` / `reliability_rts_smoother` 对比结果。默认只用 `known_z` 评估 bias/MAD，不把它喂回 smoother。
- `summarize_evaluation_suite.py`: 汇总 suite 输出目录，把 raw/robust/RTS/calibrated/direct/reliability smoother 的核心指标写成单个 `suite_metrics.csv`，其中包含 `ballistic_residual_rms_mps2`、`accel_z_rms_mps2`、`speed_rms_mps` 等动态诊断列；传 `--methods-output` 时额外写出 ReliabilityNet 每方法长表。
- `audit_reliability_methods.py`: 审查 `suite_reliability_methods.csv` 或 `sweep_reliability_methods.csv`，输出 dominant method、低覆盖方法获得 top 权重比例、低 sigma/高 bias/低 inlier 风险，用于判断排名靠前的 ReliabilityNet 是否只是学会信错一路。
- `rank_sweep_metrics.py`: 对 `sweep_metrics.csv` 按 known-distance bias/MAD、稳定性和峰峰值排序，输出 `sweep_ranking.csv`；默认 `--split auto`，有 `val` split 时只用 heldout 指标排名。`--variant all` 可把 `calibrated_smoother`、`calibrated_rts_smoother`、`robust_smooth`、`robust_rts_smooth` 和 ReliabilityNet variants 放在同一张 variant ranking 里对比。
- `select_reliability_model.py`: 合并 `sweep_metrics.csv` 和 `sweep_reliability_method_audit.csv`，输出 `recommended/caution/reject` 选择表；它不替代 heldout 指标，只负责把“指标好但方法风险高”的模型降级。
- `run_reliability_sweep.py`: 批量训练多组 ReliabilityNet 配置，并对每个 checkpoint 自动运行 suite、汇总指标和排名；可传 `--calibration` 把每方法校准 baseline 一起写进每组 suite 指标，适合明天多段 known-distance/heldout 数据做模型选择。每组 suite 会写 `suite_reliability_methods.csv`，总目录会写 `sweep_reliability_methods.csv`、`sweep_reliability_method_audit.csv` 和 `sweep_model_selection.csv`，用于横向检查模型对各测距方法的 sigma/bias/inlier/top_count、可疑权重集中和最终候选状态。
- `run_dataset_workflow.py`: 采集后的一键入口，串起 manifest 生成/预检、每方法 known-distance calibration、baseline suite 和可选 ReliabilityNet sweep，并写 `workflow_summary.json`、`workflow_report.json`、`workflow_report.md`。生成 manifest 时可加 `--stratify-known-z`；默认不把 `known_z` 喂给 smoother，需要标签辅助伪真值时才显式加 `--use-static-known-z`。
- `summarize_workflow.py`: 读取 workflow 目录或 `workflow_summary.json`，把 validation/calibration/baseline/sweep selection/audit 合成一份短报告，明确标出 no-known_z、missing-val、audit warning 和下一步动作。
- `configs/`: `run_dataset_workflow.py --configs` / `run_reliability_sweep.py --configs` 使用的固定 JSON 配置；`sweep_smoke.json` 用于链路冒烟，`sweep_known_distance_selection.json` 用于首轮 known-distance 模型选择，`sweep_dynamic_regularization.json` 用于动态片段第二轮探索。

## 当前记录能力和限制

旧版 `TrajectoryRecorder` 只记录 `x,y,z,z_mono,z_stereo,depth_method,confidence`。当前代码已经支持按 `recording.detail_level` 分级记录：

- `legacy`: 旧 CSV 列，最轻量。
- `depth_candidates`: 追加 `class_id,z_bbox_center,z_bbox_left_edge,z_bbox_right_edge,z_circle_center,z_roi_edge_centroid,z_roi_radial_center,z_roi_edge_pair_center,z_roi_corner_points,z_roi_texture_points,z_roi_binary_points,z_roi_orb_points,z_roi_brisk_points,z_roi_akaze_points,z_roi_sift_points,z_roi_iou_region_color_patch,z_roi_patch_iou_color_edge,z_roi_neural_feature,z_roi_center_patch,z_roi_multi_point,z_fallback,z_fallback_epipolar,z_fallback_template,z_fallback_feature_points` 及对应 `disparity_*`、稀疏/patch/神经特征 `support/std/confidence` 字段，同时保留 `z_yolo_bbox_pair,z_circle,z_subpixel` 旧兼容别名；`z_circle_left_edge,z_circle_right_edge` 为旧兼容列，当前默认 -1。
- `extended`: 在候选深度基础上追加左右水印/SDK 帧号、左右 bbox、左右圆心坐标和半径。

实机是否计算某种候选深度由 `detector.dual_yolo.depth_modes` 控制；`recording.detail_level` 只决定 CSV 写出哪些列，不会开启额外计算。
`recording.raw_mode=true` 时，CSV 的 `x,y,z` 写未滤波观测，`vx/vy/vz/ax/ay/az` 清零，并跳过纯 Kalman 预测帧；`false` 时写在线 Kalman 后轨迹。

`roi_orb_points/roi_brisk_points/roi_akaze_points/roi_sift_points` 使用 OpenCV 特征接口在左右 bbox ROI 内检测描述子，并用 KNN、ratio test、极线 y、正视差、动态视差门限和 MAD 一致性过滤。ORB 有 true OpenCV CUDA 路径但当前有效率和实时性不足；BRISK/AKAZE/SIFT 在当前 NX 上走 CPU，默认关闭，避免在 100fps 实机路径中隐式增加开销。`z_roi_neural_feature` 已有 TensorRT 实时入口，支持 XFeat extractor-only、direct extractor、split matcher 和 fused 点对 engine，但当前 XFeat 128/top64 只能作为专项 A/B 候选，不能作为默认深度源。

仍然缺少训练可靠性模型最关键的一部分质量特征，例如被拒绝候选、每个失败原因、完整左右检测列表和相机曝光/增益逐帧值。因此：

- 新录制数据可以开始分析 bbox/circle/subpixel/fallback 的互补关系；训练脚本会读取候选深度、候选集合统计、质量特征、同步差值、左右 bbox、圆心几何和 selected pair gate 字段。`METHOD_COLUMNS` 不使用旧在线 `z_stereo/z`，也不使用旧圆左右轮廓字段 `z_circle_left_edge/z_circle_right_edge`；训练特征同样不使用旧在线 `x/y/z/vx/vy/vz`。
- 旧 CSV 仍然只能训练/评估很弱的可靠性模型。
- 真正要让模型学会“哪种深度在当前帧可信”，还需要继续按 `schema.md` 补全被拒候选、失败原因、完整左右检测列表和相机曝光/增益。
- 在未完成新基线标定前，所有绝对深度指标只能做趋势判断，不能当最终精度结论。

训练候选方法不包含 legacy `z_stereo` / `z`。实时 C++ 仍用 `buildDepthCandidateObservations()` + `first usable` 写出 `z_stereo/z` 供兼容；这两个字段只在评估中作为旧在线选择结果和旧滤波输出的 baseline。

训练脚本使用训练集全局 feature normalizer，并把 `feature_mean` / `feature_std` 保存到 checkpoint，避免 per-sequence 归一化泄漏未来帧统计。

无真实 `known_z` 的静止片段只能验证抖动和候选一致性，不能验证绝对深度。当前训练入口提供 `--bias-reg-weight`，用于弱约束 ReliabilityNet 的方法 bias，避免纯自监督训练把所有方法一起平移到错误深度；它不能替代真实距离、平面、重投影或落点弱标签。

当前离线实验显示，直接用 ReliabilityNet consensus 生成最终 `smooth_z` 容易在无真值场景下漂移；更稳的结构是让 ReliabilityNet 只预测每种测距方法的可靠性参数，再由分组鲁棒物理 smoother 输出最终轨迹。`evaluate_reliability_smoother.py` 已按这个结构实现，明天的新数据应优先用它和 `robust_smoother.py` baseline 对比。

对带 `known_z` 的静态 clip 做公平评估时，`known_z` 只能用于报告 bias/MAD，不能作为 smoother 观测参与输出。`run_evaluation_suite.py` 和 `evaluate_reliability_smoother.py` 已默认关闭这一路更新；只有在明确生成标签辅助伪真值时才加 `--use-static-known-z`。

`robust_rts_smooth`、`calibrated_rts_smoother`、`reliability_rts_smoother` 和 `depth_polyfit_smooth` 是 suite 默认生成的离线全序列对照。它们可以利用未来帧降低抖动，适合明天 known-distance 数据到位后做伪真值候选和上限参考；如果要只看实时因果可部署路径，使用不带 `_rts_` 的 variant，或在 suite 上加 `--skip-rts-smoother` / `--skip-depth-polyfit`。

动态/抛球片段没有 `known_z` 时，优先比较 `ballistic_residual_rms_mps2`、`accel_z_rms_mps2`、`motion_jerk_rms_mps3` 和候选有效率。`gravity_y` 只有在 metadata 明确写入时才按重力轴约束使用；未写时默认为 0，等价于检查相机坐标轴下的低二阶抖动。

`--leave-one-weight` 已实现，但当前单段无真值静止片段上会加重绝对深度漂移；它应作为明天多距离/heldout 数据的 sweep 项，不作为当前默认训练配置。

## 已验证的旧 CSV baseline 结果

使用当前仓库里未跟踪的 `raw_observation_data.csv` 做脚本连通性验证：

```text
track=0 raw:    z_std=0.2307, dz_rms=0.0038, ddz_rms=0.0030, jerk_rms=0.0047
track=0 smooth: z_std=0.2312, dz_rms=0.0027, ddz_rms=0.0009, jerk_rms=0.0011
```

结论：旧 CSV 已经是在线 Kalman 输出，不是完整原始多路观测，所以离线 smoother 主要降低高阶抖动，不能显著降低整体 `z_std`。这不是模型路线失败，而是数据字段不足；后续必须记录 `schema.md` 里的原始测距和质量特征。
