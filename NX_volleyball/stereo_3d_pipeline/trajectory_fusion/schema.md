# 多测距观测记录 Schema

旧 CSV 只能支持 baseline；当前 recorder 已经支持 `legacy/depth_candidates/extended` 分级记录。下面按当前 CSV 字段和后续需要补齐的诊断字段整理。建议优先写 CSV，稳定后切 Parquet/Arrow。

## 每帧公共字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| frame_id | int | pipeline 输出帧号 |
| timestamp | float | 主机时间或相机时间，秒 |
| left_timestamp_ns | int | 左目 SDK 原始设备时间戳，当前海康 USB 实测为 ns |
| right_timestamp_ns | int | 右目 SDK 原始设备时间戳，当前海康 USB 实测为 ns |
| left_frame_counter | int | 左目水印帧计数 |
| right_frame_counter | int | 右目水印帧计数 |
| left_frame_number | int | 左目 SDK 帧号 |
| right_frame_number | int | 右目 SDK 帧号 |
| left_trigger_index | int | 左目水印外触发计数 |
| right_trigger_index | int | 右目水印外触发计数 |
| frame_counter_delta | int | 左右水印帧计数差 |
| frame_number_delta | int | 左右 SDK 帧号差 |
| timestamp_delta_us | float | 左右相机时间戳差，微秒 |
| track_id | int | 目标轨迹 ID |
| class_id | int | 检测类别 |
| dt | float | 与上一输出帧间隔 |
| raw_observation_valid | int | 当前行是否包含未滤波观测；`recording.raw_mode=true` 时只写有效观测 |

## 目标状态字段

这些字段在所有记录级别都会写出:

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| x,y,z | float | 当前在线输出位置；`raw_mode=true` 时为未滤波观测，否则为滤波后状态 |
| vx,vy,vz | float | 当前速度估计 |
| ax,ay,az | float | 当前加速度估计 |
| depth_method | int | 在线 legacy 输出的深度来源方法，0=单目,1=双目,2=融合；训练候选不要把它当标签 |
| confidence | float | 当前 3D 观测或融合置信度 |

## 检测和几何字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| left_det_valid | int | 左目 YOLO 是否有效 |
| right_det_valid | int | 右目 YOLO 是否有效 |
| left_bbox_conf | float | 左目检测置信度 |
| right_bbox_conf | float | 右目检测置信度 |
| left_bbox_cx,left_bbox_cy,left_bbox_w,left_bbox_h | float | 左目 bbox |
| right_bbox_cx,right_bbox_cy,right_bbox_w,right_bbox_h | float | 右目 bbox |
| bbox_iou_lr | float | 左右匹配框近似 IoU 或相似度 |
| epipolar_dy | float | 校正后左右中心 y 差 |
| size_ratio | float | 左右 bbox 或圆半径比例 |
| positive_disparity | int | 是否正视差 |
| pair_initial_disparity | float | 直接 YOLO pair 的 bbox 中心初始视差，fallback 为 `-1` |
| pair_epipolar_dy | float | 直接 YOLO pair 的 bbox 中心 y 残差，fallback 为 `-1` |
| pair_y_tolerance | float | 当前 pair gate 的自适应 y 容差，fallback 为 `-1` |
| pair_size_ratio | float | 直接 YOLO pair 的 bbox 宽高最大比例，fallback 为 `-1` |
| pair_shifted_iou | float | 把右框按中心视差平移到左目坐标后的 bbox IoU，fallback 为 `-1` |
| pair_score | float | 直接 YOLO pair 全局排序分数，已包含 bbox 物理视差惩罚 |
| pair_bbox_prior_penalty | float | bbox 宽度反推视差与中心视差不一致时的排序惩罚 |
| pair_positive_disparity | int | 1=直接 pair 视差为正且不超过 `max_disparity` |

## 圆心和 ROI 质量字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| left_circle_valid | int | 左 ROI 圆拟合是否有效 |
| right_circle_valid | int | 右 ROI 圆拟合是否有效 |
| left_circle_cx,left_circle_cy,left_circle_r | float | 左目圆拟合 |
| right_circle_cx,right_circle_cy,right_circle_r | float | 右目圆拟合 |
| left_circle_source | int | 0=无,1=bbox代理,2=ROI圆拟合,3=极线搜索,4=模板搜索,5=特征fallback预测窗口 |
| right_circle_source | int | 0=无,1=bbox代理,2=ROI圆拟合,3=极线搜索,4=模板搜索,5=特征fallback预测窗口 |
| left_circle_conf,right_circle_conf | float | 圆拟合置信度 |
| roi_pixels | int | ROI 像素数，防止大球性能退行 |
| roi_denoise_used | int | ROI 是否做降噪 |
| fallback_used | int | 是否使用极线 fallback |
| fallback_direction | str | `left_to_right` / `right_to_left` / `none` |

## 测距方法字段

建议每帧一行，宽表记录所有方法，便于实时写入：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| z_mono | float | bbox 单目估距 |
| z_bbox_center | float | 左右 YOLO bbox 中心视差三角测距 |
| z_bbox_left_edge | float | 左右 YOLO bbox 左边缘视差三角测距 |
| z_bbox_right_edge | float | 左右 YOLO bbox 右边缘视差三角测距 |
| z_circle_center | float | ROI 圆拟合圆心视差三角测距 |
| z_circle_left_edge | float | ROI 圆左边缘视差三角测距 |
| z_circle_right_edge | float | ROI 圆右边缘视差三角测距 |
| z_roi_edge_centroid | float | ROI 边缘梯度质心视差三角测距 |
| z_roi_radial_center | float | ROI 径向梯度中心视差三角测距 |
| z_roi_edge_pair_center | float | ROI 左右边缘成对中心视差三角测距 |
| z_roi_corner_points | float | ROI 角点特征点 ZNCC 视差三角测距 |
| z_roi_texture_points | float | ROI 纹理/梯度特征点 ZNCC 视差三角测距 |
| z_roi_binary_points | float | ROI Census/BRIEF 类二值描述子特征点视差三角测距 |
| z_roi_orb_points | float | ROI OpenCV ORB/BRIEF 二值描述子匹配视差三角测距 |
| z_roi_brisk_points | float | ROI OpenCV BRISK 二值描述子匹配视差三角测距 |
| z_roi_akaze_points | float | ROI OpenCV AKAZE/MLDB 二值描述子匹配视差三角测距 |
| z_roi_sift_points | float | ROI OpenCV CPU SIFT 实验候选视差三角测距;当前 NX 无 true CUDA SIFT 后端 |
| z_roi_iou_region_color_patch | float | ROI GPU 彩色区域 IoU + patch 视差三角测距 |
| z_roi_patch_iou_color_edge | float | ROI GPU 彩色边缘 IoU + patch 视差三角测距 |
| z_roi_cuda_template_match | float | 自研 CUDA Template/NCC 小 ROI 极线视差三角测距；主 CSV 或 sidecar `mode=cuda_template` 合并 |
| z_roi_cuda_stereo_bm | float | OpenCV CUDA StereoBM 小 ROI dense disparity 三角测距 |
| z_roi_cuda_stereo_sgm | float | OpenCV CUDA StereoSGM 小 ROI dense disparity 三角测距 |
| z_roi_vpi_template_match | float | `*.p2_diagnostic.csv` 中 `mode=vpi_template_match` 合并出的训练候选；不在主 trajectory CSV header |
| z_roi_vpi_orb | float | `*.p2_diagnostic.csv` 中 `mode=vpi_orb` 合并出的训练候选；不在主 trajectory CSV header |
| z_roi_opencv_cuda_gftt_lk | float | `*.p2_diagnostic.csv` 中 `mode=opencv_cuda_gftt_lk` 合并出的训练候选；不在主 trajectory CSV header |
| z_roi_ring_edge_profile | float | CUDA ring/edge profile 小范围极线视差三角测距 |
| z_roi_neural_feature | float | ROI TensorRT 神经特征兼容字段；split 模式下仅作旧字段兼容 |
| z_roi_neural_xfeat | float | ROI XFeat TensorRT 神经特征匹配视差三角测距；主 CSV 或 sidecar `mode=neural_xfeat` 合并 |
| z_roi_neural_superpoint | float | ROI SuperPoint TensorRT 神经特征匹配视差三角测距；主 CSV 或 sidecar `mode=neural_superpoint` 合并 |
| z_roi_center_patch | float | ROI 中心 patch ZNCC 视差三角测距 |
| z_roi_multi_point | float | ROI 多点 ZNCC 亚像素视差三角测距 |
| z_fallback | float | fallback 测距兼容汇总字段，优先 feature，其次 epipolar/template |
| z_fallback_epipolar | float | 单侧漏检时极线搜索 fallback 测距 |
| z_fallback_template | float | 单侧漏检时极线模板搜索 fallback 测距 |
| z_fallback_feature_points | float | 单侧漏检时极线特征点搜索 fallback 测距 |
| z_stereo | float | 当前在线 first-usable 兼容选择的双目测距观测，不是独立取点方法；训练候选不要读取它 |
| z | float | 当前在线输出/状态字段；基础状态字段中也会写出，用于兼容宽表读取；不要作为训练候选或标签 |
| disparity_bbox_center | float | bbox 中心视差 |
| disparity_bbox_left_edge | float | bbox 左边缘视差 |
| disparity_bbox_right_edge | float | bbox 右边缘视差 |
| disparity_circle_center | float | 圆心视差 |
| disparity_circle_left_edge | float | 圆左边缘视差 |
| disparity_circle_right_edge | float | 圆右边缘视差 |
| disparity_roi_edge_centroid | float | ROI 边缘梯度质心视差 |
| disparity_roi_radial_center | float | ROI 径向梯度中心视差 |
| disparity_roi_edge_pair_center | float | ROI 左右边缘成对中心视差 |
| disparity_roi_corner_points | float | ROI 角点特征点聚合视差 |
| disparity_roi_texture_points | float | ROI 纹理/梯度特征点聚合视差 |
| disparity_roi_binary_points | float | ROI 二值描述子特征点聚合视差 |
| disparity_roi_orb_points | float | ROI ORB 聚合视差 |
| disparity_roi_brisk_points | float | ROI BRISK 聚合视差 |
| disparity_roi_akaze_points | float | ROI AKAZE 聚合视差 |
| disparity_roi_sift_points | float | ROI OpenCV CPU SIFT 聚合视差 |
| disparity_roi_iou_region_color_patch | float | ROI 彩色区域 IoU 聚合视差 |
| disparity_roi_patch_iou_color_edge | float | ROI 彩色边缘 IoU 聚合视差 |
| disparity_roi_cuda_template_match | float | 自研 CUDA Template/NCC 聚合视差；由 `trajectory_fusion/dataset.py` 可从 sidecar 合并 |
| disparity_roi_cuda_stereo_bm | float | OpenCV CUDA StereoBM 聚合视差 |
| disparity_roi_cuda_stereo_sgm | float | OpenCV CUDA StereoSGM 聚合视差 |
| disparity_roi_vpi_template_match | float | sidecar `vpi_template_match` 聚合视差；由 `trajectory_fusion/dataset.py` 合并 |
| disparity_roi_vpi_orb | float | sidecar `vpi_orb` 聚合视差；由 `trajectory_fusion/dataset.py` 合并 |
| disparity_roi_opencv_cuda_gftt_lk | float | sidecar `opencv_cuda_gftt_lk` 聚合视差；由 `trajectory_fusion/dataset.py` 合并 |
| disparity_roi_ring_edge_profile | float | CUDA ring/edge profile 聚合视差 |
| disparity_roi_neural_feature | float | ROI 神经特征兼容聚合视差 |
| disparity_roi_neural_xfeat | float | ROI XFeat 神经特征聚合视差 |
| disparity_roi_neural_superpoint | float | ROI SuperPoint 神经特征聚合视差 |
| disparity_roi_center_patch | float | ROI 中心 patch ZNCC 视差 |
| disparity_roi_multi_point | float | ROI 多点 ZNCC 亚像素视差 |
| disparity_fallback_epipolar | float | 极线搜索 fallback 视差 |
| disparity_fallback_template | float | 极线模板搜索 fallback 视差 |
| disparity_fallback_feature_points | float | 极线特征点搜索 fallback 视差 |
| z_yolo_bbox_pair,z_circle,z_subpixel | float | 旧兼容别名，分别对应 bbox_center/circle_center/roi_multi_point |
| disparity_yolo,disparity_circle,disparity_subpixel | float | 旧兼容别名，分别对应 disparity_bbox_center/disparity_circle_center/disparity_roi_multi_point |
| subpixel_valid | int | 亚像素测距是否接受 |
| subpixel_attempted | int | 本帧是否尝试亚像素测距 |
| subpixel_support | int | 接受的多点数量 |
| subpixel_std_px | float | 多点视差标准差 |
| subpixel_confidence | float | 多点匹配综合置信度 |
| subpixel_gate_px | float | 当前动态视差门限 |
| roi_corner_points_support | int | 角点特征匹配支撑点数 |
| roi_corner_points_std_px | float | 角点特征视差标准差 |
| roi_corner_points_confidence | float | 角点特征匹配置信度 |
| roi_texture_points_support | int | 纹理特征匹配支撑点数 |
| roi_texture_points_std_px | float | 纹理特征视差标准差 |
| roi_texture_points_confidence | float | 纹理特征匹配置信度 |
| roi_binary_points_support | int | 二值描述子特征匹配支撑点数 |
| roi_binary_points_std_px | float | 二值描述子特征视差标准差 |
| roi_binary_points_confidence | float | 二值描述子特征匹配置信度 |
| roi_orb_points_support | int | ORB 特征匹配支撑点数 |
| roi_orb_points_std_px | float | ORB 特征视差标准差 |
| roi_orb_points_confidence | float | ORB 特征匹配置信度 |
| roi_brisk_points_support | int | BRISK 特征匹配支撑点数 |
| roi_brisk_points_std_px | float | BRISK 特征视差标准差 |
| roi_brisk_points_confidence | float | BRISK 特征匹配置信度 |
| roi_akaze_points_support | int | AKAZE 特征匹配支撑点数 |
| roi_akaze_points_std_px | float | AKAZE 特征视差标准差 |
| roi_akaze_points_confidence | float | AKAZE 特征匹配置信度 |
| roi_sift_points_support | int | OpenCV CPU SIFT 特征匹配支撑点数 |
| roi_sift_points_std_px | float | OpenCV CPU SIFT 特征视差标准差 |
| roi_sift_points_confidence | float | OpenCV CPU SIFT 特征匹配置信度 |
| roi_iou_region_color_patch_support | int | 彩色区域 IoU/patch 匹配支撑点数 |
| roi_iou_region_color_patch_std_px | float | 彩色区域 IoU/patch 视差标准差 |
| roi_iou_region_color_patch_confidence | float | 彩色区域 IoU/patch 匹配置信度 |
| roi_patch_iou_color_edge_support | int | 彩色边缘 IoU/patch 匹配支撑点数 |
| roi_patch_iou_color_edge_std_px | float | 彩色边缘 IoU/patch 视差标准差 |
| roi_patch_iou_color_edge_confidence | float | 彩色边缘 IoU/patch 匹配置信度 |
| roi_cuda_template_match_support | int | 自研 CUDA Template/NCC 支撑点数；主 CSV 或 sidecar `cuda_template` 合并 |
| roi_cuda_template_match_std_px | float | 自研 CUDA Template/NCC 视差标准差 |
| roi_cuda_template_match_confidence | float | 自研 CUDA Template/NCC 匹配置信度 |
| roi_cuda_stereo_sgm_support | int | OpenCV CUDA StereoSGM 有效采样点数 |
| roi_cuda_stereo_sgm_std_px | float | OpenCV CUDA StereoSGM 视差标准差 |
| roi_cuda_stereo_sgm_confidence | float | OpenCV CUDA StereoSGM 匹配置信度 |
| roi_vpi_template_match_support | int | sidecar `vpi_template_match` 支撑点数；由 `trajectory_fusion/dataset.py` 合并 |
| roi_vpi_template_match_std_px | float | sidecar `vpi_template_match` 视差标准差；由 `trajectory_fusion/dataset.py` 合并 |
| roi_vpi_template_match_confidence | float | sidecar `vpi_template_match` 置信度；由 `trajectory_fusion/dataset.py` 合并 |
| roi_vpi_orb_support | int | sidecar `vpi_orb` 支撑点数；由 `trajectory_fusion/dataset.py` 合并 |
| roi_vpi_orb_std_px | float | sidecar `vpi_orb` 视差标准差；由 `trajectory_fusion/dataset.py` 合并 |
| roi_vpi_orb_confidence | float | sidecar `vpi_orb` 置信度；由 `trajectory_fusion/dataset.py` 合并 |
| roi_opencv_cuda_gftt_lk_support | int | sidecar `opencv_cuda_gftt_lk` 支撑点数；由 `trajectory_fusion/dataset.py` 合并 |
| roi_opencv_cuda_gftt_lk_std_px | float | sidecar `opencv_cuda_gftt_lk` 视差标准差；由 `trajectory_fusion/dataset.py` 合并 |
| roi_opencv_cuda_gftt_lk_confidence | float | sidecar `opencv_cuda_gftt_lk` 置信度；由 `trajectory_fusion/dataset.py` 合并 |
| roi_ring_edge_profile_support | int | CUDA ring/edge profile 有效采样点数 |
| roi_ring_edge_profile_std_px | float | CUDA ring/edge profile 视差标准差 |
| roi_ring_edge_profile_confidence | float | CUDA ring/edge profile 匹配置信度 |
| roi_neural_feature_support | int | 神经特征匹配支撑点数 |
| roi_neural_feature_std_px | float | 神经特征视差标准差 |
| roi_neural_feature_confidence | float | 神经特征匹配置信度 |
| roi_neural_xfeat_support | int | XFeat 神经特征匹配支撑点数 |
| roi_neural_xfeat_std_px | float | XFeat 神经特征视差标准差 |
| roi_neural_xfeat_confidence | float | XFeat 神经特征匹配置信度 |
| roi_neural_superpoint_support | int | SuperPoint 神经特征匹配支撑点数 |
| roi_neural_superpoint_std_px | float | SuperPoint 神经特征视差标准差 |
| roi_neural_superpoint_confidence | float | SuperPoint 神经特征匹配置信度 |
| fallback_feature_points_support | int | 单侧漏检特征 fallback 支撑点数 |
| fallback_feature_points_std_px | float | 单侧漏检特征 fallback 视差标准差 |
| fallback_feature_points_confidence | float | 单侧漏检特征 fallback 置信度 |
| stereo_match_source | int | 0=无,1=左右YOLO,2=左到右fallback,3=右到左fallback |
| stereo_depth_source | int | legacy `z_stereo` first-usable 选择来源；0=无,1=圆心/搜索,2=ROI多点,3=bbox中心,4=中心patch,5=边缘质心,6=bbox边缘,7=模板fallback,8=径向中心,9=边缘成对中心,10=角点特征,11=纹理特征,12=特征fallback,13=二值特征,14=ORB,15=BRISK,16=AKAZE,17=SIFT,18=彩色区域IoU,19=彩色边缘IoU,20=神经特征,21=CUDA模板匹配,22=CUDA StereoBM,23=CUDA StereoSGM,24=CUDA ring/edge profile |
| depth_method | int | 在线 legacy 输出的深度来源方法，0=单目,1=双目,2=融合；只用于诊断/baseline |

## 训练标签/诊断字段

这些不是强监督真值，而是用于自监督和排查：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| predicted_z | float | update 前预测深度 |
| innovation_z | float | 在线滤波器 z 创新，`raw_z - predicted_z` |
| innovation_norm | float | 归一化创新，`innovation_z / sqrt(Pzz_prior + Rz)` |
| kalman_sigma_z | float | update 后在线估计 z 标准差 |
| rejected_reason | str | 候选级被拒原因；当前 recorder 尚未写入，需先把 pipeline 内 gate 失败原因结构化 |
| landing_x,landing_y,landing_t | float | 当前落点预测 |

## Frame Summary Sidecar

实时 recorder 会从 `output_path` 派生 `*.frames.csv`，对每个进入 `TrajectoryRecorder::record()` 的结果回调帧写一行。这个文件不替代目标级 trajectory CSV；它提供的是已发布结果帧内的无输出和误匹配退化统计，不是 USB/触发采集帧总数。被 `drop_stale_roi_frames`、async ROI stale result 或 recorder 队列满丢弃的帧不会出现在这个 sidecar 中。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| result_count | int | 本帧输出目标数，0 表示实时管线没有可记录目标 |
| tracked_count | int | `track_id>=0` 的目标数 |
| raw_observation_count | int | 有未滤波观测的目标数 |
| stereo_observation_count | int | 有有效双目深度的目标数 |
| direct_pair_count | int | `stereo_match_source=1` 的目标数 |
| fallback_l2r_count | int | 左到右 fallback 目标数 |
| fallback_r2l_count | int | 右到左 fallback 目标数 |
| pair_positive_count | int | selected pair 正视差数量 |
| pair_shifted_iou_min, pair_shifted_iou_mean | float | selected pair shifted IoU 质量 |
| pair_score_mean, pair_bbox_prior_penalty_mean | float | selected pair 排序分数和物理视差惩罚 |
| pair_epipolar_dy_max | float | selected pair 最大 y 残差 |
| roi_iou_region_color_patch_support_max | int | 彩色区域 IoU/patch 最大支撑点数 |
| roi_patch_iou_color_edge_support_max | int | 彩色边缘 IoU/patch 最大支撑点数 |
| roi_neural_feature_support_max | int | 神经特征最大支撑点数 |
| p2_candidate_observed_count | int | 本帧已写入 `Object3D` 的 P2 候选观测数；由 z/support/confidence 粗判，不代表所有未触发算法 |
| p2_candidate_valid_count | int | 本帧 P2 候选中有效 z 数量 |
| p2_feature_valid_count | int | 本帧 CPU/feature/patch/fallback feature 类 P2 有效 z 数量 |
| p2_cuda_valid_count | int | 本帧 CUDA/OpenCV CUDA Template/BM/SGM/ring-edge 类 P2 有效 z 数量 |
| p2_neural_valid_count | int | 本帧神经特征 P2 有效 z 数量 |
| best_confidence | float | 本帧最高输出置信度 |

## P2 Diagnostic Sidecar

当 `performance.p2_diagnostic_results_enabled=true` 时，运行时会写同名前缀 `*.p2_diagnostic.csv`。如果 `p2_diagnostic_results_path` 为空，`main.cpp` 会从最终 `recording.output_path` 或 `--recording-out` 自动派生，例如 `traj_001.csv` -> `traj_001.p2_diagnostic.csv`。

这个文件不替代主 trajectory CSV，不回写 `Object3D`。`trajectory_fusion/dataset.py` 当前只把 `mode=vpi_template_match` 和 `mode=vpi_orb` 按 `frame_id` 合并为训练候选字段:

| sidecar 字段 | 类型 | 说明 |
| --- | --- | --- |
| frame_id | int | 对应主 CSV 的 frame id |
| lane | str | `diagnostic` |
| mode | str | 算法名，例如 `vpi_template_match`、`vpi_orb` |
| status | str | `valid`、`invalid`、`unsupported`、`no_pair` 等 |
| valid | int | 1=本行有有效视差/深度 |
| disparity | float | 聚合视差 |
| z_m | float | 由视差三角化得到的深度 |
| confidence | float | 算法置信度 |
| stddev | float | 视差标准差 |
| support | int | 支撑点/样本数 |
| attempted | int | 尝试点/样本数 |
| left_cx,left_cy,left_w,left_h,left_conf | float | 左检测框 |
| right_cx,right_cy,right_w,right_h,right_conf | float | 右检测框 |
| anchor_cx,anchor_cy,right_anchor_cx,right_anchor_cy | float | 算法聚合 anchor |
| debug_match_count | int | artifact 可视化点对数量 |
| artifact_path | str | diagnostic artifact PNG 路径；正式采集通常为空 |
| algo_ms,queue_wait_ms,worker_elapsed_ms | float | 算法耗时、排队等待和 worker 总耗时 |
| over_deadline | int | 是否超过 diagnostic deadline |

## 最小可用版本

如果先不想大改记录器，最低限度应补：

```text
frame_id,timestamp,left_frame_counter,right_frame_counter,track_id,
left_bbox_cx,left_bbox_cy,left_bbox_w,left_bbox_h,left_bbox_conf,
right_bbox_cx,right_bbox_cy,right_bbox_w,right_bbox_h,right_bbox_conf,
left_circle_cx,left_circle_cy,right_circle_cx,right_circle_cy,
disparity_bbox_center,disparity_bbox_left_edge,disparity_bbox_right_edge,
disparity_circle_center,disparity_circle_left_edge,disparity_circle_right_edge,
disparity_roi_edge_centroid,disparity_roi_radial_center,
disparity_roi_edge_pair_center,disparity_roi_corner_points,
disparity_roi_texture_points,disparity_roi_binary_points,
disparity_roi_orb_points,disparity_roi_brisk_points,disparity_roi_akaze_points,
disparity_roi_sift_points,disparity_roi_iou_region_color_patch,
disparity_roi_patch_iou_color_edge,
disparity_roi_cuda_template_match,disparity_roi_cuda_stereo_bm,
disparity_roi_cuda_stereo_sgm,disparity_roi_opencv_cuda_gftt_lk,
disparity_roi_ring_edge_profile,
disparity_roi_neural_feature,
disparity_roi_center_patch,
disparity_roi_multi_point,disparity_fallback_epipolar,disparity_fallback_template,
disparity_fallback_feature_points,
z_mono,z_bbox_center,z_bbox_left_edge,z_bbox_right_edge,
z_circle_center,z_circle_left_edge,z_circle_right_edge,
z_roi_edge_centroid,z_roi_radial_center,z_roi_edge_pair_center,
z_roi_corner_points,z_roi_texture_points,z_roi_binary_points,
z_roi_orb_points,z_roi_brisk_points,z_roi_akaze_points,z_roi_sift_points,
z_roi_iou_region_color_patch,z_roi_patch_iou_color_edge,
z_roi_cuda_template_match,z_roi_cuda_stereo_bm,z_roi_cuda_stereo_sgm,
z_roi_opencv_cuda_gftt_lk,z_roi_ring_edge_profile,
z_roi_neural_feature,z_roi_neural_xfeat,z_roi_neural_superpoint,
z_roi_center_patch,z_roi_multi_point,
z_fallback,z_fallback_epipolar,z_fallback_template,z_fallback_feature_points,z,
epipolar_dy,size_ratio,subpixel_valid,subpixel_attempted,subpixel_support,
subpixel_std_px,subpixel_confidence,subpixel_gate_px,
roi_corner_points_support,roi_corner_points_std_px,roi_corner_points_confidence,
roi_texture_points_support,roi_texture_points_std_px,roi_texture_points_confidence,
roi_binary_points_support,roi_binary_points_std_px,roi_binary_points_confidence,
roi_orb_points_support,roi_orb_points_std_px,roi_orb_points_confidence,
roi_brisk_points_support,roi_brisk_points_std_px,roi_brisk_points_confidence,
roi_akaze_points_support,roi_akaze_points_std_px,roi_akaze_points_confidence,
roi_sift_points_support,roi_sift_points_std_px,roi_sift_points_confidence,
roi_iou_region_color_patch_support,roi_iou_region_color_patch_std_px,
roi_iou_region_color_patch_confidence,roi_patch_iou_color_edge_support,
roi_patch_iou_color_edge_std_px,roi_patch_iou_color_edge_confidence,
roi_opencv_cuda_gftt_lk_support,roi_opencv_cuda_gftt_lk_std_px,
roi_opencv_cuda_gftt_lk_confidence,
roi_ring_edge_profile_support,roi_ring_edge_profile_std_px,roi_ring_edge_profile_confidence,
roi_neural_feature_support,roi_neural_feature_std_px,roi_neural_feature_confidence,
fallback_feature_points_support,fallback_feature_points_std_px,
fallback_feature_points_confidence,
pair_initial_disparity,pair_epipolar_dy,pair_y_tolerance,pair_size_ratio,
pair_shifted_iou,pair_score,pair_bbox_prior_penalty,pair_positive_disparity,
raw_observation_valid,
predicted_z,innovation_z,innovation_norm,kalman_sigma_z,
left_circle_source,right_circle_source,stereo_match_source,stereo_depth_source,
depth_method,confidence
```

这组字段已经足够学习 bbox 抖动、圆心质量、左右匹配质量、亚像素支撑度与深度稳定性的关系。
