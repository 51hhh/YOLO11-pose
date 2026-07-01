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
| z_roi_center_patch | float | ROI 中心 patch ZNCC 视差三角测距 |
| z_roi_multi_point | float | ROI 多点 ZNCC 亚像素视差三角测距 |
| z_fallback | float | 极线 fallback 测距 |
| z_fallback_template | float | 单侧漏检时极线模板搜索 fallback 测距 |
| z_fallback_feature_points | float | 单侧漏检时极线特征点搜索 fallback 测距 |
| z_stereo | float | 当前在线选择的双目测距观测，不是独立取点方法 |
| z | float | 当前在线 HybridDepth 输出；raw_mode=true 时为未滤波观测 |
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
| disparity_roi_center_patch | float | ROI 中心 patch ZNCC 视差 |
| disparity_roi_multi_point | float | ROI 多点 ZNCC 亚像素视差 |
| disparity_fallback_template | float | 极线模板搜索 fallback 视差 |
| disparity_fallback_feature_points | float | 极线特征点搜索 fallback 视差 |
| z_yolo_bbox_pair,z_circle,z_subpixel | float | 旧兼容别名，分别对应 bbox_center/circle_center/roi_multi_point |
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
| fallback_feature_points_support | int | 单侧漏检特征 fallback 支撑点数 |
| fallback_feature_points_std_px | float | 单侧漏检特征 fallback 视差标准差 |
| fallback_feature_points_confidence | float | 单侧漏检特征 fallback 置信度 |
| stereo_match_source | int | 0=无,1=左右YOLO,2=左到右fallback,3=右到左fallback |
| stereo_depth_source | int | 0=无,1=圆心/搜索,2=ROI多点,3=bbox中心,4=中心patch,5=边缘质心,6=bbox边缘,7=模板fallback,8=径向中心,9=边缘成对中心,10=角点特征,11=纹理特征,12=特征fallback,13=二值特征,14=ORB,15=BRISK,16=AKAZE |
| depth_method | int | 在线最终选择的方法，0=单目,1=双目,2=融合 |

## 训练标签/诊断字段

这些不是强监督真值，而是用于自监督和排查：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| innovation_z | float | 在线滤波器 z 创新 |
| innovation_norm | float | 归一化创新 |
| rejected_reason | str | 被拒绝原因 |
| predicted_z | float | update 前预测深度 |
| kalman_sigma_z | float | 在线估计 z 方差 |
| landing_x,landing_y,landing_t | float | 当前落点预测 |

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
disparity_roi_center_patch,
disparity_roi_multi_point,disparity_fallback_template,
disparity_fallback_feature_points,
z_mono,z_bbox_center,z_bbox_left_edge,z_bbox_right_edge,
z_circle_center,z_circle_left_edge,z_circle_right_edge,
z_roi_edge_centroid,z_roi_radial_center,z_roi_edge_pair_center,
z_roi_corner_points,z_roi_texture_points,z_roi_binary_points,
z_roi_orb_points,z_roi_brisk_points,z_roi_akaze_points,z_roi_center_patch,z_roi_multi_point,
z_fallback,z_fallback_template,z_fallback_feature_points,z,
epipolar_dy,size_ratio,subpixel_valid,subpixel_attempted,subpixel_support,
subpixel_std_px,subpixel_confidence,subpixel_gate_px,
roi_corner_points_support,roi_corner_points_std_px,roi_corner_points_confidence,
roi_texture_points_support,roi_texture_points_std_px,roi_texture_points_confidence,
roi_binary_points_support,roi_binary_points_std_px,roi_binary_points_confidence,
roi_orb_points_support,roi_orb_points_std_px,roi_orb_points_confidence,
roi_brisk_points_support,roi_brisk_points_std_px,roi_brisk_points_confidence,
roi_akaze_points_support,roi_akaze_points_std_px,roi_akaze_points_confidence,
fallback_feature_points_support,fallback_feature_points_std_px,
fallback_feature_points_confidence,raw_observation_valid,
left_circle_source,right_circle_source,stereo_match_source,stereo_depth_source,
depth_method,confidence
```

这组字段已经足够学习 bbox 抖动、圆心质量、左右匹配质量、亚像素支撑度与深度稳定性的关系。
