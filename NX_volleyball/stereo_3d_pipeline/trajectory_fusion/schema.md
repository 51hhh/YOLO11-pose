# 多测距观测记录 Schema

当前旧 CSV 只能支持 baseline。下面是后续训练可靠性模型需要补齐的字段。建议优先写 CSV，稳定后切 Parquet/Arrow。

## 每帧公共字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| frame_id | int | pipeline 输出帧号 |
| timestamp | float | 主机时间或相机时间，秒 |
| left_frame_counter | int | 左目水印帧计数 |
| right_frame_counter | int | 右目水印帧计数 |
| left_ext_trigger_count | int | 左目外触发计数，可用时记录 |
| right_ext_trigger_count | int | 右目外触发计数，可用时记录 |
| sync_delta_frame | int | 左右帧计数差 |
| sync_delta_us | float | 左右相机时间戳差，微秒 |
| track_id | int | 目标轨迹 ID |
| class_id | int | 检测类别 |
| dt | float | 与上一输出帧间隔 |

## 检测和几何字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| left_det_valid | int | 左目 YOLO 是否有效 |
| right_det_valid | int | 右目 YOLO 是否有效 |
| left_conf | float | 左目检测置信度 |
| right_conf | float | 右目检测置信度 |
| left_cx,left_cy,left_w,left_h | float | 左目 bbox |
| right_cx,right_cy,right_w,right_h | float | 右目 bbox |
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
| z_circle | float | 左右圆心视差三角测距 |
| z_subpixel | float | ROI 多点亚像素视差测距 |
| z_fallback | float | 极线 fallback 测距 |
| z_fused_online | float | 当前在线 HybridDepth 输出 |
| disparity_circle | float | 圆心视差 |
| disparity_subpixel | float | 亚像素多点视差 |
| subpixel_valid | int | 亚像素测距是否接受 |
| subpixel_support | int | 接受的多点数量 |
| subpixel_std_px | float | 多点视差标准差 |
| subpixel_score | float | 多点匹配平均得分 |
| subpixel_gate_px | float | 当前动态视差门限 |
| method_selected | str | 在线最终选择的方法 |

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
left_cx,left_cy,left_w,left_h,left_conf,
right_cx,right_cy,right_w,right_h,right_conf,
left_circle_cx,left_circle_cy,right_circle_cx,right_circle_cy,
disparity_circle,disparity_subpixel,
z_mono,z_circle,z_subpixel,z_fused_online,
epipolar_dy,size_ratio,subpixel_valid,subpixel_support,subpixel_std_px,
fallback_used,method_selected,confidence
```

这组字段已经足够学习 bbox 抖动、圆心质量、左右匹配质量、亚像素支撑度与深度稳定性的关系。
