# ROS2 接口

最后核对: 2026-07-13

ROS2 支持是条件编译功能。只有 CMake 找到 `ament_cmake`、`rclcpp`、`geometry_msgs`、`nav_msgs`、`sensor_msgs`、`volleyball_interfaces` 时，`stereo_pipeline` 才会编译 ROS2 bridge，宏为 `HAS_ROS2`。

## 配置入口

YAML 节点:

```yaml
ros2:
  enable: true
  world_frame_id: "vision_world"
  base_frame_id: "base_link"
  odom_topic: "/odom"
  odom_timeout_sec: 0.5
```

话题配置:

```yaml
ros2:
  topics:
    ball_realtime: "/nx/debug/ball/realtime"
    ball_landing: "/nx/debug/ball/landing"
    predicted_path: "/nx/debug/ball/predicted_path"
    actual_path: "/nx/debug/ball/actual_path"
    ball_realtime_base: "/nx/debug/ball/realtime_base"
    ball_landing_base: "/nx/debug/ball/landing_base"
  nx_observation:
    enable: true
    topic: "/nx/ball/observation"
    frame_id: "nx_left_rectified_optical_frame"
    source_epoch_file: "/run/volleyball/nx_source_epoch"
```

视觉到世界平面变换:

```yaml
ros2:
  vision_to_world:
    swap_xy: false
    invert_x: false
    invert_y: false
    rotation_deg: 0.0
    translation_x: 0.0
    translation_y: 0.0
```

## GoalPoseBridge

文件:

- `src/ros/goal_pose_bridge.h`
- `src/ros/goal_pose_bridge.cpp`

发布内容:

| 话题 | 类型 | 坐标系 | 说明 |
|---|---|---|---|
| `/nx/debug/ball/realtime` | `geometry_msgs/PointStamped` | `world_frame_id` | 实时球位置 |
| `/nx/debug/ball/landing` | `geometry_msgs/PointStamped` | `world_frame_id` | 落点预测 |
| `/nx/debug/ball/predicted_path` | `nav_msgs/Path` | `world_frame_id` | 预测轨迹 |
| `/nx/debug/ball/actual_path` | `nav_msgs/Path` | `world_frame_id` | 实际轨迹 trail |
| `/nx/debug/ball/realtime_base` | `geometry_msgs/PointStamped` | `base_frame_id` | 机器人本体系实时球位置 |
| `/nx/debug/ball/landing_base` | `geometry_msgs/PointStamped` | `base_frame_id` | 机器人本体系落点 |
| `/nx/ball/observation` | `volleyball_interfaces/NxBallObservation` | `nx_left_rectified_optical_frame` | NX 当前帧原始双目观测；坐标基于当前标定的 `R1/P1` |
| `/nx/debug/auto_goal_pose` | `geometry_msgs/PoseStamped` | `base_frame_id` | 本机门控录制用 debug 目标，不接底盘 |

`/auto/goal_pose` 归 RDK `catch_controller` 独占。NX 侧代码会拒绝把本机门控目标发布到
`/auto/goal_pose`；如需录制门控结果，只能使用 `/nx/debug/auto_goal_pose`。

## Debug 控制门控

`/nx/debug/ball/landing` 保留所有有效落点候选，用于 RViz 和诊断；`/nx/debug/auto_goal_pose`
只有在以下条件全部满足时才会发布，且仅用于本机录制审计：

1. `LandingPrediction.valid=true`；
2. 默认只允许 Student-t EKF + RK4 主干，多项式 fallback 只供 RViz；
3. 置信度、Student-t 权重、速度和剩余落地时间满足阈值；
4. 同一 track 的落点连续多帧稳定，帧间跳变不超过阈值；
5. 相机平面范围合法：`|X| <= 3.6m`、`0 < depth <= 14m`；
6. `vision_to_world` 变换结果有限；
7. `/odom` 数值有效且新鲜，world 到 `base_link` 变换成功；
8. 最终 base 坐标为有限值。

配置位于 `pipeline_dual_yolo_roi.yaml`：

```yaml
ros2:
  control_goal:
    enable: false
    topic: "/nx/debug/auto_goal_pose"
    min_depth_m: 0.0
    max_depth_m: 14.0
    max_abs_x_m: 3.6
    min_confidence: 0.70
    min_time_to_land_s: 0.25
    max_time_to_land_s: 2.20
    min_speed_mps: 0.80
    min_student_w: 0.15
    stable_frames: 3
    max_stable_jump_m: 0.35
    allow_polynomial: false
    allow_fallback_observation: false
```

拒绝和成功发布都会输出节流日志。没有新鲜 `/odom` 时，RViz 落点仍发布，
但不会发送 debug gate 目标。

订阅:

| 话题 | 类型 | 说明 |
|---|---|---|
| `/odom` | `nav_msgs/Odometry` | world 到 base_link 变换 |

`base_link` 发布依赖新鲜 odom。超过 `odom_timeout_sec` 时不应把旧 odom 当作有效变换。
odom 回调会拒绝非有限位置、非有限四元数和零长度四元数，并通过互斥快照避免
ROS spin 线程与视觉结果线程并发读写 pose。

车静止、执行器关闭、临时使用零 odom 时，可运行：

```bash
bash scripts/test_control_gate_zero_odom.sh 30
```

脚本默认检测 `/auto/goal_pose` 是否已有订阅者；存在控制订阅者时会拒绝启动。

## 录制门控审计

`scripts/nx_p1_dy_regression.sh` 使用 ROS2 版本的实时管线，并在 `traj.csv`
末尾追加以下字段。即使 `recording.raw_mode=true`，这些预测字段也会写入：

- `pred_valid_ungated`、`pred_x_ungated`、`pred_y_ungated`、`pred_t_ungated`：门控前落点；
- `pred_method`、`pred_confidence`、`pred_speed_mps`、`pred_student_w`、`pred_obs_source`；
- `control_gate_selected`、`control_gate_passed`、`control_gate_reason`；
- `control_gate_stable_frames`；
- `pred_x_gated`、`pred_y_gated`、`pred_t_gated`：门控通过后的相机坐标落点；
- `control_base_x`、`control_base_y`：实际发布到 `/nx/debug/auto_goal_pose` 的 base 坐标。

`control_gate_reason` 编码：

| 值 | 原因 |
|---:|---|
| 0 | 未评估 |
| 1 | 通过并发布 |
| 2 | 控制门控未启用 |
| 3 | 非有限输入 |
| 4 | 预测质量不满足阈值 |
| 5 | 超出相机范围 |
| 6 | 连续稳定帧不足或跳变过大 |
| 7 | vision-to-world 结果无效 |
| 8 | odom 缺失或过期 |
| 9 | base 坐标无效 |

录制脚本会生成 `control_gate_report.txt`。若运行时没有 `/odom` 发布者，脚本会在
车辆静止测试条件下自动发布 20Hz 零 odom，并在 `odom_source.txt` 中记录来源。

## DiagnosticPublisher

文件:

- `src/ros/diagnostic_publisher.h`
- `src/ros/diagnostic_publisher.cpp`

配置:

```yaml
ros2:
  diagnostic:
    enable: true
    depth_full_divisor: 6
```

诊断输出:

| 话题 | 类型 | 说明 |
|---|---|---|
| `/debug/depth_full` | `sensor_msgs/Image`, `32FC1` | 完整深度图，按 divisor 降频 |
| `/debug/depth_roi` | `sensor_msgs/Image`, `32FC1` | bbox 裁剪深度图 |
| `/debug/detections` | `sensor_msgs/Image`, `8UC3` | 带 bbox 标注的左图 |
| `/debug/raw_obs` | `geometry_msgs/PoseArray` | 原始 3D 观测 |

诊断发布会引入 GPU 到 CPU 拷贝和 ROS2 序列化开销。100fps 性能测试时应分别测 `diagnostic.enable=true/false`。

## 维护要求

- 新增话题必须同步更新本页和配置示例。
- 坐标系变换只在 `GoalPoseBridge` 集中维护，不要在多个调用点重复写变换逻辑。
- 录 rosbag 做算法评估时，必须同时记录配置文件和标定文件版本。
