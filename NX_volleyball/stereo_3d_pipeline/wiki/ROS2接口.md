# ROS2 接口

最后核对: 2026-07-02

ROS2 支持是条件编译功能。只有 CMake 找到 `ament_cmake`、`rclcpp`、`geometry_msgs`、`nav_msgs`、`sensor_msgs` 时，`stereo_pipeline` 才会编译 ROS2 bridge，宏为 `HAS_ROS2`。

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
    ball_realtime: "/ball/realtime"
    ball_landing: "/ball/landing"
    predicted_path: "/ball/predicted_path"
    actual_path: "/ball/actual_path"
    ball_realtime_base: "/ball/realtime_base"
    ball_landing_base: "/ball/landing_base"
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
| `/ball/realtime` | `geometry_msgs/PointStamped` | `world_frame_id` | 实时球位置 |
| `/ball/landing` | `geometry_msgs/PointStamped` | `world_frame_id` | 落点预测 |
| `/ball/predicted_path` | `nav_msgs/Path` | `world_frame_id` | 预测轨迹 |
| `/ball/actual_path` | `nav_msgs/Path` | `world_frame_id` | 实际轨迹 trail |
| `/ball/realtime_base` | `geometry_msgs/PointStamped` | `base_frame_id` | 机器人本体系实时球位置 |
| `/ball/landing_base` | `geometry_msgs/PointStamped` | `base_frame_id` | 机器人本体系落点 |

订阅:

| 话题 | 类型 | 说明 |
|---|---|---|
| `/odom` | `nav_msgs/Odometry` | world 到 base_link 变换 |

`base_link` 发布依赖新鲜 odom。超过 `odom_timeout_sec` 时不应把旧 odom 当作有效变换。

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
