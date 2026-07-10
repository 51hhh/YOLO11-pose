# 轨迹回放 + 落点预测 (RViz)

读取 `traj.csv`，按当前落点方案做因果预测，并发布到 ROS2 / RViz。

完整说明见 wiki：

[`../stereo_3d_pipeline/wiki/轨迹回放与落点可视化.md`](../stereo_3d_pipeline/wiki/轨迹回放与落点可视化.md)

## 默认方案

```text
连续多 track 时间线
  -> bbox_center + d0 反投影
  -> Student-t EKF (Cd=0.10)
  -> RK4 落点
  -> /ball/* topics
```

- 默认 **不剔除** 远处点
- 默认 **不锁单 track**，按时间连续拼接
- 当前数据集相机高 **0.50 m**
- runtime `d0=-5.324`

## 推荐启动（Docker humble）

```bash
cd /home/rick/mid360/YOLO11-pose
SPEED=0.5 ./NX_volleyball/replay_node/start_replay_in_humble.sh \
  NX_volleyball/stereo_3d_pipeline/test_logs/trajectory_dataset/p1_dy_regression_20260710_042552/traj.csv
```

手动剔除远处误检：

```bash
MAX_RANGE_M=10 SPEED=0.5 ./NX_volleyball/replay_node/start_replay_in_humble.sh <csv>
```

## 本机 ROS2

```bash
source /opt/ros/humble/setup.bash
python3 replay_node/replay_trajectory.py \
  --csv <traj.csv> \
  --mode landing_ekf \
  --camera-height 0.50 \
  --speed 0.5 \
  --loop
```

另开终端：

```bash
rviz2 -d /home/rick/mid360/config/volleyball_view.rviz
# 或
rviz2 -d replay_node/ball_landing.rviz
```

Fixed Frame: `vision_world`

## 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--continuous` | 开 | 多 track 连续拼接 |
| `--track-id N` | 空 | 只播一个 track |
| `--max-range-m` | `0` | `>0` 才剔除更远点 |
| `--camera-height` | `0.50` | 相机高度 |
| `--mode` | `landing_ekf` | 当前落点方案 |

## 话题

`/ball/realtime` `/ball/actual_path` `/ball/filtered_path`
`/ball/predicted_path` `/ball/landing` `/ball/ground_plane`
`/ball/ball_marker` `/ball/landing_marker`

## 键盘

`Space` 暂停，`←/→` 逐帧，`+/-` 变速，`r` 重头，`q` 退出
前进续算；后退重建。
