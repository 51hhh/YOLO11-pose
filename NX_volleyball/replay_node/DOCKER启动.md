# Docker(humble) 启动：查看录制轨迹 + 预测落点

完整 wiki：[`../stereo_3d_pipeline/wiki/轨迹回放与落点可视化.md`](../stereo_3d_pipeline/wiki/轨迹回放与落点可视化.md)

## 推荐命令

```bash
docker start humble
xhost +local:docker

cd /home/rick/mid360/YOLO11-pose
SPEED=0.5 ./NX_volleyball/replay_node/start_replay_in_humble.sh \
  /home/rick/mid360/YOLO11-pose/NX_volleyball/stereo_3d_pipeline/test_logs/trajectory_dataset/p1_dy_regression_20260710_042552/traj.csv
```

默认：

- 连续多 track
- `landing_ekf` + `d0=-5.324`
- 相机高 0.50 m
- **不剔除远处点**
- 使用 `cyclonedds_local.xml`，避免旧网卡绑定失败

## 常用选项

```bash
# 慢放
SPEED=0.3 ./NX_volleyball/replay_node/start_replay_in_humble.sh <csv>

# 手动剔除 10m 外误检
MAX_RANGE_M=10 SPEED=0.5 ./NX_volleyball/replay_node/start_replay_in_humble.sh <csv>

# 只看某个 track
TRACK_ID=3 SPEED=0.5 ./NX_volleyball/replay_node/start_replay_in_humble.sh <csv>
```

## 旧 start_rviz.sh

```bash
cd /home/rick/mid360
./start_rviz.sh replay <csv>
```

若报 `192.168.31.72 does not match an available interface`，改用上面的 `start_replay_in_humble.sh`。

## RViz

Fixed Frame: `vision_world`

重点看：

- 蓝 `/ball/actual_path`：实测
- 橙 `/ball/predicted_path`：预测曲线
- 红 `/ball/landing`：预测落点

## 关闭

```bash
docker exec humble bash -c 'pkill -9 rviz2; pkill -9 -f replay_trajectory.py; pkill -9 static_transform'
```
