# 实时落点预测（Student-t EKF）

最后更新: 2026-07-10

## 在线路径

```text
bbox_center disparity (+ circle fallback/一致性)
  -> d0 反投影 3D 观测
  -> 按 track 的 Student-t EKF（重力 + 二次阻力, Cd=0.10）
  -> RK4 rollout 到地面
  -> 多项式历史拟合仅作备份
```

对应代码:

- `src/fusion/trajectory_predictor.h/.cpp`
- 配置加载: `src/main_recording_config_loaders.cpp::loadPredictorConfig`
- 运行配置: `config/pipeline_dual_yolo_roi.yaml` 的 `prediction:` 段
- Python 离线对照: `trajectory_fusion/landing_pipeline/`

## 观测形成

1. 优先 `disparity_bbox_center + left_bbox_cx/cy`，用标定 `fB` 与 runtime `d0` 反投影。
2. bbox 无效时才用 `circle_center`。
3. 再不行才退到 `raw_x/raw_y/raw_z` 或 HybridDepth 滤波状态。
4. 多路 `z_*` **不会**同时作为独立观测进入 EKF。

## 关键参数

| 参数 | 默认 | 含义 |
|---|---:|---|
| `drag_coeff` | 0.10 | 排球阻力系数（本数据速度区） |
| `student_t_nu` | 12 | Student-t 自由度，越大越接近高斯 |
| `sigma_d_px` | 0.4 | 视差噪声，用于 `σz ∝ z²/fB` |
| `q_vel` | 1.5 | 速度过程噪声 |
| `prefer_bbox` | true | bbox 主观测 |
| `ground_y` | 1.26 | 仅 `use_g_hat=false` 时的 Y-down 地面 |
| `use_g_hat` / `g_hat` / `ground_h` | 已启用 | 2026-07-09 实测平面（throws_gt） |

几何自动来自:

- `calibration.file` → fx/fy/cx/cy/fB
- `disparity_offset.file` → d0

## 输出兼容

`LandingPrediction` 仍保持旧接口:

- `x` = 落点相机 X
- `y` = 落点相机 Z（深度）
- `z` = 落点相机 Y（新增调试字段）
- `method=0` Student-t EKF 弹道
- `method=1` 多项式备份

ROS `/ball/landing` 与显示 OSD 无需改 topic 名。

## 未接入

- TinyGRU 落点残差（Python 可选模块已有；C++ 实时默认关闭，避免 torch 依赖）
- 多深度候选并行观测融合

## 验证建议（NX）

```bash
# 重新编译后
./stereo_pipeline --config config/pipeline_dual_yolo_roi.yaml
# 日志应出现:
# TrajectoryPredictor: mode=student_t_ekf Cd=0.100 ... d0=-5.324 ...
```

离线对照同一 CSV:

```bash
python -m trajectory_fusion.landing_pipeline.run_csv   test_logs/p1_dy_regression_20260709_143434/traj.csv   --config trajectory_fusion/configs/landing_pipeline_bbox_ekf.json   --no-residual
```

## 离线 RViz 回放

录制 CSV 的可视化回放与 Docker 启动方法见：

- [轨迹回放与落点可视化](轨迹回放与落点可视化.md)

默认回放链路与实时一致：`bbox_center + d0=-5.324 + Student-t EKF + RK4`。
