# 排球双目 3D 视觉系统（Jetson Orin NX → RDK）

[![Language](https://img.shields.io/badge/Language-C%2B%2B17%20%7C%20CUDA%20%7C%20Python-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20NX-lightgrey.svg)]()
[![Runtime](https://img.shields.io/badge/Runtime-ROS2%20Humble%20%7C%20TensorRT-orange.svg)]()

本仓库的当前主线是在 **Jetson Orin NX** 上以 90 Hz 硬件触发海康双目相机，完成图像校正、双路 YOLO 检测、ROI 亚像素视差、3D 观测生成，并通过 ROS2 将带采集时间戳和质量信息的观测交给 **RDK**。RDK 负责 Student-t 滤波、弹道预测与控制。

> 当前有效代码位于 [`NX_volleyball/stereo_3d_pipeline`](NX_volleyball/stereo_3d_pipeline)。
> 正式工程文档以 [`wiki/Home.md`](NX_volleyball/stereo_3d_pipeline/wiki/Home.md) 为入口；`docs/`、`.agent/` 和仓库内其他早期实现只作为历史资料或专项工具，不代表当前生产路径。

## 当前系统

```text
90 Hz GPIO/PWM 硬触发
        ↓
海康双目同步采集（1440×1080 Bayer）
        ↓
CUDA 双目校正 + BGR 转换
        ↓
左右 YOLO26 TensorRT FP16（640×640，GPU）
        ↓
双目 bbox 语义配对与极线/尺寸/IoU 门控
        ↓
ROI CUDA 多点 ZNCC + 亚像素视差（当前主深度）
        ↓
深度候选回退 + HybridDepth 关联/创新门控
        ↓
NxBallObservation（原始 3D 观测、协方差、采集时间戳）
        ↓
RDK Student-t 滤波、落点预测和控制
```

当前正式联合运行配置是 [`pipeline_rdk_joint.yaml`](NX_volleyball/stereo_3d_pipeline/config/pipeline_rdk_joint.yaml)。关键运行状态如下：

| 项目 | 当前值 |
|---|---|
| 相机 | 海康双目，1440×1080，硬件触发 |
| 触发/目标帧率 | 90 Hz / 90 FPS |
| 校正 | CUDA Remap |
| 检测 | 左右双路 YOLO26，TensorRT FP16，GPU，640×640 |
| 双目策略 | `roi_only` |
| 首选深度 | `roi_subpixel_match` → `ROI_MULTI_POINT` |
| 有效距离 | `[0.8, 15.0)` m |
| 标定基线 | 运行时以 `stereo_calib.yaml` 为准；当前参考值 853.154 mm |
| 视差零点修正 | 当前联合配置关闭；使用 `z = fB / disparity` |
| NX 在线落点预测 | 关闭 |
| ROS2 主输出 | `/nx/ball/observation` |
| 联合运行网络 | NX `10.43.0.10`，RDK `10.43.0.20`，`ROS_DOMAIN_ID=42` |

## 当前深度方法

### 1. 左右目标配对

左右相机分别运行 YOLO。候选框必须依次通过：

- 类别一致；
- 正视差且不超过 `max_disparity`；
- 校正后的极线 y 偏差门限；
- 左右框宽高比例门限；
- 按初始视差平移后的 bbox IoU 门限。

配对成功后才进入 ROI 深度计算。单侧漏检时保留有界极线搜索能力，但正式 ROS2 配置设置了 `allow_fallback: false`，退化匹配不会发布给 RDK。

### 2. 主深度：ROI 多点亚像素视差

正式配置的 `depth_solver` 为 `roi_subpixel_match`。直接双目配对成功时，管线首选 `ROI_MULTI_POINT`：

1. 在排球 ROI 内选择最多 12 个采样点；
2. 使用 CUDA patch 匹配计算多点视差；
3. 对有效匹配做亚像素细化；
4. 使用 support、confidence、视差标准差、与初始视差/深度的一致性及 1.5 ms 时间预算做质量门控；
5. 使用标定文件中的焦距和基线计算 `z = fB / d`。

该方法对应输出字段 `z_roi_multi_point`、`disparity_roi_multi_point`，以及 `stereo_depth_source=2`。

### 3. 当前启用的候选与回退

生产配置会同时计算若干廉价候选，用于质量诊断或主方法失效时回退：

| 候选 | 当前状态 | 在线用途 |
|---|---:|---|
| ROI 多点亚像素 `ROI_MULTI_POINT` | 开 | 直接配对的首选深度 |
| ROI 中心 patch | 开 | 记录/诊断候选，不主动抢占主输出 |
| 圆心三角测量 | 开 | 主方法无效后的第一回退 |
| 径向梯度中心 | 开 | 几何回退 |
| 成对边缘中心 | 开 | 几何回退 |
| 边缘质心 | 开 | 几何回退 |
| bbox 中心/边缘视差 | 开 | 最后几何回退及诊断 |
| 单侧极线搜索 | 开 | 退化回退；当前不发给 RDK |

直接配对时的实际选择逻辑是：

```text
ROI_MULTI_POINT
  → circle_center
  → roi_radial_center
  → roi_edge_pair_center
  → roi_edge_centroid
  → bbox_center / bbox_edges
```

传统全帧/稠密 SGM、CUDA StereoBM/SGM、VPI Stereo、NCC 模板、XFeat、SuperPoint、ALIKED、ORB、BRISK、AKAZE、SIFT 和其他 P2 特征方法在当前联合配置中均关闭。它们保留用于专项 A/B、离线评估或后续训练，不是现在的生产深度来源。

### 4. HybridDepth 与 RDK 的职责边界

ROI 深度进入 `HybridDepthEstimator` 后：

- 小于 0.8 m 时使用单目球尺寸深度；
- 0.8–1.2 m 过渡区对单目/双目做逆方差加权；
- 大于 1.2 m 时使用选中的双目深度；
- 9 维常加速度 Kalman 状态用于目标关联、速度估计和创新门控；
- 当前关闭 stereo bias EMA，双目尺度不再向单目 bbox 深度校正。

NX 发布的是通过上述范围、连续性与创新门控后的 `raw_x/raw_y/raw_z`，而不是把 NX 的 Kalman 平滑位置当作 RDK 测量值。消息同时携带深度方法、视差、估计协方差、左右相机设备时间戳和映射后的采集时间戳。RDK 再完成最终的 Student-t 滤波、弹道外推和控制。

## 目录结构

| 路径 | 定位 | 当前状态 |
|---|---|---|
| `NX_volleyball/stereo_3d_pipeline/` | C++/CUDA 实时双目主程序 | **当前主线** |
| `NX_volleyball/stereo_3d_pipeline/wiki/` | 架构、配置、标定、部署和测试文档 | **正式文档** |
| `NX_volleyball/stereo_3d_pipeline/config/` | 生产、诊断与采集配置 | **当前配置源** |
| `NX_volleyball/stereo_3d_pipeline/trajectory_fusion/` | 多候选可靠性、EKF/RTS、落点预测离线实验 | 活跃辅助模块 |
| `NX_volleyball/calibration/` | 本机离线标定和验证工具 | 辅助工具 |
| `volleyball_tracking/` | YOLO11-Pose 训练、圆拟合、单目追踪 | 早期训练链路 |
| `TensorYOLO8Volleyball/` | 早期 TensorRT/ROS C++ 实现，含独立 Git 元数据 | 历史实现 |
| `Stereo/` | 双目方案文档 | 历史参考 |
| `NX_volleyball/docs/`、`.agent/` | 优化报告和旧设计记录 | 历史参考 |

不要从根 README 中推断旧 `ros2_ws/src/volleyball_stereo_driver` 路径；当前程序是 `stereo_3d_pipeline` 中由 CMake 构建的 `stereo_pipeline`。

## 配置入口

| 配置文件 | 用途 |
|---|---|
| `config/pipeline_rdk_joint.yaml` | NX→RDK 正式联合运行；90 Hz、ROS2 发布、关闭本机落点预测和诊断录制 |
| `config/pipeline_dual_yolo_roi.yaml` | 双目深度算法、可视化和性能专项调试 |
| `config/pipeline_record_p0p1.yaml` | P0/P1 候选轨迹数据采集 |
| `config/pipeline.yaml` | 基础参考配置，不作为当前联合部署入口 |

修改运行路径、深度候选、CSV schema、标定流程或 ROS2 接口后，应同步更新对应 Wiki 页面。一次性实验结论和日志放入 `docs/` 或 `test_logs/`，不要覆盖正式配置语义。

## 构建

目标环境：JetPack 6.x、CUDA 12.6+、TensorRT 10.3+、VPI 3.2+、OpenCV 4.x、海康 MVS SDK、yaml-cpp、libgpiod 和 ROS2 Humble。

```bash
cd NX_volleyball/stereo_3d_pipeline
source /opt/ros/humble/setup.bash
source ~/volleyball_ros2_ws/install/setup.bash

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH=87 \
  -DREQUIRE_ROS2=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

主要构建产物：

- `stereo_pipeline`：实时双目主程序；
- `stereo_calibrate`：双目标定求解；
- `capture_chessboard`：海康双目标定采集；
- `stereo_depth_viewer`：深度算法查看与基准测试；
- `hik_frame_metadata_probe`：相机水印/时间戳检查。

## NX→RDK 联合运行

首次部署先在 RDK 配置 Chrony server，然后在 NX 安装直连 Chrony client：

```bash
cd ~/NX_volleyball/stereo_3d_pipeline
sudo scripts/install_rdk_chrony_client.sh
```

正式启动统一使用：

```bash
cd ~/NX_volleyball/stereo_3d_pipeline
VOLLEYBALL_ROS_SETUP=~/volleyball_ros2_ws/install/setup.bash \
scripts/start_rdk_joint_runtime.sh
```

启动脚本会检查 NX 与 RDK 的时钟偏差是否小于 2 ms，配置 CycloneDDS 单播直连，启动 `stereo_pipeline`，等待生成本次进程唯一的 `source_epoch`，随后在 NX 启动 `nx_time_sync_publisher`。RDK 不应重复启动该时间同步发布器。

当前相机设备时间戳已映射到 NX `CLOCK_REALTIME`。映射未预热或不确定度超过 2 ms 时，消息会标为无效并由 RDK 拒绝；曝光中点偏移仍待实测，因此 `timestamp_offset_us` 目前为 0。

## 文档导航

- [正式 Wiki 首页](NX_volleyball/stereo_3d_pipeline/wiki/Home.md)
- [项目总览](NX_volleyball/stereo_3d_pipeline/wiki/项目总览.md)
- [系统架构](NX_volleyball/stereo_3d_pipeline/wiki/系统架构.md)
- [实时管线](NX_volleyball/stereo_3d_pipeline/wiki/实时管线.md)
- [深度算法导航](NX_volleyball/stereo_3d_pipeline/wiki/深度算法导航.md)
- [配置参考](NX_volleyball/stereo_3d_pipeline/wiki/配置参考.md)
- [配置矩阵](NX_volleyball/stereo_3d_pipeline/wiki/配置矩阵.md)
- [ROS2 接口](NX_volleyball/stereo_3d_pipeline/wiki/ROS2接口.md)
- [部署与运行](NX_volleyball/stereo_3d_pipeline/wiki/部署与运行.md)
- [实机测试清单](NX_volleyball/stereo_3d_pipeline/wiki/实机测试清单.md)
- [故障排查](NX_volleyball/stereo_3d_pipeline/wiki/故障排查.md)

## 历史训练链路

`volleyball_tracking` 保留了 YOLO11-Pose 的 5 关键点数据定义、训练、圆拟合、ByteTrack/Kalman 和 TensorRT 导出代码。它仍可用于训练与算法参考，但不等同于当前 NX 生产管线；当前生产检测器是双路 YOLO26 TensorRT，深度来自双目 ROI 亚像素匹配。
