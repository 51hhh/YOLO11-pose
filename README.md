# YOLO11 Pose 排球关键点检测与双目追踪系统

[![Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B17-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20NX%20%7C%20Linux-lightgrey.svg)]()
[![Framework](https://img.shields.io/badge/Framework-ROS2%20Humble%20%7C%20TensorRT-orange.svg)]()

基于 **YOLOv11-Pose** 的排球视觉项目集合，覆盖从 **5 关键点数据标注与模型训练**、到 **TensorRT 导出与几何拟合追踪**、再到 **Jetson Orin NX + ROS2 + 海康双目相机** 的高速 3D 部署链路。仓库目前包含两条相互关联的实现路线：一条是偏算法与模型训练的 `volleyball_tracking`，另一条是面向实机部署的 `NX_volleyball`。

> **项目定位**：这是一个“训练 + 部署”一体化作品集仓库，而不是单一程序。
> - `volleyball_tracking`：单目排球关键点检测、圆拟合、ByteTrack/卡尔曼追踪、TensorRT 导出
> - `NX_volleyball`：Jetson Orin NX 上的 ROS2 双目高速追踪系统，集成 PWM 触发、海康双目同步采集、YOLO TensorRT 检测、立体匹配与 3D 卡尔曼滤波

## 项目全景

| 子项目 | 角色 | 说明 |
|------|------|------|
| **volleyball_tracking** | 算法 / 训练链路 | YOLOv11n-Pose 训练、5 关键点定义、加权最小二乘拟合圆、ByteTrack + Kalman 追踪、TensorRT 导出 |
| **NX_volleyball** | 实机部署链路 | Jetson Orin NX + ROS2 Humble + TensorRT + 海康双目相机，高速 3D 排球追踪 |

## 特性

- **YOLOv11-Pose 排球关键点方案**：将排球定义为 1 类目标 + 5 个关键点，便于后续几何建模与轨迹估计
- **几何拟合增强定位**：基于关键点做加权最小二乘圆拟合，Center 点加权，提高圆心与半径估计精度
- **追踪模块解耦**：Python 侧提供几何拟合、ByteTrack、卡尔曼滤波和可视化组件，便于单独验证算法
- **TensorRT 部署链路**：提供从 `.pt` 到 `.engine` 的转换脚本，适配 Jetson Orin NX 推理部署
- **双目 3D 追踪系统**：ROS2 包整合 PWM 触发、海康双目同步采集、YOLO 检测、立体匹配、3D 位置/速度发布
- **高频低延迟架构**：`NX_volleyball` 中实现双缓冲 + 条件变量 + CUDA Bayer 预处理 + Batch=2 推理优化
- **工程化脚本与文档**：包含训练配置、部署依赖、启动脚本、标定工具与多份项目说明文档

## 仓库结构

```text
YOLO11-pose/
├── README.md                          # 仓库总览（本文档）
├── volleyball_tracking/               # 训练与单目追踪链路
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── PROJECT_SUMMARY.md
│   ├── data/
│   │   └── dataset.yaml               # YOLO-Pose 数据集配置（5关键点）
│   ├── train/
│   │   ├── train.py                   # Ultralytics YOLOv11-Pose 训练脚本
│   │   ├── config.yaml                # 训练超参数配置
│   │   └── requirements.txt
│   ├── deploy/
│   │   ├── export_tensorrt.py         # TensorRT 引擎导出
│   │   ├── geometry.py                # 圆拟合算法
│   │   ├── tracker.py                 # ByteTrack + Kalman 追踪
│   │   ├── visualizer.py              # 结果可视化
│   │   └── requirements_nx.txt
│   ├── docs/                          # 数据/训练/导出/部署文档
│   ├── demo/                          # 预留演示目录
│   ├── tools/                         # 预留工具目录
│   └── models/                        # 模型目录说明
└── NX_volleyball/                     # Jetson Orin NX 双目高速部署系统
    ├── README.md
    ├── calibration/                   # 双目标定工具与标定文件
    ├── scripts/                       # 测试/优化/模型转换脚本
    └── ros2_ws/src/volleyball_stereo_driver/
        ├── CMakeLists.txt
        ├── package.xml
        ├── include/volleyball_stereo_driver/
        ├── src/
        │   ├── volleyball_tracker_node.cpp
        │   ├── volleyball_visualizer_node.cpp
        │   ├── yolo_detector.cpp
        │   ├── yolo_preprocessor.cu
        │   ├── stereo_matcher.cpp
        │   ├── kalman_filter_3d.cpp
        │   ├── high_precision_pwm.cpp
        │   ├── hik_camera_wrapper.cpp
        │   └── roi_manager.cpp
        ├── config/
        ├── calibration/
        ├── model/
        └── scripts/
```

## 核心技术路线

### 1. `volleyball_tracking`：关键点检测 + 圆拟合 + 追踪

```text
输入图像
  ↓
YOLOv11n-Pose
  ↓
5关键点 (Center / Top / Bottom / Left / Right)
  ↓
加权最小二乘拟合圆
  ↓
ByteTrack 多目标追踪
  ↓
卡尔曼滤波平滑
  ↓
输出 (cx, cy, r, vx, vy, track_id)
```

关键点定义来自 `volleyball_tracking/data/dataset.yaml:15`：

```text
     Top
      ↑
Left ← Center → Right
      ↓
    Bottom
```

其中：
- `Center` 为球心点，拟合时会额外加权，见 `volleyball_tracking/deploy/geometry.py:53`
- `Left/Right` 在水平翻转时会互换，见 `volleyball_tracking/data/dataset.yaml:26`
- 追踪器使用 **ByteTrack 两阶段匹配 + 8 维状态卡尔曼滤波**，见 `volleyball_tracking/deploy/tracker.py:74`、`volleyball_tracking/deploy/tracker.py:233`

### 2. `NX_volleyball`：双目同步 + 3D 追踪

```text
PWM 触发
  ↓
海康双目相机同步采集
  ↓
双缓冲 + 条件变量唤醒
  ↓
YOLO TensorRT 检测（支持 Batch=2）
  ↓
双目立体匹配 / 三角测量
  ↓
3D 卡尔曼滤波
  ↓
ROS2 发布 pose / velocity / 调试图像
```

`volleyball_tracker_node` 是一个 **All-in-One 主节点**，在同一个节点中完成：
- PWM 触发
- 双目相机采集
- TensorRT 推理
- 立体匹配
- 3D 追踪

可从 `NX_volleyball/ros2_ws/src/volleyball_stereo_driver/CMakeLists.txt:76` 看到该节点的构建入口；相关发布器定义位于 `NX_volleyball/ros2_ws/src/volleyball_stereo_driver/src/volleyball_tracker_node.cpp:342`。

## `volleyball_tracking` 详解

### 训练配置

训练脚本基于 Ultralytics YOLO，入口为：
- `volleyball_tracking/train/train.py:1`
- `volleyball_tracking/train/config.yaml:1`

当前默认训练配置：
- 模型：`yolov11n-pose.pt`，见 `volleyball_tracking/train/config.yaml:5`
- 任务类型：`pose`，见 `volleyball_tracking/train/config.yaml:6`
- 输入尺寸：`640`，见 `volleyball_tracking/train/config.yaml:10`
- 训练轮数：`100`，见 `volleyball_tracking/train/config.yaml:13`
- 优化器：`AdamW`，见 `volleyball_tracking/train/config.yaml:20`
- pose loss 权重：`12.0`，见 `volleyball_tracking/train/config.yaml:35`

训练完成后，脚本会把最佳模型复制到 `../models/volleyball_best.pt`，见 `volleyball_tracking/train/train.py:162`。

### 几何拟合

`volleyball_tracking/deploy/geometry.py:10` 提供 `CircleFitter`，支持：
- `weighted_lsq`：加权最小二乘
- `ransac`：鲁棒拟合
- `algebraic`：代数拟合

其中默认方法会对球心关键点加权两倍，见 `volleyball_tracking/deploy/geometry.py:53`。

### 追踪模块

`volleyball_tracking/deploy/tracker.py:36` 的 `VolleyballTracker` 实现了：
- 高分/低分检测分离的 ByteTrack 两阶段匹配，见 `volleyball_tracking/deploy/tracker.py:74`
- 基于圆 IoU 的匹配策略，见 `volleyball_tracking/deploy/tracker.py:163`
- 8 维状态卡尔曼滤波，见 `volleyball_tracking/deploy/tracker.py:233`

### TensorRT 导出

`volleyball_tracking/deploy/export_tensorrt.py:1` 用 Ultralytics 导出 TensorRT 引擎，支持：
- `--fp16`
- `--int8`
- `--workspace`

导出成功后会把 `.engine` 复制到 `../models/`，见 `volleyball_tracking/deploy/export_tensorrt.py:78`。

## `NX_volleyball` 详解

### ROS2 包组成

核心 ROS2 包为 `volleyball_stereo_driver`：
- CMake：`NX_volleyball/ros2_ws/src/volleyball_stereo_driver/CMakeLists.txt:1`
- 包描述：`NX_volleyball/ros2_ws/src/volleyball_stereo_driver/package.xml:1`

主要节点：
- `volleyball_tracker_node`：主追踪节点，见 `CMakeLists.txt:81`
- `volleyball_visualizer_node`：可视化节点，见 `CMakeLists.txt:118`

### 主要依赖

从 `CMakeLists.txt` 和 `package.xml` 可确认依赖：
- ROS2 Humble
- OpenCV
- libgpiod
- CUDA（必需），见 `CMakeLists.txt:26`
- TensorRT（必需），见 `CMakeLists.txt:38`
- 海康 MVS SDK `/opt/MVS`，见 `CMakeLists.txt:53`

### 发布话题

`volleyball_tracker_node` 会发布：
- `volleyball/pose_3d`，见 `volleyball_tracker_node.cpp:342`
- `volleyball/velocity`，见 `volleyball_tracker_node.cpp:344`
- `camera_trigger`，见 `volleyball_tracker_node.cpp:346`
- `volleyball/debug_info`，见 `volleyball_tracker_node.cpp:350`
- `stereo/left/image_raw`，见 `volleyball_tracker_node.cpp:354`
- `stereo/right/image_raw`，见 `volleyball_tracker_node.cpp:356`
- `volleyball/detection_image`，见 `volleyball_tracker_node.cpp:358`

### 性能优化点

从实现与子项目文档可看出，`NX_volleyball` 的重点优化方向包括：
- **双缓冲 + 条件变量**：相机回调写缓冲并直接唤醒推理线程，见 `volleyball_tracker_node.cpp:365`、`volleyball_tracker_node.cpp:425`
- **CUDA Bayer 预处理**：`yolo_preprocessor.cu`
- **Batch=2 推理**：`yolo_detector.cpp:219`、`yolo_detector.cpp:693`
- **低延迟 QoS**：`SensorDataQoS`，见 `volleyball_tracker_node.cpp:337`

## 环境要求

### `volleyball_tracking` 训练环境

适合 PC / 服务器：
- Python 3.8+
- PyTorch
- Ultralytics
- CUDA 环境（训练和导出推荐）

### `volleyball_tracking` 部署环境

`deploy/requirements_nx.txt` 指向 Jetson Orin NX 部署依赖：
- JetPack 5.1.2+
- Python 3.8+
- numpy / scipy / filterpy / pycuda
- TensorRT 随 JetPack 安装，见 `volleyball_tracking/deploy/requirements_nx.txt:20`

### `NX_volleyball` 实机环境

- Jetson Orin NX 16GB
- ROS2 Humble
- CUDA
- TensorRT
- OpenCV 4.x
- libgpiod
- 海康 MVS SDK (`/opt/MVS`)

## 快速开始

### 一、训练 YOLOv11-Pose 排球模型

```bash
cd volleyball_tracking/train
pip install -r requirements.txt
python train.py --config config.yaml
```

训练脚本入口：`volleyball_tracking/train/train.py:42`

### 二、导出 TensorRT 引擎

```bash
cd volleyball_tracking/deploy
python export_tensorrt.py \
    --weights ../models/volleyball_best.pt \
    --imgsz 640 \
    --fp16
```

### 三、编译 ROS2 双目追踪包

```bash
cd NX_volleyball/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash
```

### 四、启动主追踪节点

可以直接运行：

```bash
sudo -E bash -c "source /opt/ros/humble/setup.bash && \
source ~/NX_volleyball/ros2_ws/install/setup.bash && \
ros2 run volleyball_stereo_driver volleyball_tracker_node"
```

对应脚本：`NX_volleyball/启动追踪节点.sh:12`

### 五、启动可视化节点

```bash
source /opt/ros/humble/setup.bash
source ~/NX_volleyball/ros2_ws/install/setup.bash
ros2 run volleyball_stereo_driver volleyball_visualizer_node
```

对应脚本：`NX_volleyball/启动可视化节点.sh:12`

## 适合展示的作品点

如果把这个仓库作为作品集项目，它能体现的能力包括：

- **计算机视觉模型设计**：将排球检测转化为 5 关键点 pose 任务，而不是仅做 bbox 检测
- **几何建模能力**：利用关键点做圆拟合，输出比单 bbox 更稳定的球心与半径
- **追踪算法工程化**：把 ByteTrack、卡尔曼滤波、IoU 匹配和可视化拆成可复用模块
- **边缘部署能力**：完成 YOLOv11-Pose → TensorRT → Jetson Orin NX 的完整落地
- **机器人系统集成能力**：把 GPIO 触发、相机采集、推理、立体匹配、ROS2 消息流串起来
- **性能优化意识**：双缓冲、条件变量、CUDA 预处理、Batch=2 推理、低延迟 QoS

## 当前公开仓库状态

这个仓库更接近“训练链路 + 部署链路”的组合式作品集，而不是单一可直接运行的完整应用。公开仓库当前状态可以概括为：

- `volleyball_tracking` 已包含训练、导出、几何拟合、追踪与可视化核心代码，但 `demo/`、`tools/` 仍是预留目录
- `NX_volleyball` 已包含 ROS2 主节点、可视化节点、参数文件、启动脚本和优化说明，适合展示 Jetson 端实时部署能力
- 仓库未附带实际数据集、训练产物、TensorRT engine 与完整实机标定产物，使用时需要自行准备
- 部分更细的实现记录保存在子目录文档中；如果文档描述与代码不一致，应以 `train/`、`deploy/`、`ros2_ws/src/volleyball_stereo_driver/` 下的实际代码为准

## 相关文档

- `volleyball_tracking/README.md`：训练、几何拟合与单目追踪链路说明
- `volleyball_tracking/QUICKSTART.md`：偏操作步骤的训练/导出快速上手
- `NX_volleyball/README.md`：Jetson Orin NX 双目高速追踪系统说明
- `NX_volleyball/ros2_ws/src/volleyball_stereo_driver/README.md`：ROS2 包使用说明
- `NX_volleyball/架构优化_双缓冲流水线.md`：双缓冲与条件变量优化记录
- `NX_volleyball/Bayer_CUDA加速说明.md`：CUDA Bayer 预处理说明

## 许可证

仓库根目录当前未看到明确的 `LICENSE` 文件；如需开源发布，建议后续补充统一许可证说明。