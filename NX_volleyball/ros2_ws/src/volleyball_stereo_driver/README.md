# 🏐 volleyball_stereo_driver

双目排球追踪系统 ROS2 包

---

## 📦 节点架构

本包包含 **2 个节点**：

### 1. volleyball_tracker_node (主追踪节点)

**All-in-One 设计**，集成所有功能：
- PWM 触发 (100Hz 内置)
- 双目相机同步采集
- YOLO TensorRT 检测
- 立体匹配 (三角测量)
- 3D 卡尔曼滤波追踪

**发布话题**：
| 话题 | 类型 | 说明 |
|------|------|------|
| `/volleyball/pose_3d` | PoseStamped | 3D 位置 |
| `/volleyball/velocity` | Vector3Stamped | 3D 速度 |
| `/volleyball/debug_info` | String | 调试信息 (JSON) |
| `/stereo/left/image_raw` | Image | 左相机图像 |
| `/stereo/right/image_raw` | Image | 右相机图像 |
| `/volleyball/detection_image` | Image | 检测可视化图像 |
| `/camera_trigger` | Header | PWM 触发时间戳 |

### 2. volleyball_visualizer_node (可视化节点)

**调试可视化**：
- 显示相机实时图像
- 显示 YOLO 检测框
- 显示目标 3D 位置
- 显示轨迹历史
- 支持截图和快捷键

**订阅话题**：
| 话题 | 类型 | 说明 |
|------|------|------|
| `/stereo/left/image_raw` | Image | 左相机图像 |
| `/stereo/right/image_raw` | Image | 右相机图像 |
| `/volleyball/detection_image` | Image | 检测可视化 |
| `/volleyball/pose_3d` | PoseStamped | 3D 位置 |
| `/volleyball/velocity` | Vector3Stamped | 3D 速度 |
| `/volleyball/debug_info` | String | 调试信息 |

---

## 📂 目录结构

```
volleyball_stereo_driver/
├── include/volleyball_stereo_driver/
│   ├── volleyball_tracker_node.hpp     # 追踪节点
│   ├── volleyball_visualizer_node.hpp  # 可视化节点
│   ├── high_precision_pwm.hpp          # PWM
│   ├── hik_camera_wrapper.hpp          # 相机
│   ├── yolo_detector.hpp               # YOLO
│   ├── stereo_matcher.hpp              # 立体匹配
│   ├── kalman_filter_3d.hpp            # 卡尔曼
│   └── roi_manager.hpp                 # ROI
│
├── src/
│   ├── volleyball_tracker_node.cpp
│   ├── volleyball_visualizer_node.cpp
│   ├── high_precision_pwm.cpp
│   ├── hik_camera_wrapper.cpp
│   ├── yolo_detector.cpp
│   ├── stereo_matcher.cpp
│   ├── kalman_filter_3d.cpp
│   └── roi_manager.cpp
│
├── config/
│   ├── tracker_params.yaml      # 追踪节点配置
│   └── visualizer_params.yaml   # 可视化节点配置
│
├── model/                       # YOLO 模型
│   └── yolo11n.engine
│
├── calibration/                 # 标定文件
│   └── stereo_calib.yaml
│
├── CMakeLists.txt
└── package.xml
```

---

## 🚀 使用方法

### 编译

```bash
cd ~/NX_volleyball/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash
```

### ⭐ 快速启动（推荐）

**终端1 - 主追踪节点**
```bash
sudo -E bash -c "source /opt/ros/humble/setup.bash && \
source ~/NX_volleyball/ros2_ws/install/setup.bash && \
ros2 run volleyball_stereo_driver volleyball_tracker_node"
```

**终端2 - 可视化节点**
```bash
source /opt/ros/humble/setup.bash
source ~/NX_volleyball/ros2_ws/install/setup.bash
ros2 run volleyball_stereo_driver volleyball_visualizer_node
```

> 💡 节点会自动使用内置默认参数

### 自定义配置（可选）

如需修改参数，可指定配置文件：

```bash
# 主追踪节点
sudo -E ros2 run volleyball_stereo_driver volleyball_tracker_node \
  --ros-args --params-file install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/tracker_params.yaml

# 可视化节点
ros2 run volleyball_stereo_driver volleyball_visualizer_node \
  --ros-args --params-file install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/visualizer_params.yaml
```

---

## ⌨️ 可视化节点快捷键

| 按键 | 功能 |
|------|------|
| `Q` / `ESC` | 退出 |
| `S` | 截图 |
| `T` | 切换轨迹显示 |
| `C` | 清除轨迹 |

---

## 📋 依赖

### 必需
- ROS2 Humble
- OpenCV 4.x
- libgpiod

### 追踪节点额外依赖
- 海康 MVS SDK (`/opt/MVS`)
- TensorRT
- CUDA

### 可视化节点
- 无额外依赖（可在任何有 OpenCV 的环境运行）

---

## 📊 性能指标

| 指标 | 目标值 |
|------|--------|
| 采集帧率 | 100 Hz |
| 检测延迟 | <10 ms |
| 端到端延迟 | <15 ms |
| 深度精度 (3m) | ±1.2 cm |
| 深度精度 (15m) | ±31 cm |

---

*更新: 2026-01-25*
