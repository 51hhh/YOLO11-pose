# NX_volleyball：Jetson Orin NX 双目排球高速追踪系统

`NX_volleyball` 是这个仓库中面向实机部署的一条实现路线，重点是把排球检测、双目同步采集、立体测距和 ROS2 消息发布整合到 Jetson Orin NX 上的实时系统里。

它与同仓库下的 `../volleyball_tracking` 形成互补关系：
- `volleyball_tracking` 负责 5 关键点数据定义、YOLOv11-Pose 训练、几何拟合和单目追踪验证
- `NX_volleyball` 负责 Jetson Orin NX + ROS2 + 海康双目相机上的高速 3D 部署

## 系统流程

```text
PWM 触发
  ↓
海康双目相机同步采集
  ↓
双缓冲 + 条件变量唤醒
  ↓
YOLO TensorRT 检测（支持 Batch=2）
  ↓
立体匹配 / 三角测量
  ↓
3D 卡尔曼滤波
  ↓
ROS2 发布 pose / velocity / 调试图像
```

## 当前实现内容

| 模块 | 说明 |
|------|------|
| `volleyball_tracker_node` | All-in-One 主节点，整合 PWM、相机采集、YOLO TensorRT、立体匹配和 3D 追踪 |
| `volleyball_visualizer_node` | 独立可视化节点，用于显示检测图像、双目图像、3D 位置和轨迹历史 |
| `yolo_preprocessor.cu` | CUDA Bayer 预处理与推理前图像准备 |
| `yolo_detector.cpp` | TensorRT 推理封装，支持双路图像批量处理 |
| `stereo_matcher.cpp` | 双目匹配与三角测量 |
| `kalman_filter_3d.cpp` | 3D 位置/速度滤波 |
| `high_precision_pwm.cpp` | GPIO PWM 触发控制 |
| `hik_camera_wrapper.cpp` | 海康工业相机封装 |
| `config/tracker_params.yaml` | 运行参数入口，包括模型、PWM、相机、立体和调试配置 |

## 目录结构

```text
NX_volleyball/
├── README.md
├── calibration/                         # 双目标定工具与标定相关脚本
├── scripts/
│   └── convert_yolo_to_tensorrt.py      # PT/ONNX/TensorRT 转换脚本
├── 启动追踪节点.sh
├── 启动可视化节点.sh
├── 架构优化_双缓冲流水线.md
├── Bayer_CUDA加速说明.md
├── 性能优化报告_320模型.md
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

## 构建依赖

### 必需依赖
- ROS2 Humble
- OpenCV 4.x
- libgpiod
- C++17

### 主追踪节点额外依赖
- CUDA
- TensorRT
- 海康 MVS SDK（默认路径 `/opt/MVS`）

根据 `ros2_ws/src/volleyball_stereo_driver/CMakeLists.txt` 的构建逻辑：
- 只有在 **TensorRT** 和 **海康 MVS SDK** 都存在时，`volleyball_tracker_node` 才会被构建
- `volleyball_visualizer_node` 不依赖 TensorRT 和相机 SDK，可单独编译运行

## 编译

```bash
cd NX_volleyball/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash
```

## 运行

### 1. 启动主追踪节点

主节点需要 GPIO 权限，通常通过 `sudo` 启动：

```bash
sudo -E bash -c "source /opt/ros/humble/setup.bash && \
source ~/NX_volleyball/ros2_ws/install/setup.bash && \
ros2 run volleyball_stereo_driver volleyball_tracker_node"
```

仓库根目录也提供了便捷脚本：

```bash
cd NX_volleyball
bash 启动追踪节点.sh
```

### 2. 启动可视化节点

```bash
source /opt/ros/humble/setup.bash
source ~/NX_volleyball/ros2_ws/install/setup.bash
ros2 run volleyball_stereo_driver volleyball_visualizer_node
```

或使用：

```bash
cd NX_volleyball
bash 启动可视化节点.sh
```

可视化窗口快捷键：
- `Q` / `ESC`：退出
- `S`：截图
- `T`：切换轨迹显示
- `C`：清除轨迹历史

## 模型与配置

默认运行参数位于：
- `ros2_ws/src/volleyball_stereo_driver/config/tracker_params.yaml`
- `ros2_ws/src/volleyball_stereo_driver/config/visualizer_params.yaml`

当前默认检测模型配置为：

```yaml
volleyball_tracker:
  ros__parameters:
    detector:
      model_path: "model/yolo_320.engine"
      input_size: 0
      confidence_threshold: 0.5
      nms_threshold: 0.4
      roi_size: 320
      global_size: 640
```

其中：
- `model_path` 默认指向 `model/yolo_320.engine`
- 注释中同时保留了 `yolo11n.engine`、`yolo11n_batch2.engine` 等模型变体说明
- 立体标定默认读取 `calibration/stereo_calib.yaml`
- 调试图像和日志频率也在该配置文件中统一控制

## ROS2 接口

### `volleyball_tracker_node` 发布话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `volleyball/pose_3d` | `geometry_msgs/msg/PoseStamped` | 排球 3D 位置 |
| `volleyball/velocity` | `geometry_msgs/msg/Vector3Stamped` | 排球 3D 速度 |
| `camera_trigger` | `std_msgs/msg/Header` | 触发时间戳 |
| `volleyball/debug_info` | `std_msgs/msg/String` | 调试信息 |
| `stereo/left/image_raw` | `sensor_msgs/msg/Image` | 左相机图像 |
| `stereo/right/image_raw` | `sensor_msgs/msg/Image` | 右相机图像 |
| `volleyball/detection_image` | `sensor_msgs/msg/Image` | 检测可视化图像 |

### `volleyball_visualizer_node` 订阅话题

- `stereo/left/image_raw`
- `stereo/right/image_raw`
- `volleyball/detection_image`
- `volleyball/pose_3d`
- `volleyball/velocity`
- `volleyball/debug_info`

## 工程特点

- 双目系统围绕 **100Hz PWM 同步触发** 设计
- 主节点采用 **双缓冲 + 条件变量** 减少轮询等待开销
- 预处理链路包含 **CUDA Bayer 转换与图像准备**
- 推理侧支持 **Batch=2 TensorRT**，用于左右相机合批处理
- 图像与位姿接口使用低延迟的 **SensorDataQoS**
- 可视化节点与主追踪节点解耦，便于单独调试链路

## 适合作品集展示的点

- 将工业相机同步采集、PWM 触发、YOLO 推理、立体测距和 ROS2 发布整合到单个部署项目中
- 从代码结构上体现了对 **低延迟实时系统** 的工程关注：缓冲设计、线程唤醒、GPU 预处理、批量推理
- 具备从 2D 检测到 3D 位置/速度输出的完整闭环，更适合作为机器人视觉部署作品展示

## 当前使用边界

- 该子项目默认面向 **Jetson Orin NX + ROS2 Humble + 海康双目相机** 环境
- 若缺少 TensorRT 或 `/opt/MVS`，主追踪节点不会被构建
- 运行前需要准备好可用的 TensorRT engine 与双目标定文件
- 仓库中还保留了一些历史优化记录文档；实际运行以 `ros2_ws/src/volleyball_stereo_driver/` 下的代码、`CMakeLists.txt` 和 `config/` 为准

## 相关文档

- `../README.md`：仓库总览
- `../volleyball_tracking/README.md`：训练与单目追踪链路
- `ros2_ws/src/volleyball_stereo_driver/README.md`：ROS2 包级说明
- `架构优化_双缓冲流水线.md`：双缓冲架构设计记录
- `Bayer_CUDA加速说明.md`：CUDA Bayer 预处理说明
- `性能优化报告_320模型.md`：320 输入模型优化记录
