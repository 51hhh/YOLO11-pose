# 🚀 Jetson NX 编译和测试指南

## 📋 前置条件检查

### 1. 系统环境
```bash
# 检查 ROS2 Humble
source /opt/ros/humble/setup.bash
ros2 --version

# 检查 OpenCV
pkg-config --modversion opencv4

# 检查 libgpiod
pkg-config --modversion libgpiod

# 检查海康 SDK
ls /opt/MVS/lib/aarch64/libMvCameraControl.so
```

### 2. 硬件连接
- ✅ 2x 海康 MV-CA016-10UC 相机已连接
- ✅ GPIO PWM 引脚已连接到相机触发线
- ✅ 相机供电正常

---

## 🔧 编译步骤

### 1. 进入工作空间
```bash
cd ~/desktop/yolo/yoloProject/NX_volleyball/ros2_ws
```

### 2. 清理旧的编译文件 (可选)
```bash
rm -rf build install log
```

### 3. 编译包
```bash
# 设置环境
source /opt/ros/humble/setup.bash

# 编译 (Release 模式)
colcon build --packages-select volleyball_stereo_driver \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

# 如果编译失败，查看详细错误
colcon build --packages-select volleyball_stereo_driver \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --event-handlers console_direct+
```

### 4. 加载环境
```bash
source install/setup.bash
```

---

## 🧪 测试流程

### 测试 1: 基础节点 (PWM + 相机)

#### 启动节点
```bash
# 终端 1: 启动基础节点
ros2 run volleyball_stereo_driver stereo_system_node \
  --ros-args --params-file src/volleyball_stereo_driver/config/system_params.yaml
```

#### 验证输出
```bash
# 终端 2: 查看话题
ros2 topic list

# 应该看到:
# /camera_trigger
# /stereo/left/image_raw
# /stereo/right/image_raw

# 查看触发频率
ros2 topic hz /camera_trigger

# 查看图像
ros2 topic echo /stereo/left/image_raw --once
```

#### 预期结果
- ✅ PWM 触发频率 ~100Hz
- ✅ 左右相机图像正常发布
- ✅ 无错误日志

---

### 测试 2: 完整追踪节点 (需要标定文件)

#### 准备标定文件
```bash
# 如果有 .npz 格式标定文件，先转换
cd ~/desktop/yolo/yoloProject/NX_volleyball/ros2_ws/src/volleyball_stereo_driver
python3 scripts/convert_calibration.py \
  calibration/stereo_calib.npz \
  calibration/stereo_calib.yaml

# 或者创建测试用的默认标定文件 (仅用于测试编译)
cat > calibration/stereo_calib.yaml << 'EOF'
%YAML:1.0
---
K1: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1000., 0., 640., 0., 1000., 360., 0., 0., 1. ]
D1: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ 0., 0., 0., 0., 0. ]
K2: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1000., 0., 640., 0., 1000., 360., 0., 0., 1. ]
D2: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ 0., 0., 0., 0., 0. ]
P1: !!opencv-matrix
   rows: 3
   cols: 4
   dt: d
   data: [ 1000., 0., 640., 0., 0., 1000., 360., 0., 0., 0., 1., 0. ]
P2: !!opencv-matrix
   rows: 3
   cols: 4
   dt: d
   data: [ 1000., 0., 640., -250., 0., 1000., 360., 0., 0., 0., 1., 0. ]
baseline: 0.25
EOF
```

#### 启动追踪节点
```bash
# 终端 1: 启动追踪节点
ros2 run volleyball_stereo_driver volleyball_tracker_node \
  --ros-args --params-file src/volleyball_stereo_driver/config/tracker_params.yaml
```

#### 验证输出
```bash
# 终端 2: 查看话题
ros2 topic list

# 应该看到:
# /volleyball/pose_3d
# /volleyball/velocity
# /volleyball/debug_info

# 查看 3D 位置
ros2 topic echo /volleyball/pose_3d

# 查看速度
ros2 topic echo /volleyball/velocity

# 查看调试信息
ros2 topic echo /volleyball/debug_info
```

#### 预期结果
- ✅ 节点启动无错误
- ✅ 相机采集正常
- ✅ 检测器加载成功 (即使是占位符)
- ✅ 立体匹配器加载标定文件
- ✅ 卡尔曼滤波器初始化
- ✅ 话题正常发布

---

## 🐛 常见问题排查

### 问题 1: 编译失败 - OpenCV 未找到
```bash
# 安装 OpenCV
sudo apt install libopencv-dev python3-opencv

# 或指定 OpenCV 路径
colcon build --packages-select volleyball_stereo_driver \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4
```

### 问题 2: 海康 SDK 未找到
```bash
# 检查 SDK 路径
ls /opt/MVS/lib/aarch64/

# 如果路径不同，修改 CMakeLists.txt 中的 HIK_SDK_DIR
```

### 问题 3: TensorRT 未找到
```bash
# TensorRT 是可选的，节点会自动跳过
# 如果需要 YOLO 检测，确保 TensorRT 已安装
dpkg -l | grep tensorrt
```

### 问题 4: 相机无法打开
```bash
# 检查相机连接
ls /dev/video*

# 检查 USB 权限
sudo chmod 666 /dev/bus/usb/*/*

# 或添加 udev 规则
sudo cp src/volleyball_stereo_driver/scripts/99-hikrobot.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### 问题 5: GPIO 权限不足
```bash
# 添加用户到 gpio 组
sudo usermod -a -G gpio $USER

# 重新登录或
newgrp gpio

# 或临时使用 sudo
sudo -E ros2 run volleyball_stereo_driver stereo_system_node ...
```

---

## 📊 性能监控

### 查看系统资源
```bash
# CPU 使用率
htop

# GPU 使用率
tegrastats

# 内存使用
free -h
```

### 查看节点性能
```bash
# 查看话题频率
ros2 topic hz /volleyball/pose_3d

# 查看节点信息
ros2 node info /volleyball_tracker

# 查看参数
ros2 param list /volleyball_tracker
```

---

## 🎯 下一步优化

### 1. 完成双目标定
```bash
cd ~/desktop/yolo/yoloProject/NX_volleyball/calibration
# 采集棋盘格图像
# 运行标定程序
# 转换为 YAML 格式
```

### 2. 训练/获取 YOLO 模型
- 训练 YOLO11n 模型
- 导出 ONNX
- 转换为 TensorRT Engine
- 放置到 `model/yolo11n.engine`

### 3. 实现完整的 YOLO TensorRT 推理
- 更新 `src/yolo_detector.cpp`
- 实现真实的推理逻辑

### 4. 端到端测试
- 使用真实排球进行测试
- 调整参数优化性能
- 记录测试数据

---

## 📝 快速命令参考

```bash
# 编译
cd ~/desktop/yolo/yoloProject/NX_volleyball/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash

# 运行基础节点
ros2 run volleyball_stereo_driver stereo_system_node \
  --ros-args --params-file src/volleyball_stereo_driver/config/system_params.yaml

# 运行追踪节点
ros2 run volleyball_stereo_driver volleyball_tracker_node \
  --ros-args --params-file src/volleyball_stereo_driver/config/tracker_params.yaml

# 查看话题
ros2 topic list
ros2 topic hz /camera_trigger
ros2 topic echo /volleyball/pose_3d

# 停止节点
Ctrl+C
```

---

**准备好在 NX 上测试了！** 🚀

如果遇到任何问题，请查看日志输出并参考上述排查步骤。
