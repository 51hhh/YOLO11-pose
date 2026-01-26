# 🎥 双目相机 ROS2 节点 - 使用指南

## 📦 完整系统架构

```
┌─────────────────────────────────────────────────┐
│              Orin NX 系统                        │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────────┐    ┌──────────────────┐ │
│  │ PWM Trigger Node │───▶│ GPIO Line 7      │ │
│  │  (100 Hz)        │    │  (硬件触发)       │ │
│  └────────┬─────────┘    └────────┬─────────┘ │
│           │                       │            │
│           │ /camera_trigger       │            │
│           ↓                       ↓            │
│  ┌──────────────────┐    ┌──────────────────┐ │
│  │ Stereo Camera    │◀───│ 左相机 + 右相机   │ │
│  │ Node             │    │ (Line0 触发)      │ │
│  └────────┬─────────┘    └──────────────────┘ │
│           │                                    │
│           ├─▶ /stereo/left/image_raw          │
│           └─▶ /stereo/right/image_raw         │
│                                                │
└─────────────────────────────────────────────────┘
```

---

## 🔧 在 NX 上编译

### 1. 确认海康 SDK 已安装

```bash
# 检查 SDK 路径
ls /opt/MVS/lib/64/libMvCameraControl.so
ls /opt/MVS/include/MvCameraControl.h

# 如果没有，安装 SDK
cd /opt/MVS/bin
sudo ./setup.sh
```

### 2. 编译 ROS2 包

```bash
cd ~/NX_volleyball/ros2_ws

# 编译
colcon build --packages-select volleyball_stereo_driver --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source 环境
source install/setup.bash
```

**预期输出**:
```
Starting >>> volleyball_stereo_driver
Finished <<< volleyball_stereo_driver [15.3s]

Summary: 1 package finished [15.4s]
```

---

## 🚀 运行完整系统

### 方法 1: 分别启动两个节点

#### 终端 1: PWM 触发节点

```bash
cd ~/NX_volleyball/ros2_ws
source install/setup.bash

# 运行 PWM 节点
sudo -E ros2 run volleyball_stereo_driver pwm_trigger_node
```

#### 终端 2: 双目相机节点

```bash
cd ~/NX_volleyball/ros2_ws
source install/setup.bash

# 运行相机节点
ros2 run volleyball_stereo_driver stereo_camera_node
```

### 方法 2: 使用 tmux 一键启动

```bash
# 创建 tmux 会话
tmux new -s volleyball

# 分割窗口
Ctrl+B "

# 上窗口: PWM 节点
sudo -E ros2 run volleyball_stereo_driver pwm_trigger_node

# 切换到下窗口
Ctrl+B ↓

# 下窗口: 相机节点
ros2 run volleyball_stereo_driver stereo_camera_node
```

---

## 📊 监控和验证

### 查看话题列表

```bash
ros2 topic list
```

**预期输出**:
```
/camera_trigger
/stereo/left/image_raw
/stereo/right/image_raw
```

### 查看图像话题频率

```bash
# 左相机
ros2 topic hz /stereo/left/image_raw

# 右相机
ros2 topic hz /stereo/right/image_raw
```

**预期**:
```
average rate: 100.012
    min: 0.009s max: 0.011s std dev: 0.00015s
```

### 查看图像信息

```bash
ros2 topic echo /stereo/left/image_raw --once
```

### 保存图像 (测试)

```bash
# 安装 image_tools
sudo apt install ros-humble-image-tools

# 保存左相机图像
ros2 run image_tools showimage --ros-args --remap image:=/stereo/left/image_raw
```

---

## ⚙️ 参数配置

### 修改相机参数

编辑 `config/camera_params.yaml`:

```yaml
stereo_camera_node:
  ros__parameters:
    left_camera_index: 0
    right_camera_index: 1
    exposure_time: 800.0    # 修改曝光时间
    gain: 5.0               # 修改增益
    trigger_mode: true
    trigger_source: "Line0"
    trigger_activation: "RisingEdge"
```

使用配置文件运行:

```bash
ros2 run volleyball_stereo_driver stereo_camera_node \
    --ros-args --params-file install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/camera_params.yaml
```

### 命令行参数

```bash
# 自定义曝光时间
ros2 run volleyball_stereo_driver stereo_camera_node \
    --ros-args -p exposure_time:=1000.0

# 自定义增益
ros2 run volleyball_stereo_driver stereo_camera_node \
    --ros-args -p gain:=10.0
```

---

## 🐛 故障排除

### Q1: 编译错误 "找不到 MvCameraControl.h"

```bash
# 检查 SDK 路径
ls /opt/MVS/include/MvCameraControl.h

# 如果不存在，安装 SDK
cd /opt/MVS/bin
sudo ./setup.sh
```

### Q2: 运行时错误 "未找到相机"

```bash
# 检查相机连接
lsusb | grep Hikvision

# 或使用 MVS 工具
/opt/MVS/bin/MVViewer

# 检查相机权限
sudo chmod 666 /dev/bus/usb/*/*
```

### Q3: 图像采集超时

```bash
# 检查 PWM 触发是否运行
ros2 topic hz /camera_trigger

# 检查相机触发配置
# 确认 Line0 已连接到 GPIO Line 7
```

### Q4: 时间戳不同步

```bash
# 检查触发延迟
ros2 topic echo /stereo/left/image_raw | grep stamp

# 应该看到相同的时间戳
```

---

## 📈 性能验证

### 系统延迟测试

```bash
# 运行完整系统
# 观察日志中的 "触发延迟"

# 预期: <10ms
```

### 帧率测试

```bash
# 左相机
ros2 topic hz /stereo/left/image_raw

# 右相机  
ros2 topic hz /stereo/right/image_raw

# 预期: 99-101 Hz
```

### 同步误差测试

```bash
# 记录时间戳
ros2 topic echo /stereo/left/image_raw | grep stamp > left_stamps.txt &
ros2 topic echo /stereo/right/image_raw | grep stamp > right_stamps.txt &

# 等待 10 秒
sleep 10

# 停止记录
killall ros2

# 分析时间戳差异
# 预期: <1ms
```

---

## 🎯 快速命令参考

```bash
# === 编译 ===
cd ~/NX_volleyball/ros2_ws
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash

# === 运行 ===
# 终端 1: PWM 节点
sudo -E ros2 run volleyball_stereo_driver pwm_trigger_node

# 终端 2: 相机节点
ros2 run volleyball_stereo_driver stereo_camera_node

# === 监控 ===
ros2 topic list
ros2 topic hz /stereo/left/image_raw
ros2 topic hz /stereo/right/image_raw
ros2 topic echo /camera_trigger

# === 参数调整 ===
ros2 run volleyball_stereo_driver stereo_camera_node \
    --ros-args -p exposure_time:=1000.0 -p gain:=5.0
```

---

## 📝 下一步

1. ✅ 编译相机节点
2. ✅ 测试双目采集
3. ⏳ 相机标定
4. ⏳ 立体匹配节点
5. ⏳ 追踪节点

---

**准备好测试了吗？按照上面的步骤在 NX 上编译和运行！**
