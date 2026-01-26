# 🚀 ROS2 PWM 触发节点 - 纯 C++ 构建指南

## 📦 项目结构

```
volleyball_stereo_driver/
├── CMakeLists.txt              # 纯 C++ 构建配置
├── package.xml                 # ROS2 包依赖
├── include/
│   └── volleyball_stereo_driver/
│       └── high_precision_pwm.hpp
├── src/
│   ├── high_precision_pwm.cpp  # 高精度 PWM 实现
│   └── pwm_trigger_node.cpp    # ROS2 节点
├── config/
│   └── pwm_params.yaml         # 参数配置
└── scripts/
    └── start_pwm_node.sh       # Bash 启动脚本
```

**特点**:
- ✅ 纯 C++ 实现，无 Python 依赖
- ✅ 使用 colcon 构建
- ✅ 高性能，低延迟
- ✅ 简单易用

---

## 🔧 编译步骤

### 1. 安装依赖

```bash
# ROS2 依赖
sudo apt install -y \
    ros-humble-rclcpp \
    ros-humble-std-msgs \
    libgpiod-dev

# 如果还没安装 ROS2
# 参考: https://docs.ros.org/en/humble/Installation.html
```

### 2. 编译包

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
Finished <<< volleyball_stereo_driver [8.2s]

Summary: 1 package finished [8.3s]
```

---

## 🚀 运行节点

### 方法 1: 直接运行 (推荐)

```bash
# Source 环境
source ~/NX_volleyball/ros2_ws/install/setup.bash

# 使用 sudo 运行 (提升优先级)
sudo -E ros2 run volleyball_stereo_driver pwm_trigger_node
```

### 方法 2: 使用启动脚本

```bash
# 默认配置 (100Hz, 50%)
cd ~/NX_volleyball/ros2_ws
sudo ./install/volleyball_stereo_driver/share/volleyball_stereo_driver/scripts/start_pwm_node.sh

# 自定义参数
sudo ./install/volleyball_stereo_driver/share/volleyball_stereo_driver/scripts/start_pwm_node.sh \
    --frequency 120.0 \
    --duty-cycle 60.0
```

### 方法 3: 带参数运行

```bash
sudo -E ros2 run volleyball_stereo_driver pwm_trigger_node \
    --ros-args \
    -p frequency:=120.0 \
    -p duty_cycle:=60.0 \
    -p gpio_chip:=gpiochip2 \
    -p gpio_line:=7
```

---

## 📊 监控和调试

### 查看话题

```bash
# 查看触发时间戳
ros2 topic echo /camera_trigger

# 查看话题频率
ros2 topic hz /camera_trigger
```

**预期输出**:
```
average rate: 100.012
    min: 0.009s max: 0.011s std dev: 0.00015s window: 100
```

### 动态参数调整

```bash
# 查看参数
ros2 param list /pwm_trigger_node

# 修改频率
ros2 param set /pwm_trigger_node frequency 120.0

# 修改占空比
ros2 param set /pwm_trigger_node duty_cycle 60.0
```

### 节点信息

```bash
# 节点列表
ros2 node list

# 节点详情
ros2 node info /pwm_trigger_node
```

---

## ⚙️ 配置文件

编辑 `config/pwm_params.yaml`:

```yaml
pwm_trigger_node:
  ros__parameters:
    gpio_chip: "gpiochip2"
    gpio_line: 7
    frequency: 100.0
    duty_cycle: 50.0
    auto_start: true
```

使用配置文件运行:

```bash
sudo -E ros2 run volleyball_stereo_driver pwm_trigger_node \
    --ros-args --params-file install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/pwm_params.yaml
```

---

## 🐛 故障排除

### Q1: 编译错误 "找不到 gpiod.h"

```bash
sudo apt install libgpiod-dev pkg-config
```

### Q2: 编译错误 "找不到 rclcpp"

```bash
# 确认 ROS2 已安装
source /opt/ros/humble/setup.bash

# 重新编译
colcon build --packages-select volleyball_stereo_driver
```

### Q3: 运行时错误 "无法打开 GPIO 芯片"

```bash
# 检查 GPIO
ls /dev/gpiochip*
gpioinfo gpiochip2

# 检查权限
sudo usermod -a -G gpio $USER
# 重新登录
```

### Q4: 频率不准确

```bash
# 1. 优化系统
cd ~/NX_volleyball/scripts
sudo ./optimize_system.sh

# 2. 使用 sudo 运行
sudo -E ros2 run volleyball_stereo_driver pwm_trigger_node

# 3. 检查 CPU 频率
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

---

## 📈 性能验证

### 示波器测量

连接示波器到 gpiochip2 line 7:
- 频率: 99.9-100.1 Hz ✅
- 占空比: 49.5-50.5% ✅
- 抖动: <0.05 ms ✅

### ROS2 话题频率

```bash
ros2 topic hz /camera_trigger
```

**预期**:
```
average rate: 100.012
    min: 0.009s max: 0.011s std dev: 0.00015s window: 100
```

---

## 🎯 快速命令参考

```bash
# === 编译 ===
cd ~/NX_volleyball/ros2_ws
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash

# === 运行 ===
# 方法 1: 直接运行
sudo -E ros2 run volleyball_stereo_driver pwm_trigger_node

# 方法 2: 使用脚本
sudo ./install/volleyball_stereo_driver/share/volleyball_stereo_driver/scripts/start_pwm_node.sh

# 方法 3: 自定义参数
sudo -E ros2 run volleyball_stereo_driver pwm_trigger_node \
    --ros-args -p frequency:=120.0

# === 监控 ===
ros2 topic hz /camera_trigger
ros2 param list /pwm_trigger_node
ros2 node info /pwm_trigger_node

# === 调整参数 ===
ros2 param set /pwm_trigger_node frequency 120.0
ros2 param set /pwm_trigger_node duty_cycle 60.0
```

---

## 🔄 重新编译

```bash
cd ~/NX_volleyball/ros2_ws

# 清理
rm -rf build install log

# 重新编译
colcon build --packages-select volleyball_stereo_driver

# Source
source install/setup.bash
```

---

## 📝 下一步

1. ✅ 编译 PWM 触发节点
2. ✅ 测试 PWM 输出
3. ⏳ 创建相机驱动节点
4. ⏳ 集成双目同步采集

---

**准备好编译了吗？运行 `colcon build` 开始！**
