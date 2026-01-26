# 🎉 双目排球追踪系统 - 整合节点使用指南

## ✅ 已完成的优化

### 1. 修复配置文件读取
- ✅ 默认参数值改为测试的最佳值
- ✅ 曝光时间: 9867.0 us
- ✅ 增益: 10.9854 dB

### 2. 解决编译警告
- ✅ 修复成员初始化顺序警告
- ✅ 调整 `HighPrecisionPWM` 初始化列表顺序

### 3. 整合 PWM 和相机为一个节点
- ✅ 创建 `stereo_system_node` 整合节点
- ✅ 一个节点同时控制 PWM 和双目相机
- ✅ 简化部署和使用

---

## 🚀 在 NX 上使用

### 步骤 1: 传输文件

```bash
# 在本地机器
cd /home/rick/desktop/yolo/yoloProject
scp -r ./NX_volleyball/ nvidia@10.42.0.148:~
```

### 步骤 2: 编译

```bash
# 在 NX 上
cd ~/NX_volleyball/ros2_ws
rm -rf build install log
colcon build --packages-select volleyball_stereo_driver --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

**预期输出** (无警告):
```
Starting >>> volleyball_stereo_driver
Finished <<< volleyball_stereo_driver [25.3s]

Summary: 1 package finished [25.4s]
```

### 步骤 3: 运行整合节点

```bash
# 使用 sudo 运行 (提升 PWM 优先级)
sudo -E ros2 run volleyball_stereo_driver stereo_system_node
```

**预期输出**:
```
========================================
双目排球追踪系统 - 整合节点
========================================
PWM 配置:
  GPIO: gpiochip2 line 7
  频率: 100.0 Hz
  占空比: 50.0%
相机配置:
  左相机: 0 | 右相机: 1
  曝光: 9867.0 us | 增益: 10.9854 dB
========================================
✅ 海康 SDK 已初始化
✅ 高精度 PWM 已启动: 100 Hz, 50%
  ✅ 线程优先级已提升 (SCHED_FIFO)
✅ 相机已打开: 1440x1080
✅ 曝光时间: 9867 us
✅ 增益: 10.9854 dB
✅ 触发模式: On
✅ 触发源: Line0
✅ 开始采集
✅ 系统已启动，开始采集...
双目排球追踪系统正在运行...
按 Ctrl+C 停止

  周期: 500 | 实际频率: 99.98 Hz | 误差: -0.02 Hz
已采集 500 帧 | PWM 频率: 99.98 Hz
```

---

## 📊 验证系统

### 查看话题

```bash
# 新终端
ros2 topic list
```

**输出**:
```
/stereo/left/image_raw
/stereo/right/image_raw
```

### 查看帧率

```bash
ros2 topic hz /stereo/left/image_raw
ros2 topic hz /stereo/right/image_raw
```

**预期**: `average rate: 100.012`

---

## ⚙️ 配置文件

使用配置文件运行:

```bash
sudo -E ros2 run volleyball_stereo_driver stereo_system_node \
    --ros-args --params-file install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/system_params.yaml
```

修改配置: 编辑 `config/system_params.yaml`

---

## 📁 项目结构 (简化后)

```
volleyball_stereo_driver/
├── src/
│   ├── stereo_system_node.cpp      # ✅ 整合节点 (主要)
│   ├── high_precision_pwm.cpp      # PWM 实现
│   ├── hik_camera_wrapper.cpp      # 相机封装
│   ├── pwm_trigger_node.cpp        # (已注释，保留用于测试)
│   └── stereo_camera_node.cpp      # (已注释，保留用于测试)
├── include/
│   └── volleyball_stereo_driver/
│       ├── high_precision_pwm.hpp
│       └── hik_camera_wrapper.hpp
├── config/
│   └── system_params.yaml          # ✅ 整合配置
└── CMakeLists.txt                  # ✅ 只编译整合节点
```

---

## 🎯 快速命令参考

```bash
# === 编译 ===
cd ~/NX_volleyball/ros2_ws
rm -rf build install log
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash

# === 运行 ===
sudo -E ros2 run volleyball_stereo_driver stereo_system_node

# === 监控 ===
ros2 topic list
ros2 topic hz /stereo/left/image_raw

# === 参数调整 ===
ros2 run volleyball_stereo_driver stereo_system_node \
    --ros-args -p pwm_frequency:=120.0 -p exposure_time:=8000.0
```

---

## 🔄 与之前的区别

| 之前 | 现在 |
|------|------|
| 2 个节点 (PWM + 相机) | 1 个整合节点 |
| 需要 2 个终端 | 只需 1 个终端 |
| 参数分散在 2 个配置文件 | 统一配置文件 |
| 编译 2 个可执行文件 | 编译 1 个可执行文件 |

---

## 📝 下一步

1. ✅ 编译整合节点
2. ✅ 测试双目采集
3. ⏳ 相机标定
4. ⏳ 立体匹配节点
5. ⏳ 追踪节点

---

**准备好测试了吗？运行 `colcon build` 开始！**
