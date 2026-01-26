# 🏐 排球追踪系统 - 简洁启动指南

## 🚀 快速启动 (自动加载配置)

### 基础节点 (PWM + 相机)
```bash
ros2 run volleyball_stereo_driver stereo_system_node
```

### 追踪节点 (完整功能)
```bash
ros2 run volleyball_stereo_driver volleyball_tracker_node
```

---

## 📋 说明

- ✅ **自动加载配置**: 节点会自动从安装目录加载配置文件
  - `stereo_system_node` → `config/system_params.yaml`
  - `volleyball_tracker_node` → `config/tracker_params.yaml`

- ✅ **无需指定路径**: 使用 `ament_index_cpp` 自动定位配置文件

- ✅ **开发模式**: 如果在源码目录运行，会自动查找相对路径的配置文件

---

## 🔧 修改配置

### 方法 1: 修改安装目录的配置文件 (推荐)
```bash
cd ~/desktop/yolo/yoloProject/NX_volleyball/ros2_ws
nano install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/system_params.yaml
```

### 方法 2: 修改源码配置文件后重新编译
```bash
cd ~/desktop/yolo/yoloProject/NX_volleyball/ros2_ws
nano src/volleyball_stereo_driver/config/system_params.yaml
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash
```

---

## 📊 查看运行状态

```bash
# 查看话题列表
ros2 topic list

# 查看触发频率
ros2 topic hz /camera_trigger

# 查看 3D 位置
ros2 topic echo /volleyball/pose_3d

# 查看节点信息
ros2 node info /stereo_system_node
ros2 node info /volleyball_tracker
```

---

## 🎯 完整流程

```bash
# 1. 编译
cd ~/desktop/yolo/yoloProject/NX_volleyball
./quick_build_test.sh

# 2. 运行节点 (选择一个)
ros2 run volleyball_stereo_driver stereo_system_node
# 或
ros2 run volleyball_stereo_driver volleyball_tracker_node

# 3. 查看输出
ros2 topic list
```

---

**简洁、高效、无需手动指定配置文件路径！** ✨
