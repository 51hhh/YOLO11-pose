# 🚀 自动配置文件加载 - 使用指南

## ✅ 新特性：自动加载配置文件

节点现在会**自动查找并加载**配置文件，无需手动指定！

---

## 🔍 配置文件查找顺序

节点会按以下顺序查找 `system_params.yaml`：

1. **安装目录** (最常用)
   ```
   ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/system_params.yaml
   ```

2. **源码目录**
   ```
   ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/config/system_params.yaml
   ```

3. **当前目录**
   ```
   ./config/system_params.yaml
   ./system_params.yaml
   ```

找到第一个存在的文件后，立即加载并停止查找。

---

## 🚀 使用方法

### 方法 1: 直接运行 (推荐)

```bash
# 节点会自动查找配置文件
sudo -E ros2 run volleyball_stereo_driver stereo_system_node
```

**输出示例**:
```
[INFO] [stereo_system_node]: 自动加载配置文件: /home/nvidia/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/system_params.yaml
[INFO] [stereo_system_node]: 已加载 11 个参数
========================================
双目排球追踪系统 - 整合节点
========================================
PWM 配置:
  GPIO: gpiochip2 line 7
  频率: 100.0 Hz
相机配置:
  曝光: 9867.0 us | 增益: 10.9854 dB
========================================
```

### 方法 2: 使用启动脚本

```bash
cd ~/NX_volleyball/ros2_ws
sudo ./install/volleyball_stereo_driver/share/volleyball_stereo_driver/scripts/start_system.sh
```

### 方法 3: 手动指定配置文件 (可选)

如果需要使用自定义配置文件：

```bash
sudo -E ros2 run volleyball_stereo_driver stereo_system_node \
    --ros-args --params-file /path/to/custom_config.yaml
```

---

## 📝 配置文件位置

### 推荐：编辑源文件

```bash
# 编辑源文件
nano ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/config/system_params.yaml

# 重新编译 (复制到安装目录)
cd ~/NX_volleyball/ros2_ws
colcon build --packages-select volleyball_stereo_driver

# 运行 (自动加载)
sudo -E ros2 run volleyball_stereo_driver stereo_system_node
```

### 临时测试：直接编辑安装文件

```bash
# 直接编辑安装目录的文件
nano ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/system_params.yaml

# 运行 (无需重新编译)
sudo -E ros2 run volleyball_stereo_driver stereo_system_node
```

---

## ⚠️ 如果未找到配置文件

节点会输出警告并尝试使用命令行参数：

```
[WARN] [stereo_system_node]: 未找到配置文件，将使用命令行参数
[ERROR] [stereo_system_node]: Parameter 'gpio_chip' not set
```

**解决方法**：
1. 确保配置文件存在
2. 或通过命令行提供所有参数

---

## 🎯 优势

| 特性 | 之前 | 现在 |
|------|------|------|
| **启动命令** | 需要 `--params-file` | 直接运行 |
| **配置文件** | 必须手动指定 | 自动查找 |
| **使用复杂度** | 高 | 低 |
| **灵活性** | 中 | 高 (支持多个查找路径) |

---

## 📊 完整示例

### 编译和运行

```bash
# 1. 编译
cd ~/NX_volleyball/ros2_ws
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash

# 2. 运行 (自动加载配置)
sudo -E ros2 run volleyball_stereo_driver stereo_system_node

# 3. 验证 (新终端)
ros2 topic hz /stereo/left/image_raw
```

### 修改参数

```bash
# 1. 编辑配置文件
nano src/volleyball_stereo_driver/config/system_params.yaml

# 2. 重新编译
colcon build --packages-select volleyball_stereo_driver

# 3. 运行 (自动加载新配置)
sudo -E ros2 run volleyball_stereo_driver stereo_system_node
```

---

## 🔧 故障排除

### Q1: 节点使用了错误的配置文件

**查看日志确认加载的文件**:
```
[INFO] [stereo_system_node]: 自动加载配置文件: /path/to/config.yaml
```

**解决**: 删除或移动不需要的配置文件

### Q2: 配置文件存在但未加载

**检查文件权限**:
```bash
ls -l ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/system_params.yaml
```

**检查文件格式**:
```bash
cat ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/system_params.yaml
```

### Q3: 想使用自定义配置文件

**方法 1**: 放到查找路径中
```bash
cp my_config.yaml ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/config/system_params.yaml
```

**方法 2**: 手动指定
```bash
sudo -E ros2 run volleyball_stereo_driver stereo_system_node \
    --ros-args --params-file my_config.yaml
```

---

## 📋 快速命令参考

```bash
# === 编译 ===
cd ~/NX_volleyball/ros2_ws
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash

# === 运行 (自动加载配置) ===
sudo -E ros2 run volleyball_stereo_driver stereo_system_node

# === 或使用脚本 ===
sudo ./install/volleyball_stereo_driver/share/volleyball_stereo_driver/scripts/start_system.sh

# === 监控 ===
ros2 topic hz /stereo/left/image_raw
```

---

**现在启动节点更简单了！无需手动指定配置文件路径。**
