# 🔧 海康 SDK 集成修复说明

## ✅ 已修复的问题

### 1. 库路径错误
- ❌ 错误: `/opt/MVS/lib/64/libMvCameraControl.so` (x86_64)
- ✅ 正确: `/opt/MVS/lib/aarch64/libMvCameraControl.so` (ARM64)

### 2. API 使用错误
根据海康官方示例代码，修复了以下 API 调用：

#### 添加 SDK 初始化
```cpp
// 在构造函数中添加
MV_CC_Initialize();  // 初始化 SDK (全局一次)
```

#### 修复图像获取方法
```cpp
// 旧方法 (错误)
MV_CC_GetOneFrameTimeout(handle, data, size, &frame_info, timeout);

// 新方法 (正确)
MV_CC_GetImageBuffer(handle, &stImageInfo, timeout);
// ... 使用图像 ...
MV_CC_FreeImageBuffer(handle, &stImageInfo);  // 释放缓冲区
```

---

## 📦 在 NX 上执行步骤

### 步骤 1: 传输更新的文件

```bash
# 在本地机器上
cd /home/rick/desktop/yolo/yoloProject
scp -r ./NX_volleyball/ nvidia@10.42.0.148:~
```

### 步骤 2: 在 NX 上编译

```bash
# SSH 到 NX
ssh nvidia@10.42.0.148

# 进入工作空间
cd ~/NX_volleyball/ros2_ws

# 清理旧的编译文件
rm -rf build/volleyball_stereo_driver install/volleyball_stereo_driver log/volleyball_stereo_driver

# 重新编译
colcon build --packages-select volleyball_stereo_driver --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source 环境
source install/setup.bash
```

**预期输出**:
```
Starting >>> volleyball_stereo_driver
Finished <<< volleyball_stereo_driver [18.5s]

Summary: 1 package finished [18.6s]
```

### 步骤 3: 测试 PWM 节点

```bash
# 终端 1
sudo -E ros2 run volleyball_stereo_driver pwm_trigger_node
```

### 步骤 4: 测试相机节点

```bash
# 终端 2 (新终端)
cd ~/NX_volleyball/ros2_ws
source install/setup.bash

# 运行相机节点
ros2 run volleyball_stereo_driver stereo_camera_node
```

**预期输出**:
```
✅ 海康 SDK 已初始化
[INFO] [stereo_camera_node]: 双目相机节点初始化
[INFO] [stereo_camera_node]:   左相机索引: 0
[INFO] [stereo_camera_node]:   右相机索引: 1
[INFO] [stereo_camera_node]:   曝光时间: 9867.0 us
✅ 相机已打开: 1440x1080
✅ 曝光时间: 9867 us
✅ 增益: 10.9854 dB
✅ 触发模式: On
✅ 触发源: Line0
✅ 开始采集
[INFO] [stereo_camera_node]: ✅ 双目相机节点已启动
```

---

## 🎯 验证系统

### 查看话题

```bash
# 新终端
ros2 topic list
```

**应该看到**:
```
/camera_trigger
/stereo/left/image_raw
/stereo/right/image_raw
```

### 查看帧率

```bash
ros2 topic hz /stereo/left/image_raw
```

**预期**: `average rate: 100.012`

---

## 🐛 可能的问题

### 问题 1: 编译仍然失败

**检查库文件是否存在**:
```bash
ls -l /opt/MVS/lib/aarch64/libMvCameraControl.so
```

如果不存在，检查实际路径:
```bash
find /opt/MVS -name "libMvCameraControl.so"
```

### 问题 2: 运行时找不到库

**设置 LD_LIBRARY_PATH**:
```bash
export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH
```

或添加到 `.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 问题 3: 未找到相机

**检查相机连接**:
```bash
lsusb | grep Hikvision

# 设置 USB 权限
sudo chmod 666 /dev/bus/usb/*/*
```

---

## 📝 关键修改总结

| 文件 | 修改内容 |
|------|---------|
| `CMakeLists.txt` | 库路径: `/opt/MVS/lib/64` → `/opt/MVS/lib/aarch64` |
| `hik_camera_wrapper.cpp` | 添加 `MV_CC_Initialize()` |
| `hik_camera_wrapper.cpp` | 使用 `MV_CC_GetImageBuffer()` + `MV_CC_FreeImageBuffer()` |
| `hik_camera_wrapper.cpp` | 添加 `<mutex>` 头文件 |

---

**现在可以在 NX 上重新编译和测试了！**
