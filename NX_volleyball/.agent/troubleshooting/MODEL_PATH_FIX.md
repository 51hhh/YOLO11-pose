# 🔧 模型文件路径问题修复

## 问题描述

节点运行时报错：
```
❌ 无法打开引擎文件: model/yolo11n.engine
```

**原因**: 节点使用相对路径 `model/yolo11n.engine`，但运行时当前目录不是源码目录。

---

## ✅ 快速解决方案（在 NX 上执行）

### 方法 1: 手动复制模型文件

```bash
cd ~/NX_volleyball/ros2_ws

# 创建模型目录
mkdir -p install/volleyball_stereo_driver/share/volleyball_stereo_driver/model

# 复制模型文件
cp src/volleyball_stereo_driver/model/yolo11n.engine \
   install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/

# 验证
ls -lh install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/

# 重新运行
source install/setup.bash
ros2 run volleyball_stereo_driver volleyball_tracker_node
```

### 方法 2: 重新编译（推荐）

如果模型文件已经在源码目录，重新编译会自动复制：

```bash
cd ~/NX_volleyball/ros2_ws

# 确保模型文件存在
ls src/volleyball_stereo_driver/model/yolo11n.engine

# 重新编译
colcon build --packages-select volleyball_stereo_driver

# 验证安装
ls install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/

# 运行
source install/setup.bash
ros2 run volleyball_stereo_driver volleyball_tracker_node
```

---

## 🔍 验证模型文件

```bash
# 检查源码目录
ls -lh ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/

# 检查安装目录
ls -lh ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/

# 应该看到:
# yolo11n.engine (大小约 10-20 MB)
```

---

## 📋 完整流程

### 如果模型文件不在 NX 上

```bash
# 在本地机器上转换模型
cd ~/desktop/yolo/yoloProject/NX_volleyball
python3 scripts/convert_yolo_to_tensorrt.py \
  --model ros2_ws/src/volleyball_stereo_driver/model/best.pt \
  --output ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine \
  --fp16

# 传输到 NX
scp ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine \
  nvidia@10.42.0.148:~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/

# 在 NX 上重新编译
ssh nvidia@10.42.0.148
cd ~/NX_volleyball/ros2_ws
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash
ros2 run volleyball_stereo_driver volleyball_tracker_node
```

---

## 🎯 预期结果

成功运行后应该看到：

```
[INFO] [volleyball_tracker]: 🏐 排球追踪节点初始化...
[INFO] [volleyball_tracker]: 📄 加载配置文件: ...
[INFO] [volleyball_tracker]: 📷 初始化相机...
[INFO] [volleyball_tracker]: 🎯 初始化 YOLO 检测器...
🎯 初始化 YOLO 检测器...
   模型: model/yolo11n.engine
   置信度阈值: 0.5
   NMS 阈值: 0.4
📦 加载引擎文件: XX.XX MB
✅ Engine 加载成功
   输入尺寸: 640x640
   输出尺寸: 84x8400
📊 缓冲区分配:
   输入: X.XX MB
   输出: X.XX MB
✅ YOLO 检测器初始化成功
[INFO] [volleyball_tracker]: 📐 初始化立体匹配器...
[INFO] [volleyball_tracker]: 🎲 初始化卡尔曼滤波器...
[INFO] [volleyball_tracker]: ✂️  初始化 ROI 管理器...
[INFO] [volleyball_tracker]: ✅ 排球追踪节点已启动
[INFO] [volleyball_tracker]:    状态: GLOBAL_SEARCH
[INFO] [volleyball_tracker]:    等待触发信号...
```

---

## 💡 提示

1. **模型文件大小**: yolo11n.engine 通常 10-20 MB
2. **转换时间**: 首次转换需要 5-10 分钟
3. **FP16 精度**: 已启用，模型更小更快

---

**执行上述方法 1 或方法 2，然后重新运行节点！** 🚀
