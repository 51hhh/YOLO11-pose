# 🎯 YOLO11n 集成完整指南

## 📋 概述

本指南将帮助你将 YOLOv11n 目标检测模型集成到排球追踪系统中。

---

## 🔄 模型转换流程

### 步骤 1: 准备 PyTorch 模型

你已经有了 `best.pt` 模型文件。

### 步骤 2: 转换为 TensorRT Engine

```bash
cd ~/desktop/yolo/yoloProject/NX_volleyball

# 运行转换脚本
python3 scripts/convert_yolo_to_tensorrt.py \
  --model ros2_ws/src/volleyball_stereo_driver/model/best.pt \
  --output ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine \
  --imgsz 640 \
  --fp16

# 预期输出:
# ✅ ONNX 模型已生成
# ✅ TensorRT Engine 已生成
```

### 步骤 3: 传输到 NX

```bash
# 只传输模型文件
scp ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine \
  nvidia@10.42.0.148:~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/

# 或者传输整个源码（不包含 build/install）
cd ~/desktop/yolo/yoloProject
scp -r --exclude='build' --exclude='install' --exclude='log' \
  ./NX_volleyball/ nvidia@10.42.0.148:~
```

---

## 🔧 在 NX 上编译

```bash
# SSH 到 NX
ssh nvidia@10.42.0.148

# 进入工作空间
cd ~/NX_volleyball/ros2_ws

# 清理旧的编译文件
rm -rf build install log

# 编译
colcon build --packages-select volleyball_stereo_driver

# 加载环境
source install/setup.bash
```

---

## 🚀 运行系统

### 方法 1: 运行基础节点（仅相机）

```bash
ros2 run volleyball_stereo_driver stereo_system_node
```

### 方法 2: 运行完整追踪节点（含 YOLO）

```bash
ros2 run volleyball_stereo_driver volleyball_tracker_node
```

---

## 📊 验证 YOLO 检测

### 查看话题

```bash
# 查看所有话题
ros2 topic list

# 应该看到:
# /volleyball/pose_3d
# /volleyball/velocity
# /volleyball/debug_info

# 查看调试信息
ros2 topic echo /volleyball/debug_info
```

### 预期输出

```json
{
  "pos": [x, y, z],
  "vel": [vx, vy, vz],
  "depth": depth,
  "state": "GLOBAL" 或 "ROI"
}
```

---

## 🎯 模型要求

### 输入
- **格式**: RGB 图像
- **尺寸**: 640x640
- **归一化**: [0, 1]
- **布局**: CHW (Channel, Height, Width)

### 输出
- **格式**: [84, 8400]
- **内容**: [cx, cy, w, h, obj_conf, class0, ..., class79]
- **坐标**: 相对于 640x640 的归一化坐标

---

## 🔍 故障排查

### 问题 1: 模型转换失败

```bash
# 检查 ultralytics 是否安装
pip3 install ultralytics

# 检查 TensorRT 工具
which trtexec

# 如果没有，安装 TensorRT
sudo apt install tensorrt
```

### 问题 2: 编译失败 - CUDA 头文件未找到

```bash
# 检查 CUDA 路径
ls /usr/local/cuda/include/cuda_runtime_api.h

# 如果不存在，创建软链接
sudo ln -s /usr/include/aarch64-linux-gnu /usr/local/cuda/include
```

### 问题 3: 运行时找不到模型

```bash
# 检查模型文件是否存在
ls ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/yolo11n.engine

# 如果不存在，手动复制
cp ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine \
   ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/
```

### 问题 4: 检测不到目标

检查配置文件中的阈值：
```bash
nano ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/config/tracker_params.yaml

# 调整:
detector:
  confidence_threshold: 0.3  # 降低阈值试试
  nms_threshold: 0.4
```

---

## 📈 性能优化

### 1. 使用 FP16 精度

已在转换脚本中默认启用：
```bash
--fp16
```

### 2. 调整 ROI 大小

在 `tracker_params.yaml` 中：
```yaml
detector:
  roi_size: 320  # 较小的 ROI 更快
  global_size: 640
```

### 3. 降低处理频率

如果 100Hz 太快，可以在代码中跳帧处理。

---

## 🎓 下一步

### 1. 测试目标检测

使用真实排球测试检测效果。

### 2. 替换为 Pose 模型

如果需要关键点检测：
1. 训练 YOLOv11n-pose 模型
2. 修改 `yolo_detector.cpp` 的后处理逻辑
3. 输出格式变为: [56, 8400] (bbox + 17 关键点)

### 3. 优化追踪算法

- 调整卡尔曼滤波参数
- 优化 ROI 切换逻辑
- 添加多目标追踪

---

## 📝 快速命令参考

```bash
# 1. 转换模型（本地）
python3 scripts/convert_yolo_to_tensorrt.py --model model/best.pt --output model/yolo11n.engine

# 2. 传输到 NX
scp model/yolo11n.engine nvidia@10.42.0.148:~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/

# 3. 在 NX 上编译
ssh nvidia@10.42.0.148
cd ~/NX_volleyball/ros2_ws
rm -rf build install log
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash

# 4. 运行
ros2 run volleyball_stereo_driver volleyball_tracker_node
```

---

**准备好开始了！** 🚀

按照上述步骤，你的系统应该能够成功运行 YOLO 目标检测。
