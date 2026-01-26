# 📁 NX_volleyball 项目文档索引

> 更新时间: 2026-01-26

---

## 📊 项目当前状态

### ✅ 已完成 (100%)
- ✅ ROS2 包结构 (`volleyball_stereo_driver`)
- ✅ 核心组件实现 (7个头文件，8个源文件)
- ✅ PWM 触发器 (100Hz, ±0.05%精度)
- ✅ 海康双目相机驱动 (回调模式，零等待)
- ✅ 帧同步系统 (100%同步率，0.06%丢帧)
- ✅ Batch=2 TensorRT推理 (9.5-12ms延迟)
- ✅ 立体匹配与3D追踪
- ✅ **NX部署验证通过** (2026-01-26)

### ⏳ 待优化
- [ ] **双目标定** - 当前使用默认参数
- [ ] **YOLO 真实模型** - 当前使用占位符
- [ ] **FP16推理** - 预期提升30-50%性能

---

## 📂 文档结构

```
.agent/
├── README.md                    # 📋 本文件 - 项目状态和文档索引
├── DEPLOYMENT_VALIDATION.md     # ✅ NX部署验证报告 (2026-01-26)
├── IMPLEMENTATION_SUMMARY.md    # ✅ 实现完成总结
├── PROGRESS.md                  # 📊 详细进度追踪
│
├── reference/                   # 📖 技术参考文档
│   ├── ROS2_ARCHITECTURE_FINAL.md      # 最终架构设计
│   ├── PERFORMANCE_OPTIMIZATION.md     # 🚀 性能优化完整指南（已整合）
│   ├── NX_BUILD_TEST_GUIDE.md          # NX 编译测试指南
│   ├── YOLO_INTEGRATION_GUIDE.md       # YOLO 集成指南
│   ├── YOLO_COMPLETE_SUMMARY.md        # YOLO 集成总结
│   ├── CAMERA_PARAMS_GUIDE.md          # 相机参数说明
│   └── INTEGRATED_NODE_GUIDE.md        # 整合节点使用
│
├── troubleshooting/             # 🔧 故障排查
│   ├── CUDA_FIX.md                     # CUDA 编译问题
│   ├── GPIO_CUSTOM_BOARD.md            # 自定义载板 GPIO
│   ├── HIK_SDK_FIX.md                  # 海康 SDK 问题
│   ├── MODEL_PATH_FIX.md               # 模型路径问题
│   ├── OPENCV_FIX.md                   # OpenCV 兼容性
│   ├── PWM_PRECISION_GUIDE.md          # PWM 精度优化
│   └── TENSORRT_VERSION_GUIDE.md       # TensorRT 版本
│
└── archive/                     # 📦 归档文档 (旧版本/已整合/已完成)
    ├── STAGE2_IMPLEMENTATION_PLAN.md   # 阶段2实施计划（已完成）
    ├── PWM_TIMESTAMP_SYNC_DESIGN.md    # PWM同步设计（已实现）
    ├── 性能瓶颈分析.md                 # 性能瓶颈分析（已过时）
    ├── 重新部署说明.md                 # 重新部署说明（已过时）
    ├── NX性能优化指南.md               # 旧性能指南（已整合）
    ├── 性能优化详解.md                 # 旧性能详解（已整合）
    ├── ROS2_ARCHITECTURE.md            # 旧版架构
    ├── ROS2_NODE_DESIGN_SIMPLIFIED.md  # 旧版节点设计
    ├── ARCHITECTURE_SUMMARY.md         # 旧架构总结
    └── ...
```

---

## 🚀 快速开始

### 在 NX 上编译运行

```bash
cd ~/NX_volleyball
./quick_build_test.sh
```

### 手动步骤

```bash
# 1. 编译
cd ~/NX_volleyball/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash

# 2. 运行基础节点 (PWM + 相机)
sudo -E ros2 run volleyball_stereo_driver stereo_system_node

# 3. 运行完整追踪节点 (需要标定和模型)
ros2 run volleyball_stereo_driver volleyball_tracker_node
```

---

## 🎯 核心架构

```
volleyball_stereo_driver/  (唯一的 ROS2 包)
├── stereo_system_node     # 基础节点: PWM + 相机
└── volleyball_tracker_node # All-in-One: 完整追踪

5 个核心组件:
├── HighPrecisionPWM       # PWM 触发 (100Hz)
├── HikCameraWrapper       # 海康相机封装
├── YOLODetector          # TensorRT 推理
├── StereoMatcher         # 三角测量
├── KalmanFilter3D        # 3D 追踪
└── ROIManager            # ROI 管理
```

---

## 📖 关键文档快速链接

| 需求 | 文档 |
|------|------|
| 了解系统架构 | [reference/ROS2_ARCHITECTURE_FINAL.md](reference/ROS2_ARCHITECTURE_FINAL.md) |
| 在 NX 上编译测试 | [reference/NX_BUILD_TEST_GUIDE.md](reference/NX_BUILD_TEST_GUIDE.md) |
| 集成 YOLO 模型 | [reference/YOLO_INTEGRATION_GUIDE.md](reference/YOLO_INTEGRATION_GUIDE.md) |
| 解决编译问题 | [troubleshooting/](troubleshooting/) |
| 查看历史设计 | [archive/](archive/) |

---

## 📝 下一步行动

1. **双目标定**
   ```bash
   cd ~/NX_volleyball/calibration
   python3 capture_chessboard.py  # 采集棋盘格图像
   # TODO: 运行标定程序
   ```

2. **准备 YOLO 模型**
   ```bash
   python3 scripts/convert_yolo_to_tensorrt.py \
     --model model/best.pt \
     --output ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine \
     --fp16
   ```

3. **端到端测试**
   ```bash
   ros2 run volleyball_stereo_driver volleyball_tracker_node
   ```
