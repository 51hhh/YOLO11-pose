# 🏐 海康双目高速排球追踪系统

> **Jetson Orin NX 16GB** | **ROS2 Humble** | **TensorRT** | **100+ FPS**

---

## 📊 项目状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| ✅ 代码实现 | **完成** | ROS2 包、所有组件 |
| ⏳ 双目标定 | 待执行 | 采集棋盘格图像 |
| ⏳ YOLO 模型 | 待准备 | 训练/导出 TensorRT |
| ⏳ NX 测试 | 待执行 | 真实硬件测试 |

📖 **详细进度**: [.agent/README.md](.agent/README.md)

---

## 🚀 快速开始

```bash
# 一键编译测试
cd ~/NX_volleyball
./quick_build_test.sh

# 或手动运行
cd ~/NX_volleyball/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash

# 运行节点
sudo -E ros2 run volleyball_stereo_driver stereo_system_node
```

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────┐
│      volleyball_tracker_node             │
│         (All-in-One 追踪节点)             │
├──────────────────────────────────────────┤
│  PWM 触发 (100Hz)                        │
│      ↓                                   │
│  双目相机同步采集 (海康 MV-CA016-10UC)    │
│      ↓                                   │
│  YOLO11n TensorRT 检测 (全图/ROI)        │
│      ↓                                   │
│  立体匹配 (三角测量)                      │
│      ↓                                   │
│  3D 卡尔曼滤波                           │
│      ↓                                   │
│  发布 /volleyball/pose_3d                │
└──────────────────────────────────────────┘
```

---

## 📦 项目结构

```
NX_volleyball/
├── README.md                # 本文件
├── quick_build_test.sh      # 快速编译脚本
├── deploy_to_nx.sh          # 部署到 NX 脚本
│
├── ros2_ws/                 # ROS2 工作空间
│   └── src/volleyball_stereo_driver/  # 唯一的 ROS2 包
│       ├── include/         # 7 个头文件
│       ├── src/             # 10 个源文件
│       ├── config/          # 配置文件
│       ├── model/           # YOLO 模型目录
│       └── calibration/     # 标定文件目录
│
├── scripts/                 # 工具脚本
│   ├── hik_camera.py        # 相机驱动
│   ├── test_pwm_*.py        # PWM 测试
│   └── convert_yolo_to_tensorrt.py  # 模型转换
│
├── calibration/             # 标定工具
│   └── capture_chessboard.py
│
└── .agent/                  # 📖 详细文档
    ├── README.md            # 文档索引和项目状态
    ├── reference/           # 技术参考文档
    ├── troubleshooting/     # 故障排查
    └── archive/             # 归档文档
```

---

## 🔧 硬件配置

| 组件 | 型号/参数 |
|------|----------|
| **平台** | Jetson Orin NX 16GB |
| **相机** | 2x Hikvision MV-CA016-10UC (IMX273, 1440x1080) |
| **镜头** | 5mm 广角 (FOV ~52°) |
| **基线** | 25cm |
| **触发** | GPIO PWM → 相机 Line0 (硬触发) |

---

## 📊 性能指标

| 距离 | 深度精度 | 说明 |
|------|----------|------|
| 3m | ±1.2cm | 极高精度 |
| 9m | ±11cm | 高精度 |
| 15m | ±31cm | 中等精度 (物理限制) |

---

## 📖 文档

| 需求 | 文档路径 |
|------|----------|
| 项目状态和文档索引 | [.agent/README.md](.agent/README.md) |
| 架构设计 | [.agent/reference/ROS2_ARCHITECTURE_FINAL.md](.agent/reference/ROS2_ARCHITECTURE_FINAL.md) |
| NX 编译测试 | [.agent/reference/NX_BUILD_TEST_GUIDE.md](.agent/reference/NX_BUILD_TEST_GUIDE.md) |
| YOLO 集成 | [.agent/reference/YOLO_INTEGRATION_GUIDE.md](.agent/reference/YOLO_INTEGRATION_GUIDE.md) |
| 故障排查 | [.agent/troubleshooting/](.agent/troubleshooting/) |

---

## 📝 下一步

1. **双目标定**: 采集棋盘格图像，运行标定程序
2. **YOLO 模型**: 训练排球检测模型，导出 TensorRT
3. **NX 测试**: 真实硬件端到端测试

---

*更新: 2026-01-25*
