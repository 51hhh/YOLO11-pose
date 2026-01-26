# 🏐 双目排球追踪 ROS2 架构设计 (最终版)

## 📅 更新时间
**最后更新**: 2026-01-24

---

## 🎯 设计原则

- **极简架构**: 仅 **1 个 ROS2 包**，**2 个节点**
- **C++ 实现**: 高性能，遵循现有代码风格
- **All-in-One 集成**: 所有功能集成在一个节点中
- **Colcon 构建**: 标准 ROS2 构建流程
- **无 Launch 文件**: 直接运行可执行文件

---

## 🏗️ 最终架构

```
┌────────────────────────────────────────────────────────────┐
│              ROS2 节点拓扑图 (最终版)                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  [stereo_system_node]  ← 基础节点 (仅 PWM + 相机)          │
│   ├─ PWM 触发 (100Hz)                                      │
│   ├─ 双目相机采集                                          │
│   └─ 发布原始图像                                          │
│                                                            │
│                      OR                                    │
│                                                            │
│  [volleyball_tracker_node]  ← All-in-One 节点             │
│   ├─ PWM 触发 (100Hz)                                      │
│   ├─ 双目相机同步采集                                      │
│   ├─ YOLO11n 检测 (全图/ROI)                               │
│   ├─ 立体匹配 (三角测量)                                   │
│   ├─ 3D 卡尔曼滤波                                         │
│   └─ 发布 3D 位置和速度                                    │
│          ↓                                                 │
│   /volleyball/pose_3d                                      │
│   /volleyball/velocity                                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 📦 包结构 (单包设计)

```
volleyball_stereo_driver/  # 唯一的 ROS2 包
├── include/volleyball_stereo_driver/
│   ├── high_precision_pwm.hpp           # ✅ PWM 实现
│   ├── hik_camera_wrapper.hpp           # ✅ 海康相机封装
│   ├── yolo_detector.hpp                # 🆕 YOLO TensorRT
│   ├── stereo_matcher.hpp               # 🆕 立体匹配
│   ├── kalman_filter_3d.hpp             # 🆕 3D 卡尔曼滤波
│   ├── roi_manager.hpp                  # 🆕 ROI 管理
│   └── volleyball_tracker_node.hpp      # 🆕 主追踪节点
│
├── src/
│   ├── stereo_system_node.cpp           # ✅ 基础节点 (PWM + 相机)
│   ├── volleyball_tracker_node.cpp      # 🆕 All-in-One 追踪节点
│   ├── high_precision_pwm.cpp           # ✅ PWM 实现
│   ├── hik_camera_wrapper.cpp           # ✅ 相机封装
│   ├── yolo_detector.cpp                # 🆕 YOLO 实现
│   ├── stereo_matcher.cpp               # 🆕 立体匹配实现
│   ├── kalman_filter_3d.cpp             # 🆕 卡尔曼滤波实现
│   └── roi_manager.cpp                  # 🆕 ROI 管理实现
│
├── config/
│   ├── pwm_params.yaml                  # ✅ PWM 参数
│   ├── camera_params.yaml               # ✅ 相机参数
│   ├── system_params.yaml               # ✅ 系统参数
│   └── tracker_params.yaml              # 🆕 追踪系统参数
│
├── model/
│   └── yolo11n.engine                   # 🆕 YOLO TensorRT 模型
│
├── calibration/
│   └── stereo_calib.npz                 # 🆕 双目标定文件
│
├── scripts/
│   ├── start_pwm_node.sh                # ✅ 启动 PWM 节点
│   └── start_system.sh                  # ✅ 启动系统节点
│
├── CMakeLists.txt                       # ✅ 已更新
└── package.xml                          # ✅ 已更新
```

---

## 🔧 节点详细设计

### 节点 1: stereo_system_node ✅
**功能**: 基础节点 (仅 PWM 触发 + 双目相机采集)  
**状态**: 已实现  
**用途**: 测试相机同步，或为其他节点提供图像流

#### 发布话题
- `/camera_trigger` (std_msgs/Header) - PWM 触发时间戳
- `/stereo/left/image_raw` (sensor_msgs/Image) - 左图
- `/stereo/right/image_raw` (sensor_msgs/Image) - 右图

---

### 节点 2: volleyball_tracker_node 🆕
**功能**: All-in-One 排球追踪节点  
**状态**: 待实现 (框架已创建)  
**语言**: C++

#### 集成功能
1. **PWM 触发**: 内置 100Hz PWM 生成
2. **相机采集**: 双目同步采集
3. **YOLO 检测**: TensorRT 推理，状态机切换
4. **立体匹配**: 稀疏去畸变 + 三角测量
5. **3D 追踪**: 卡尔曼滤波，速度估计

#### 发布话题
| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/volleyball/pose_3d` | `geometry_msgs/PoseStamped` | 100Hz | 3D 位置 |
| `/volleyball/velocity` | `geometry_msgs/Vector3Stamped` | 100Hz | 3D 速度 |
| `/volleyball/debug_info` | `std_msgs/String` | 10Hz | 调试信息 (JSON) |

#### 参数配置
所有参数统一在 `config/tracker_params.yaml` 中配置：
- 相机参数 (索引、曝光、增益、触发)
- YOLO 参数 (模型路径、阈值、ROI 大小)
- 立体匹配参数 (标定文件、视差范围)
- 追踪参数 (卡尔曼噪声、最大丢失帧数)
- 调试参数 (日志间隔)

---

## 🚀 编译和使用

### 编译
```bash
cd ~/desktop/yolo/yoloProject/NX_volleyball/ros2_ws
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash
```

### 运行基础节点 (测试用)
```bash
ros2 run volleyball_stereo_driver stereo_system_node \
  --ros-args --params-file src/volleyball_stereo_driver/config/system_params.yaml
```

### 运行追踪节点 (完整功能)
```bash
ros2 run volleyball_stereo_driver volleyball_tracker_node \
  --ros-args --params-file src/volleyball_stereo_driver/config/tracker_params.yaml
```

---

## 📋 实现状态

### ✅ 已完成
- [x] 包结构创建
- [x] CMakeLists.txt 配置
- [x] package.xml 配置
- [x] 所有头文件定义
- [x] 配置文件模板
- [x] PWM 触发实现
- [x] 海康相机封装

### 🔄 进行中
- [ ] volleyball_tracker_node.cpp 主节点实现
- [ ] yolo_detector.cpp TensorRT 推理
- [ ] stereo_matcher.cpp 立体匹配
- [ ] kalman_filter_3d.cpp 卡尔曼滤波
- [ ] roi_manager.cpp ROI 管理

### 📅 待完成
- [ ] YOLO11n 模型训练和转换
- [ ] 双目标定 (生成 stereo_calib.npz)
- [ ] 集成测试
- [ ] 性能优化

---

## 🎯 下一步行动

### 立即执行
1. **实现 ROI 管理器** (最简单)
   ```bash
   # 创建 src/roi_manager.cpp
   ```

2. **实现卡尔曼滤波器**
   ```bash
   # 创建 src/kalman_filter_3d.cpp
   ```

3. **实现立体匹配**
   ```bash
   # 创建 src/stereo_matcher.cpp
   # 需要先完成双目标定
   ```

4. **实现 YOLO 检测器**
   ```bash
   # 创建 src/yolo_detector.cpp
   # 需要 TensorRT 模型
   ```

5. **实现主节点**
   ```bash
   # 创建 src/volleyball_tracker_node.cpp
   # 集成所有组件
   ```

---

## 📊 优势总结

### 架构优势
1. **极简**: 只有 1 个包，2 个节点
2. **高效**: 单进程内处理，无 ROS 通信开销
3. **统一**: 全部 C++，代码风格一致
4. **灵活**: 可选择运行基础节点或完整追踪节点

### 开发优势
1. **易维护**: 所有代码在一个包中
2. **易调试**: 单进程调试
3. **易部署**: 一次编译，两个可执行文件
4. **易配置**: 统一的 YAML 配置文件

---

## 📝 技术要点

### 依赖管理
- **必需**: ROS2 Humble, OpenCV, libgpiod, 海康 SDK
- **可选**: TensorRT (用于 volleyball_tracker_node)
- **条件编译**: CMake 自动检测依赖，缺少时跳过对应节点

### 性能优化
- **TensorRT FP16**: 加速推理
- **ROI 裁切**: 减少计算量
- **状态机**: 全图/ROI 动态切换
- **动态噪声**: 根据深度调整卡尔曼滤波

---

**文档版本**: v3.0 (最终版)  
**最后更新**: 2026-01-24  
**包名**: `volleyball_stereo_driver`  
**节点**: `stereo_system_node`, `volleyball_tracker_node`
