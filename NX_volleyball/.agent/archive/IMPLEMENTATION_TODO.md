# 🏐 双目排球追踪系统 - 完整实现 TODO

## 📋 项目配置

### 硬件参数确认
- ✅ 相机: 海康 MV-CA016-10UC × 2
- ✅ 帧率: **100 FPS**
- ✅ 触发: **外部上升沿触发** (NX PWM → 两相机并联)
- ✅ 基线: 25cm
- ✅ 镜头: 5mm 广角

### 软件配置
- ✅ ROS2: Humble (推荐) 或 Foxy
- ✅ 平台: Orin NX 16GB
- ✅ 模型: 当前使用普通检测模型，预留 Pose 接口

---

## 🎯 实现路线图

### 阶段 0: 环境准备 (1-2 天)

#### 0.1 ROS2 环境安装
- [ ] 安装 ROS2 Humble
- [ ] 配置 workspace
- [ ] 安装依赖包

#### 0.2 海康 SDK 安装
- [ ] 下载 MVS SDK (Linux ARM64)
- [ ] 安装驱动和库
- [ ] 测试相机连接

#### 0.3 PWM 配置
- [ ] 配置 NX GPIO/PWM 引脚
- [ ] 编写 PWM 驱动脚本
- [ ] 测试 100Hz 触发信号

---

### 阶段 1: 相机标定工具 (2-3 天)

#### 1.1 标定数据采集
- [ ] 编写棋盘格采集脚本
- [ ] 实现同步触发采集
- [ ] 自动保存图像对

#### 1.2 标定计算
- [ ] OpenCV 双目标定
- [ ] 验证重投影误差
- [ ] 保存标定参数

#### 1.3 标定验证
- [ ] 测距精度测试 (3m, 9m, 15m)
- [ ] 可视化极线约束

**产出**: `stereo_calib.npz` (K1, D1, K2, D2, R, T, P1, P2)

---

### 阶段 2: ROS2 消息和包结构 (1 天)

#### 2.1 创建 ROS2 包
```bash
volleyball_stereo_msgs/    # 消息定义
volleyball_stereo_driver/  # 相机驱动
volleyball_detector/       # 检测节点
volleyball_tracker/        # 追踪节点
volleyball_viz/            # 可视化
```

#### 2.2 定义消息
- [ ] `VolleyballPose3D.msg`
- [ ] `StereoDetection.msg`
- [ ] `VolleyballDebug.msg`
- [ ] `CameraInfo.msg` (扩展)

#### 2.3 编译测试
- [ ] `colcon build`
- [ ] 消息导入测试

---

### 阶段 3: 相机驱动节点 (3-4 天)

#### 3.1 PWM 触发器
- [ ] 实现 100Hz PWM 生成
- [ ] 可配置频率和占空比
- [ ] 发布触发时间戳

#### 3.2 海康相机封装
- [ ] Python/C++ 接口封装
- [ ] 配置外部触发模式
- [ ] 图像采集和缓存

#### 3.3 同步节点
- [ ] 双目图像时间戳对齐
- [ ] 发布 `StereoImage`
- [ ] 监控同步误差

**测试**: 验证时间戳差异 <1ms

---

### 阶段 4: 检测节点 (4-5 天)

#### 4.1 TensorRT 推理引擎
- [ ] 加载 TensorRT 模型
- [ ] 预留 Pose 模型接口
- [ ] Batch 推理 (左右图)

#### 4.2 状态机实现
- [ ] 全图检测模式
- [ ] ROI 追踪模式
- [ ] 自动切换逻辑

#### 4.3 几何拟合 (当前用 BBox)
- [ ] BBox 中心提取
- [ ] **预留**: Pose 关键点拟合接口
- [ ] 坐标还原 (ROI → 原图)

#### 4.4 双目匹配
- [ ] 左右目标匹配 (IoU)
- [ ] 发布 `StereoDetection`

**测试**: 检测精度和帧率

---

### 阶段 5: 立体匹配节点 (2-3 天)

#### 5.1 稀疏去畸变
- [ ] 加载标定参数
- [ ] `undistortPoints` 实现
- [ ] 批量处理优化

#### 5.2 三角测量
- [ ] `triangulatePoints` 封装
- [ ] 深度计算
- [ ] 置信度评估

#### 5.3 坐标转换
- [ ] 相机坐标系 → 世界坐标系
- [ ] 发布 `VolleyballPose3D` (raw)

**测试**: 深度精度验证

---

### 阶段 6: 追踪节点 (2-3 天)

#### 6.1 3D 卡尔曼滤波
- [ ] 9 维状态向量 (x,y,z, vx,vy,vz, ax,ay,az)
- [ ] 动态观测噪声调整
- [ ] 预测和更新

#### 6.2 轨迹管理
- [ ] 多目标追踪 (ByteTrack 3D)
- [ ] ID 分配和保持
- [ ] 轨迹历史记录

#### 6.3 速度估计
- [ ] 3D 速度计算
- [ ] 发布平滑后的 `VolleyballPose3D`

**测试**: 轨迹平滑度

---

### 阶段 7: 可视化工具 (2 天)

#### 7.1 RViz2 配置
- [ ] 3D 轨迹显示
- [ ] 速度向量
- [ ] 相机位姿

#### 7.2 调试界面
- [ ] 左右图像叠加
- [ ] 关键点/BBox 可视化
- [ ] 实时性能监控

#### 7.3 rqt 插件
- [ ] 参数调节面板
- [ ] 状态监控

---

### 阶段 8: 测试和优化 (3-4 天)

#### 8.1 精度测试
- [ ] 静态测距精度
- [ ] 动态追踪精度
- [ ] 不同距离测试 (3m, 9m, 15m)

#### 8.2 性能测试
- [ ] 端到端延迟
- [ ] CPU/GPU 占用率
- [ ] 帧率稳定性

#### 8.3 鲁棒性测试
- [ ] 遮挡处理
- [ ] 快速运动
- [ ] 光照变化

---

### 阶段 9: 文档和部署 (1-2 天)

#### 9.1 使用文档
- [ ] 安装指南
- [ ] 标定流程
- [ ] 启动教程

#### 9.2 Launch 文件
- [ ] 完整系统启动
- [ ] 参数配置
- [ ] 调试模式

#### 9.3 Docker 镜像 (可选)
- [ ] 构建 Docker
- [ ] 一键部署

---

## 📦 项目结构

```
NX_volleyball/
├── IMPLEMENTATION_TODO.md          # 本文件
├── ROS2_ARCHITECTURE.md            # 架构设计
├── README.md                       # 项目说明
│
├── ros2_ws/                        # ROS2 工作空间
│   └── src/
│       ├── volleyball_stereo_msgs/     # 消息定义
│       │   ├── msg/
│       │   │   ├── VolleyballPose3D.msg
│       │   │   ├── StereoDetection.msg
│       │   │   └── VolleyballDebug.msg
│       │   ├── CMakeLists.txt
│       │   └── package.xml
│       │
│       ├── volleyball_stereo_driver/   # 相机驱动
│       │   ├── volleyball_stereo_driver/
│       │   │   ├── pwm_trigger.py
│       │   │   ├── hik_camera.py
│       │   │   └── stereo_sync_node.py
│       │   ├── config/
│       │   │   └── camera_params.yaml
│       │   ├── launch/
│       │   │   └── stereo_driver.launch.py
│       │   └── package.xml
│       │
│       ├── volleyball_detector/        # 检测节点
│       │   ├── volleyball_detector/
│       │   │   ├── trt_detector.py
│       │   │   ├── detection_node.py
│       │   │   └── stereo_matcher.py
│       │   ├── models/
│       │   │   └── volleyball.engine
│       │   ├── config/
│       │   │   └── detector_params.yaml
│       │   └── package.xml
│       │
│       ├── volleyball_tracker/         # 追踪节点
│       │   ├── volleyball_tracker/
│       │   │   ├── kalman_3d.py
│       │   │   └── tracking_node.py
│       │   ├── config/
│       │   │   └── tracker_params.yaml
│       │   └── package.xml
│       │
│       └── volleyball_viz/             # 可视化
│           ├── volleyball_viz/
│           │   ├── rviz_publisher.py
│           │   └── debug_node.py
│           ├── rviz/
│           │   └── volleyball.rviz
│           └── package.xml
│
├── calibration/                    # 标定工具
│   ├── capture_chessboard.py
│   ├── stereo_calibrate.py
│   ├── validate_calibration.py
│   └── data/
│       ├── left/
│       ├── right/
│       └── stereo_calib.npz
│
├── scripts/                        # 工具脚本
│   ├── setup_environment.sh
│   ├── test_pwm.py
│   ├── test_camera.py
│   └── benchmark.py
│
└── docs/                          # 文档
    ├── INSTALLATION.md
    ├── CALIBRATION.md
    ├── USAGE.md
    └── TROUBLESHOOTING.md
```

---

## 🔧 关键技术点

### 1. PWM 触发配置

**NX GPIO 引脚映射**:
```python
# Jetson Orin NX
# Pin 32 (GPIO09) - PWM0
# Pin 33 (GPIO12) - PWM2
```

**100Hz PWM 生成**:
```python
import Jetson.GPIO as GPIO

PWM_PIN = 32  # GPIO09
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PWM_PIN, GPIO.OUT)

pwm = GPIO.PWM(PWM_PIN, 100)  # 100 Hz
pwm.start(50)  # 50% 占空比
```

### 2. 海康相机触发配置

```python
# MVS SDK 配置
camera.set_enum_value("TriggerMode", 1)  # On
camera.set_enum_value("TriggerSource", 0)  # Line0
camera.set_enum_value("TriggerActivation", 0)  # RisingEdge
camera.set_float_value("ExposureTime", 800)  # 800us
```

### 3. 模型接口设计

```python
class DetectorInterface:
    """检测器接口 - 支持普通检测和 Pose 检测"""
    
    def __init__(self, model_type='bbox'):
        self.model_type = model_type  # 'bbox' or 'pose'
    
    def detect(self, image):
        if self.model_type == 'bbox':
            return self.detect_bbox(image)
        else:
            return self.detect_pose(image)
    
    def detect_bbox(self, image):
        """当前实现: BBox 检测"""
        # 返回: cx, cy, w, h, conf
        pass
    
    def detect_pose(self, image):
        """预留接口: Pose 检测"""
        # 返回: cx, cy, r, keypoints, conf
        # keypoints: (5, 3) - [x, y, visibility]
        pass
    
    def get_center(self, detection):
        """统一获取中心点"""
        if self.model_type == 'bbox':
            return detection['cx'], detection['cy']
        else:
            return detection['keypoints'][0]  # Center 点
```

---

## 📊 性能目标

| 指标 | 目标值 | 验收标准 |
|------|--------|----------|
| **采集帧率** | 100 FPS | 稳定 ±2 FPS |
| **同步误差** | <1ms | 99% 帧 <1ms |
| **检测延迟** | <8ms | 平均值 |
| **端到端延迟** | <15ms | 采集→发布 |
| **深度精度 (3m)** | ±2cm | 静态测试 |
| **深度精度 (9m)** | ±15cm | 静态测试 |
| **深度精度 (15m)** | ±40cm | 静态测试 |
| **CPU 占用** | <50% | 4 核平均 |
| **GPU 占用** | <60% | - |

---

## ✅ 验收清单

### 功能验收
- [ ] 双目相机同步采集 (100 FPS)
- [ ] PWM 触发稳定工作
- [ ] 检测模型正常推理
- [ ] 3D 坐标准确发布
- [ ] 卡尔曼滤波平滑轨迹
- [ ] RViz2 可视化正常

### 性能验收
- [ ] 端到端延迟 <15ms
- [ ] 帧率稳定 >95 FPS
- [ ] 深度精度符合目标
- [ ] 系统资源占用合理

### 鲁棒性验收
- [ ] 连续运行 1 小时无崩溃
- [ ] 处理遮挡和跟丢
- [ ] 光照变化适应

---

## 🚀 快速开始 (完成后)

```bash
# 1. 启动 PWM 触发
ros2 run volleyball_stereo_driver pwm_trigger

# 2. 启动相机驱动
ros2 launch volleyball_stereo_driver stereo_driver.launch.py

# 3. 启动检测和追踪
ros2 launch volleyball_detector detector.launch.py

# 4. 启动可视化
rviz2 -d src/volleyball_viz/rviz/volleyball.rviz

# 或一键启动
ros2 launch volleyball_stereo_driver full_system.launch.py
```

---

## 📝 下一步行动

**立即开始**: 阶段 0 - 环境准备

1. 安装 ROS2 Humble
2. 配置海康 SDK
3. 测试 PWM 输出

**预计总时长**: 20-25 天 (全职开发)

---

**准备好开始了吗？我可以立即为你生成第一阶段的代码！**
