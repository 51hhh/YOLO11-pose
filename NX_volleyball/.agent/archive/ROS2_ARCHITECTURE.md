# 🏐 双目排球追踪 ROS2 节点方案

## 📋 方案确认

### 硬件配置
- **相机**: 2x 海康 MV-CA016-10UC (IMX273, 1440x1080, 全局快门)
- **镜头**: 5mm 广角 (FOV ~52°)
- **基线**: 25cm
- **平台**: Orin NX 16GB
- **同步**: 硬触发 (GPIO/Line0)

### 性能目标
- **帧率**: >100 FPS (ROI 模式)
- **延迟**: <15ms (采集→发布)
- **精度**: 
  - 3m: ±1.2cm
  - 9m: ±11cm
  - 15m: ±31cm

---

## 🏗️ ROS2 节点架构

### 节点拓扑

```
┌─────────────────────────────────────────────────┐
│         volleyball_stereo_detector              │
│                                                 │
│  ┌──────────────┐      ┌──────────────┐       │
│  │  Left Camera │      │ Right Camera │       │
│  │   (Hikvision)│      │  (Hikvision) │       │
│  └──────┬───────┘      └──────┬───────┘       │
│         │                     │                │
│         │  Hardware Trigger   │                │
│         └──────────┬──────────┘                │
│                    ↓                            │
│         ┌──────────────────────┐               │
│         │  Stereo Sync Node    │               │
│         │  (时间戳对齐)         │               │
│         └──────────┬───────────┘               │
│                    ↓                            │
│         ┌──────────────────────┐               │
│         │  Detection Node      │               │
│         │  (YOLO + 几何拟合)    │               │
│         └──────────┬───────────┘               │
│                    ↓                            │
│         ┌──────────────────────┐               │
│         │  Stereo Matching     │               │
│         │  (三角测量)           │               │
│         └──────────┬───────────┘               │
│                    ↓                            │
│         ┌──────────────────────┐               │
│         │  Tracking Node       │               │
│         │  (卡尔曼滤波 3D)      │               │
│         └──────────┬───────────┘               │
│                    ↓                            │
│         ┌──────────────────────┐               │
│         │  Publisher           │               │
│         │  /volleyball/pose_3d │               │
│         └──────────────────────┘               │
└─────────────────────────────────────────────────┘
```

---

## 📦 ROS2 消息定义

### 自定义消息 `VolleyballPose3D.msg`

```msg
# VolleyballPose3D.msg
# 排球 3D 位置和速度

std_msgs/Header header

# 3D 位置 (相机坐标系, 单位: 米)
geometry_msgs/Point position
  float64 x  # 左右方向
  float64 y  # 上下方向
  float64 z  # 深度方向

# 3D 速度 (单位: m/s)
geometry_msgs/Vector3 velocity
  float64 x
  float64 y
  float64 z

# 圆形参数 (像素坐标系)
float32 radius_left   # 左图半径
float32 radius_right  # 右图半径

# 置信度
float32 detection_confidence  # 检测置信度
float32 stereo_confidence     # 立体匹配置信度

# 追踪信息
int32 track_id
int32 track_age
```

### 可视化消息 `VolleyballDebug.msg`

```msg
# VolleyballDebug.msg
# 调试信息

std_msgs/Header header

# 左右图像 (压缩)
sensor_msgs/CompressedImage left_image
sensor_msgs/CompressedImage right_image

# 关键点 (左图)
geometry_msgs/Point[] keypoints_left

# 关键点 (右图)
geometry_msgs/Point[] keypoints_right

# 视差
float32 disparity

# 处理耗时 (ms)
float32 detection_time
float32 stereo_time
float32 total_time
```

---

## 🔧 核心节点实现

### 1. Stereo Sync Node (同步节点)

**功能**: 
- 硬触发采集
- 时间戳对齐
- 图像缓存

**关键代码**:
```python
class StereoSyncNode(Node):
    def __init__(self):
        super().__init__('stereo_sync')
        
        # 初始化相机
        self.cam_left = HikCamera(0)
        self.cam_right = HikCamera(1)
        
        # 配置硬触发
        self.setup_hardware_trigger()
        
        # 发布同步图像对
        self.pub_stereo = self.create_publisher(
            StereoImage, 
            '/stereo/images', 
            10
        )
        
        # 定时器 (150 Hz)
        self.timer = self.create_timer(
            1.0/150.0, 
            self.capture_callback
        )
    
    def setup_hardware_trigger(self):
        """配置硬触发"""
        for cam in [self.cam_left, self.cam_right]:
            cam.set_trigger_mode('On')
            cam.set_trigger_source('Line0')
            cam.set_exposure_time(800)  # 800us
    
    def capture_callback(self):
        """采集回调"""
        # 发送触发信号
        self.send_trigger()
        
        # 同时采集
        img_left = self.cam_left.grab()
        img_right = self.cam_right.grab()
        
        # 检查时间戳差异
        ts_diff = abs(img_left.timestamp - img_right.timestamp)
        if ts_diff > 1e-3:  # >1ms
            self.get_logger().warn(f'时间戳不同步: {ts_diff*1000:.2f}ms')
        
        # 发布
        msg = StereoImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.left_image = self.to_ros_image(img_left)
        msg.right_image = self.to_ros_image(img_right)
        self.pub_stereo.publish(msg)
```

---

### 2. Detection Node (检测节点)

**功能**:
- 状态机 (全图/ROI)
- YOLO 推理
- 几何拟合

**状态机逻辑**:
```python
class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection')
        
        # TensorRT 引擎
        self.detector = TRTDetector('volleyball.engine')
        self.fitter = CircleFitter()
        
        # 状态机
        self.state = 'GLOBAL_SEARCH'  # or 'ROI_TRACKING'
        self.predicted_pos = None
        
        # 订阅同步图像
        self.sub_stereo = self.create_subscription(
            StereoImage,
            '/stereo/images',
            self.detect_callback,
            10
        )
        
        # 发布检测结果
        self.pub_detection = self.create_publisher(
            StereoDetection,
            '/volleyball/detection',
            10
        )
    
    def detect_callback(self, msg):
        """检测回调"""
        img_left = self.from_ros_image(msg.left_image)
        img_right = self.from_ros_image(msg.right_image)
        
        if self.state == 'GLOBAL_SEARCH':
            # 全图检测 (下采样到 640x640)
            det_left = self.detect_global(img_left)
            det_right = self.detect_global(img_right)
            
            if det_left and det_right:
                # 切换到 ROI 模式
                self.state = 'ROI_TRACKING'
                self.predicted_pos = (det_left.cx, det_left.cy)
        
        else:  # ROI_TRACKING
            # ROI 检测
            roi_left, offset_left = self.crop_roi(
                img_left, self.predicted_pos
            )
            roi_right, offset_right = self.crop_roi(
                img_right, self.predicted_pos
            )
            
            # Batch 推理
            det_left = self.detect_roi(roi_left, offset_left)
            det_right = self.detect_roi(roi_right, offset_right)
            
            if not det_left or not det_right:
                # 跟丢，切回全图
                self.state = 'GLOBAL_SEARCH'
                return
        
        # 发布检测结果
        self.publish_detection(det_left, det_right, msg.header)
    
    def crop_roi(self, img, center, size=320):
        """裁切 ROI"""
        cx, cy = int(center[0]), int(center[1])
        x1 = max(0, cx - size//2)
        y1 = max(0, cy - size//2)
        x2 = min(img.shape[1], x1 + size)
        y2 = min(img.shape[0], y1 + size)
        
        roi = img[y1:y2, x1:x2]
        offset = (x1, y1)
        return roi, offset
```

---

### 3. Stereo Matching Node (立体匹配节点)

**功能**:
- 关键点去畸变
- 三角测量
- 深度计算

**核心代码**:
```python
class StereoMatchingNode(Node):
    def __init__(self):
        super().__init__('stereo_matching')
        
        # 加载标定参数
        self.load_calibration('/path/to/stereo_calib.npz')
        
        # 订阅检测结果
        self.sub_detection = self.create_subscription(
            StereoDetection,
            '/volleyball/detection',
            self.match_callback,
            10
        )
        
        # 发布 3D 位置
        self.pub_pose3d = self.create_publisher(
            VolleyballPose3D,
            '/volleyball/pose_3d_raw',
            10
        )
    
    def load_calibration(self, path):
        """加载标定参数"""
        data = np.load(path)
        self.K1, self.D1 = data['K1'], data['D1']
        self.K2, self.D2 = data['K2'], data['D2']
        self.P1, self.P2 = data['P1'], data['P2']
        self.baseline = data['baseline']  # 25cm
    
    def match_callback(self, msg):
        """立体匹配回调"""
        # 提取关键点
        kpts_left = self.extract_keypoints(msg.left)
        kpts_right = self.extract_keypoints(msg.right)
        
        # 去畸变 (稀疏)
        kpts_left_undist = cv2.undistortPoints(
            kpts_left, self.K1, self.D1, P=self.P1
        )
        kpts_right_undist = cv2.undistortPoints(
            kpts_right, self.K2, self.D2, P=self.P2
        )
        
        # 三角测量 (只用 Center 点)
        center_left = kpts_left_undist[0]
        center_right = kpts_right_undist[0]
        
        point_4d = cv2.triangulatePoints(
            self.P1, self.P2,
            center_left.reshape(2, 1),
            center_right.reshape(2, 1)
        )
        
        # 归一化
        point_3d = point_4d[:3] / point_4d[3]
        
        # 计算置信度 (基于视差)
        disparity = kpts_left[0, 0] - kpts_right[0, 0]
        stereo_conf = self.compute_confidence(disparity, point_3d[2])
        
        # 发布
        pose_msg = VolleyballPose3D()
        pose_msg.header = msg.header
        pose_msg.position.x = float(point_3d[0])
        pose_msg.position.y = float(point_3d[1])
        pose_msg.position.z = float(point_3d[2])
        pose_msg.stereo_confidence = stereo_conf
        
        self.pub_pose3d.publish(pose_msg)
    
    def compute_confidence(self, disparity, depth):
        """计算立体匹配置信度"""
        # 基于视差和深度的置信度
        if disparity < 10:  # 视差太小
            return 0.1
        elif depth > 15.0:  # 超出范围
            return 0.3
        else:
            return min(1.0, disparity / 100.0)
```

---

### 4. Tracking Node (追踪节点)

**功能**:
- 3D 卡尔曼滤波
- 速度估计
- 轨迹平滑

**核心代码**:
```python
class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking')
        
        # 3D 卡尔曼滤波器
        self.kf = self.create_3d_kalman()
        self.is_initialized = False
        
        # 订阅原始 3D 位置
        self.sub_pose = self.create_subscription(
            VolleyballPose3D,
            '/volleyball/pose_3d_raw',
            self.track_callback,
            10
        )
        
        # 发布平滑后的 3D 位置
        self.pub_tracked = self.create_publisher(
            VolleyballPose3D,
            '/volleyball/pose_3d',
            10
        )
    
    def create_3d_kalman(self):
        """创建 3D 卡尔曼滤波器"""
        from filterpy.kalman import KalmanFilter
        
        # 状态: [x, y, z, vx, vy, vz, ax, ay, az]
        kf = KalmanFilter(dim_x=9, dim_z=3)
        
        dt = 1.0 / 150.0  # 150 Hz
        
        # 状态转移矩阵
        kf.F = np.array([
            [1, 0, 0, dt, 0,  0,  0.5*dt**2, 0, 0],
            [0, 1, 0, 0,  dt, 0,  0, 0.5*dt**2, 0],
            [0, 0, 1, 0,  0,  dt, 0, 0, 0.5*dt**2],
            [0, 0, 0, 1,  0,  0,  dt, 0, 0],
            [0, 0, 0, 0,  1,  0,  0, dt, 0],
            [0, 0, 0, 0,  0,  1,  0, 0, dt],
            [0, 0, 0, 0,  0,  0,  1, 0, 0],
            [0, 0, 0, 0,  0,  0,  0, 1, 0],
            [0, 0, 0, 0,  0,  0,  0, 0, 1],
        ])
        
        # 观测矩阵
        kf.H = np.eye(3, 9)
        
        # 过程噪声
        kf.Q *= 0.01
        
        # 观测噪声 (动态调整)
        kf.R = np.diag([0.01, 0.01, 0.05])
        
        return kf
    
    def track_callback(self, msg):
        """追踪回调"""
        z = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z
        ])
        
        if not self.is_initialized:
            # 初始化
            self.kf.x[:3] = z
            self.is_initialized = True
            return
        
        # 预测
        self.kf.predict()
        
        # 动态调整观测噪声
        depth = z[2]
        if depth < 5.0:
            self.kf.R = np.diag([0.01, 0.01, 0.01])
        elif depth > 12.0:
            self.kf.R = np.diag([0.1, 0.1, 0.5])
        else:
            self.kf.R = np.diag([0.05, 0.05, 0.2])
        
        # 更新
        self.kf.update(z)
        
        # 发布
        tracked_msg = VolleyballPose3D()
        tracked_msg.header = msg.header
        tracked_msg.position.x = self.kf.x[0]
        tracked_msg.position.y = self.kf.x[1]
        tracked_msg.position.z = self.kf.x[2]
        tracked_msg.velocity.x = self.kf.x[3]
        tracked_msg.velocity.y = self.kf.x[4]
        tracked_msg.velocity.z = self.kf.x[5]
        
        self.pub_tracked.publish(tracked_msg)
```

---

## 📊 话题列表

| 话题名称 | 消息类型 | 频率 | 说明 |
|---------|---------|------|------|
| `/stereo/images` | `StereoImage` | 150 Hz | 同步双目图像 |
| `/volleyball/detection` | `StereoDetection` | 150 Hz | 双目检测结果 |
| `/volleyball/pose_3d_raw` | `VolleyballPose3D` | 150 Hz | 原始 3D 位置 |
| `/volleyball/pose_3d` | `VolleyballPose3D` | 150 Hz | 平滑后 3D 位置 |
| `/volleyball/debug` | `VolleyballDebug` | 30 Hz | 调试信息 |

---

## 🚀 下一步行动

1. **相机标定**: 采集棋盘格，运行 `stereoCalibrate`
2. **消息定义**: 创建 ROS2 消息包
3. **节点实现**: 按上述架构实现各节点
4. **集成测试**: 验证端到端延迟和精度

**需要我继续实现完整的 ROS2 代码吗？**
