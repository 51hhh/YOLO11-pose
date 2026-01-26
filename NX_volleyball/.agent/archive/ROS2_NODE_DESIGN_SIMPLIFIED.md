# 🏐 双目排球追踪 ROS2 节点设计方案 (精简版)

## 📅 更新时间
**最后更新**: 2026-01-24

---

## 🎯 设计原则

- **精简架构**: 仅 2-3 个核心节点
- **C++ 实现**: 高性能，遵循现有代码风格
- **单节点集成**: 相机驱动 + YOLO 检测 + 立体匹配 + 追踪 合并为一个节点
- **Colcon 构建**: 不使用 Python 和 Launch 文件
- **独立可视化**: 可视化功能单独一个包

---

## 🏗️ 精简架构

```
┌──────────────────────────────────────────────────────────────┐
│                    ROS2 节点拓扑图 (精简版)                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [PWM Trigger Node]  →  /camera_trigger                     │
│                              ↓                               │
│                                                              │
│  [Volleyball Tracker Node]  ← 监听触发                       │
│   ├─ 双目相机采集                                            │
│   ├─ YOLO11n 检测                                            │
│   ├─ 立体匹配                                                │
│   ├─ 卡尔曼滤波                                              │
│   └─ 发布 3D 位置                                            │
│                              ↓                               │
│                    /volleyball/pose_3d                       │
│                              ↓                               │
│                                                              │
│  [Visualizer Node] (可选)                                    │
│   └─ RViz2 可视化 + 调试图像                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 📦 ROS2 包结构

```
ros2_ws/src/
├── volleyball_stereo_driver/     # ✅ 已有包 (保留 PWM 节点)
│   ├── src/
│   │   ├── pwm_trigger_node.cpp          # ✅ PWM 触发节点
│   │   └── high_precision_pwm.cpp        # ✅ PWM 实现
│   └── ...
│
├── volleyball_tracker/           # 🆕 核心追踪包 (All-in-One)
│   ├── include/volleyball_tracker/
│   │   ├── hik_camera_wrapper.hpp        # 海康相机封装
│   │   ├── yolo_detector.hpp             # YOLO TensorRT 推理
│   │   ├── stereo_matcher.hpp            # 立体匹配
│   │   ├── kalman_filter_3d.hpp          # 3D 卡尔曼滤波
│   │   └── roi_manager.hpp               # ROI 管理
│   │
│   ├── src/
│   │   ├── volleyball_tracker_node.cpp   # 🆕 主节点 (集成所有功能)
│   │   ├── hik_camera_wrapper.cpp
│   │   ├── yolo_detector.cpp
│   │   ├── stereo_matcher.cpp
│   │   ├── kalman_filter_3d.cpp
│   │   └── roi_manager.cpp
│   │
│   ├── config/
│   │   └── tracker_params.yaml           # 统一配置文件
│   │
│   ├── model/
│   │   └── yolo11n.engine                # YOLO 模型
│   │
│   ├── calibration/
│   │   └── stereo_calib.npz              # 标定文件
│   │
│   ├── CMakeLists.txt
│   └── package.xml
│
└── volleyball_viz/               # 🆕 可视化包 (可选)
    ├── src/
    │   └── visualizer_node.cpp           # C++ 可视化节点
    ├── rviz/
    │   └── volleyball.rviz
    ├── CMakeLists.txt
    └── package.xml
```

---

## 🔧 节点详细设计

### 节点 1: pwm_trigger_node ✅
**包**: `volleyball_stereo_driver`  
**状态**: 已实现，保持不变  
**语言**: C++

#### 功能
- 生成 100Hz PWM 信号
- 发布触发时间戳

#### 发布话题
- `/camera_trigger` (std_msgs/Header)

---

### 节点 2: volleyball_tracker_node 🆕 (核心节点)
**包**: `volleyball_tracker`  
**状态**: 待实现  
**语言**: C++

#### 功能 (All-in-One)
1. **相机采集**: 监听触发信号，同步采集左右图像
2. **YOLO 检测**: TensorRT 推理，状态机切换 (全图/ROI)
3. **立体匹配**: 稀疏去畸变 + 三角测量
4. **3D 追踪**: 卡尔曼滤波，速度估计
5. **结果发布**: 发布 3D 位置和速度

#### 订阅话题
| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/camera_trigger` | `std_msgs/Header` | 100Hz | PWM 触发时间戳 |

#### 发布话题
| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/volleyball/pose_3d` | `geometry_msgs/PoseStamped` | 100Hz | 3D 位置 |
| `/volleyball/velocity` | `geometry_msgs/Vector3Stamped` | 100Hz | 3D 速度 |
| `/volleyball/debug_info` | `std_msgs/String` | 10Hz | 调试信息 (JSON) |

#### 参数配置 (tracker_params.yaml)
```yaml
volleyball_tracker:
  # 相机参数
  camera:
    left_index: 0
    right_index: 1
    exposure_time: 9867.0
    gain: 10.9854
    trigger_mode: true
    trigger_source: "Line0"
    trigger_activation: "RisingEdge"
  
  # YOLO 检测参数
  detector:
    model_path: "model/yolo11n.engine"
    confidence_threshold: 0.5
    nms_threshold: 0.4
    roi_size: 320
    global_size: 640
    enable_roi_mode: true
    max_lost_frames: 10
  
  # 立体匹配参数
  stereo:
    calibration_file: "calibration/stereo_calib.npz"
    min_disparity: 10.0
    max_depth: 15.0
    min_confidence: 0.3
  
  # 追踪参数
  tracker:
    process_noise: 0.01
    measurement_noise_near: [0.01, 0.01, 0.01]  # < 5m
    measurement_noise_mid: [0.05, 0.05, 0.2]    # 5-12m
    measurement_noise_far: [0.1, 0.1, 0.5]      # > 12m
    max_lost_frames: 30
```

#### 核心类设计

```cpp
class VolleyballTrackerNode : public rclcpp::Node {
public:
    VolleyballTrackerNode();
    ~VolleyballTrackerNode();

private:
    // 回调函数
    void triggerCallback(const std_msgs::msg::Header::SharedPtr msg);
    void processFrame();
    
    // 处理流程
    void captureImages();
    bool detectVolleyball();
    bool computeStereoMatch();
    void updateTracker();
    void publishResults();
    
    // 组件
    std::unique_ptr<HikCameraWrapper> cam_left_;
    std::unique_ptr<HikCameraWrapper> cam_right_;
    std::unique_ptr<YOLODetector> detector_;
    std::unique_ptr<StereoMatcher> stereo_matcher_;
    std::unique_ptr<KalmanFilter3D> tracker_;
    std::unique_ptr<ROIManager> roi_manager_;
    
    // ROS 接口
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr sub_trigger_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pub_velocity_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_debug_;
    
    // 状态
    enum State { GLOBAL_SEARCH, ROI_TRACKING };
    State state_;
    cv::Mat img_left_, img_right_;
    rclcpp::Time last_trigger_stamp_;
    
    // 统计
    size_t frame_count_;
    std::chrono::high_resolution_clock::time_point last_stat_time_;
};
```

---

### 节点 3: visualizer_node 🆕 (可选)
**包**: `volleyball_viz`  
**状态**: 待实现  
**语言**: C++

#### 功能
- 订阅 3D 位置
- 发布 RViz2 Marker (轨迹、速度向量)
- 可选：叠加检测框到图像

#### 订阅话题
| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/volleyball/pose_3d` | `geometry_msgs/PoseStamped` | 100Hz | 3D 位置 |
| `/volleyball/velocity` | `geometry_msgs/Vector3Stamped` | 100Hz | 3D 速度 |

#### 发布话题
| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/volleyball/trajectory` | `visualization_msgs/MarkerArray` | 30Hz | 3D 轨迹 |
| `/volleyball/velocity_arrow` | `visualization_msgs/Marker` | 30Hz | 速度向量 |

---

## 🔧 核心组件实现

### 1. HikCameraWrapper (复用现有代码)
```cpp
class HikCameraWrapper {
public:
    HikCameraWrapper(int camera_index);
    ~HikCameraWrapper();
    
    bool open();
    void close();
    bool startGrabbing();
    void stopGrabbing();
    
    void setTriggerMode(bool enable);
    void setTriggerSource(const std::string& source);
    void setTriggerActivation(const std::string& activation);
    void setExposureTime(double time_us);
    void setGain(double gain);
    
    cv::Mat grabImage(int timeout_ms);
    
private:
    void* camera_handle_;
    int camera_index_;
};
```

### 2. YOLODetector (TensorRT)
```cpp
struct Detection {
    float cx, cy;        // 中心点
    float width, height; // 宽高
    float confidence;    // 置信度
    bool valid;
};

class YOLODetector {
public:
    YOLODetector(const std::string& engine_path);
    ~YOLODetector();
    
    // 全图检测 (640x640)
    Detection detectGlobal(const cv::Mat& image);
    
    // ROI 检测 (320x320)
    Detection detectROI(const cv::Mat& roi, const cv::Point2f& offset);
    
private:
    void* engine_;
    void* context_;
    void preprocess(const cv::Mat& image, float* input_buffer);
    Detection postprocess(float* output_buffer);
};
```

### 3. StereoMatcher
```cpp
struct StereoPoint {
    cv::Point3f position_3d;  // 3D 坐标
    float confidence;         // 置信度
    float disparity;          // 视差
    bool valid;
};

class StereoMatcher {
public:
    StereoMatcher(const std::string& calib_file);
    
    StereoPoint triangulate(
        const cv::Point2f& pt_left,
        const cv::Point2f& pt_right
    );
    
private:
    cv::Mat K1_, D1_, P1_;  // 左相机参数
    cv::Mat K2_, D2_, P2_;  // 右相机参数
    float baseline_;
    
    cv::Point2f undistortPoint(const cv::Point2f& pt, 
                                const cv::Mat& K, 
                                const cv::Mat& D, 
                                const cv::Mat& P);
};
```

### 4. KalmanFilter3D
```cpp
class KalmanFilter3D {
public:
    KalmanFilter3D();
    
    void init(const cv::Point3f& initial_position);
    void predict();
    void update(const cv::Point3f& measurement, float depth);
    
    cv::Point3f getPosition() const;
    cv::Point3f getVelocity() const;
    cv::Point3f getAcceleration() const;
    
private:
    cv::Mat state_;       // [x, y, z, vx, vy, vz, ax, ay, az]
    cv::Mat covariance_;  // P
    cv::Mat F_;           // 状态转移矩阵
    cv::Mat H_;           // 观测矩阵
    cv::Mat Q_;           // 过程噪声
    cv::Mat R_;           // 观测噪声 (动态调整)
    
    void updateMeasurementNoise(float depth);
};
```

### 5. ROIManager
```cpp
class ROIManager {
public:
    ROIManager(int roi_size = 320);
    
    // 基于预测位置裁切 ROI
    cv::Mat cropROI(const cv::Mat& image, 
                    const cv::Point2f& predicted_center,
                    cv::Point2f& offset);
    
    // 坐标还原
    cv::Point2f mapToOriginal(const cv::Point2f& roi_point, 
                              const cv::Point2f& offset);
    
private:
    int roi_size_;
};
```

---

## 🚀 实现流程

### volleyball_tracker_node 主循环

```cpp
void VolleyballTrackerNode::triggerCallback(
    const std_msgs::msg::Header::SharedPtr msg
) {
    last_trigger_stamp_ = msg->stamp;
    processFrame();
}

void VolleyballTrackerNode::processFrame() {
    // 1. 采集图像
    captureImages();
    
    // 2. YOLO 检测
    if (!detectVolleyball()) {
        // 检测失败，切换到全图搜索
        state_ = GLOBAL_SEARCH;
        return;
    }
    
    // 3. 立体匹配
    if (!computeStereoMatch()) {
        return;
    }
    
    // 4. 更新追踪器
    updateTracker();
    
    // 5. 发布结果
    publishResults();
    
    // 6. 统计
    frame_count_++;
    if (frame_count_ % 100 == 0) {
        printStatistics();
    }
}

void VolleyballTrackerNode::captureImages() {
    img_left_ = cam_left_->grabImage(100);
    img_right_ = cam_right_->grabImage(100);
}

bool VolleyballTrackerNode::detectVolleyball() {
    Detection det_left, det_right;
    
    if (state_ == GLOBAL_SEARCH) {
        // 全图检测
        det_left = detector_->detectGlobal(img_left_);
        det_right = detector_->detectGlobal(img_right_);
        
        if (det_left.valid && det_right.valid) {
            state_ = ROI_TRACKING;
            // 保存检测结果
            return true;
        }
        return false;
        
    } else {  // ROI_TRACKING
        // 基于预测位置裁切 ROI
        cv::Point2f predicted_pos = tracker_->getPosition();
        cv::Point2f offset_left, offset_right;
        
        cv::Mat roi_left = roi_manager_->cropROI(
            img_left_, predicted_pos, offset_left
        );
        cv::Mat roi_right = roi_manager_->cropROI(
            img_right_, predicted_pos, offset_right
        );
        
        // ROI 检测
        det_left = detector_->detectROI(roi_left, offset_left);
        det_right = detector_->detectROI(roi_right, offset_right);
        
        if (!det_left.valid || !det_right.valid) {
            // 跟丢，切换到全图
            state_ = GLOBAL_SEARCH;
            return false;
        }
        
        return true;
    }
}

bool VolleyballTrackerNode::computeStereoMatch() {
    // 使用检测到的中心点进行三角测量
    cv::Point2f pt_left(det_left_.cx, det_left_.cy);
    cv::Point2f pt_right(det_right_.cx, det_right_.cy);
    
    StereoPoint stereo_pt = stereo_matcher_->triangulate(pt_left, pt_right);
    
    if (!stereo_pt.valid) {
        return false;
    }
    
    // 保存 3D 点
    current_3d_point_ = stereo_pt.position_3d;
    current_depth_ = stereo_pt.position_3d.z;
    
    return true;
}

void VolleyballTrackerNode::updateTracker() {
    if (!tracker_initialized_) {
        tracker_->init(current_3d_point_);
        tracker_initialized_ = true;
    } else {
        tracker_->predict();
        tracker_->update(current_3d_point_, current_depth_);
    }
}

void VolleyballTrackerNode::publishResults() {
    // 发布 3D 位置
    auto pose_msg = geometry_msgs::msg::PoseStamped();
    pose_msg.header.stamp = last_trigger_stamp_;
    pose_msg.header.frame_id = "camera_frame";
    
    cv::Point3f pos = tracker_->getPosition();
    pose_msg.pose.position.x = pos.x;
    pose_msg.pose.position.y = pos.y;
    pose_msg.pose.position.z = pos.z;
    
    pub_pose_->publish(pose_msg);
    
    // 发布速度
    auto vel_msg = geometry_msgs::msg::Vector3Stamped();
    vel_msg.header = pose_msg.header;
    
    cv::Point3f vel = tracker_->getVelocity();
    vel_msg.vector.x = vel.x;
    vel_msg.vector.y = vel.y;
    vel_msg.vector.z = vel.z;
    
    pub_velocity_->publish(vel_msg);
}
```

---

## 📋 编译配置

### CMakeLists.txt (volleyball_tracker)

```cmake
cmake_minimum_required(VERSION 3.8)
project(volleyball_tracker)

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# 依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS /usr/lib/aarch64-linux-gnu)

# 海康 SDK
set(HIK_SDK_DIR "/opt/MVS")
include_directories(${HIK_SDK_DIR}/include)
link_directories(${HIK_SDK_DIR}/lib/aarch64)

# 头文件
include_directories(include)

# 源文件
add_executable(volleyball_tracker_node
  src/volleyball_tracker_node.cpp
  src/hik_camera_wrapper.cpp
  src/yolo_detector.cpp
  src/stereo_matcher.cpp
  src/kalman_filter_3d.cpp
  src/roi_manager.cpp
)

ament_target_dependencies(volleyball_tracker_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
)

target_link_libraries(volleyball_tracker_node
  ${OpenCV_LIBS}
  ${TENSORRT_LIBRARY}
  MvCameraControl
)

# 安装
install(TARGETS volleyball_tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config model calibration
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
```

---

## 🎯 使用方式

### 编译
```bash
cd ~/desktop/yolo/yoloProject/NX_volleyball/ros2_ws
colcon build --packages-select volleyball_tracker
source install/setup.bash
```

### 运行
```bash
# 终端 1: 启动 PWM 触发
ros2 run volleyball_stereo_driver pwm_trigger_node \
  --ros-args --params-file src/volleyball_stereo_driver/config/pwm_params.yaml

# 终端 2: 启动追踪节点
ros2 run volleyball_tracker volleyball_tracker_node \
  --ros-args --params-file src/volleyball_tracker/config/tracker_params.yaml

# 终端 3 (可选): 启动可视化
rviz2 -d src/volleyball_viz/rviz/volleyball.rviz
```

---

## 📊 优势

1. **架构简单**: 只有 2 个核心节点，易于维护
2. **性能优异**: 单节点内部处理，无 ROS 通信开销
3. **代码统一**: 全部 C++，风格一致
4. **配置集中**: 一个 YAML 文件管理所有参数
5. **调试方便**: 所有逻辑在一个进程内

---

## 📝 下一步行动

1. **创建 volleyball_tracker 包**
2. **实现 HikCameraWrapper** (复用现有代码)
3. **实现 YOLODetector** (TensorRT)
4. **实现 StereoMatcher**
5. **实现 KalmanFilter3D**
6. **集成到 volleyball_tracker_node**
7. **测试和优化**

---

**文档版本**: v2.0 (精简版)  
**最后更新**: 2026-01-24
