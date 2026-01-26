/**
 * @file volleyball_tracker_node.cpp
 * @brief 排球追踪主节点 (All-in-One)
 *
 * 集成功能:
 * - PWM 触发 (内置，不依赖外部触发话题)
 * - 双目相机同步采集
 * - YOLO 检测 (全图/ROI 状态机)
 * - 立体匹配 (三角测量)
 * - 3D 卡尔曼滤波
 * - 发布 3D 位置、速度和图像
 */

#include "volleyball_stereo_driver/volleyball_tracker_node.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <rclcpp/qos.hpp>

namespace volleyball {

// ==================== 构造函数 ====================
VolleyballTrackerNode::VolleyballTrackerNode(const rclcpp::NodeOptions & options)
    : Node("volleyball_tracker", options),
      state_(GLOBAL_SEARCH),
      lost_frames_(0),
      max_lost_frames_(10),
      running_(false),
      current_depth_(0.0f),
      frame_count_(0),
      total_detection_time_(0.0),
      total_stereo_time_(0.0),
      total_tracking_time_(0.0),
      total_capture_time_(0.0),
      enable_debug_(true),
      log_interval_(100),
      publish_images_(true),
      publish_detection_image_(true) {
    
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "🏐 排球追踪节点 (All-in-One)");
    RCLCPP_INFO(this->get_logger(), "========================================");
    
    // 1. 声明和加载参数
    declareParameters();
    loadParameters();

    // 2. 初始化各组件
    bool init_success = true;
    
    if (!initializePWM()) {
        RCLCPP_ERROR(this->get_logger(), "❌ PWM 初始化失败");
        init_success = false;
    }
    
    if (init_success && !initializeCameras()) {
        RCLCPP_ERROR(this->get_logger(), "❌ 相机初始化失败");
        init_success = false;
    }
    
    if (init_success && !initializeDetector()) {
        RCLCPP_WARN(this->get_logger(), "⚠️  YOLO 检测器初始化失败，使用占位符模式");
    }
    
    if (init_success && !initializeStereoMatcher()) {
        RCLCPP_WARN(this->get_logger(), "⚠️  立体匹配器初始化失败");
    }
    
    initializeKalmanFilter();
    initializeROIManager();

    // 3. 创建 ROS 发布器
    createPublishers();

    if (!init_success) {
        RCLCPP_ERROR(this->get_logger(), "❌ 初始化失败，节点无法启动");
        return;
    }

    // 4. 启动推理线程 (轮询模式，参考RC_Volleyball_vision)
    running_ = true;
    inference_thread_ = std::thread(&VolleyballTrackerNode::inferenceLoop, this);

    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "✅ 排球追踪节点已启动 (PWM时间戳同步)");
    RCLCPP_INFO(this->get_logger(), "   PWM: %.1f Hz", pwm_frequency_);
    RCLCPP_INFO(this->get_logger(), "   同步策略: 帧号差≤3 且 时间差<25ms");
    RCLCPP_INFO(this->get_logger(), "========================================");
}

// ==================== 析构函数 ====================
VolleyballTrackerNode::~VolleyballTrackerNode() {
    running_ = false;
    
    // 等待推理线程结束
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
    
    if (pwm_) {
        pwm_->stop();
    }
    
    if (cam_left_) {
        cam_left_->stopGrabbing();
        cam_left_->close();
    }
    
    if (cam_right_) {
        cam_right_->stopGrabbing();
        cam_right_->close();
    }
    
    RCLCPP_INFO(this->get_logger(), "排球追踪节点已关闭");
}

// ==================== 参数声明 ====================
void VolleyballTrackerNode::declareParameters() {
    // PWM 参数
    this->declare_parameter("pwm.gpio_chip", "gpiochip2");
    this->declare_parameter("pwm.gpio_line", 7);
    this->declare_parameter("pwm.frequency", 100.0);
    this->declare_parameter("pwm.duty_cycle", 50.0);
    
    // 相机参数
    this->declare_parameter("camera.left_index", 0);
    this->declare_parameter("camera.right_index", 1);
    this->declare_parameter("camera.exposure_time", 9867.0);
    this->declare_parameter("camera.gain", 10.9854);
    
    // 检测参数
    this->declare_parameter("detector.model_path", "model/yolo11n_batch2.engine");
    this->declare_parameter("detector.input_size", 0);  // 0=自动检测
    this->declare_parameter("detector.confidence_threshold", 0.5);
    this->declare_parameter("detector.nms_threshold", 0.4);
    this->declare_parameter("detector.roi_size", 320);
    this->declare_parameter("detector.global_size", 640);
    this->declare_parameter("detector.max_lost_frames", 10);
    
    // 立体视觉参数
    this->declare_parameter("stereo.calibration_file", "calibration/stereo_calib.yaml");
    this->declare_parameter("stereo.min_disparity", 10.0);
    this->declare_parameter("stereo.max_depth", 15.0);
    
    // 追踪参数
    this->declare_parameter("tracker.process_noise", 0.01);
    this->declare_parameter("tracker.measurement_noise", 0.01);
    
    // 调试参数
    this->declare_parameter("debug.enable_logging", true);
    this->declare_parameter("debug.log_interval", 100);
    this->declare_parameter("debug.publish_images", true);
    this->declare_parameter("debug.publish_detection_image", true);
}

// ==================== 加载参数 ====================
void VolleyballTrackerNode::loadParameters() {
    // PWM 参数
    gpio_chip_ = this->get_parameter("pwm.gpio_chip").as_string();
    gpio_line_ = this->get_parameter("pwm.gpio_line").as_int();
    pwm_frequency_ = this->get_parameter("pwm.frequency").as_double();
    pwm_duty_cycle_ = this->get_parameter("pwm.duty_cycle").as_double();
    
    // 相机参数
    left_camera_index_ = this->get_parameter("camera.left_index").as_int();
    right_camera_index_ = this->get_parameter("camera.right_index").as_int();
    exposure_time_ = this->get_parameter("camera.exposure_time").as_double();
    gain_ = this->get_parameter("camera.gain").as_double();
    
    // 检测参数
    model_path_ = this->get_parameter("detector.model_path").as_string();
    input_size_ = this->get_parameter("detector.input_size").as_int();
    conf_threshold_ = this->get_parameter("detector.confidence_threshold").as_double();
    nms_threshold_ = this->get_parameter("detector.nms_threshold").as_double();
    roi_size_ = this->get_parameter("detector.roi_size").as_int();
    global_size_ = this->get_parameter("detector.global_size").as_int();
    max_lost_frames_ = this->get_parameter("detector.max_lost_frames").as_int();
    
    // 立体视觉参数
    calibration_file_ = this->get_parameter("stereo.calibration_file").as_string();
    min_disparity_ = this->get_parameter("stereo.min_disparity").as_double();
    max_depth_ = this->get_parameter("stereo.max_depth").as_double();
    
    // 追踪参数
    process_noise_ = this->get_parameter("tracker.process_noise").as_double();
    measurement_noise_ = this->get_parameter("tracker.measurement_noise").as_double();
    
    // 调试参数
    enable_debug_ = this->get_parameter("debug.enable_logging").as_bool();
    log_interval_ = this->get_parameter("debug.log_interval").as_int();
    publish_images_ = this->get_parameter("debug.publish_images").as_bool();
    publish_detection_image_ = this->get_parameter("debug.publish_detection_image").as_bool();
    
    RCLCPP_INFO(this->get_logger(), "参数已加载:");
    RCLCPP_INFO(this->get_logger(), "  PWM: %s line %d, %.1f Hz", 
                gpio_chip_.c_str(), gpio_line_, pwm_frequency_);
    RCLCPP_INFO(this->get_logger(), "  相机: L=%d R=%d, 曝光=%.1f us", 
                left_camera_index_, right_camera_index_, exposure_time_);
}

// ==================== 初始化 PWM ====================
bool VolleyballTrackerNode::initializePWM() {
    RCLCPP_INFO(this->get_logger(), "🔧 初始化 PWM...");
    
    pwm_ = std::make_unique<HighPrecisionPWM>(
        gpio_chip_,
        static_cast<unsigned int>(gpio_line_),
        pwm_frequency_,
        pwm_duty_cycle_
    );
    
    if (!pwm_->start()) {
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "✅ PWM 已启动: %.1f Hz, %.1f%%", 
                pwm_frequency_, pwm_duty_cycle_);
    return true;
}

// ==================== 初始化相机 ====================
bool VolleyballTrackerNode::initializeCameras() {
    RCLCPP_INFO(this->get_logger(), "📷 初始化相机...");
    
    cam_left_ = std::make_unique<HikCamera>(left_camera_index_);
    cam_right_ = std::make_unique<HikCamera>(right_camera_index_);
    
    if (!cam_left_->open() || !cam_right_->open()) {
        return false;
    }
    
    // 配置相机
    cam_left_->setTriggerMode(true);
    cam_left_->setTriggerSource("Line0");
    cam_left_->setTriggerActivation("RisingEdge");
    cam_left_->setExposureTime(exposure_time_);
    cam_left_->setGain(gain_);
    
    cam_right_->setTriggerMode(true);
    cam_right_->setTriggerSource("Line0");
    cam_right_->setTriggerActivation("RisingEdge");
    cam_right_->setExposureTime(exposure_time_);
    cam_right_->setGain(gain_);
    
    if (!cam_left_->startGrabbing() || !cam_right_->startGrabbing()) {
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "✅ 相机已启动");
    return true;
}

// ==================== 初始化检测器 ====================
bool VolleyballTrackerNode::initializeDetector() {
    RCLCPP_INFO(this->get_logger(), "🎯 初始化 YOLO 检测器...");
    RCLCPP_INFO(this->get_logger(), "   配置的模型路径: %s", model_path_.c_str());
    
    std::string resolved_path = findFilePath(model_path_);
    
    if (resolved_path.empty()) {
        RCLCPP_ERROR(this->get_logger(), "❌ 未找到模型文件: %s", model_path_.c_str());
        RCLCPP_ERROR(this->get_logger(), "   请确认文件存在于 install/share/volleyball_stereo_driver/%s", model_path_.c_str());
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "   模型路径: %s", resolved_path.c_str());
    if (input_size_ > 0) {
        RCLCPP_INFO(this->get_logger(), "   指定输入尺寸: %dx%d", input_size_, input_size_);
    }
    
    detector_ = std::make_unique<YOLODetector>(
        resolved_path, conf_threshold_, nms_threshold_, input_size_);
    
    // 打印实际使用的模式
    if (detector_->isBatch2Model()) {
        RCLCPP_INFO(this->get_logger(), "🚀 使用 Batch=2 批量推理模式");
    } else {
        RCLCPP_INFO(this->get_logger(), "⚡ 使用双流并行推理模式 (batch=1)");
    }
    
    return true;
}

// ==================== 初始化立体匹配器 ====================
bool VolleyballTrackerNode::initializeStereoMatcher() {
    RCLCPP_INFO(this->get_logger(), "📐 初始化立体匹配器...");
    
    std::string resolved_path = findFilePath(calibration_file_);
    if (resolved_path.empty()) {
        RCLCPP_WARN(this->get_logger(), "⚠️  未找到标定文件: %s", calibration_file_.c_str());
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "   标定文件: %s", resolved_path.c_str());
    
    stereo_matcher_ = std::make_unique<StereoMatcher>(
        resolved_path, min_disparity_, max_depth_);
    
    return true;
}

// ==================== 初始化卡尔曼滤波器 ====================
void VolleyballTrackerNode::initializeKalmanFilter() {
    RCLCPP_INFO(this->get_logger(), "🎲 初始化卡尔曼滤波器...");
    tracker_ = std::make_unique<KalmanFilter3D>(process_noise_, measurement_noise_);
}

// ==================== 初始化 ROI 管理器 ====================
void VolleyballTrackerNode::initializeROIManager() {
    RCLCPP_INFO(this->get_logger(), "✂️  初始化 ROI 管理器...");
    roi_manager_ = std::make_unique<ROIManager>(roi_size_);
}

// ==================== 创建发布器 ====================
void VolleyballTrackerNode::createPublishers() {
    // ✅ 使用 SensorDataQoS: BEST_EFFORT + KEEP_LAST(5)
    // 特点: 无重传、低延迟，适合实时传感器数据
    rclcpp::SensorDataQoS sensor_qos;
    
    // 追踪结果 (使用低延迟 QoS)
    pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "volleyball/pose_3d", sensor_qos);
    pub_velocity_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
        "volleyball/velocity", sensor_qos);
    pub_trigger_ = this->create_publisher<std_msgs::msg::Header>(
        "camera_trigger", sensor_qos);
    
    // 调试信息 (可靠传输)
    pub_debug_ = this->create_publisher<std_msgs::msg::String>(
        "volleyball/debug_info", 10);
    
    // 图像话题 (使用低延迟 QoS，避免图像积压)
    pub_left_image_ = this->create_publisher<sensor_msgs::msg::Image>(
        "stereo/left/image_raw", sensor_qos);
    pub_right_image_ = this->create_publisher<sensor_msgs::msg::Image>(
        "stereo/right/image_raw", sensor_qos);
    pub_detection_image_ = this->create_publisher<sensor_msgs::msg::Image>(
        "volleyball/detection_image", sensor_qos);
    
    RCLCPP_INFO(this->get_logger(), "📡 发布器已创建 (SensorDataQoS: 低延迟模式)");
}

// ==================== 帧更新方法 (轮询相机回调，参考RC项目) ====================
void VolleyballTrackerNode::updateLeftFrame() {
    // 轮询左相机回调是否有新帧 (非阻塞)
    if (!cam_left_->waitForNewFrame(FRAME_WAIT_TIMEOUT_MS)) {
        return;  // 无新帧，直接返回
    }
    
    std::lock_guard<std::mutex> lock(frame_sync_mutex_);
    
    // 如果上一帧还未被处理，丢弃 (推理太慢)
    if (left_frame_.ready.load()) {
        left_dropped_++;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), LOG_THROTTLE_MS,
                            "⚠️  Left frame dropped (inference too slow)");
    }
    
    // 获取新帧 + 元数据
    left_frame_.image = cam_left_->getLatestImage().clone();
    left_frame_.metadata = cam_left_->getFrameMetadata();
    left_frame_.ready.store(true);
}

void VolleyballTrackerNode::updateRightFrame() {
    // 轮询右相机回调是否有新帧 (非阻塞)
    if (!cam_right_->waitForNewFrame(FRAME_WAIT_TIMEOUT_MS)) {
        return;  // 无新帧，直接返回
    }
    
    std::lock_guard<std::mutex> lock(frame_sync_mutex_);
    
    // 如果上一帧还未被处理，丢弃
    if (right_frame_.ready.load()) {
        right_dropped_++;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), LOG_THROTTLE_MS,
                            "⚠️  Right frame dropped (inference too slow)");
    }
    
    // 获取新帧 + 元数据
    right_frame_.image = cam_right_->getLatestImage().clone();
    right_frame_.metadata = cam_right_->getFrameMetadata();
    right_frame_.ready.store(true);
}

// ==================== PWM时间戳同步核心逻辑 ====================
bool VolleyballTrackerNode::waitForSyncedPair(
    cv::Mat& left, cv::Mat& right,
    FrameMetadata& left_meta, FrameMetadata& right_meta)
{
    std::lock_guard<std::mutex> lock(frame_sync_mutex_);
    
    // 检查两帧是否都ready
    if (!left_frame_.ready.load() || !right_frame_.ready.load()) {
        return false;
    }
    
    // ========== PWM时间戳同步：帧号差异 + 接收时间双重检验 ==========
    auto left_recv = left_frame_.metadata.receive_time;
    auto right_recv = right_frame_.metadata.receive_time;
    uint32_t left_num = left_frame_.metadata.frame_number;
    uint32_t right_num = right_frame_.metadata.frame_number;
    
    // 1. 检查帧号差异（PWM同步触发，帧号应该相同或相差1）
    int frame_diff = std::abs(static_cast<int>(left_num) - static_cast<int>(right_num));
    
    // 2. 计算接收时间差异 (微秒)
    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
        left_recv > right_recv ? (left_recv - right_recv) : (right_recv - left_recv)
    ).count();
    
    // 同步条件：
    // - 帧号差异 <= SYNC_MAX_FRAME_DIFF（容忍USB传输顺序固定导致的系统性延迟）
    // - 接收时间差异 < SYNC_MAX_TIME_DIFF_US（容忍USB传输延迟，PWM周期10ms × 2.5倍）
    
    if (frame_diff > SYNC_MAX_FRAME_DIFF || time_diff > SYNC_MAX_TIME_DIFF_US) {
        sync_mismatch_count_++;
        
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), LOG_THROTTLE_MS,
            "⚠️  同步失败: L#%u vs R#%u | ΔFrame=%d Δ=%ldμs",
            left_num, right_num, frame_diff, time_diff);
        
        // 丢弃旧帧（帧号小的，或帧号相同时接收时间早的）
        if (left_num < right_num || (left_num == right_num && left_recv < right_recv)) {
            left_frame_.reset();
        } else {
            right_frame_.reset();
        }
        return false;
    }
    
    // 同步成功
    sync_success_count_++;
    
    // 拷贝数据
    left = left_frame_.image.clone();
    right = right_frame_.image.clone();
    left_meta = left_frame_.metadata;
    right_meta = right_frame_.metadata;
    
    // 重置标志位
    left_frame_.reset();
    right_frame_.reset();
    
    return true;
}

// ==================== 推理线程 (轮询标志位同步，参考RC项目) ====================
void VolleyballTrackerNode::inferenceLoop() {
    RCLCPP_INFO(this->get_logger(), "🧠 推理线程已启动 (PWM时间戳同步模式)");
    
    last_stat_time_ = std::chrono::high_resolution_clock::now();
    
    while (running_ && rclcpp::ok()) {
        // ========== 1. 更新相机帧 (轮询回调) ==========
        updateLeftFrame();
        updateRightFrame();
        
        // ========== 2. 等待同步帧对 (帧号匹配) ==========
        cv::Mat left, right;
        FrameMetadata left_meta, right_meta;
        
        if (!waitForSyncedPair(left, right, left_meta, right_meta)) {
            // 无同步帧，短暂休眠后重试
            std::this_thread::sleep_for(std::chrono::microseconds(SYNC_RETRY_SLEEP_US));
            continue;
        }
        
        // 更新当前帧数据
        img_left_ = left;
        img_right_ = right;
        current_stamp_ = this->get_clock()->now();
        
        // ========== 3. 发布触发时间戳 ==========
        auto trigger_msg = std_msgs::msg::Header();
        trigger_msg.stamp = current_stamp_;
        trigger_msg.frame_id = "camera_frame";
        pub_trigger_->publish(trigger_msg);
        
        // ========== 4. 发布原始图像 (降低频率) ==========
        if (publish_images_ && frame_count_ % RAW_IMAGE_PUBLISH_INTERVAL == 0) {
            publishImages();
        }
        
        // ========== 5. YOLO Batch=2 检测 ==========
        auto detect_start = std::chrono::high_resolution_clock::now();
        bool detected = detectVolleyball(left, right);
        auto detect_end = std::chrono::high_resolution_clock::now();
        total_detection_time_ += std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
        
        if (!detected) {
            lost_frames_++;
            frame_count_++;
            if (enable_debug_ && frame_count_ % log_interval_ == 0) {
                printStatistics();
            }
            continue;
        }
        
        lost_frames_ = 0;
        
        // ========== 6. 立体匹配 ==========
        auto stereo_start = std::chrono::high_resolution_clock::now();
        bool matched = computeStereoMatch();
        auto stereo_end = std::chrono::high_resolution_clock::now();
        total_stereo_time_ += std::chrono::duration<double, std::milli>(stereo_end - stereo_start).count();
        
        if (!matched) {
            lost_frames_++;
            frame_count_++;
            if (enable_debug_ && frame_count_ % log_interval_ == 0) {
                printStatistics();
            }
            continue;
        }
        
        // ========== 7. 更新追踪器 ==========
        auto track_start = std::chrono::high_resolution_clock::now();
        updateTracker();
        auto track_end = std::chrono::high_resolution_clock::now();
        total_tracking_time_ += std::chrono::duration<double, std::milli>(track_end - track_start).count();
        
        // ========== 8. 发布结果 ==========
        publishResults();
        
        // 8. 统计
        frame_count_++;
        if (enable_debug_ && frame_count_ % log_interval_ == 0) {
            printStatistics();
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "🧠 推理线程已退出");
}

// ==================== YOLO 检测 ====================
bool VolleyballTrackerNode::detectVolleyball(const cv::Mat& left, const cv::Mat& right) {
    // img_left_ 和 img_right_ 已在调用前赋值，无需重复操作
    
    if (!detector_) {
        // 占位符模式：返回假检测结果
        return false;
    }
    
    // ✅ 使用双流并行推理（替代原来的串行检测）
    // 原代码: det_left_ = detectGlobal(); det_right_ = detectGlobal(); // 串行 17-21ms
    // 新代码: detectDual() 同时处理两张图, 预期 9-12ms
    auto [det_left, det_right] = detector_->detectDual(img_left_, img_right_, global_size_);
    det_left_ = det_left;
    det_right_ = det_right;
    
    if (det_left_.valid && det_right_.valid) {
        return true;
    }
    
    return false;
}

// ==================== 立体匹配 ====================
bool VolleyballTrackerNode::computeStereoMatch() {
    if (!stereo_matcher_) {
        return false;
    }
    
    cv::Point2f pt_left(det_left_.cx, det_left_.cy);
    cv::Point2f pt_right(det_right_.cx, det_right_.cy);
    
    StereoPoint stereo_pt = stereo_matcher_->triangulate(pt_left, pt_right);
    
    if (!stereo_pt.valid) {
        return false;
    }
    
    current_3d_point_ = stereo_pt.position_3d;
    current_depth_ = stereo_pt.position_3d.z;
    
    return true;
}

// ==================== 更新追踪器 ====================
void VolleyballTrackerNode::updateTracker() {
    if (!tracker_->isInitialized()) {
        tracker_->init(current_3d_point_);
        RCLCPP_INFO(this->get_logger(), "🎲 卡尔曼滤波器已初始化");
    } else {
        tracker_->predict();
        tracker_->update(current_3d_point_, current_depth_);
    }
}

// ==================== 发布结果 ====================
void VolleyballTrackerNode::publishResults() {
    if (!tracker_->isInitialized()) {
        return;
    }
    
    // 发布 3D 位置
    auto pose_msg = geometry_msgs::msg::PoseStamped();
    pose_msg.header.stamp = current_stamp_;
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
    
    // 发布调试信息
    if (enable_debug_) {
        auto debug_msg = std_msgs::msg::String();
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss << "{\"pos\":[" << pos.x << "," << pos.y << "," << pos.z << "],";
        ss << "\"vel\":[" << vel.x << "," << vel.y << "," << vel.z << "],";
        ss << "\"depth\":" << current_depth_ << ",";
        ss << "\"state\":\"TRACKING\",";
        ss << "\"det_left\":{\"cx\":" << det_left_.cx << ",\"cy\":" << det_left_.cy << "},";
        ss << "\"det_right\":{\"cx\":" << det_right_.cx << ",\"cy\":" << det_right_.cy << "}}";
        debug_msg.data = ss.str();
        pub_debug_->publish(debug_msg);
    }
    
    // 发布检测可视化图像（降低频率减少负载）
    if (publish_detection_image_ && !img_left_.empty() && 
        pub_detection_image_->get_subscription_count() > 0 &&
        frame_count_ % DETECTION_IMAGE_PUBLISH_INTERVAL == 0) {
        cv::Mat vis_img = drawDetections(img_left_, det_left_, true);
        auto img_msg = cvMatToRosImage(vis_img, "bgr8", current_stamp_);
        pub_detection_image_->publish(*img_msg);
    }
}

// ==================== 发布图像 ====================
void VolleyballTrackerNode::publishImages() {
    // 只有当有订阅者时才发布，避免不必要的开销
    if (pub_left_image_->get_subscription_count() > 0 && !img_left_.empty()) {
        auto left_msg = cvMatToRosImage(img_left_, "bgr8", current_stamp_);
        pub_left_image_->publish(*left_msg);
    }
    
    if (pub_right_image_->get_subscription_count() > 0 && !img_right_.empty()) {
        auto right_msg = cvMatToRosImage(img_right_, "bgr8", current_stamp_);
        pub_right_image_->publish(*right_msg);
    }
}

// ==================== 绘制检测结果 ====================
cv::Mat VolleyballTrackerNode::drawDetections(const cv::Mat& image, const Detection& det, bool /*is_left*/) {
    cv::Mat vis = image.clone();
    
    if (det.valid) {
        // 绘制检测框
        int x1 = static_cast<int>(det.cx - det.width / 2);
        int y1 = static_cast<int>(det.cy - det.height / 2);
        int x2 = static_cast<int>(det.cx + det.width / 2);
        int y2 = static_cast<int>(det.cy + det.height / 2);
        
        cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        
        // 绘制中心点
        cv::circle(vis, cv::Point(static_cast<int>(det.cx), static_cast<int>(det.cy)), 
                   5, cv::Scalar(0, 0, 255), -1);
        
        // 绘制置信度
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << det.confidence;
        cv::putText(vis, ss.str(), cv::Point(x1, y1 - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    
    // 绘制状态
    cv::putText(vis, "TRACKING", cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2);
    
    // 绘制 3D 位置
    if (tracker_ && tracker_->isInitialized()) {
        cv::Point3f pos = tracker_->getPosition();
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "Pos: [" << pos.x << ", " << pos.y << ", " << pos.z << "] m";
        cv::putText(vis, ss.str(), cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);
    }
    
    return vis;
}

// ==================== 打印统计 ====================
void VolleyballTrackerNode::printStatistics() {
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_stat_time_).count();
    
    double fps = frame_count_ / elapsed;
    double avg_detection = total_detection_time_ / frame_count_;
    double avg_stereo = total_stereo_time_ / frame_count_;
    double avg_tracking = total_tracking_time_ / frame_count_;
    double avg_total = avg_detection + avg_stereo + avg_tracking;
    
    // ✅ 新增：同步统计
    uint64_t sync_success = sync_success_count_.load();
    uint64_t sync_mismatch = sync_mismatch_count_.load();
    uint64_t left_drop = left_dropped_.load();
    uint64_t right_drop = right_dropped_.load();
    double sync_rate = 0.0;
    if (sync_success + sync_mismatch > 0) {
        sync_rate = sync_success * 100.0 / (sync_success + sync_mismatch);
    }
    
    RCLCPP_INFO(this->get_logger(),
                "🧠 [推理线程 %ld帧] FPS: %.1f | 检测: %.2fms | 立体: %.2fms | 追踪: %.2fms | 总计: %.2fms",
                frame_count_, fps, avg_detection, avg_stereo, avg_tracking, avg_total);
    
    RCLCPP_INFO(this->get_logger(),
                "   🔄 同步成功: %lu | 失配: %lu | 丢帧: L=%lu R=%lu | 同步率: %.1f%%",
                sync_success, sync_mismatch, left_drop, right_drop, sync_rate);
    
    if (tracker_ && tracker_->isInitialized()) {
        cv::Point3f pos = tracker_->getPosition();
        cv::Point3f vel = tracker_->getVelocity();
        float speed = std::sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        
        // 计算检测率（在重置前）
        double detection_rate = 0.0;
        if (frame_count_ > 0) {
            long successful_detections = frame_count_ - lost_frames_;
            if (successful_detections >= 0) {
                detection_rate = successful_detections * 100.0 / frame_count_;
            }
        }
        
        RCLCPP_INFO(this->get_logger(),
                    "   位置: [%.3f, %.3f, %.3f] m | 速度: %.2f m/s | 检测率: %.1f%%",
                    pos.x, pos.y, pos.z, speed, detection_rate);
    }
    
    // 重置统计
    frame_count_ = 0;
    lost_frames_ = 0;
    total_capture_time_ = 0.0;
    total_detection_time_ = 0.0;
    total_stereo_time_ = 0.0;
    total_tracking_time_ = 0.0;
    last_stat_time_ = now;
}

// ==================== CV Mat 转 ROS Image ====================
sensor_msgs::msg::Image::SharedPtr VolleyballTrackerNode::cvMatToRosImage(
    const cv::Mat& cv_image,
    const std::string& encoding,
    const rclcpp::Time& stamp) {
    
    auto ros_image = std::make_shared<sensor_msgs::msg::Image>();
    
    ros_image->header.stamp = stamp;
    ros_image->header.frame_id = "camera_frame";
    ros_image->height = cv_image.rows;
    ros_image->width = cv_image.cols;
    ros_image->encoding = encoding;
    ros_image->is_bigendian = false;
    ros_image->step = cv_image.cols * cv_image.elemSize();
    
    size_t size = ros_image->step * cv_image.rows;
    ros_image->data.resize(size);
    memcpy(&ros_image->data[0], cv_image.data, size);
    
    return ros_image;
}

// ==================== 查找文件路径 ====================
std::string VolleyballTrackerNode::findFilePath(const std::string& relative_path) {
    std::vector<std::string> search_paths;
    
    // 如果是绝对路径，直接检查
    if (relative_path[0] == '/') {
        std::ifstream f(relative_path);
        if (f.good()) {
            RCLCPP_DEBUG(this->get_logger(), "找到文件（绝对路径）: %s", relative_path.c_str());
            return relative_path;
        }
        return "";
    }
    
    // 1. 当前工作目录
    search_paths.push_back(relative_path);
    
    // 2. install 目录（通过 ament_index）
    try {
        std::string share_dir = ament_index_cpp::get_package_share_directory("volleyball_stereo_driver");
        search_paths.push_back(share_dir + "/" + relative_path);
    } catch (const std::exception& e) {
        RCLCPP_DEBUG(this->get_logger(), "无法获取包路径: %s", e.what());
    }
    
    // 3. src 目录（相对于当前位置）
    search_paths.push_back("../src/volleyball_stereo_driver/" + relative_path);
    
    // 4. 工作空间 src 目录（常见路径）
    std::vector<std::string> workspace_prefixes = {
        std::string(getenv("HOME") ? getenv("HOME") : "") + "/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/",
        std::string(getenv("HOME") ? getenv("HOME") : "") + "/ros2_ws/src/volleyball_stereo_driver/",
        "/home/nvidia/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/",
        "../../src/volleyball_stereo_driver/",
        "../../../src/volleyball_stereo_driver/"
    };
    
    for (const auto& prefix : workspace_prefixes) {
        if (!prefix.empty()) {
            search_paths.push_back(prefix + relative_path);
        }
    }
    
    // 遍历所有可能的路径
    for (const auto& path : search_paths) {
        std::ifstream f(path);
        if (f.good()) {
            RCLCPP_INFO(this->get_logger(), "✅ 找到文件: %s", path.c_str());
            return path;
        }
    }
    
    // 输出所有尝试的路径（调试用）
    RCLCPP_WARN(this->get_logger(), "未找到文件: %s", relative_path.c_str());
    RCLCPP_DEBUG(this->get_logger(), "尝试的路径:");
    for (const auto& path : search_paths) {
        RCLCPP_DEBUG(this->get_logger(), "  - %s", path.c_str());
    }
    
    return "";
}

}  // namespace volleyball

// ==================== Main ====================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    // ✅ 自动加载YAML配置文件
    rclcpp::NodeOptions options;
    try {
        std::string share_dir = ament_index_cpp::get_package_share_directory("volleyball_stereo_driver");
        std::string config_file = share_dir + "/config/tracker_params.yaml";
        options.arguments({"--ros-args", "--params-file", config_file});
    } catch (const std::exception& e) {
        std::cerr << "⚠️  未找到配置文件，使用默认值: " << e.what() << std::endl;
    }
    
    auto node = std::make_shared<volleyball::VolleyballTrackerNode>(options);
    
    RCLCPP_INFO(node->get_logger(), "🏐 排球追踪节点正在运行...");
    RCLCPP_INFO(node->get_logger(), "   按 Ctrl+C 停止");
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}
