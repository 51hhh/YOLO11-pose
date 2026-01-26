/**
 * @file volleyball_tracker_node.hpp
 * @brief 排球追踪主节点 (All-in-One)
 * 
 * 集成功能:
 * - PWM 触发 (内置)
 * - 双目相机采集
 * - YOLO 检测
 * - 立体匹配
 * - 3D 卡尔曼滤波
 */

#ifndef VOLLEYBALL_STEREO_DRIVER__VOLLEYBALL_TRACKER_NODE_HPP_
#define VOLLEYBALL_STEREO_DRIVER__VOLLEYBALL_TRACKER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <array>

#include "volleyball_stereo_driver/high_precision_pwm.hpp"
#include "volleyball_stereo_driver/hik_camera_wrapper.hpp"
#include "volleyball_stereo_driver/yolo_detector.hpp"
#include "volleyball_stereo_driver/stereo_matcher.hpp"
#include "volleyball_stereo_driver/kalman_filter_3d.hpp"
#include "volleyball_stereo_driver/roi_manager.hpp"

namespace volleyball {

/**
 * @struct CameraFrame
 * @brief 相机帧数据结构 (包含图像 + 元数据 + 标志位)
 * 参考: RC_Volleyball_vision 的全局帧缓冲设计
 */
struct CameraFrame {
    cv::Mat image;                    // 图像数据
    FrameMetadata metadata;           // 帧元数据 (帧号、时间戳)
    std::atomic<bool> ready{false};   // 就绪标志位
    
    void reset() {
        image.release();
        metadata = FrameMetadata();
        ready.store(false);
    }
};

/**
 * @brief 追踪状态
 */
enum TrackingState {
    GLOBAL_SEARCH,   // 全图搜索模式
    ROI_TRACKING     // ROI 追踪模式
};

/**
 * @brief 排球追踪主节点
 */
class VolleyballTrackerNode : public rclcpp::Node {
public:
    explicit VolleyballTrackerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~VolleyballTrackerNode();

private:
    // ==================== 初始化 ====================
    void declareParameters();
    void loadParameters();
    bool initializePWM();
    bool initializeCameras();
    bool initializeDetector();
    bool initializeStereoMatcher();
    void initializeKalmanFilter();
    void initializeROIManager();
    void createPublishers();
    
    // ==================== 线程方法 ====================
    void inferenceLoop();    // 推理线程: 轮询标志位同步
    
    // ==================== 帧同步方法 (PWM时间戳同步) ====================
    bool waitForSyncedPair(cv::Mat& left, cv::Mat& right,
                          FrameMetadata& left_meta,
                          FrameMetadata& right_meta);
    void updateLeftFrame();   // 左相机回调处理
    void updateRightFrame();  // 右相机回调处理
    
    // ==================== 处理流程 (在推理线程中) ====================
    bool detectVolleyball(const cv::Mat& left, const cv::Mat& right);
    bool computeStereoMatch();
    void updateTracker();
    void publishResults();
    void publishImages();
    void printStatistics();
    
    // ==================== 工具函数 ====================
    sensor_msgs::msg::Image::SharedPtr cvMatToRosImage(
        const cv::Mat& cv_image, 
        const std::string& encoding,
        const rclcpp::Time& stamp);
    std::string findFilePath(const std::string& relative_path);
    cv::Mat drawDetections(const cv::Mat& image, const Detection& det, bool is_left);
    
    // ==================== 常量定义 ====================
    // 同步参数
    static constexpr int SYNC_MAX_FRAME_DIFF = 3;           // 帧号差容忍度
    static constexpr int64_t SYNC_MAX_TIME_DIFF_US = 25000; // 25ms时间差容忍度
    static constexpr int FRAME_WAIT_TIMEOUT_MS = 0;          // ⚡ 优化: 0ms = 非阻塞轮询
    static constexpr int SYNC_RETRY_SLEEP_US = 1;            // ⚡ 优化: 降到1us（最小CPU yield）
    
    // 日志节流
    static constexpr int LOG_THROTTLE_MS = 1000;             // 日志节流周期(ms)
    
    // ⚡ 性能分析统计
    struct PerformanceStats {
        double total_loop_time = 0;
        double total_wait_left_time = 0;
        double total_wait_right_time = 0;
        double total_sync_check_time = 0;
        double total_sync_retry_time = 0;
        int sync_retry_count = 0;
        std::chrono::high_resolution_clock::time_point last_frame_time;
    };
    PerformanceStats perf_stats_;
    
    // 图像发布频率控制
    static constexpr int RAW_IMAGE_PUBLISH_INTERVAL = 10;    // 原始图像每10帧发布一次
    static constexpr int DETECTION_IMAGE_PUBLISH_INTERVAL = 5; // 检测图像每5帧发布一次
    
    // ==================== PWM 组件 ====================
    std::unique_ptr<HighPrecisionPWM> pwm_;
    std::string gpio_chip_;
    int gpio_line_;
    double pwm_frequency_;
    double pwm_duty_cycle_;
    
    // ==================== 相机组件 ====================
    std::unique_ptr<HikCamera> cam_left_;
    std::unique_ptr<HikCamera> cam_right_;
    int left_camera_index_;
    int right_camera_index_;
    double exposure_time_;
    double gain_;
    
    // ==================== 检测组件 ====================
    std::unique_ptr<YOLODetector> detector_;
    std::string model_path_;
    int input_size_;  // 模型输入分辨率 (0=自动检测, 320/640=手动指定)
    float conf_threshold_;
    float nms_threshold_;
    int roi_size_;
    int global_size_;
    
    // ==================== 立体视觉组件 ====================
    std::unique_ptr<StereoMatcher> stereo_matcher_;
    std::string calibration_file_;
    float min_disparity_;
    float max_depth_;
    
    // ==================== 追踪组件 ====================
    std::unique_ptr<KalmanFilter3D> tracker_;
    std::unique_ptr<ROIManager> roi_manager_;
    double process_noise_;
    double measurement_noise_;
    
    // ==================== ROS 发布器 ====================
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pub_velocity_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_debug_;
    rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr pub_trigger_;
    // 图像发布 (供可视化节点使用)
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_image_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_right_image_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_detection_image_;
    
    // ==================== 状态 ====================
    TrackingState state_;
    int lost_frames_;
    int max_lost_frames_;
    std::atomic<bool> running_;
    std::thread inference_thread_;   // 推理线程 (轮询标志位)
    
    // ==================== 全局帧缓冲 + 标志位同步 (参考RC项目) ====================
    CameraFrame left_frame_;         // 左相机帧缓冲
    CameraFrame right_frame_;        // 右相机帧缓冲
    std::mutex frame_sync_mutex_;    // 同步锁
    
    // ==================== 同步统计 ====================
    std::atomic<uint64_t> sync_success_count_{0};   // 帧号匹配成功次数
    std::atomic<uint64_t> sync_mismatch_count_{0};  // 帧号失配次数
    std::atomic<uint64_t> left_dropped_{0};         // 左帧丢弃次数
    std::atomic<uint64_t> right_dropped_{0};        // 右帧丢弃次数
    
    // ==================== 当前帧数据 ====================
    cv::Mat img_left_, img_right_;
    rclcpp::Time current_stamp_;
    Detection det_left_, det_right_;
    cv::Point3f current_3d_point_;
    float current_depth_;
    
    // ==================== 统计 ====================
    size_t frame_count_;
    std::chrono::high_resolution_clock::time_point last_stat_time_;
    double total_detection_time_;
    double total_stereo_time_;
    double total_tracking_time_;
    double total_capture_time_;
    
    // ==================== 调试参数 ====================
    bool enable_debug_;
    int log_interval_;
    bool publish_images_;
    bool publish_detection_image_;
};

}  // namespace volleyball

#endif  // VOLLEYBALL_STEREO_DRIVER__VOLLEYBALL_TRACKER_NODE_HPP_
