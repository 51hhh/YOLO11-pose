/**
 * @file volleyball_visualizer_node.cpp
 * @brief 排球追踪可视化节点
 *
 * 功能:
 * - 显示相机实时图像 (左/右/检测结果)
 * - 显示目标 3D 位置和速度
 * - 显示轨迹历史
 * - 调试信息面板
 */

#include "volleyball_stereo_driver/volleyball_visualizer_node.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <rclcpp/qos.hpp>

namespace volleyball {

// ==================== 构造函数 ====================
VolleyballVisualizerNode::VolleyballVisualizerNode()
    : Node("volleyball_visualizer"),
      max_trajectory_length_(100),
      show_stereo_view_(true),
      show_trajectory_(true),
      window_width_(1280),
      window_height_(720),
      display_fps_(30.0),
      frame_count_(0),
      current_fps_(0.0) {
    
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "🎥 排球追踪可视化节点");
    RCLCPP_INFO(this->get_logger(), "========================================");
    
    // 声明参数
    this->declare_parameter("show_stereo_view", true);
    this->declare_parameter("show_trajectory", true);
    this->declare_parameter("window_width", 1280);
    this->declare_parameter("window_height", 720);
    this->declare_parameter("display_fps", 30.0);
    this->declare_parameter("max_trajectory_length", 100);
    
    // 加载参数
    show_stereo_view_ = this->get_parameter("show_stereo_view").as_bool();
    show_trajectory_ = this->get_parameter("show_trajectory").as_bool();
    window_width_ = this->get_parameter("window_width").as_int();
    window_height_ = this->get_parameter("window_height").as_int();
    display_fps_ = this->get_parameter("display_fps").as_double();
    max_trajectory_length_ = this->get_parameter("max_trajectory_length").as_int();
    
    // ✅ 使用 SensorDataQoS 匹配发布器 (必须一致才能通信)
    rclcpp::SensorDataQoS sensor_qos;
    
    // 创建订阅器 (使用低延迟 QoS)
    sub_left_image_ = this->create_subscription<sensor_msgs::msg::Image>(
        "stereo/left/image_raw", sensor_qos,
        std::bind(&VolleyballVisualizerNode::leftImageCallback, this, std::placeholders::_1));
    
    sub_right_image_ = this->create_subscription<sensor_msgs::msg::Image>(
        "stereo/right/image_raw", sensor_qos,
        std::bind(&VolleyballVisualizerNode::rightImageCallback, this, std::placeholders::_1));
    
    sub_detection_image_ = this->create_subscription<sensor_msgs::msg::Image>(
        "volleyball/detection_image", sensor_qos,
        std::bind(&VolleyballVisualizerNode::detectionImageCallback, this, std::placeholders::_1));
    
    sub_pose_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "volleyball/pose_3d", sensor_qos,
        std::bind(&VolleyballVisualizerNode::poseCallback, this, std::placeholders::_1));
    
    sub_velocity_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
        "volleyball/velocity", sensor_qos,
        std::bind(&VolleyballVisualizerNode::velocityCallback, this, std::placeholders::_1));
    
    // 调试信息使用可靠传输
    sub_debug_ = this->create_subscription<std_msgs::msg::String>(
        "volleyball/debug_info", 10,
        std::bind(&VolleyballVisualizerNode::debugCallback, this, std::placeholders::_1));
    
    // 创建显示定时器
    auto period = std::chrono::duration<double>(1.0 / display_fps_);
    display_timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::milliseconds>(period),
        std::bind(&VolleyballVisualizerNode::displayLoop, this));
    
    last_frame_time_ = std::chrono::high_resolution_clock::now();
    
    // 创建窗口
    cv::namedWindow("Volleyball Tracker", cv::WINDOW_NORMAL);
    cv::resizeWindow("Volleyball Tracker", window_width_, window_height_);
    
    if (show_stereo_view_) {
        cv::namedWindow("Stereo View", cv::WINDOW_NORMAL);
        cv::resizeWindow("Stereo View", 1280, 480);
    }
    
    RCLCPP_INFO(this->get_logger(), "✅ 可视化节点已启动");
    RCLCPP_INFO(this->get_logger(), "   显示帧率: %.1f FPS", display_fps_);
    RCLCPP_INFO(this->get_logger(), "   窗口大小: %dx%d", window_width_, window_height_);
    RCLCPP_INFO(this->get_logger(), "========================================");
}

// ==================== 析构函数 ====================
VolleyballVisualizerNode::~VolleyballVisualizerNode() {
    cv::destroyAllWindows();
    RCLCPP_INFO(this->get_logger(), "可视化节点已关闭");
}

// ==================== 左图像回调 ====================
void VolleyballVisualizerNode::leftImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    img_left_ = rosImageToCvMat(msg);
}

// ==================== 右图像回调 ====================
void VolleyballVisualizerNode::rightImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    img_right_ = rosImageToCvMat(msg);
}

// ==================== 检测图像回调 ====================
void VolleyballVisualizerNode::detectionImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    img_detection_ = rosImageToCvMat(msg);
}

// ==================== 位置回调 ====================
void VolleyballVisualizerNode::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    last_pose_ = msg;
    
    // 添加到轨迹历史
    if (show_trajectory_) {
        cv::Point3f pos(
            static_cast<float>(msg->pose.position.x),
            static_cast<float>(msg->pose.position.y),
            static_cast<float>(msg->pose.position.z)
        );
        trajectory_history_.push_back(pos);
        
        // 限制轨迹长度
        while (trajectory_history_.size() > max_trajectory_length_) {
            trajectory_history_.pop_front();
        }
    }
}

// ==================== 速度回调 ====================
void VolleyballVisualizerNode::velocityCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    last_velocity_ = msg;
}

// ==================== 调试信息回调 ====================
void VolleyballVisualizerNode::debugCallback(const std_msgs::msg::String::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    debug_info_ = msg->data;
}

// ==================== 显示循环 ====================
void VolleyballVisualizerNode::displayLoop() {
    cv::Mat display_frame;
    cv::Mat stereo_frame;
    
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // 主显示窗口：使用检测图像或左图像
        if (!img_detection_.empty()) {
            display_frame = img_detection_.clone();
        } else if (!img_left_.empty()) {
            display_frame = img_left_.clone();
        }
        
        // 双目显示
        if (show_stereo_view_ && !img_left_.empty() && !img_right_.empty()) {
            cv::hconcat(img_left_, img_right_, stereo_frame);
        }
    }
    
    // 显示主窗口
    if (!display_frame.empty()) {
        // 绘制信息面板
        drawInfoPanel(display_frame);
        
        // 调整大小
        cv::Mat resized;
        cv::resize(display_frame, resized, cv::Size(window_width_, window_height_));
        
        cv::imshow("Volleyball Tracker", resized);
    }
    
    // 显示双目窗口
    if (show_stereo_view_ && !stereo_frame.empty()) {
        cv::imshow("Stereo View", stereo_frame);
    }
    
    // 处理键盘事件
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {  // 'q' 或 ESC
        RCLCPP_INFO(this->get_logger(), "用户请求退出");
        rclcpp::shutdown();
    } else if (key == 's') {  // 截图
        std::string filename = "screenshot_" + std::to_string(std::time(nullptr)) + ".png";
        if (!display_frame.empty()) {
            cv::imwrite(filename, display_frame);
            RCLCPP_INFO(this->get_logger(), "📸 截图已保存: %s", filename.c_str());
        }
    } else if (key == 't') {  // 切换轨迹显示
        show_trajectory_ = !show_trajectory_;
        RCLCPP_INFO(this->get_logger(), "轨迹显示: %s", show_trajectory_ ? "开" : "关");
    } else if (key == 'c') {  // 清除轨迹
        std::lock_guard<std::mutex> lock(data_mutex_);
        trajectory_history_.clear();
        RCLCPP_INFO(this->get_logger(), "轨迹已清除");
    }
    
    // 更新帧率统计
    frame_count_++;
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_frame_time_).count();
    if (elapsed >= 1.0) {
        current_fps_ = frame_count_ / elapsed;
        frame_count_ = 0;
        last_frame_time_ = now;
    }
}

// ==================== 绘制信息面板 ====================
void VolleyballVisualizerNode::drawInfoPanel(cv::Mat& frame) {
    int panel_height = 150;
    int panel_width = 300;
    int margin = 10;
    
    // 绘制半透明背景
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, 
                  cv::Point(margin, frame.rows - panel_height - margin),
                  cv::Point(panel_width + margin, frame.rows - margin),
                  cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);
    
    int y = frame.rows - panel_height;
    int line_height = 25;
    cv::Scalar text_color(255, 255, 255);
    double font_scale = 0.5;
    
    // 标题
    cv::putText(frame, "Volleyball Tracker", cv::Point(margin + 5, y + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    y += line_height + 5;
    
    // 显示 FPS
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "Display FPS: " << current_fps_;
    cv::putText(frame, ss.str(), cv::Point(margin + 5, y + 20),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1);
    y += line_height;
    
    // 显示位置
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (last_pose_) {
            ss.str("");
            ss << "Pos: [" << last_pose_->pose.position.x << ", "
               << last_pose_->pose.position.y << ", "
               << last_pose_->pose.position.z << "] m";
            cv::putText(frame, ss.str(), cv::Point(margin + 5, y + 20),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1);
            y += line_height;
        }
        
        // 显示速度
        if (last_velocity_) {
            float speed = std::sqrt(
                last_velocity_->vector.x * last_velocity_->vector.x +
                last_velocity_->vector.y * last_velocity_->vector.y +
                last_velocity_->vector.z * last_velocity_->vector.z
            );
            ss.str("");
            ss << "Speed: " << speed << " m/s";
            cv::putText(frame, ss.str(), cv::Point(margin + 5, y + 20),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1);
            y += line_height;
        }
        
        // 显示轨迹点数
        if (show_trajectory_) {
            ss.str("");
            ss << "Trajectory: " << trajectory_history_.size() << " points";
            cv::putText(frame, ss.str(), cv::Point(margin + 5, y + 20),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1);
        }
    }
    
    // 绘制快捷键提示
    int help_y = margin + 20;
    cv::putText(frame, "Keys: Q-Quit | S-Screenshot | T-Trajectory | C-Clear",
                cv::Point(margin + 5, help_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
}

// ==================== ROS Image 转 CV Mat ====================
cv::Mat VolleyballVisualizerNode::rosImageToCvMat(const sensor_msgs::msg::Image::SharedPtr& msg) {
    cv::Mat image;
    
    if (msg->encoding == "bgr8") {
        image = cv::Mat(msg->height, msg->width, CV_8UC3, 
                        const_cast<uint8_t*>(msg->data.data()), msg->step).clone();
    } else if (msg->encoding == "rgb8") {
        cv::Mat rgb(msg->height, msg->width, CV_8UC3,
                    const_cast<uint8_t*>(msg->data.data()), msg->step);
        cv::cvtColor(rgb, image, cv::COLOR_RGB2BGR);
    } else if (msg->encoding == "mono8") {
        image = cv::Mat(msg->height, msg->width, CV_8UC1,
                        const_cast<uint8_t*>(msg->data.data()), msg->step).clone();
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    } else {
        RCLCPP_WARN_ONCE(this->get_logger(), "不支持的图像编码: %s", msg->encoding.c_str());
    }
    
    return image;
}

}  // namespace volleyball

// ==================== Main ====================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<volleyball::VolleyballVisualizerNode>();
    
    RCLCPP_INFO(node->get_logger(), "🎥 可视化节点正在运行...");
    RCLCPP_INFO(node->get_logger(), "   按 Q 或 ESC 退出");
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}
