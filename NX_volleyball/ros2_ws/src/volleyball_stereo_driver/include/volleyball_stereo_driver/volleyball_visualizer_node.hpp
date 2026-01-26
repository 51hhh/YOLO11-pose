/**
 * @file volleyball_visualizer_node.hpp
 * @brief 排球追踪可视化节点
 * 
 * 功能:
 * - 显示相机实时图像
 * - 显示 YOLO 检测结果
 * - 显示目标 3D 位置
 * - 调试信息面板
 */

#ifndef VOLLEYBALL_STEREO_DRIVER__VOLLEYBALL_VISUALIZER_NODE_HPP_
#define VOLLEYBALL_STEREO_DRIVER__VOLLEYBALL_VISUALIZER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>
#include <deque>

namespace volleyball {

/**
 * @brief 可视化节点
 */
class VolleyballVisualizerNode : public rclcpp::Node {
public:
    VolleyballVisualizerNode();
    ~VolleyballVisualizerNode();

private:
    // ==================== 回调函数 ====================
    void leftImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void rightImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void detectionImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void velocityCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);
    void debugCallback(const std_msgs::msg::String::SharedPtr msg);
    
    // ==================== 显示函数 ====================
    void displayLoop();
    void drawInfoPanel(cv::Mat& frame);
    void drawTrajectory(cv::Mat& frame);
    cv::Mat rosImageToCvMat(const sensor_msgs::msg::Image::SharedPtr& msg);
    
    // ==================== 订阅器 ====================
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_left_image_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_right_image_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_detection_image_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr sub_velocity_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_debug_;
    
    // ==================== 数据缓存 ====================
    cv::Mat img_left_, img_right_, img_detection_;
    geometry_msgs::msg::PoseStamped::SharedPtr last_pose_;
    geometry_msgs::msg::Vector3Stamped::SharedPtr last_velocity_;
    std::string debug_info_;
    std::mutex data_mutex_;
    
    // ==================== 轨迹历史 ====================
    std::deque<cv::Point3f> trajectory_history_;
    size_t max_trajectory_length_;
    
    // ==================== 显示定时器 ====================
    rclcpp::TimerBase::SharedPtr display_timer_;
    
    // ==================== 参数 ====================
    bool show_stereo_view_;
    bool show_trajectory_;
    int window_width_;
    int window_height_;
    double display_fps_;
    
    // ==================== 统计 ====================
    size_t frame_count_;
    std::chrono::high_resolution_clock::time_point last_frame_time_;
    double current_fps_;
};

}  // namespace volleyball

#endif  // VOLLEYBALL_STEREO_DRIVER__VOLLEYBALL_VISUALIZER_NODE_HPP_
