/**
 * @file diagnostic_publisher.h
 * @brief ROS2 诊断数据发布器 — 深度图 + 检测框 + 原始观测
 *
 * 用于录制 rosbag，离线分析深度取点策略效果。
 * 发布话题:
 *   /debug/depth_full   (sensor_msgs/Image, 32FC1, 10Hz) — 完整深度图
 *   /debug/depth_roi    (sensor_msgs/Image, 32FC1, 60Hz) — bbox裁剪深度图
 *   /debug/detections   (sensor_msgs/Image, 8UC3)        — 带bbox标注的左图(10Hz)
 *   /debug/raw_obs      (geometry_msgs/PoseArray)         — 原始3D观测
 */

#ifndef STEREO3D_DIAGNOSTIC_PUBLISHER_H_
#define STEREO3D_DIAGNOSTIC_PUBLISHER_H_

#ifdef HAS_ROS2

#include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include "../pipeline/frame_slot.h"

namespace stereo3d {

struct DiagnosticPublisherConfig {
    bool enabled = false;
    int depth_full_divisor = 6;     ///< 全帧深度发布频率 = pipeline_fps / divisor (60/6=10Hz)
    std::string frame_id = "zed_left";
};

class DiagnosticPublisher {
public:
    DiagnosticPublisher(const std::shared_ptr<rclcpp::Node>& node,
                        const DiagnosticPublisherConfig& cfg);

    /**
     * @brief 发布一帧诊断数据
     * @param frame_id      帧号
     * @param depth_gpu     ZED 深度图 GPU 指针 (float*, meters)
     * @param depth_pitch   深度图行字节跨度
     * @param depth_width   深度图宽度
     * @param depth_height  深度图高度
     * @param img_width     全分辨率图像宽度 (检测坐标系)
     * @param img_height    全分辨率图像高度
     * @param detections    YOLO 检测结果
     * @param results       3D 定位结果 (含 z_mono, z_stereo 等)
     */
    void publish(int frame_id,
                 const float* depth_gpu, int depth_pitch,
                 int depth_width, int depth_height,
                 int img_width, int img_height,
                 const std::vector<Detection>& detections,
                 const std::vector<Object3D>& results);

    bool enabled() const { return cfg_.enabled; }

private:
    void publishDepthFull(int frame_id, const float* depth_gpu, int depth_pitch,
                          int depth_width, int depth_height);
    void publishDepthROI(int frame_id, const float* depth_gpu, int depth_pitch,
                         int depth_width, int depth_height,
                         int img_width, int img_height,
                         const std::vector<Detection>& detections);
    void publishRawObs(int frame_id, const std::vector<Object3D>& results);

    std::shared_ptr<rclcpp::Node> node_;
    DiagnosticPublisherConfig cfg_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_full_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_roi_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr raw_obs_pub_;

    int publish_count_ = 0;

    // CPU 缓冲区 (避免每帧分配)
    std::vector<float> depth_cpu_buf_;
};

}  // namespace stereo3d

#endif  // HAS_ROS2
#endif  // STEREO3D_DIAGNOSTIC_PUBLISHER_H_
