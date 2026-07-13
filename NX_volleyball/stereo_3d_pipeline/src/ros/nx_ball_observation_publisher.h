#pragma once
#ifdef HAS_ROS2
#include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <volleyball_interfaces/msg/nx_ball_observation.hpp>
#include "../pipeline/frame_slot.h"
namespace stereo3d {
class NxBallObservationPublisher {
public:
    NxBallObservationPublisher(const std::shared_ptr<rclcpp::Node>& node,
                               const std::string& topic,
                               const std::string& frame_id);
    void publish(int frame_id, const std::vector<Object3D>& results,
                 const FrameMetadata& metadata);
private:
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Publisher<volleyball_interfaces::msg::NxBallObservation>::SharedPtr pub_;
    std::string frame_id_;
    uint32_t source_epoch_{0};
};
}
#endif
