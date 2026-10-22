#include "volleyball/adapters/ros2/observation_publisher.hpp"

#include <rclcpp/rclcpp.hpp>
#include <volleyball_interfaces/msg/nx_ball_observation.hpp>

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>

namespace volleyball::adapters::ros2 {
namespace {

using NxMessage = volleyball_interfaces::msg::NxBallObservation;

NxMessage toRosMessage(const ObservationMessage& source) {
    NxMessage output;
    output.header.stamp.sec = source.stamp_sec;
    output.header.stamp.nanosec = source.stamp_nanosec;
    output.header.frame_id = source.header_frame_id;
    output.source_epoch = source.source_epoch;
    output.frame_id = source.frame_index;
    output.track_id = source.track_id;
    output.class_id = source.class_id;
    output.detection_confidence =
        static_cast<float>(source.detection_confidence);
    for (std::size_t index = 0; index < source.bbox_left_xyxy.size(); ++index) {
        output.bbox_left_xyxy[index] =
            static_cast<float>(source.bbox_left_xyxy[index]);
        output.bbox_right_xyxy[index] =
            static_cast<float>(source.bbox_right_xyxy[index]);
    }
    output.position.x = source.position_m.x;
    output.position.y = source.position_m.y;
    output.position.z = source.position_m.z;
    for (std::size_t index = 0;
         index < source.position_covariance_m2.size();
         ++index) {
        output.position_covariance[index] =
            source.position_covariance_m2[index];
    }
    output.depth_m = static_cast<float>(source.depth_m);
    output.disparity_px = static_cast<float>(source.disparity_px);
    output.depth_sigma_m = static_cast<float>(source.depth_sigma_m);
    output.match_confidence = static_cast<float>(source.match_confidence);
    output.depth_method = source.depth_method_code;
    output.detection_valid = source.detection_valid;
    output.stereo_valid = source.stereo_valid;
    output.fallback_observation = source.fallback_observation;
    output.left_device_timestamp = source.left_device_timestamp_ns;
    output.right_device_timestamp = source.right_device_timestamp_ns;
    output.stereo_timestamp_delta_ns =
        source.stereo_hardware_timestamp_residual_ns;
    output.capture_timestamp_mapping_valid =
        source.capture_timestamp_mapping_valid;
    output.capture_timestamp_uncertainty_ns =
        source.capture_timestamp_uncertainty_ns;
    return output;
}

}  // namespace

struct Ros2ObservationPublisher::Impl {
    std::shared_ptr<rclcpp::Node> node;
    ObservationMessageConfig config;
    rclcpp::Publisher<NxMessage>::SharedPtr publisher;
};

Ros2ObservationPublisher::Ros2ObservationPublisher(
    std::shared_ptr<rclcpp::Node> node,
    std::string topic,
    ObservationMessageConfig config)
    : impl_(std::make_unique<Impl>()) {
    if (!node || topic.empty() || !isValid(config)) {
        throw std::invalid_argument(
            "ROS2 observation publisher requires node, topic, and valid mapping config");
    }
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
                   .best_effort()
                   .durability_volatile()
                   .deadline(rclcpp::Duration::from_seconds(0.05));
    impl_->publisher = node->create_publisher<NxMessage>(topic, qos);
    if (!impl_->publisher) {
        throw std::runtime_error("ROS2 observation publisher creation failed");
    }
    impl_->node = std::move(node);
    impl_->config = std::move(config);
}

Ros2ObservationPublisher::~Ros2ObservationPublisher() = default;
Ros2ObservationPublisher::Ros2ObservationPublisher(
    Ros2ObservationPublisher&&) noexcept = default;
Ros2ObservationPublisher& Ros2ObservationPublisher::operator=(
    Ros2ObservationPublisher&&) noexcept = default;

ports::PublishResult Ros2ObservationPublisher::publish(
    const ports::ObservationPublication& publication) {
    try {
        const auto mapped = makeObservationMessage(publication, impl_->config);
        if (!mapped) {
            return {
                false,
                mapped.detail.empty()
                    ? "direct observation cannot be represented by ROS message"
                    : mapped.detail,
            };
        }
        impl_->publisher->publish(toRosMessage(*mapped.message));
        return {true, {}};
    } catch (const std::exception& error) {
        return {false, std::string{"ROS2 observation publish failed: "} + error.what()};
    } catch (...) {
        return {false, "ROS2 observation publish failed with unknown exception"};
    }
}

}  // namespace volleyball::adapters::ros2
