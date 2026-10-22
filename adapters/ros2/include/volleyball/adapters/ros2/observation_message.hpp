#pragma once

#include "volleyball/ports/observation_publisher.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace volleyball::adapters::ros2 {

// 这里描述的是 ROS 消息的无中间件等价物。映射逻辑因此可以在没有 ROS2
// 的主机 CI 上完整测试，真正的 rclcpp 适配器只负责逐字段搬运。
struct ObservationMessageConfig {
    std::string header_frame_id;
    core::ClockDomainId ros_time_clock_domain;
    core::StereoDepthMethodId primary_depth_method;
    std::uint32_t source_epoch{0};
    std::int32_t class_id{0};
};

[[nodiscard]] bool isValid(
    const ObservationMessageConfig& config) noexcept;

struct ObservationMessage {
    std::int32_t stamp_sec{0};
    std::uint32_t stamp_nanosec{0};
    std::string header_frame_id;
    std::uint32_t source_epoch{0};
    std::int32_t frame_index{0};
    // 直接观测尚未经过跟踪；0 是明确的“无 track”哨兵值。
    std::int32_t track_id{0};
    std::int32_t class_id{0};
    double detection_confidence{0.0};
    std::array<double, 4> bbox_left_xyxy{};
    std::array<double, 4> bbox_right_xyxy{};
    core::Vec3 position_m;
    std::array<double, 9> position_covariance_m2{};
    double depth_m{0.0};
    double disparity_px{0.0};
    double depth_sigma_m{0.0};
    double match_confidence{0.0};
    std::uint8_t depth_method_code{0};
    bool detection_valid{false};
    bool stereo_valid{false};
    bool fallback_observation{false};
    std::uint64_t left_device_timestamp_ns{0};
    std::uint64_t right_device_timestamp_ns{0};
    std::int64_t stereo_hardware_timestamp_residual_ns{0};
    bool capture_timestamp_mapping_valid{false};
    std::uint64_t capture_timestamp_uncertainty_ns{0};
};

enum class ObservationMessageError {
    none,
    invalid_config,
    invalid_domain_record,
    inconsistent_reference,
    missing_required_evidence,
    unsupported_depth_method,
    numeric_range,
};

struct ObservationMessageResult {
    std::optional<ObservationMessage> message;
    ObservationMessageError error{ObservationMessageError::none};
    std::string detail;

    [[nodiscard]] explicit operator bool() const noexcept {
        return message.has_value();
    }
};

// 与现有 volleyball_interfaces/NxBallObservation.depth_method 数值兼容。
// 未知方法必须拒绝，不能静默发布为 0 或猜测相近方法。
[[nodiscard]] std::optional<std::uint8_t> depthMethodWireCode(
    const core::StereoDepthMethodId& method) noexcept;

[[nodiscard]] ObservationMessageResult makeObservationMessage(
    const ports::ObservationPublication& publication,
    const ObservationMessageConfig& config);

}  // namespace volleyball::adapters::ros2
