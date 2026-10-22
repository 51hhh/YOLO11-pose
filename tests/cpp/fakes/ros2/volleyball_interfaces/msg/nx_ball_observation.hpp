#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace volleyball_interfaces::msg {

struct NxBallObservation {
    struct Stamp {
        std::int32_t sec{0};
        std::uint32_t nanosec{0};
    };
    struct Header {
        Stamp stamp;
        std::string frame_id;
    };
    struct Point {
        double x{0.0};
        double y{0.0};
        double z{0.0};
    };

    Header header;
    std::uint32_t source_epoch{0};
    std::int32_t frame_id{0};
    std::int32_t track_id{0};
    std::int32_t class_id{0};
    float detection_confidence{0.0F};
    std::array<float, 4> bbox_left_xyxy{};
    std::array<float, 4> bbox_right_xyxy{};
    Point position;
    std::array<double, 9> position_covariance{};
    float depth_m{0.0F};
    float disparity_px{0.0F};
    float depth_sigma_m{0.0F};
    float match_confidence{0.0F};
    std::uint8_t depth_method{0};
    bool detection_valid{false};
    bool stereo_valid{false};
    bool fallback_observation{false};
    std::uint64_t left_device_timestamp{0};
    std::uint64_t right_device_timestamp{0};
    std::int64_t stereo_timestamp_delta_ns{0};
    bool capture_timestamp_mapping_valid{false};
    std::uint64_t capture_timestamp_uncertainty_ns{0};
};

}  // namespace volleyball_interfaces::msg
