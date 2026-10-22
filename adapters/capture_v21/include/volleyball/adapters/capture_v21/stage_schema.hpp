#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>

namespace volleyball::adapters::capture_v21 {

inline constexpr int kCaptureV21StageSchemaVersion = 4;

inline constexpr std::array<std::string_view, 25> kCaptureV21FrameColumns = {
    "frame_id",
    "host_capture_timestamp_ns",
    "callback_timestamp_ns",
    "left_receive_monotonic_ns",
    "right_receive_monotonic_ns",
    "left_receive_realtime_ns",
    "right_receive_realtime_ns",
    "left_sdk_host_timestamp_raw",
    "right_sdk_host_timestamp_raw",
    "left_device_timestamp_ns",
    "right_device_timestamp_ns",
    "stereo_timestamp_residual_ns",
    "trigger_sequence",
    "trigger_scheduled_monotonic_ns",
    "trigger_edge_monotonic_ns",
    "trigger_edge_realtime_ns",
    "trigger_write_uncertainty_ns",
    "left_frame_number",
    "right_frame_number",
    "left_frame_counter",
    "right_frame_counter",
    "left_trigger_index",
    "right_trigger_index",
    "raw_left",
    "raw_right",
};

inline constexpr std::array<std::string_view, 10>
    kCaptureV21DetectionColumns = {
        "frame_id",
        "camera_role",
        "candidate_id",
        "cx_px",
        "cy_px",
        "width_px",
        "height_px",
        "confidence",
        "class_id",
        "selected",
    };

inline constexpr std::array<std::string_view, 13> kCaptureV21PairColumns = {
    "pair_id",
    "frame_id",
    "left_candidate_id",
    "right_candidate_id",
    "disparity_px",
    "epipolar_error_px",
    "pair_score",
    "semantic_confidence",
    "x_rig_m",
    "y_rig_m",
    "z_rig_m",
    "accepted",
    "reject_reason",
};

struct CaptureV21FrameRow {
    int frame_id = -1;
    std::uint64_t host_capture_timestamp_ns = 0;
    std::uint64_t callback_timestamp_ns = 0;
    std::uint64_t left_receive_monotonic_ns = 0;
    std::uint64_t right_receive_monotonic_ns = 0;
    std::uint64_t left_receive_realtime_ns = 0;
    std::uint64_t right_receive_realtime_ns = 0;
    std::int64_t left_sdk_host_timestamp_raw = 0;
    std::int64_t right_sdk_host_timestamp_raw = 0;
    std::uint64_t left_device_timestamp_ns = 0;
    std::uint64_t right_device_timestamp_ns = 0;
    std::int64_t stereo_timestamp_residual_ns = 0;
    std::uint64_t trigger_sequence = 0;
    std::uint64_t trigger_scheduled_monotonic_ns = 0;
    std::uint64_t trigger_edge_monotonic_ns = 0;
    std::uint64_t trigger_edge_realtime_ns = 0;
    std::uint64_t trigger_write_uncertainty_ns = 0;
    std::uint32_t left_frame_number = 0;
    std::uint32_t right_frame_number = 0;
    std::uint32_t left_frame_counter = 0;
    std::uint32_t right_frame_counter = 0;
    std::uint32_t left_trigger_index = 0;
    std::uint32_t right_trigger_index = 0;
    std::string raw_left;
    std::string raw_right;
};

struct CaptureV21DetectionRow {
    int frame_id = -1;
    std::string camera_role;
    std::uint64_t candidate_id = 0;
    float cx_px = 0.0F;
    float cy_px = 0.0F;
    float width_px = 0.0F;
    float height_px = 0.0F;
    float confidence = 0.0F;
    int class_id = 0;
    bool selected = false;
};

struct CaptureV21PairRow {
    std::uint64_t pair_id = 0;
    int frame_id = -1;
    int left_candidate_id = -1;
    int right_candidate_id = -1;
    float disparity_px = 0.0F;
    float epipolar_error_px = 0.0F;
    float pair_score = 0.0F;
    float semantic_confidence = 0.0F;
    std::optional<std::array<float, 3>> position_rig_m;
    bool accepted = false;
    std::string reject_reason;
};

void writeCaptureV21FrameHeader(std::ostream& output);
void writeCaptureV21DetectionHeader(std::ostream& output);
void writeCaptureV21PairHeader(std::ostream& output);

[[nodiscard]] bool writeCaptureV21FrameRow(
    std::ostream& output, const CaptureV21FrameRow& row);
[[nodiscard]] bool writeCaptureV21DetectionRow(
    std::ostream& output, const CaptureV21DetectionRow& row);
[[nodiscard]] bool writeCaptureV21PairRow(
    std::ostream& output, const CaptureV21PairRow& row);

}  // namespace volleyball::adapters::capture_v21
