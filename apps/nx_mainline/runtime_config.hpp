#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace volleyball::apps::nx_mainline {

struct RuntimeSettings {
    std::string mode;
    double nominal_fps{0.0};
    unsigned int source_timeout_ms{0};
    std::size_t max_consecutive_timeouts{0};
    std::size_t queue_capacity{0};
    std::string overflow_policy;
    std::optional<std::uint64_t> frame_limit;
    std::optional<std::filesystem::path> capture_stage;
};

struct CameraSettings {
    std::string left_id;
    std::string right_id;
    std::string left_serial;
    std::string right_serial;
    int left_index{0};
    int right_index{0};
    int width{0};
    int height{0};
    double exposure_us{0.0};
    double gain_db{0.0};
    int image_node_num{0};
    int embedded_info_clear_rows{0};
    std::string trigger_source;
    std::string trigger_activation;
    std::string exposure_timestamp_source;
    std::string shutter_type;
    std::string exposure_delay_status;
    std::uint64_t exposure_delay_ns{0};
    std::optional<std::uint64_t> exposure_delay_uncertainty_ns;
};

struct ClockSettings {
    std::string reference;
    std::string left_device;
    std::string right_device;
    std::string capture_realtime;
    std::string host_monotonic;
};

struct TriggerSettings {
    std::string gpio_chip;
    unsigned int gpio_line{0};
    double frequency_hz{0.0};
    double duty_cycle{0.0};
};

struct CalibrationSettings {
    std::filesystem::path file;
    std::string sha256;
    std::string rig_frame;
    int output_width{0};
    int output_height{0};
};

struct DetectorSettings {
    std::filesystem::path engine_file;
    std::string engine_sha256;
    std::string pipeline_id;
    int input_size{0};
    std::string output_contract;
    int class_count{0};
    double confidence_threshold{0.0};
    double nms_iou_threshold{0.0};
    std::size_t max_detections{0};
    int class_id{0};
    bool use_dla{false};
    int dla_core{0};
};

struct AssociationSettings {
    std::int64_t max_capture_delta_ns{0};
    double epipolar_tolerance_px{0.0};
    double adaptive_epipolar_height_ratio{0.0};
    double max_size_ratio{0.0};
    double min_shifted_iou{0.0};
    std::uint32_t max_disparity_px{0};
    double ball_diameter_m{0.0};
    double bbox_scale{0.0};
    double bbox_disparity_consistency_ratio{0.0};
    double bbox_disparity_consistency_min_px{0.0};
    double bbox_disparity_penalty_scale{0.0};
};

struct DepthSettings {
    std::vector<std::string> method_order;
    std::uint32_t circle_max_roi_pixels{0};
    double anchor_stddev_px{0.0};
    double disparity_stddev_px{0.0};
    double min_depth_m{0.0};
    double max_depth_m{0.0};
};

struct PublisherSettings {
    bool enabled{false};
    std::string topic;
    std::string frame_id;
    std::filesystem::path source_epoch_file;
};

struct MainlineRuntimeConfig {
    std::string profile;
    std::string config_sha256;
    RuntimeSettings runtime;
    CameraSettings camera;
    ClockSettings clocks;
    TriggerSettings trigger;
    CalibrationSettings calibration;
    DetectorSettings detector;
    AssociationSettings association;
    DepthSettings depth;
    PublisherSettings publisher;
    std::size_t metrics_interval_frames{0};
};

[[nodiscard]] bool loadRuntimeConfig(
    const std::filesystem::path& path,
    MainlineRuntimeConfig& output,
    std::string& error);

}  // namespace volleyball::apps::nx_mainline
