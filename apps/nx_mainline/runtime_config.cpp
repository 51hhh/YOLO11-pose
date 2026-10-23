#include "runtime_config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>

namespace volleyball::apps::nx_mainline {
namespace {

using KeySet = std::set<std::string>;

void exactKeys(
    const YAML::Node& node,
    std::initializer_list<const char*> keys,
    std::string_view context) {
    if (!node.IsMap()) {
        throw std::runtime_error(std::string{context} + " must be a mapping");
    }
    KeySet expected;
    for (const char* key : keys) expected.emplace(key);
    for (const auto& entry : node) {
        const std::string key = entry.first.as<std::string>();
        if (expected.count(key) == 0) {
            throw std::runtime_error(
                std::string{context} + " contains unknown key " + key);
        }
    }
    for (const auto& key : expected) {
        const YAML::Node value = node[key];
        if (!value.IsDefined()) {
            throw std::runtime_error(
                std::string{context} + " is missing key " + key);
        }
    }
}

template <typename Value>
Value scalar(
    const YAML::Node& node,
    const char* key,
    std::string_view context) {
    const YAML::Node value = node[key];
    if (!value.IsDefined() || value.IsNull() || !value.IsScalar()) {
        throw std::runtime_error(
            std::string{context} + "." + key + " must be a scalar");
    }
    try {
        return value.as<Value>();
    } catch (const YAML::Exception& error) {
        throw std::runtime_error(
            std::string{context} + "." + key + ": " + error.what());
    }
}

template <typename Value>
std::optional<Value> nullableScalar(
    const YAML::Node& node,
    const char* key,
    std::string_view context) {
    const YAML::Node value = node[key];
    if (!value.IsDefined()) {
        throw std::runtime_error(
            std::string{context} + " is missing key " + key);
    }
    if (value.IsNull()) return std::nullopt;
    return scalar<Value>(node, key, context);
}

std::filesystem::path absolutePath(
    const YAML::Node& node,
    const char* key,
    std::string_view context) {
    std::filesystem::path path = scalar<std::string>(node, key, context);
    if (!path.is_absolute() || path.empty()) {
        throw std::runtime_error(
            std::string{context} + "." + key + " must be an absolute path");
    }
    return path.lexically_normal();
}

bool finite(double value) noexcept { return std::isfinite(value); }

bool sha256(std::string_view value) noexcept {
    return value.size() == 64 && std::all_of(
        value.begin(), value.end(), [](unsigned char character) {
            return (character >= '0' && character <= '9') ||
                   (character >= 'a' && character <= 'f');
        });
}

std::vector<std::string> stringSequence(
    const YAML::Node& node,
    const char* key,
    std::string_view context) {
    const YAML::Node sequence = node[key];
    if (!sequence.IsDefined() || !sequence.IsSequence() || sequence.size() == 0) {
        throw std::runtime_error(
            std::string{context} + "." + key + " must be a non-empty sequence");
    }
    std::vector<std::string> output;
    output.reserve(sequence.size());
    for (const auto& value : sequence) {
        if (!value.IsScalar()) {
            throw std::runtime_error(
                std::string{context} + "." + key + " entries must be strings");
        }
        output.push_back(value.as<std::string>());
    }
    return output;
}

void parseRuntime(const YAML::Node& node, RuntimeSettings& output) {
    exactKeys(
        node,
        {"mode", "nominal_fps", "source_timeout_ms",
         "max_consecutive_timeouts", "queue_capacity", "overflow_policy",
         "frame_limit", "capture_stage"},
        "runtime");
    output.mode = scalar<std::string>(node, "mode", "runtime");
    output.nominal_fps = scalar<double>(node, "nominal_fps", "runtime");
    output.source_timeout_ms =
        scalar<unsigned int>(node, "source_timeout_ms", "runtime");
    output.max_consecutive_timeouts =
        scalar<std::size_t>(node, "max_consecutive_timeouts", "runtime");
    output.queue_capacity =
        scalar<std::size_t>(node, "queue_capacity", "runtime");
    output.overflow_policy =
        scalar<std::string>(node, "overflow_policy", "runtime");
    output.frame_limit =
        nullableScalar<std::uint64_t>(node, "frame_limit", "runtime");
    const auto stage =
        nullableScalar<std::string>(node, "capture_stage", "runtime");
    if (stage) {
        output.capture_stage = std::filesystem::path{*stage};
        if (!output.capture_stage->is_absolute()) {
            throw std::runtime_error("runtime.capture_stage must be absolute");
        }
    }
}

void parseCamera(const YAML::Node& node, CameraSettings& output) {
    exactKeys(
        node,
        {"left_id", "right_id", "left_serial", "right_serial", "left_index",
         "right_index", "width", "height", "exposure_us", "gain_db",
         "image_node_num", "embedded_info_clear_rows", "trigger_source",
         "trigger_activation", "exposure_timestamp_source", "shutter_type",
         "exposure_delay_status", "exposure_delay_ns",
         "exposure_delay_uncertainty_ns"},
        "camera");
    output.left_id = scalar<std::string>(node, "left_id", "camera");
    output.right_id = scalar<std::string>(node, "right_id", "camera");
    output.left_serial = scalar<std::string>(node, "left_serial", "camera");
    output.right_serial = scalar<std::string>(node, "right_serial", "camera");
    output.left_index = scalar<int>(node, "left_index", "camera");
    output.right_index = scalar<int>(node, "right_index", "camera");
    output.width = scalar<int>(node, "width", "camera");
    output.height = scalar<int>(node, "height", "camera");
    output.exposure_us = scalar<double>(node, "exposure_us", "camera");
    output.gain_db = scalar<double>(node, "gain_db", "camera");
    output.image_node_num = scalar<int>(node, "image_node_num", "camera");
    output.embedded_info_clear_rows =
        scalar<int>(node, "embedded_info_clear_rows", "camera");
    output.trigger_source =
        scalar<std::string>(node, "trigger_source", "camera");
    output.trigger_activation =
        scalar<std::string>(node, "trigger_activation", "camera");
    output.exposure_timestamp_source =
        scalar<std::string>(node, "exposure_timestamp_source", "camera");
    output.shutter_type =
        scalar<std::string>(node, "shutter_type", "camera");
    output.exposure_delay_status =
        scalar<std::string>(node, "exposure_delay_status", "camera");
    output.exposure_delay_ns =
        scalar<std::uint64_t>(node, "exposure_delay_ns", "camera");
    output.exposure_delay_uncertainty_ns = nullableScalar<std::uint64_t>(
        node, "exposure_delay_uncertainty_ns", "camera");
}

void parseClocks(const YAML::Node& node, ClockSettings& output) {
    exactKeys(
        node,
        {"reference", "left_device", "right_device", "capture_realtime",
         "host_monotonic"},
        "clocks");
    output.reference = scalar<std::string>(node, "reference", "clocks");
    output.left_device = scalar<std::string>(node, "left_device", "clocks");
    output.right_device = scalar<std::string>(node, "right_device", "clocks");
    output.capture_realtime =
        scalar<std::string>(node, "capture_realtime", "clocks");
    output.host_monotonic =
        scalar<std::string>(node, "host_monotonic", "clocks");
}

void parseTrigger(const YAML::Node& node, TriggerSettings& output) {
    exactKeys(
        node,
        {"gpio_chip", "gpio_line", "frequency_hz", "duty_cycle"},
        "trigger");
    output.gpio_chip = scalar<std::string>(node, "gpio_chip", "trigger");
    output.gpio_line = scalar<unsigned int>(node, "gpio_line", "trigger");
    output.frequency_hz = scalar<double>(node, "frequency_hz", "trigger");
    output.duty_cycle = scalar<double>(node, "duty_cycle", "trigger");
}

void parseCalibration(const YAML::Node& node, CalibrationSettings& output) {
    exactKeys(
        node,
        {"file", "rig_frame", "output_width", "output_height"},
        "calibration");
    output.file = absolutePath(node, "file", "calibration");
    output.rig_frame =
        scalar<std::string>(node, "rig_frame", "calibration");
    output.output_width = scalar<int>(node, "output_width", "calibration");
    output.output_height = scalar<int>(node, "output_height", "calibration");
}

void parseDetector(const YAML::Node& node, DetectorSettings& output) {
    exactKeys(
        node,
        {"engine_file", "engine_sha256", "pipeline_id", "input_size",
         "output_contract", "class_count",
         "confidence_threshold", "nms_iou_threshold", "max_detections",
         "class_id", "use_dla", "dla_core"},
        "detector");
    output.engine_file = absolutePath(node, "engine_file", "detector");
    output.engine_sha256 =
        scalar<std::string>(node, "engine_sha256", "detector");
    output.pipeline_id = scalar<std::string>(node, "pipeline_id", "detector");
    output.input_size = scalar<int>(node, "input_size", "detector");
    output.output_contract =
        scalar<std::string>(node, "output_contract", "detector");
    output.class_count = scalar<int>(node, "class_count", "detector");
    output.confidence_threshold =
        scalar<double>(node, "confidence_threshold", "detector");
    output.nms_iou_threshold =
        scalar<double>(node, "nms_iou_threshold", "detector");
    output.max_detections =
        scalar<std::size_t>(node, "max_detections", "detector");
    output.class_id = scalar<int>(node, "class_id", "detector");
    output.use_dla = scalar<bool>(node, "use_dla", "detector");
    output.dla_core = scalar<int>(node, "dla_core", "detector");
}

void parseAssociation(
    const YAML::Node& node,
    AssociationSettings& output) {
    exactKeys(
        node,
        {"max_capture_delta_ns", "epipolar_tolerance_px",
         "adaptive_epipolar_height_ratio", "max_size_ratio",
         "min_shifted_iou", "max_disparity_px", "ball_diameter_m",
         "bbox_scale", "bbox_disparity_consistency_ratio",
         "bbox_disparity_consistency_min_px", "bbox_disparity_penalty_scale"},
        "association");
    output.max_capture_delta_ns =
        scalar<std::int64_t>(node, "max_capture_delta_ns", "association");
    output.epipolar_tolerance_px =
        scalar<double>(node, "epipolar_tolerance_px", "association");
    output.adaptive_epipolar_height_ratio = scalar<double>(
        node, "adaptive_epipolar_height_ratio", "association");
    output.max_size_ratio =
        scalar<double>(node, "max_size_ratio", "association");
    output.min_shifted_iou =
        scalar<double>(node, "min_shifted_iou", "association");
    output.max_disparity_px =
        scalar<std::uint32_t>(node, "max_disparity_px", "association");
    output.ball_diameter_m =
        scalar<double>(node, "ball_diameter_m", "association");
    output.bbox_scale = scalar<double>(node, "bbox_scale", "association");
    output.bbox_disparity_consistency_ratio = scalar<double>(
        node, "bbox_disparity_consistency_ratio", "association");
    output.bbox_disparity_consistency_min_px = scalar<double>(
        node, "bbox_disparity_consistency_min_px", "association");
    output.bbox_disparity_penalty_scale = scalar<double>(
        node, "bbox_disparity_penalty_scale", "association");
}

void parseDepth(const YAML::Node& node, DepthSettings& output) {
    exactKeys(
        node,
        {"method_order", "patch_radius", "search_radius_px", "max_points",
         "min_points", "min_confidence", "max_disparity_delta_px",
         "max_disparity_delta_ratio", "max_depth_delta_m",
         "max_disparity_stddev_px", "circle_max_roi_pixels", "min_depth_m",
         "max_depth_m"},
        "depth");
    output.method_order = stringSequence(node, "method_order", "depth");
    output.patch_radius = scalar<std::uint32_t>(node, "patch_radius", "depth");
    output.search_radius_px =
        scalar<std::uint32_t>(node, "search_radius_px", "depth");
    output.max_points = scalar<std::uint32_t>(node, "max_points", "depth");
    output.min_points = scalar<std::uint32_t>(node, "min_points", "depth");
    output.min_confidence = scalar<double>(node, "min_confidence", "depth");
    output.max_disparity_delta_px =
        scalar<double>(node, "max_disparity_delta_px", "depth");
    output.max_disparity_delta_ratio =
        scalar<double>(node, "max_disparity_delta_ratio", "depth");
    output.max_depth_delta_m =
        scalar<double>(node, "max_depth_delta_m", "depth");
    output.max_disparity_stddev_px =
        scalar<double>(node, "max_disparity_stddev_px", "depth");
    output.circle_max_roi_pixels =
        scalar<std::uint32_t>(node, "circle_max_roi_pixels", "depth");
    output.min_depth_m = scalar<double>(node, "min_depth_m", "depth");
    output.max_depth_m = scalar<double>(node, "max_depth_m", "depth");
}

void parsePublisher(const YAML::Node& node, PublisherSettings& output) {
    exactKeys(
        node,
        {"enabled", "topic", "frame_id", "source_epoch_file"},
        "publisher");
    output.enabled = scalar<bool>(node, "enabled", "publisher");
    output.topic = scalar<std::string>(node, "topic", "publisher");
    output.frame_id = scalar<std::string>(node, "frame_id", "publisher");
    output.source_epoch_file =
        absolutePath(node, "source_epoch_file", "publisher");
}

void validate(const MainlineRuntimeConfig& value) {
    if (value.profile.empty() || !sha256(value.config_sha256)) {
        throw std::runtime_error("profile/config_sha256 is invalid");
    }
    const bool capture = value.runtime.mode == "capture";
    if (!capture && value.runtime.mode != "runtime") {
        throw std::runtime_error("runtime.mode must be runtime or capture");
    }
    if (!finite(value.runtime.nominal_fps) || value.runtime.nominal_fps <= 0.0 ||
        value.runtime.nominal_fps >
            static_cast<double>(std::numeric_limits<int>::max()) ||
        std::floor(value.runtime.nominal_fps) != value.runtime.nominal_fps ||
        value.runtime.source_timeout_ms == 0 ||
        value.runtime.max_consecutive_timeouts > 1000U ||
        value.runtime.queue_capacity == 0 ||
        value.runtime.queue_capacity > 4096U ||
        (value.runtime.overflow_policy != "block" &&
         value.runtime.overflow_policy != "drop_oldest")) {
        throw std::runtime_error("runtime timing/queue configuration is invalid");
    }
    if (capture != value.runtime.frame_limit.has_value() ||
        capture != value.runtime.capture_stage.has_value() ||
        (capture && (*value.runtime.frame_limit == 0 ||
                     value.runtime.overflow_policy != "block"))) {
        throw std::runtime_error(
            "capture requires finite frame_limit, stage, and blocking policy");
    }
    if (value.camera.left_id.empty() || value.camera.right_id.empty() ||
        value.camera.left_id == value.camera.right_id ||
        value.camera.left_serial.empty() || value.camera.right_serial.empty() ||
        value.camera.left_serial == value.camera.right_serial ||
        value.camera.left_index < 0 || value.camera.right_index < 0 ||
        value.camera.left_index == value.camera.right_index ||
        value.camera.width <= 0 || value.camera.height <= 0 ||
        !finite(value.camera.exposure_us) || value.camera.exposure_us <= 0.0 ||
        !finite(value.camera.gain_db) ||
        value.camera.exposure_us >
            static_cast<double>(std::numeric_limits<float>::max()) ||
        std::abs(value.camera.gain_db) >
            static_cast<double>(std::numeric_limits<float>::max()) ||
        value.camera.image_node_num < 2 ||
        value.camera.image_node_num > 16 ||
        value.camera.embedded_info_clear_rows <= 0 ||
        value.camera.embedded_info_clear_rows > value.camera.height ||
        value.camera.trigger_source != "Line0" ||
        value.camera.trigger_activation != "RisingEdge" ||
        value.camera.exposure_timestamp_source != "nx_gpio_rising_edge" ||
        value.camera.shutter_type != "global") {
        throw std::runtime_error("camera configuration is invalid");
    }
    const bool measured_delay =
        value.camera.exposure_delay_status == "measured" &&
        value.camera.exposure_delay_uncertainty_ns.has_value();
    const bool unmeasured_delay =
        value.camera.exposure_delay_status == "assumed_zero_unmeasured" &&
        value.camera.exposure_delay_ns == 0 &&
        !value.camera.exposure_delay_uncertainty_ns.has_value();
    if (!measured_delay && !unmeasured_delay) {
        throw std::runtime_error("camera exposure delay semantics are invalid");
    }
    const KeySet clocks{
        value.clocks.reference,
        value.clocks.left_device,
        value.clocks.right_device,
        value.clocks.capture_realtime,
        value.clocks.host_monotonic,
    };
    if (clocks.size() != 5 || clocks.count("") != 0) {
        throw std::runtime_error("clock identities must be non-empty and unique");
    }
    if (value.trigger.gpio_chip.empty() ||
        !finite(value.trigger.frequency_hz) ||
        !finite(value.trigger.duty_cycle) || value.trigger.frequency_hz <= 0.0 ||
        value.trigger.duty_cycle <= 0.0 || value.trigger.duty_cycle >= 100.0 ||
        std::abs(value.trigger.frequency_hz - value.runtime.nominal_fps) > 1e-9) {
        throw std::runtime_error("trigger configuration is invalid");
    }
    if (value.calibration.rig_frame.empty() ||
        value.calibration.output_width != value.camera.width ||
        value.calibration.output_height != value.camera.height) {
        throw std::runtime_error("calibration output geometry is invalid");
    }
    if (!sha256(value.detector.engine_sha256) ||
        value.detector.pipeline_id.empty() || value.detector.input_size <= 0 ||
        (value.detector.output_contract != "yolo_pre_nms" &&
         value.detector.output_contract != "yolo_dfl") ||
        value.detector.class_count <= 0 ||
        !finite(value.detector.confidence_threshold) ||
        value.detector.confidence_threshold < 0.0 ||
        value.detector.confidence_threshold > 1.0 ||
        !finite(value.detector.nms_iou_threshold) ||
        value.detector.nms_iou_threshold < 0.0 ||
        value.detector.nms_iou_threshold > 1.0 ||
        value.detector.max_detections == 0 || value.detector.class_id < 0 ||
        value.detector.class_id >= value.detector.class_count ||
        value.detector.dla_core < 0) {
        throw std::runtime_error("detector configuration is invalid");
    }
    const auto& association = value.association;
    if (association.max_capture_delta_ns < 0 ||
        !finite(association.epipolar_tolerance_px) ||
        association.epipolar_tolerance_px < 0.0 ||
        !finite(association.adaptive_epipolar_height_ratio) ||
        association.adaptive_epipolar_height_ratio < 0.0 ||
        !finite(association.max_size_ratio) || association.max_size_ratio < 1.0 ||
        !finite(association.min_shifted_iou) ||
        association.min_shifted_iou < 0.0 || association.min_shifted_iou > 1.0 ||
        association.max_disparity_px == 0 ||
        !finite(association.ball_diameter_m) || association.ball_diameter_m <= 0.0 ||
        !finite(association.bbox_scale) || association.bbox_scale <= 0.0 ||
        !finite(association.bbox_disparity_consistency_ratio) ||
        association.bbox_disparity_consistency_ratio <= 0.0 ||
        !finite(association.bbox_disparity_consistency_min_px) ||
        association.bbox_disparity_consistency_min_px <= 0.0 ||
        !finite(association.bbox_disparity_penalty_scale) ||
        association.bbox_disparity_penalty_scale < 0.0) {
        throw std::runtime_error("association configuration is invalid");
    }
    const KeySet supported_methods{
        "circle_center",
        "radial_center", "edge_pair_center", "edge_centroid",
        "bbox_center", "bbox_edges"};
    KeySet configured_methods;
    for (const auto& method : value.depth.method_order) {
        if (supported_methods.count(method) == 0 ||
            !configured_methods.insert(method).second) {
            throw std::runtime_error("depth.method_order is unsupported or repeated");
        }
    }
    const auto& depth = value.depth;
    if (depth.method_order.empty() || depth.patch_radius == 0 ||
        depth.search_radius_px == 0 || depth.max_points == 0 ||
        depth.min_points == 0 || depth.min_points > depth.max_points ||
        !finite(depth.min_confidence) || depth.min_confidence < 0.0 ||
        depth.min_confidence > 1.0 ||
        !finite(depth.max_disparity_delta_px) ||
        depth.max_disparity_delta_px <= 0.0 ||
        !finite(depth.max_disparity_delta_ratio) ||
        depth.max_disparity_delta_ratio < 0.0 ||
        !finite(depth.max_depth_delta_m) || depth.max_depth_delta_m <= 0.0 ||
        !finite(depth.max_disparity_stddev_px) ||
        depth.max_disparity_stddev_px <= 0.0 ||
        depth.circle_max_roi_pixels == 0 || !finite(depth.min_depth_m) ||
        !finite(depth.max_depth_m) || depth.min_depth_m < 0.0 ||
        depth.max_depth_m <= depth.min_depth_m) {
        throw std::runtime_error("depth configuration is invalid");
    }
    if (capture && value.publisher.enabled) {
        throw std::runtime_error("capture mode cannot publish ROS2 observations");
    }
    if (value.publisher.enabled &&
        (value.publisher.topic.empty() || value.publisher.frame_id.empty() ||
         value.publisher.frame_id != value.calibration.rig_frame)) {
        throw std::runtime_error("publisher configuration is invalid");
    }
    if (value.metrics_interval_frames == 0) {
        throw std::runtime_error("diagnostics interval must be positive");
    }
}

}  // namespace

bool loadRuntimeConfig(
    const std::filesystem::path& path,
    MainlineRuntimeConfig& output,
    std::string& error) {
    output = {};
    error.clear();
    try {
        const YAML::Node root = YAML::LoadFile(path.string());
        exactKeys(
            root,
            {"schema_version", "profile", "config_sha256", "runtime",
             "camera", "clocks", "trigger", "calibration", "detector",
             "association", "depth", "publisher", "diagnostics"},
            "root");
        if (scalar<int>(root, "schema_version", "root") != 2) {
            throw std::runtime_error("unsupported mainline config schema_version");
        }
        output.profile = scalar<std::string>(root, "profile", "root");
        output.config_sha256 =
            scalar<std::string>(root, "config_sha256", "root");
        parseRuntime(root["runtime"], output.runtime);
        parseCamera(root["camera"], output.camera);
        parseClocks(root["clocks"], output.clocks);
        parseTrigger(root["trigger"], output.trigger);
        parseCalibration(root["calibration"], output.calibration);
        parseDetector(root["detector"], output.detector);
        parseAssociation(root["association"], output.association);
        parseDepth(root["depth"], output.depth);
        parsePublisher(root["publisher"], output.publisher);
        const YAML::Node diagnostics = root["diagnostics"];
        exactKeys(diagnostics, {"metrics_interval_frames"}, "diagnostics");
        output.metrics_interval_frames = scalar<std::size_t>(
            diagnostics, "metrics_interval_frames", "diagnostics");
        validate(output);
        return true;
    } catch (const std::exception& exception) {
        output = {};
        error = exception.what();
        return false;
    } catch (...) {
        output = {};
        error = "unknown error while parsing mainline runtime configuration";
        return false;
    }
}

}  // namespace volleyball::apps::nx_mainline
