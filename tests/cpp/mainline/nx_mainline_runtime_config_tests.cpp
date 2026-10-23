#include "runtime_config.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>

namespace app = volleyball::apps::nx_mainline;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

std::string validCaptureYaml() {
    return R"YAML(schema_version: 2
profile: free-flight
config_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
runtime:
  mode: capture
  nominal_fps: 90
  source_timeout_ms: 1000
  max_consecutive_timeouts: 0
  queue_capacity: 8
  overflow_policy: block
  frame_limit: 90
  capture_stage: /tmp/volleyball-stage
camera:
  left_id: left
  right_id: right
  left_serial: serial-left
  right_serial: serial-right
  left_index: 0
  right_index: 1
  width: 1440
  height: 1080
  exposure_us: 9867
  gain_db: 11.9906
  image_node_num: 3
  embedded_info_clear_rows: 2
  trigger_source: Line0
  trigger_activation: RisingEdge
  exposure_timestamp_source: nx_gpio_rising_edge
  shutter_type: global
  exposure_delay_status: assumed_zero_unmeasured
  exposure_delay_ns: 0
  exposure_delay_uncertainty_ns: null
clocks:
  reference: hik-shared
  left_device: left-device
  right_device: right-device
  capture_realtime: nx-realtime
  host_monotonic: nx-monotonic
trigger:
  gpio_chip: gpiochip2
  gpio_line: 7
  frequency_hz: 90
  duty_cycle: 50
calibration:
  file: /tmp/stereo.yaml
  rig_frame: left-optical
  output_width: 1440
  output_height: 1080
detector:
  engine_file: /tmp/model.engine
  engine_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  pipeline_id: detector
  input_size: 640
  output_contract: yolo_pre_nms
  class_count: 1
  confidence_threshold: 0.3
  nms_iou_threshold: 0.4
  max_detections: 10
  class_id: 0
  use_dla: false
  dla_core: 0
association:
  max_capture_delta_ns: 100000
  epipolar_tolerance_px: 12
  adaptive_epipolar_height_ratio: 0.35
  max_size_ratio: 2
  min_shifted_iou: 0.05
  max_disparity_px: 2048
  ball_diameter_m: 0.21
  bbox_scale: 0.95
  bbox_disparity_consistency_ratio: 0.3
  bbox_disparity_consistency_min_px: 45
  bbox_disparity_penalty_scale: 0.75
depth:
  method_order: [roi_multi_point, bbox_center]
  patch_radius: 5
  search_radius_px: 8
  max_points: 12
  min_points: 4
  min_confidence: 0.25
  max_disparity_delta_px: 2
  max_disparity_delta_ratio: 0.03
  max_depth_delta_m: 0.5
  max_disparity_stddev_px: 1
  circle_max_roi_pixels: 18000
  min_depth_m: 0.8
  max_depth_m: 15
publisher:
  enabled: false
  topic: ""
  frame_id: left-optical
  source_epoch_file: /tmp/source-epoch
diagnostics:
  metrics_interval_frames: 90
)YAML";
}

void replaceOnce(
    std::string& value,
    const std::string& before,
    const std::string& after) {
    const auto position = value.find(before);
    if (position == std::string::npos) {
        ++failures;
        std::cerr << "FAILED: fixture token not found: " << before << '\n';
        return;
    }
    value.replace(position, before.size(), after);
}

std::filesystem::path writeFixture(const std::string& content, int sequence) {
    const auto path = std::filesystem::temp_directory_path() /
        ("volleyball-mainline-config-" + std::to_string(::getpid()) + "-" +
         std::to_string(sequence) + ".yaml");
    std::ofstream output(path);
    output << content;
    output.close();
    return path;
}

bool load(const std::string& content, int sequence, std::string& error) {
    const auto path = writeFixture(content, sequence);
    app::MainlineRuntimeConfig config;
    const bool result = app::loadRuntimeConfig(path, config, error);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return result;
}

void testValidCaptureConfig() {
    const auto path = writeFixture(validCaptureYaml(), 1);
    app::MainlineRuntimeConfig config;
    std::string error;
    check(app::loadRuntimeConfig(path, config, error),
          "strict parser must accept a complete capture configuration");
    check(error.empty() && config.runtime.frame_limit == 90 &&
              config.runtime.capture_stage &&
              config.detector.output_contract == "yolo_pre_nms" &&
              config.detector.class_count == 1 &&
              config.association.max_capture_delta_ns == 100000,
          "accepted config must preserve typed launch facts");
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

void testValidRuntimeConfig() {
    std::string yaml = validCaptureYaml();
    replaceOnce(yaml, "profile: free-flight", "profile: nx-rdk");
    replaceOnce(yaml, "mode: capture", "mode: runtime");
    replaceOnce(yaml, "overflow_policy: block", "overflow_policy: drop_oldest");
    replaceOnce(yaml, "frame_limit: 90", "frame_limit: null");
    replaceOnce(
        yaml,
        "capture_stage: /tmp/volleyball-stage",
        "capture_stage: null");
    replaceOnce(yaml, "enabled: false", "enabled: true");
    replaceOnce(yaml, "topic: \"\"", "topic: /nx/ball/observation");
    std::string error;
    check(load(yaml, 2, error),
          "strict parser must accept runtime drop-oldest publication mode");
}

void testRejectsUnknownAndIllTypedFields() {
    std::string error;
    check(!load(validCaptureYaml() + "unexpected: true\n", 3, error) &&
              error.find("unknown key") != std::string::npos,
          "unknown root keys must fail closed");

    std::string fractional = validCaptureYaml();
    replaceOnce(
        fractional,
        "max_capture_delta_ns: 100000",
        "max_capture_delta_ns: 100000.5");
    error.clear();
    check(!load(fractional, 4, error),
          "fractional nanoseconds must not enter an integer timestamp field");

    std::string unhashed = validCaptureYaml();
    replaceOnce(
        unhashed,
        "engine_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "engine_sha256: null");
    error.clear();
    check(!load(unhashed, 5, error),
          "an unpinned TensorRT engine must fail before hardware startup");

    std::string out_of_range_class = validCaptureYaml();
    replaceOnce(out_of_range_class, "class_id: 0", "class_id: 1");
    error.clear();
    check(!load(out_of_range_class, 6, error),
          "target class id outside the manifest class table must fail closed");

    std::string unsupported_output = validCaptureYaml();
    replaceOnce(
        unsupported_output,
        "output_contract: yolo_pre_nms",
        "output_contract: yolo_post_nms");
    error.clear();
    check(!load(unsupported_output, 7, error),
          "output contracts without a statically auditable class table must fail");
}

}  // namespace

int main() {
    testValidCaptureConfig();
    testValidRuntimeConfig();
    testRejectsUnknownAndIllTypedFields();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "NX mainline runtime config tests passed\n";
    return EXIT_SUCCESS;
}
