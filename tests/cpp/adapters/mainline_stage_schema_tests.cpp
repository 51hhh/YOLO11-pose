#include "volleyball/adapters/capture_stage/mainline_stage_schema.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace capture_stage = volleyball::adapters::capture_stage;
namespace core = volleyball::core;
namespace ports = volleyball::ports;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

std::string lineFrom(std::ostringstream& output) {
    std::string line = output.str();
    if (!line.empty() && line.back() == '\n') line.pop_back();
    return line;
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = line.find(',', start);
        if (comma == std::string::npos) {
            fields.push_back(line.substr(start));
            return fields;
        }
        fields.push_back(line.substr(start, comma - start));
        start = comma + 1;
    }
}

class HostImage final : public ports::ImageBuffer {
public:
    [[nodiscard]] ports::MemoryKind memoryKind() const noexcept override {
        return ports::MemoryKind::host;
    }
    [[nodiscard]] std::uint32_t width() const noexcept override { return 4; }
    [[nodiscard]] std::uint32_t height() const noexcept override { return 3; }
    [[nodiscard]] std::size_t strideBytes() const noexcept override { return 8; }
    [[nodiscard]] core::PixelFormat pixelFormat() const noexcept override {
        return core::PixelFormat::bayer8;
    }
    [[nodiscard]] std::uintptr_t nativeHandle() const noexcept override {
        return 1;
    }
    [[nodiscard]] std::size_t sizeBytes() const noexcept override { return 24; }
};

core::Timestamp timestamp(std::int64_t value, const char* clock) {
    return {value, core::ClockDomainId{clock}};
}

ports::CapturedFrame richFrame() {
    ports::CapturedFrame output;
    auto& value = output.metadata;
    value.camera_id = core::CameraId{"left-camera"};
    value.role = core::CameraRole::left;
    value.frame_index = 42;
    value.hardware_timestamp = timestamp(100, "left-device");
    value.exposure_start = timestamp(200, "nx-realtime");
    value.capture_timestamp = timestamp(250, "nx-realtime");
    value.receive_timestamp = timestamp(300, "nx-realtime");
    value.inference_done_timestamp = timestamp(400, "nx-realtime");
    value.exposure_us = 100.5;
    value.gain_db = 3.25;
    value.roi = {1, 2, 4, 3};
    value.pixel_format = core::PixelFormat::bayer8;
    auto& evidence = value.acquisition;
    evidence.receive_monotonic_timestamp = timestamp(150, "nx-monotonic");
    evidence.sdk_host_timestamp_raw = 500;
    evidence.sensor_frame_number = 41;
    evidence.sensor_frame_counter = 42;
    evidence.sensor_trigger_index = 43;
    evidence.stereo_hardware_timestamp_residual_ns = -4;
    evidence.trigger_sequence = 42;
    evidence.trigger_scheduled_monotonic = timestamp(120, "nx-monotonic");
    evidence.trigger_edge_monotonic = timestamp(125, "nx-monotonic");
    evidence.trigger_edge_realtime = timestamp(195, "nx-realtime");
    evidence.trigger_write_uncertainty_ns = 10;
    evidence.capture_timestamp_uncertainty_ns = 20;
    evidence.capture_timestamp_mapping_valid = true;
    output.image = std::make_shared<HostImage>();
    return output;
}

void testHeadersAndFrame() {
    check(capture_stage::kMainlineStageSchemaVersion == 6,
          "mainline staging version must be explicit");
    std::ostringstream header;
    capture_stage::writeMainlineStageFrameHeader(header);
    check(splitCsv(lineFrom(header)).size() ==
              capture_stage::kMainlineStageFrameColumns.size(),
          "frame header width must match its declaration");

    std::ostringstream output;
    const auto frame = richFrame();
    check(capture_stage::writeMainlineStageFrameRow(
              output, frame, "raw/left/00000042.pgm"),
          "complete frame evidence must serialize");
    const auto fields = splitCsv(lineFrom(output));
    check(fields.size() == capture_stage::kMainlineStageFrameColumns.size(),
          "frame row width must match its header");
    check(fields[0] == "left-camera" && fields[1] == "left" &&
              fields[2] == "42",
          "frame identity must retain camera, role, and index");
    check(fields[3] == "100" && fields[4] == "left-device" &&
              fields[7] == "250" && fields[8] == "nx-realtime",
          "device and mapped capture clocks must remain separate");
    check(fields[22] == "150" && fields[23] == "nx-monotonic" &&
              fields[29] == "42" && fields[38] == "true",
          "acquisition and trigger evidence must keep named columns");
    check(fields[39] == "4" && fields[41] == "8" &&
              fields[44] == "host" && fields[45] == "raw/left/00000042.pgm",
          "image layout and raw reference must be bound to the frame");

    std::ostringstream invalid;
    check(!capture_stage::writeMainlineStageFrameRow(
              invalid, frame, "raw,left.pgm"),
          "unsafe raw path token must be rejected");
}

void testDetection() {
    std::ostringstream header;
    capture_stage::writeMainlineStageDetectionHeader(header);
    check(splitCsv(lineFrom(header)).size() ==
              capture_stage::kMainlineStageDetectionColumns.size(),
          "detection header width must match its declaration");

    core::Detection2D value;
    value.frame = {
        core::CameraId{"left-camera"}, core::CameraRole::left, 42};
    value.candidate_id = 7;
    value.center_px = {10.25, 20.5};
    value.radius_px = 4.5;
    value.bounding_box = {5.0, 15.0, 10.0, 12.0};
    value.pixel_velocity_pxps = core::Vec2{1.0, 2.0};
    value.score = 0.75;
    value.covariance_px2 = core::Covariance2D{{1.0, 0.25, 2.0}};
    value.selected = true;
    value.source_pipeline = core::PipelineId{"detector-a"};
    std::ostringstream output;
    check(capture_stage::writeMainlineStageDetectionRow(output, value),
          "detection with optional evidence must serialize");
    const auto fields = splitCsv(lineFrom(output));
    check(fields.size() ==
              capture_stage::kMainlineStageDetectionColumns.size(),
          "detection row width must match its header");
    check(fields[6] == "4.5" && fields[11] == "1" &&
              fields[17] == "true" && fields[18] == "detector-a",
          "detection optional evidence must retain its columns");
}

void testAssociationAndObservation() {
    core::StereoAssociationAttempt association;
    association.pair_id = 9;
    association.left_frame_index = 42;
    association.right_frame_index = 42;
    association.left_candidate_id = 7;
    association.right_candidate_id = 8;
    association.left_capture = timestamp(250, "nx-realtime");
    association.right_capture = timestamp(260, "nx-realtime");
    association.delta_t_lr_ns = 10;
    association.disparity_px = 30.0;
    association.epipolar_error_px = 0.5;
    association.decision = core::AssociationDecision::accepted;
    association.reject_reason = core::AssociationRejectReason::none;
    std::ostringstream association_output;
    check(capture_stage::writeMainlineStageAssociationRow(
              association_output, association),
          "accepted association must serialize independently of 3D");
    const auto association_fields = splitCsv(lineFrom(association_output));
    check(association_fields.size() ==
              capture_stage::kMainlineStageAssociationColumns.size(),
          "association row width must match its header");
    check(association_fields[12] == "accepted" &&
              association_fields[13].empty(),
          "accepted association must have no rejection token");

    ports::EstimatedDirectObservation observation;
    observation.measurement.observation_id = 3;
    observation.measurement.source_pair_id = 9;
    observation.measurement.capture_reference = timestamp(255, "nx-realtime");
    observation.measurement.coordinate_frame =
        core::CoordinateFrameId{"stereo-rig"};
    observation.measurement.position_m = {1.0, 2.0, 3.0};
    observation.measurement.covariance_m2 =
        core::Covariance3D{{1.0, 0.1, 0.2, 2.0, 0.3, 3.0}};
    observation.measurement.confidence = 0.9;
    observation.measurement.reprojection_error_px = 0.5;
    observation.evidence.method = core::StereoDepthMethodId{"circle_center"};
    observation.evidence.disparity_px = 30.25;
    observation.evidence.disparity_stddev_px = 0.2;
    observation.evidence.support_points = 8;
    observation.evidence.left_anchor_px = core::Vec2{710.25, 490.5};
    std::ostringstream observation_output;
    check(capture_stage::writeMainlineStageObservationRow(
              observation_output, observation),
          "direct observation must serialize without merging into association");
    const auto observation_fields = splitCsv(lineFrom(observation_output));
    check(observation_fields.size() ==
              capture_stage::kMainlineStageObservationColumns.size(),
          "observation row width must match its header");
    check(observation_fields[1] == "9" && observation_fields[4] == "stereo-rig" &&
              observation_fields[14] == "0.90000000000000002" &&
              observation_fields[16] == "circle_center" &&
              observation_fields[17] == "30.25" &&
              observation_fields[19] == "8",
          "observation provenance, precision, and method evidence must remain");

    std::ostringstream measurement_header;
    capture_stage::writeStereoMeasurementHeader(measurement_header);
    check(splitCsv(lineFrom(measurement_header)).size() ==
              capture_stage::kStereoMeasurementColumns.size(),
          "stereo measurement header width must match its declaration");
    std::ostringstream measurement_output;
    check(capture_stage::writeStereoMeasurementRow(
              measurement_output,
              observation.measurement.source_pair_id,
              observation.evidence),
          "stereo measurement inputs must serialize independently of XYZ");
    const auto measurement_fields = splitCsv(lineFrom(measurement_output));
    check(measurement_fields.size() ==
              capture_stage::kStereoMeasurementColumns.size() &&
              measurement_fields[0] ==
                  capture_stage::kStereoMeasurementSchemaVersion &&
              measurement_fields[1] == "9" &&
              measurement_fields[2] == "circle_center" &&
              measurement_fields[3] == "710.25" &&
              measurement_fields[4] == "490.5" &&
              measurement_fields[5] == "30.25",
          "stereo measurement row must preserve versioned UVD inputs");

    auto missing_anchor = observation.evidence;
    missing_anchor.left_anchor_px.reset();
    std::ostringstream invalid_measurement;
    check(!capture_stage::writeStereoMeasurementRow(
              invalid_measurement,
              observation.measurement.source_pair_id,
              missing_anchor),
          "stereo measurement row must reject a missing input anchor");
}

}  // namespace

int main() {
    testHeadersAndFrame();
    testDetection();
    testAssociationAndObservation();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "mainline capture staging schema tests passed\n";
    return EXIT_SUCCESS;
}
