#include "volleyball/adapters/ros2/observation_message.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace adapter = volleyball::adapters::ros2;
namespace core = volleyball::core;
namespace ports = volleyball::ports;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

core::Timestamp timestamp(std::int64_t nanoseconds) {
    return {nanoseconds, core::ClockDomainId{"nx-realtime"}};
}

core::CameraFrameRecord frame(
    core::CameraRole role,
    const char* camera_id,
    std::int64_t capture_ns,
    std::int64_t hardware_ns) {
    core::CameraFrameRecord output;
    output.camera_id = core::CameraId{camera_id};
    output.role = role;
    output.frame_index = 42;
    output.hardware_timestamp = {
        hardware_ns,
        core::ClockDomainId{
            role == core::CameraRole::left ? "left-device" : "right-device"},
    };
    output.capture_timestamp = timestamp(capture_ns);
    output.roi = {0, 0, 1440, 1080};
    output.pixel_format = core::PixelFormat::bayer8;
    output.acquisition.stereo_hardware_timestamp_residual_ns = -4;
    return output;
}

core::Detection2D detection(
    core::CameraRole role,
    const char* camera_id,
    core::CandidateId candidate_id,
    double x,
    double score) {
    core::Detection2D output;
    output.frame = {core::CameraId{camera_id}, role, 42};
    output.candidate_id = candidate_id;
    output.center_px = {x, 500.0};
    output.bounding_box = {x - 20.0, 480.0, 40.0, 40.0};
    output.score = score;
    output.selected = true;
    output.source_pipeline = core::PipelineId{"yolo-left-right"};
    return output;
}

ports::ObservationPublication publication() {
    ports::ObservationPublication output;
    output.left_frame = frame(core::CameraRole::left, "left", 2'000'000'000, 100);
    output.right_frame =
        frame(core::CameraRole::right, "right", 2'000'000'002, 104);
    output.left_detection =
        detection(core::CameraRole::left, "left", 3, 700.0, 0.9);
    output.right_detection =
        detection(core::CameraRole::right, "right", 4, 650.0, 0.8);
    output.association.pair_id = 9;
    output.association.left_frame_index = 42;
    output.association.right_frame_index = 42;
    output.association.left_candidate_id = 3;
    output.association.right_candidate_id = 4;
    output.association.left_capture = output.left_frame.capture_timestamp;
    output.association.right_capture = output.right_frame.capture_timestamp;
    output.association.delta_t_lr_ns = 2;
    output.association.disparity_px = 50.0;
    output.association.epipolar_error_px = 0.0;
    output.association.decision = core::AssociationDecision::accepted;
    output.association.reject_reason = core::AssociationRejectReason::none;

    auto& measurement = output.observation;
    measurement.observation_id = 11;
    measurement.source_pair_id = 9;
    measurement.capture_reference = timestamp(2'000'000'001);
    measurement.coordinate_frame =
        core::CoordinateFrameId{"nx_left_rectified_optical_frame"};
    measurement.position_m = {0.1, 0.2, 5.0};
    measurement.covariance_m2 = core::Covariance3D{{
        0.01,
        0.0,
        0.0,
        0.02,
        0.0,
        0.04,
    }};
    measurement.confidence = 0.75;
    measurement.reprojection_error_px = 0.0;
    output.estimation.method = core::StereoDepthMethodId{"roi_multi_point"};
    output.estimation.disparity_px = 49.75;
    output.estimation.disparity_stddev_px = 0.2;
    output.estimation.support_points = 8;
    return output;
}

adapter::ObservationMessageConfig config() {
    return {
        "nx_left_rectified_optical_frame",
        core::ClockDomainId{"nx-realtime"},
        core::StereoDepthMethodId{"roi_multi_point"},
        17,
        0,
    };
}

void testAcceptedPublicationMapsWithoutInventingState() {
    const auto mapped = adapter::makeObservationMessage(publication(), config());
    check(static_cast<bool>(mapped), "valid direct observation must map");
    if (!mapped) return;
    const auto& value = *mapped.message;
    check(value.stamp_sec == 2 && value.stamp_nanosec == 1,
          "ROS stamp must use the exact stereo midpoint");
    check(value.header_frame_id == "nx_left_rectified_optical_frame" &&
              value.source_epoch == 17 && value.frame_index == 42,
          "message identity must come from config and frame evidence");
    check(value.track_id == 0,
          "direct observation publisher must not invent a track identity");
    check(std::abs(value.detection_confidence - 0.8) < 1.0e-12 &&
              std::abs(value.match_confidence - 0.75) < 1.0e-12,
          "detection and observation confidence must remain distinct");
    check(value.bbox_left_xyxy == std::array<double, 4>{680, 480, 720, 520} &&
              value.bbox_right_xyxy ==
                  std::array<double, 4>{630, 480, 670, 520},
          "bbox must convert from xywh to xyxy without approximation");
    check(value.position_covariance_m2 == std::array<double, 9>{
              0.01, 0.0, 0.0,
              0.0, 0.02, 0.0,
              0.0, 0.0, 0.04,
          },
          "full symmetric covariance must be preserved");
    check(std::abs(value.depth_sigma_m - 0.2) < 1.0e-12 &&
              value.depth_method_code == 2 &&
              !value.fallback_observation,
          "depth uncertainty and method must come from measured evidence");
    check(value.left_device_timestamp_ns == 100 &&
              value.right_device_timestamp_ns == 104 &&
              value.stereo_hardware_timestamp_residual_ns == -4,
          "device timing evidence must retain nanosecond semantics");
    check(!value.capture_timestamp_mapping_valid &&
              value.capture_timestamp_uncertainty_ns == 0,
          "unmeasured exposure delay must remain explicitly invalid");
}

void testFallbackAndMeasuredTimestampUncertaintyAreExplicit() {
    auto input = publication();
    input.estimation.method = core::StereoDepthMethodId{"circle_center"};
    input.left_frame.acquisition.capture_timestamp_mapping_valid = true;
    input.right_frame.acquisition.capture_timestamp_mapping_valid = true;
    input.left_frame.acquisition.capture_timestamp_uncertainty_ns = 12;
    input.right_frame.acquisition.capture_timestamp_uncertainty_ns = 15;
    const auto mapped = adapter::makeObservationMessage(input, config());
    check(static_cast<bool>(mapped), "known fallback method must map");
    if (!mapped) return;
    check(mapped.message->depth_method_code == 1 &&
              mapped.message->fallback_observation,
          "non-primary method must be marked as fallback");
    check(mapped.message->capture_timestamp_mapping_valid &&
              mapped.message->capture_timestamp_uncertainty_ns == 15,
          "one message must conservatively retain the larger camera uncertainty");
}

void testCrossReferenceAndCoordinateMismatchesAreRejected() {
    auto input = publication();
    input.right_detection.frame.frame_index = 41;
    auto mapped = adapter::makeObservationMessage(input, config());
    check(!mapped &&
              mapped.error == adapter::ObservationMessageError::inconsistent_reference,
          "candidate from another frame must be rejected");

    input = publication();
    input.observation.coordinate_frame =
        core::CoordinateFrameId{"renamed-without-transform"};
    mapped = adapter::makeObservationMessage(input, config());
    check(!mapped &&
              mapped.error == adapter::ObservationMessageError::inconsistent_reference,
          "ROS header cannot relabel an untransformed measurement");
}

void testMissingOrUnrepresentableEvidenceIsRejected() {
    auto input = publication();
    input.observation.covariance_m2.reset();
    auto mapped = adapter::makeObservationMessage(input, config());
    check(!mapped && mapped.error ==
              adapter::ObservationMessageError::missing_required_evidence,
          "publisher must not synthesize a missing covariance");

    input = publication();
    input.estimation.method = core::StereoDepthMethodId{"new_unknown_method"};
    mapped = adapter::makeObservationMessage(input, config());
    check(!mapped && mapped.error ==
              adapter::ObservationMessageError::unsupported_depth_method,
          "unknown method must not silently use depth code zero");

    input = publication();
    input.right_frame.acquisition.stereo_hardware_timestamp_residual_ns = -3;
    mapped = adapter::makeObservationMessage(input, config());
    check(!mapped && mapped.error ==
              adapter::ObservationMessageError::missing_required_evidence,
          "conflicting stereo residual evidence must be rejected");
}

void testDepthMethodCodebookIsFrozen() {
    const std::pair<const char*, std::uint8_t> cases[] = {
        {"circle_center", 1},
        {"roi_multi_point", 2},
        {"bbox_center", 3},
        {"detection_center", 3},
        {"center_patch", 4},
        {"edge_centroid", 5},
        {"bbox_edges", 6},
        {"epipolar_fallback", 7},
        {"radial_center", 8},
        {"edge_pair_center", 9},
        {"cuda_template_match", 21},
    };
    for (const auto& [name, expected] : cases) {
        const auto actual = adapter::depthMethodWireCode(
            core::StereoDepthMethodId{name});
        check(actual && *actual == expected,
              "depth method wire code changed unexpectedly");
    }
}

}  // namespace

int main() {
    testAcceptedPublicationMapsWithoutInventingState();
    testFallbackAndMeasuredTimestampUncertaintyAreExplicit();
    testCrossReferenceAndCoordinateMismatchesAreRejected();
    testMissingOrUnrepresentableEvidenceIsRejected();
    testDepthMethodCodebookIsFrozen();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "ROS2 observation message mapping tests passed\n";
    return EXIT_SUCCESS;
}
