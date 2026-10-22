#include "volleyball/adapters/ros2/observation_message.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <utility>

namespace volleyball::adapters::ros2 {
namespace {

ObservationMessageResult failure(
    ObservationMessageError error,
    std::string detail) {
    return {std::nullopt, error, std::move(detail)};
}

bool referencesFrame(
    const core::Detection2D& detection,
    const core::CameraFrameRecord& frame) noexcept {
    return detection.frame.camera_id == frame.camera_id &&
           detection.frame.role == frame.role &&
           detection.frame.frame_index == frame.frame_index;
}

std::int64_t integerMidpoint(
    std::int64_t left,
    std::int64_t right) noexcept {
    return left / 2 + right / 2 + (left % 2 + right % 2) / 2;
}

std::array<double, 4> bboxXyxy(const core::BoundingBox2D& box) noexcept {
    return {box.x, box.y, box.x + box.width, box.y + box.height};
}

std::array<double, 9> covarianceMatrix(
    const core::Covariance3D& covariance) noexcept {
    const auto& value = covariance.upper;
    return {
        value[0], value[1], value[2],
        value[1], value[3], value[4],
        value[2], value[4], value[5],
    };
}

bool sameOptionalResidual(
    const std::optional<std::int64_t>& left,
    const std::optional<std::int64_t>& right) noexcept {
    return left && right && *left == *right;
}

}  // namespace

std::optional<std::uint8_t> legacyDepthMethodCode(
    const core::StereoDepthMethodId& method) noexcept {
    const std::string_view value = method.value();
    if (value == "circle_center") return 1;
    if (value == "roi_multi_point") return 2;
    if (value == "bbox_center" || value == "detection_center") return 3;
    if (value == "center_patch") return 4;
    if (value == "edge_centroid") return 5;
    if (value == "bbox_edges") return 6;
    if (value == "epipolar_fallback") return 7;
    if (value == "radial_center") return 8;
    if (value == "edge_pair_center") return 9;
    if (value == "cuda_template_match") return 21;
    return std::nullopt;
}

ObservationMessageResult makeObservationMessage(
    const ports::ObservationPublication& publication,
    const ObservationMessageConfig& config) {
    if (config.header_frame_id.empty() ||
        config.ros_time_clock_domain.empty() ||
        config.primary_depth_method.empty() || config.source_epoch == 0 ||
        config.class_id < 0) {
        return failure(
            ObservationMessageError::invalid_config,
            "ROS observation mapping config is incomplete");
    }

    const auto& validated = publication.observation;
    const auto& measurement = validated.measurement;
    const auto& left = publication.left_frame;
    const auto& right = publication.right_frame;
    const auto& left_detection = publication.left_detection;
    const auto& right_detection = publication.right_detection;
    const auto& association = publication.association;
    const auto& estimation = publication.estimation;

    if (validated.status != core::ValidationStatus::accepted ||
        !validated.reasons.empty() || !core::validate(measurement) ||
        !core::validate(left) || !core::validate(right) ||
        !core::validate(left_detection) ||
        !core::validate(right_detection) ||
        !core::validate(association) || !ports::isValid(estimation)) {
        return failure(
            ObservationMessageError::invalid_domain_record,
            "publication contains a rejected or invalid domain record");
    }
    if (left.dropped || right.dropped ||
        left.role != core::CameraRole::left ||
        right.role != core::CameraRole::right ||
        left.camera_id == right.camera_id ||
        left.frame_index != right.frame_index ||
        !referencesFrame(left_detection, left) ||
        !referencesFrame(right_detection, right) ||
        !left_detection.selected || !right_detection.selected ||
        association.decision != core::AssociationDecision::accepted ||
        association.reject_reason != core::AssociationRejectReason::none ||
        association.left_frame_index != left.frame_index ||
        association.right_frame_index != right.frame_index ||
        association.left_candidate_id != left_detection.candidate_id ||
        association.right_candidate_id != right_detection.candidate_id ||
        association.left_capture.nanoseconds !=
            left.capture_timestamp.nanoseconds ||
        association.left_capture.clock_domain !=
            left.capture_timestamp.clock_domain ||
        association.right_capture.nanoseconds !=
            right.capture_timestamp.nanoseconds ||
        association.right_capture.clock_domain !=
            right.capture_timestamp.clock_domain ||
        measurement.source_pair_id != association.pair_id) {
        return failure(
            ObservationMessageError::inconsistent_reference,
            "publication cross-references do not describe one accepted pair");
    }

    const auto expected_capture = integerMidpoint(
        association.left_capture.nanoseconds,
        association.right_capture.nanoseconds);
    if (measurement.capture_reference.nanoseconds != expected_capture ||
        measurement.capture_reference.clock_domain !=
            association.left_capture.clock_domain ||
        measurement.capture_reference.clock_domain !=
            config.ros_time_clock_domain ||
        measurement.coordinate_frame.value() != config.header_frame_id) {
        return failure(
            ObservationMessageError::inconsistent_reference,
            "measurement timestamp or coordinate frame disagrees with ROS header");
    }
    if (!left_detection.bounding_box || !right_detection.bounding_box ||
        !left_detection.score || !right_detection.score ||
        !measurement.covariance_m2 || !measurement.confidence ||
        !sameOptionalResidual(
            left.acquisition.stereo_hardware_timestamp_residual_ns,
            right.acquisition.stereo_hardware_timestamp_residual_ns)) {
        return failure(
            ObservationMessageError::missing_required_evidence,
            "ROS compatibility message requires bbox, confidence, covariance, "
            "and stereo hardware residual evidence");
    }
    if (left.acquisition.capture_timestamp_mapping_valid !=
        right.acquisition.capture_timestamp_mapping_valid) {
        return failure(
            ObservationMessageError::inconsistent_reference,
            "left/right capture timestamp mapping validity disagrees");
    }

    const auto method_code = legacyDepthMethodCode(estimation.method);
    if (!method_code) {
        return failure(
            ObservationMessageError::unsupported_depth_method,
            "depth method has no frozen NxBallObservation code");
    }

    constexpr auto kBillion = std::int64_t{1'000'000'000};
    const auto capture_ns = measurement.capture_reference.nanoseconds;
    const auto seconds = capture_ns / kBillion;
    if (capture_ns < 0 || seconds > std::numeric_limits<std::int32_t>::max() ||
        left.frame_index > static_cast<core::FrameIndex>(
            std::numeric_limits<std::int32_t>::max()) ||
        left.hardware_timestamp.nanoseconds < 0 ||
        right.hardware_timestamp.nanoseconds < 0 ||
        measurement.position_m.z <= 0.0) {
        return failure(
            ObservationMessageError::numeric_range,
            "timestamp, frame index, device timestamp, or depth is out of range");
    }

    ObservationMessage output;
    output.stamp_sec = static_cast<std::int32_t>(seconds);
    output.stamp_nanosec = static_cast<std::uint32_t>(capture_ns % kBillion);
    output.header_frame_id = config.header_frame_id;
    output.source_epoch = config.source_epoch;
    output.frame_index = static_cast<std::int32_t>(left.frame_index);
    output.track_id = 0;
    output.class_id = config.class_id;
    output.detection_confidence = std::min(
        *left_detection.score, *right_detection.score);
    output.bbox_left_xyxy = bboxXyxy(*left_detection.bounding_box);
    output.bbox_right_xyxy = bboxXyxy(*right_detection.bounding_box);
    output.position_m = measurement.position_m;
    output.position_covariance_m2 =
        covarianceMatrix(*measurement.covariance_m2);
    output.depth_m = measurement.position_m.z;
    output.disparity_px = estimation.disparity_px;
    output.depth_sigma_m = std::sqrt(
        measurement.covariance_m2->upper[5]);
    output.match_confidence = *measurement.confidence;
    output.depth_method_code = *method_code;
    output.detection_valid = true;
    output.stereo_valid = true;
    output.fallback_observation =
        estimation.method != config.primary_depth_method;
    output.left_device_timestamp_ns = static_cast<std::uint64_t>(
        left.hardware_timestamp.nanoseconds);
    output.right_device_timestamp_ns = static_cast<std::uint64_t>(
        right.hardware_timestamp.nanoseconds);
    output.stereo_hardware_timestamp_residual_ns =
        *left.acquisition.stereo_hardware_timestamp_residual_ns;
    output.capture_timestamp_mapping_valid =
        left.acquisition.capture_timestamp_mapping_valid;
    const auto left_uncertainty =
        left.acquisition.capture_timestamp_uncertainty_ns.value_or(0);
    const auto right_uncertainty =
        right.acquisition.capture_timestamp_uncertainty_ns.value_or(0);
    output.capture_timestamp_uncertainty_ns =
        std::max(left_uncertainty, right_uncertainty);
    return {std::move(output), ObservationMessageError::none, {}};
}

}  // namespace volleyball::adapters::ros2
