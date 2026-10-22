#include "volleyball/core/observation.hpp"

#include <algorithm>
#include <cmath>

namespace volleyball::core {
namespace {

DomainValidationResult ok() noexcept { return {}; }

DomainValidationResult failure(DomainError error, std::string message) noexcept {
    return DomainValidationResult{error, std::move(message)};
}

bool sameClock(
    const ClockDomainId& expected,
    const std::optional<Timestamp>& timestamp) noexcept {
    return !timestamp || timestamp->clock_domain == expected;
}

bool nonnegative(const std::optional<Timestamp>& timestamp) noexcept {
    return !timestamp || timestamp->nanoseconds >= 0;
}

bool finite(const Covariance2D& covariance) noexcept {
    return std::all_of(
        covariance.upper.begin(), covariance.upper.end(),
        [](double value) { return std::isfinite(value); });
}

bool finite(const Covariance3D& covariance) noexcept {
    return std::all_of(
        covariance.upper.begin(), covariance.upper.end(),
        [](double value) { return std::isfinite(value); });
}

bool positiveSemidefinite(const Covariance2D& covariance) noexcept {
    if (!finite(covariance)) {
        return false;
    }
    const auto& values = covariance.upper;
    const double uu = values[0];
    const double uv = values[1];
    const double vv = values[2];
    if (uu < 0.0 || vv < 0.0) {
        return false;
    }
    const double scale = std::max({std::abs(uu), std::abs(uv), std::abs(vv)});
    if (scale == 0.0) {
        return true;
    }
    const double normalized_uu = uu / scale;
    const double normalized_uv = uv / scale;
    const double normalized_vv = vv / scale;
    return normalized_uu * normalized_vv -
               normalized_uv * normalized_uv >=
           -1.0e-9;
}

bool positiveSemidefinite(const Covariance3D& covariance) noexcept {
    if (!finite(covariance)) {
        return false;
    }
    const auto& values = covariance.upper;
    const double scale = std::max({
        std::abs(values[0]), std::abs(values[1]), std::abs(values[2]),
        std::abs(values[3]), std::abs(values[4]), std::abs(values[5]),
    });
    if (scale == 0.0) {
        return true;
    }
    const double xx = values[0] / scale;
    const double xy = values[1] / scale;
    const double xz = values[2] / scale;
    const double yy = values[3] / scale;
    const double yz = values[4] / scale;
    const double zz = values[5] / scale;
    if (xx < 0.0 || yy < 0.0 || zz < 0.0 ||
        xx * yy - xy * xy < -1.0e-9 ||
        xx * zz - xz * xz < -1.0e-9 ||
        yy * zz - yz * yz < -1.0e-9) {
        return false;
    }
    const double determinant =
        xx * yy * zz + 2.0 * xy * xz * yz - xx * yz * yz -
        yy * xz * xz - zz * xy * xy;
    return determinant >= -1.0e-9;
}

}  // namespace

DomainValidationResult validate(const CameraFrameRecord& frame) noexcept {
    if (frame.camera_id.empty() || frame.capture_timestamp.clock_domain.empty()) {
        return failure(DomainError::missing_identifier, "frame identifiers are required");
    }
    const auto& clock = frame.capture_timestamp.clock_domain;
    if (frame.hardware_timestamp.clock_domain != clock ||
        !sameClock(clock, frame.exposure_start) ||
        !sameClock(clock, frame.receive_timestamp) ||
        !sameClock(clock, frame.inference_done_timestamp)) {
        return failure(
            DomainError::clock_domain_mismatch,
            "all camera frame timestamps must use one declared clock domain");
    }
    if (frame.hardware_timestamp.nanoseconds < 0 ||
        frame.capture_timestamp.nanoseconds < 0 ||
        !nonnegative(frame.exposure_start) ||
        !nonnegative(frame.receive_timestamp) ||
        !nonnegative(frame.inference_done_timestamp)) {
        return failure(
            DomainError::invalid_numeric_value,
            "camera frame timestamps must be non-negative");
    }
    std::int64_t previous = frame.exposure_start
        ? frame.exposure_start->nanoseconds
        : frame.hardware_timestamp.nanoseconds;
    const std::optional<Timestamp> ordered[] = {
        frame.capture_timestamp,
        frame.receive_timestamp,
        frame.inference_done_timestamp,
    };
    for (const auto& timestamp : ordered) {
        if (timestamp && timestamp->nanoseconds < previous) {
            return failure(
                DomainError::timestamp_order,
                "frame timestamps must be monotonically ordered");
        }
        if (timestamp) {
            previous = timestamp->nanoseconds;
        }
    }
    if (frame.roi.width == 0 || frame.roi.height == 0) {
        return failure(DomainError::invalid_roi, "frame ROI must be non-empty");
    }
    if (frame.dropped && (!frame.drop_reason || frame.drop_reason->empty())) {
        return failure(
            DomainError::missing_drop_reason,
            "dropped frame requires a reason");
    }
    if ((frame.exposure_us &&
         (!std::isfinite(*frame.exposure_us) || *frame.exposure_us < 0.0)) ||
        (frame.gain_db && !std::isfinite(*frame.gain_db))) {
        return failure(
            DomainError::invalid_numeric_value,
            "frame exposure and gain must be finite");
    }
    return ok();
}

DomainValidationResult validate(const Detection2D& detection) noexcept {
    if (detection.frame.camera_id.empty() || detection.source_pipeline.empty()) {
        return failure(
            DomainError::missing_identifier,
            "detection camera and pipeline identifiers are required");
    }
    if (!isFinite(detection.center_px) ||
        (detection.pixel_velocity_pxps &&
         !isFinite(*detection.pixel_velocity_pxps)) ||
        (detection.radius_px &&
         (!std::isfinite(*detection.radius_px) || *detection.radius_px <= 0.0)) ||
        (detection.score &&
         (!std::isfinite(*detection.score) || *detection.score < 0.0 ||
          *detection.score > 1.0)) ||
        (detection.covariance_px2 &&
         !positiveSemidefinite(*detection.covariance_px2))) {
        return failure(
            DomainError::invalid_numeric_value,
            "detection numeric values are invalid");
    }
    if (detection.bounding_box &&
        (!std::isfinite(detection.bounding_box->x) ||
         !std::isfinite(detection.bounding_box->y) ||
         !std::isfinite(detection.bounding_box->width) ||
         !std::isfinite(detection.bounding_box->height) ||
         detection.bounding_box->width <= 0.0 ||
         detection.bounding_box->height <= 0.0)) {
        return failure(
            DomainError::invalid_numeric_value,
            "detection bounding box is invalid");
    }
    return ok();
}

DomainValidationResult validate(
    const StereoAssociationAttempt& attempt) noexcept {
    if (attempt.left_capture.clock_domain != attempt.right_capture.clock_domain) {
        return failure(
            DomainError::clock_domain_mismatch,
            "stereo captures must use the same camera clock domain");
    }
    if (attempt.left_capture.nanoseconds < 0 ||
        attempt.right_capture.nanoseconds < 0) {
        return failure(
            DomainError::invalid_numeric_value,
            "stereo capture timestamps must be non-negative");
    }
    const auto measured_delta =
        attempt.right_capture.nanoseconds - attempt.left_capture.nanoseconds;
    if (measured_delta != attempt.delta_t_lr_ns) {
        return failure(
            DomainError::timestamp_order,
            "delta_t_lr_ns must equal right minus left capture timestamp");
    }
    if ((attempt.disparity_px && !std::isfinite(*attempt.disparity_px)) ||
        (attempt.epipolar_error_px &&
         (!std::isfinite(*attempt.epipolar_error_px) ||
          *attempt.epipolar_error_px < 0.0))) {
        return failure(
            DomainError::invalid_numeric_value,
            "association geometry must be finite and non-negative where applicable");
    }
    const bool accepted = attempt.decision == AssociationDecision::accepted;
    if ((accepted && attempt.reject_reason != AssociationRejectReason::none) ||
        (!accepted && attempt.reject_reason == AssociationRejectReason::none)) {
        return failure(
            DomainError::invalid_decision_reason,
            "association decision and rejection reason disagree");
    }
    if (accepted &&
        (!attempt.disparity_px || *attempt.disparity_px <= 0.0 ||
         !attempt.epipolar_error_px)) {
        return failure(
            DomainError::invalid_numeric_value,
            "accepted association requires positive disparity and epipolar error");
    }
    return ok();
}

DomainValidationResult validate(
    const TriangulatedObservation& observation) noexcept {
    if (observation.capture_reference.clock_domain.empty() ||
        observation.coordinate_frame.empty()) {
        return failure(
            DomainError::missing_identifier,
            "observation clock and coordinate frame are required");
    }
    if (!isFinite(observation.position_m) ||
        (observation.covariance_m2 &&
         !positiveSemidefinite(*observation.covariance_m2)) ||
        (observation.confidence &&
         (!std::isfinite(*observation.confidence) ||
          *observation.confidence < 0.0 || *observation.confidence > 1.0)) ||
        (observation.reprojection_error_px &&
         (!std::isfinite(*observation.reprojection_error_px) ||
          *observation.reprojection_error_px < 0.0))) {
        return failure(
            DomainError::invalid_numeric_value,
            "triangulated observation numeric values are invalid");
    }
    return ok();
}

}  // namespace volleyball::core

