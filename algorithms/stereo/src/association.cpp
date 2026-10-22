#include "volleyball/algorithms/stereo/association.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace volleyball::algorithms::stereo {
namespace {

double scoreOf(const core::Detection2D& detection) noexcept {
    return detection.score.value_or(0.0);
}

double dimensionRatio(double left, double right) noexcept {
    if (left <= 0.0 || right <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(left / right, right / left);
}

double sizeRatio(const core::BoundingBox2D& left,
                 const core::BoundingBox2D& right) noexcept {
    return std::max(
        dimensionRatio(left.width, right.width),
        dimensionRatio(left.height, right.height));
}

double shiftedIou(const core::BoundingBox2D& left,
                  const core::BoundingBox2D& right,
                  double disparity_px) noexcept {
    const double right_x = right.x + disparity_px;
    const double x0 = std::max(left.x, right_x);
    const double y0 = std::max(left.y, right.y);
    const double x1 = std::min(left.x + left.width, right_x + right.width);
    const double y1 = std::min(left.y + left.height, right.y + right.height);
    const double intersection =
        std::max(0.0, x1 - x0) * std::max(0.0, y1 - y0);
    const double union_area = left.width * left.height +
                              right.width * right.height - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

core::AssociationRejectReason classify(
    const core::Detection2D& left,
    const core::Detection2D& right,
    const AssociationConfig& config,
    double disparity,
    double epipolar_error,
    std::int64_t capture_delta_ns,
    bool capture_delta_valid) noexcept {
    if (!core::validate(left) || !core::validate(right)) {
        return core::AssociationRejectReason::quality_gate;
    }
    if (!capture_delta_valid || config.max_capture_delta_ns < 0 ||
        std::abs(capture_delta_ns) > config.max_capture_delta_ns) {
        return core::AssociationRejectReason::temporal_mismatch;
    }
    if (disparity <= 0.0) {
        return core::AssociationRejectReason::nonpositive_disparity;
    }
    if (disparity > config.max_disparity_px) {
        return core::AssociationRejectReason::disparity_out_of_range;
    }
    const double adaptive_epipolar_tolerance = std::max(
        config.epipolar_tolerance_px,
        config.adaptive_epipolar_height_ratio * std::max(
            left.bounding_box ? left.bounding_box->height : 0.0,
            right.bounding_box ? right.bounding_box->height : 0.0));
    if (epipolar_error > adaptive_epipolar_tolerance) {
        return core::AssociationRejectReason::epipolar_error;
    }
    if (!left.bounding_box || !right.bounding_box) {
        return config.require_bounding_box
            ? core::AssociationRejectReason::quality_gate
            : core::AssociationRejectReason::none;
    }
    if (sizeRatio(*left.bounding_box, *right.bounding_box) >
        config.max_size_ratio) {
        return core::AssociationRejectReason::size_ratio;
    }
    if (shiftedIou(*left.bounding_box, *right.bounding_box, disparity) <
        config.min_shifted_iou) {
        return core::AssociationRejectReason::shifted_iou;
    }
    return core::AssociationRejectReason::none;
}

double expectedDisparityFromBox(
    const core::Detection2D& detection,
    const AssociationConfig& config) noexcept {
    if (!detection.bounding_box || detection.bounding_box->width <= 1.0 ||
        config.baseline_m <= 0.0 || config.ball_diameter_m <= 0.0 ||
        config.bbox_scale <= 0.0) {
        return -1.0;
    }
    return std::clamp(
        config.baseline_m * detection.bounding_box->width * config.bbox_scale /
            config.ball_diameter_m,
        1.0,
        config.max_disparity_px);
}

double bboxDisparityPenalty(
    const core::Detection2D& left,
    const core::Detection2D& right,
    double disparity,
    const AssociationConfig& config) noexcept {
    if (config.bbox_disparity_penalty_scale <= 0.0 || disparity <= 0.0) {
        return 0.0;
    }
    const double left_expected = expectedDisparityFromBox(left, config);
    const double right_expected = expectedDisparityFromBox(right, config);
    double expected = -1.0;
    if (left_expected > 0.0 && right_expected > 0.0) {
        expected = 0.5 * (left_expected + right_expected);
    } else if (left_expected > 0.0) {
        expected = left_expected;
    } else if (right_expected > 0.0) {
        expected = right_expected;
    }
    if (expected <= 0.0) return 0.0;

    const double tolerance = std::max(
        config.bbox_disparity_consistency_min_px,
        expected * config.bbox_disparity_consistency_ratio);
    const double excess = std::abs(disparity - expected) - tolerance;
    if (excess <= 0.0) return 0.0;
    return config.bbox_disparity_penalty_scale * excess /
           std::max(1.0, tolerance);
}

}  // namespace

bool isValidAssociationConfig(const AssociationConfig& config) noexcept {
    return config.max_capture_delta_ns >= 0 &&
           std::isfinite(config.epipolar_tolerance_px) &&
           config.epipolar_tolerance_px >= 0.0 &&
           std::isfinite(config.adaptive_epipolar_height_ratio) &&
           config.adaptive_epipolar_height_ratio >= 0.0 &&
           std::isfinite(config.max_size_ratio) &&
           config.max_size_ratio >= 1.0 &&
           std::isfinite(config.min_shifted_iou) &&
           config.min_shifted_iou >= 0.0 && config.min_shifted_iou <= 1.0 &&
           std::isfinite(config.max_disparity_px) &&
           config.max_disparity_px > 0.0 &&
           std::isfinite(config.bbox_disparity_penalty_scale) &&
           config.bbox_disparity_penalty_scale >= 0.0 &&
           std::isfinite(config.bbox_disparity_consistency_ratio) &&
           config.bbox_disparity_consistency_ratio > 0.0 &&
           std::isfinite(config.bbox_disparity_consistency_min_px) &&
           config.bbox_disparity_consistency_min_px > 0.0 &&
           std::isfinite(config.bbox_scale) && config.bbox_scale > 0.0 &&
           (config.bbox_disparity_penalty_scale == 0.0 ||
            (std::isfinite(config.baseline_m) && config.baseline_m > 0.0 &&
             std::isfinite(config.ball_diameter_m) &&
             config.ball_diameter_m > 0.0));
}

AssociationBatch associateCandidates(
    const std::vector<core::Detection2D>& left,
    const std::vector<core::Detection2D>& right,
    const core::Timestamp& left_capture,
    const core::Timestamp& right_capture,
    const AssociationConfig& config,
    core::PairId first_pair_id) {
    AssociationBatch output;
    if (!isValidAssociationConfig(config)) {
        output.message = "invalid stereo association configuration";
        return output;
    }
    if (left_capture.clock_domain.empty() ||
        left_capture.clock_domain != right_capture.clock_domain ||
        left_capture.nanoseconds < 0 || right_capture.nanoseconds < 0) {
        output.message = "invalid stereo association capture timestamps";
        return output;
    }
    if (!right.empty() &&
        left.size() > std::numeric_limits<std::size_t>::max() / right.size()) {
        output.message = "stereo association candidate product overflows";
        return output;
    }
    const std::size_t attempt_count = left.size() * right.size();
    if (attempt_count > output.attempts.max_size() ||
        attempt_count >
            std::numeric_limits<core::PairId>::max() - first_pair_id) {
        output.message = "stereo association pair ID range is exhausted";
        return output;
    }
    output.attempts.reserve(attempt_count);
    double best_semantic = -std::numeric_limits<double>::infinity();
    double best_score = std::numeric_limits<double>::infinity();
    const bool capture_delta_valid = true;
    const std::int64_t capture_delta_ns =
        right_capture.nanoseconds - left_capture.nanoseconds;

    for (const auto& left_candidate : left) {
        for (const auto& right_candidate : right) {
            const double disparity =
                left_candidate.center_px.x - right_candidate.center_px.x;
            const double epipolar_error = std::abs(
                left_candidate.center_px.y - right_candidate.center_px.y);
            const auto rejection = classify(
                left_candidate,
                right_candidate,
                config,
                disparity,
                epipolar_error,
                capture_delta_ns,
                capture_delta_valid);

            core::StereoAssociationAttempt attempt;
            attempt.pair_id = first_pair_id + output.attempts.size();
            attempt.left_frame_index = left_candidate.frame.frame_index;
            attempt.right_frame_index = right_candidate.frame.frame_index;
            attempt.left_candidate_id = left_candidate.candidate_id;
            attempt.right_candidate_id = right_candidate.candidate_id;
            attempt.left_capture = left_capture;
            attempt.right_capture = right_capture;
            attempt.delta_t_lr_ns = capture_delta_ns;
            attempt.disparity_px = disparity;
            attempt.epipolar_error_px = epipolar_error;
            attempt.decision = rejection == core::AssociationRejectReason::none
                ? core::AssociationDecision::accepted
                : core::AssociationDecision::rejected;
            attempt.reject_reason = rejection;

            const double semantic_confidence = std::sqrt(std::max(
                0.0,
                scoreOf(left_candidate) * scoreOf(right_candidate)));
            const double adaptive_epipolar_tolerance = std::max(
                config.epipolar_tolerance_px,
                config.adaptive_epipolar_height_ratio * std::max(
                    left_candidate.bounding_box
                        ? left_candidate.bounding_box->height
                        : 0.0,
                    right_candidate.bounding_box
                        ? right_candidate.bounding_box->height
                        : 0.0));
            const double normalized_epipolar =
                adaptive_epipolar_tolerance > 0.0
                ? epipolar_error / adaptive_epipolar_tolerance
                : epipolar_error;
            double score = normalized_epipolar;
            if (left_candidate.bounding_box && right_candidate.bounding_box) {
                const double width_ratio = dimensionRatio(
                    left_candidate.bounding_box->width,
                    right_candidate.bounding_box->width);
                const double height_ratio = dimensionRatio(
                    left_candidate.bounding_box->height,
                    right_candidate.bounding_box->height);
                score += std::abs(std::log(width_ratio)) +
                         std::abs(std::log(height_ratio));
            }
            score -= 0.25 * scoreOf(right_candidate);
            score += bboxDisparityPenalty(
                left_candidate, right_candidate, disparity, config);
            output.attempts.push_back({attempt, score, semantic_confidence});
            const std::size_t index = output.attempts.size() - 1;
            if (attempt.decision == core::AssociationDecision::accepted &&
                (semantic_confidence > best_semantic ||
                 (semantic_confidence == best_semantic &&
                  score < best_score))) {
                best_semantic = semantic_confidence;
                best_score = score;
                output.best_accepted_index = index;
            }
        }
    }
    return output;
}

}  // namespace volleyball::algorithms::stereo
