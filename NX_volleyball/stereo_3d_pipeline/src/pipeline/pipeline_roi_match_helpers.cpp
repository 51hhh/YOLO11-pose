#include "pipeline_roi_match_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace stereo3d {

CircleFit2D circleFromGpuCandidate(const DualYoloGpuCircle& in,
                                   const Detection& fallback) {
    if (in.valid) {
        CircleFit2D out;
        out.cx = in.cx;
        out.cy = in.cy;
        out.radius = in.radius;
        out.confidence = in.confidence;
        out.source = in.source;
        out.valid = true;
        return out;
    }
    return circleFromDetectionCPU(fallback);
}

PointMeasure2D pointFromGpuCandidate(const DualYoloGpuPointMeasure& in) {
    PointMeasure2D out;
    if (in.valid) {
        out.cx = in.cx;
        out.cy = in.cy;
        out.confidence = in.confidence;
        out.valid = true;
    }
    return out;
}

DualYoloGpuDetection makeGpuDetection(const Detection& det) {
    DualYoloGpuDetection out;
    out.cx = det.cx;
    out.cy = det.cy;
    out.width = det.width;
    out.height = det.height;
    out.confidence = det.confidence;
    out.class_id = det.class_id;
    return out;
}

SubpixelDisparityResult subpixelFromGpuCandidate(const DualYoloGpuDisparity& in) {
    SubpixelDisparityResult out;
    out.valid = in.valid != 0;
    out.low_confidence = in.low_confidence != 0;
    out.disparity = in.disparity;
    out.confidence = in.confidence;
    out.stddev = in.stddev;
    out.delta_gate_px = in.delta_gate_px;
    out.support = in.support;
    out.attempted = in.attempted;
    return out;
}

SparseFeatureDisparityResult sparseFromGpuCandidate(const DualYoloGpuDisparity& in) {
    SparseFeatureDisparityResult out;
    out.valid = in.valid != 0;
    out.low_confidence = in.low_confidence != 0;
    out.disparity = in.disparity;
    out.confidence = in.confidence;
    out.stddev = in.stddev;
    out.anchor_cx = in.anchor_cx;
    out.anchor_cy = in.anchor_cy;
    out.support = in.support;
    out.attempted = in.attempted;
    return out;
}

float estimateDisparityFromBBoxCPU(
    const Detection& det,
    float baseline,
    const HybridDepthConfig& depth_cfg,
    int max_disparity)
{
    if (det.width <= 1.0f || depth_cfg.object_diameter <= 0.01f ||
        baseline <= 0.0f || max_disparity <= 0) {
        return -1.0f;
    }

    const float disp = baseline * det.width * depth_cfg.bbox_scale /
                       depth_cfg.object_diameter;
    return std::clamp(disp, 1.0f, static_cast<float>(max_disparity));
}

float bboxDisparityConsistencyPenaltyCPU(
    const Detection& left,
    const Detection& right,
    float pair_disparity,
    float baseline,
    const HybridDepthConfig& depth_cfg,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity)
{
    if (!std::isfinite(pair_disparity) || pair_disparity <= 0.0f) {
        return 0.0f;
    }
    const float left_expected =
        estimateDisparityFromBBoxCPU(left, baseline, depth_cfg, max_disparity);
    const float right_expected =
        estimateDisparityFromBBoxCPU(right, baseline, depth_cfg, max_disparity);

    float expected = -1.0f;
    if (left_expected > 0.0f && right_expected > 0.0f) {
        expected = 0.5f * (left_expected + right_expected);
    } else if (left_expected > 0.0f) {
        expected = left_expected;
    } else if (right_expected > 0.0f) {
        expected = right_expected;
    }
    if (expected <= 0.0f) return 0.0f;

    const float ratio_tol =
        std::max(0.05f, dual_cfg.bbox_disparity_consistency_ratio);
    const float abs_tol =
        std::max(5.0f, dual_cfg.bbox_disparity_consistency_min_px);
    const float tolerance = std::max(abs_tol, expected * ratio_tol);
    const float excess = std::abs(pair_disparity - expected) - tolerance;
    if (excess <= 0.0f) return 0.0f;

    const float scale = std::max(0.0f, dual_cfg.bbox_disparity_penalty_scale);
    return scale * excess / std::max(1.0f, tolerance);
}

void stampFrameMetadata(FrameSlot& slot)
{
    const int64_t frame_counter_delta =
        static_cast<int64_t>(slot.left_frame_counter) -
        static_cast<int64_t>(slot.right_frame_counter);
    const int64_t frame_number_delta =
        static_cast<int64_t>(slot.left_frame_number) -
        static_cast<int64_t>(slot.right_frame_number);
    const int64_t timestamp_delta_raw =
        static_cast<int64_t>(slot.left_timestamp_us) -
        static_cast<int64_t>(slot.right_timestamp_us);
    for (auto& obj : slot.results) {
        obj.left_timestamp_us = slot.left_timestamp_us;
        obj.right_timestamp_us = slot.right_timestamp_us;
        obj.left_frame_number = slot.left_frame_number;
        obj.right_frame_number = slot.right_frame_number;
        obj.left_frame_counter = slot.left_frame_counter;
        obj.right_frame_counter = slot.right_frame_counter;
        obj.left_trigger_index = slot.left_trigger_index;
        obj.right_trigger_index = slot.right_trigger_index;
        obj.frame_counter_delta = frame_counter_delta;
        obj.frame_number_delta = frame_number_delta;
        obj.timestamp_delta_us = timestamp_delta_raw / 1000;
    }
}

}  // namespace stereo3d
