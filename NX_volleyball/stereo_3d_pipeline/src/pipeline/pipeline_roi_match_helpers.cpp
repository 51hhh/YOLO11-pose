#include "pipeline_roi_match_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stereo3d {

namespace {

struct SubpixelSampleOffset {
    float dx = 0.0f;
    float dy = 0.0f;
};

std::vector<SubpixelSampleOffset> makeSubpixelSampleOffsetsCPU(
    float radius,
    int max_points,
    int patch_radius)
{
    std::vector<SubpixelSampleOffset> offsets;
    max_points = std::clamp(max_points, 1, 64);
    offsets.reserve(static_cast<size_t>(max_points));
    offsets.push_back({});

    const float usable_radius = std::max(static_cast<float>(patch_radius + 2),
                                         radius * 0.70f);
    const float ring_fracs[] = {0.28f, 0.48f, 0.66f};
    const int angle_count = max_points <= 12 ? 4 : 8;
    constexpr float kPi = 3.14159265358979323846f;

    for (float frac : ring_fracs) {
        const float r = usable_radius * frac;
        for (int i = 0; i < angle_count; ++i) {
            if (static_cast<int>(offsets.size()) >= max_points) return offsets;
            const float angle = 2.0f * kPi * static_cast<float>(i) /
                                static_cast<float>(angle_count);
            offsets.push_back({r * std::cos(angle), r * std::sin(angle)});
        }
    }
    return offsets;
}

}  // namespace

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
    out.debug_match_count = std::clamp(
        in.debug_match_count,
        0,
        std::min(kMaxSparseFeatureDebugMatches,
                 kMaxDualYoloGpuDebugMatches));
    for (int i = 0; i < out.debug_match_count; ++i) {
        auto& m = out.debug_matches[static_cast<size_t>(i)];
        m.left_x = in.debug_left_x[i];
        m.left_y = in.debug_left_y[i];
        m.right_x = in.debug_right_x[i];
        m.right_y = in.debug_right_y[i];
        m.disparity = in.debug_disparity[i];
        m.score = in.debug_score[i];
    }
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

std::vector<DualYoloGpuDetectionPair> buildGpuDetectionPairsForRefine(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections,
    const StereoRoiPairGateConfig& roi_pair_gate,
    float baseline,
    const HybridDepthConfig& depth_cfg,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity,
    std::size_t max_pairs)
{
    std::vector<StereoRoiPair> roi_pairs =
        collectStereoRoiPairCandidates(
            left_detections,
            right_detections,
            roi_pair_gate,
            left_detections.size() * right_detections.size());
    for (auto& roi_pair : roi_pairs) {
        roi_pair.score += bboxDisparityConsistencyPenaltyCPU(
            roi_pair.left,
            roi_pair.right,
            roi_pair.initial_disparity,
            baseline,
            depth_cfg,
            dual_cfg,
            max_disparity);
    }
    std::sort(roi_pairs.begin(), roi_pairs.end(),
              [](const StereoRoiPair& a, const StereoRoiPair& b) {
                  return a.score < b.score;
              });
    if (roi_pairs.size() > max_pairs) {
        roi_pairs.resize(max_pairs);
    }

    std::vector<DualYoloGpuDetectionPair> gpu_pairs;
    gpu_pairs.reserve(roi_pairs.size());
    for (const auto& roi_pair : roi_pairs) {
        DualYoloGpuDetectionPair pair;
        pair.left = makeGpuDetection(roi_pair.left);
        pair.right = makeGpuDetection(roi_pair.right);
        pair.left_index = roi_pair.left_index;
        pair.right_index = roi_pair.right_index;
        pair.epipolar_y_delta_px = roi_pair.epipolar_y_delta;
        gpu_pairs.push_back(pair);
    }
    return gpu_pairs;
}

CircleFit2D searchTemplateOnEpipolarCPU(
    const uint8_t* source_img, int source_pitch,
    const uint8_t* target_img, int target_pitch,
    int img_w, int img_h,
    const CircleFit2D& source_circle,
    float predicted_cx, float predicted_cy,
    float y_tolerance,
    const PipelineConfig::DualYoloConfig& dual_cfg)
{
    CircleFit2D out;
    if (!source_img || !target_img || source_pitch <= 0 || target_pitch <= 0 ||
        !source_circle.valid) {
        return out;
    }

    const int patch_radius = std::clamp(
        static_cast<int>(std::lround(std::min(source_circle.radius * 0.35f, 10.0f))),
        4, 10);
    const int source_x = static_cast<int>(std::lround(source_circle.cx));
    const int source_y = static_cast<int>(std::lround(source_circle.cy));
    if (!patchInsideCPU(img_w, img_h, source_x, source_y,
                        patch_radius, dual_cfg.roi_denoise)) {
        return out;
    }

    const float max_width = std::max(32.0f, static_cast<float>(dual_cfg.fallback_max_width_px));
    const float margin = std::min(
        std::max(4.0f, static_cast<float>(dual_cfg.fallback_search_margin_px)),
        max_width * 0.5f);
    const int x_start = std::max(patch_radius + 1,
        static_cast<int>(std::floor(predicted_cx - margin)));
    const int x_end = std::min(img_w - patch_radius - 2,
        static_cast<int>(std::ceil(predicted_cx + margin)));
    const int y_start = std::max(patch_radius + 1,
        static_cast<int>(std::floor(predicted_cy - y_tolerance)));
    const int y_end = std::min(img_h - patch_radius - 2,
        static_cast<int>(std::ceil(predicted_cy + y_tolerance)));
    if (x_start >= x_end || y_start >= y_end) return out;

    auto score_at = [&](int x, int y) -> float {
        if (!patchInsideCPU(img_w, img_h, x, y, patch_radius, dual_cfg.roi_denoise)) {
            return -2.0f;
        }
        return znccPatchCPU(source_img, source_pitch, target_img, target_pitch,
                            source_x, source_y, x, y,
                            patch_radius, dual_cfg.roi_denoise);
    };

    float best_score = -2.0f;
    float second_score = -2.0f;
    int best_x = -1;
    int best_y = -1;
    const int coarse_step = (x_end - x_start) > 64 ? 2 : 1;
    for (int y = y_start; y <= y_end; y += coarse_step) {
        for (int x = x_start; x <= x_end; x += coarse_step) {
            const float score = score_at(x, y);
            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_x = x;
                best_y = y;
            } else if (score > second_score) {
                second_score = score;
            }
        }
    }
    if (best_x < 0 || best_y < 0) return out;

    const int refine_x1 = std::max(x_start, best_x - coarse_step);
    const int refine_x2 = std::min(x_end, best_x + coarse_step);
    const int refine_y1 = std::max(y_start, best_y - coarse_step);
    const int refine_y2 = std::min(y_end, best_y + coarse_step);
    for (int y = refine_y1; y <= refine_y2; ++y) {
        for (int x = refine_x1; x <= refine_x2; ++x) {
            const float score = score_at(x, y);
            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_x = x;
                best_y = y;
            } else if (score > second_score && (x != best_x || y != best_y)) {
                second_score = score;
            }
        }
    }

    const float min_score = std::max(0.12f, dual_cfg.subpixel_min_confidence * 0.55f);
    const float uniqueness = second_score > -1.5f ? best_score - second_score : 1.0f;
    if (best_score < min_score || (uniqueness < 0.005f && best_score < 0.70f)) {
        return out;
    }

    float sub_x = static_cast<float>(best_x);
    if (best_x > x_start && best_x < x_end) {
        const float s_minus = score_at(best_x - 1, best_y);
        const float s_plus = score_at(best_x + 1, best_y);
        const float denom = s_minus - 2.0f * best_score + s_plus;
        if (s_minus > -1.5f && s_plus > -1.5f && denom < -1e-5f) {
            sub_x += std::clamp(0.5f * (s_minus - s_plus) / denom, -1.0f, 1.0f);
        }
    }

    if (std::abs(sub_x - predicted_cx) > margin ||
        std::abs(static_cast<float>(best_y) - predicted_cy) > y_tolerance) {
        return out;
    }

    out.cx = sub_x;
    out.cy = static_cast<float>(best_y);
    out.radius = source_circle.radius;
    out.confidence = std::max(0.2f,
        std::clamp((best_score - min_score) / std::max(0.01f, 1.0f - min_score),
                   0.0f, 1.0f));
    out.source = kCircleSourceTemplateSearch;
    out.valid = true;
    return out;
}

SubpixelDisparityResult refineDisparityByROICenterPatchCPU(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    const CircleFit2D& left_circle,
    const CircleFit2D& right_circle,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity,
    float focal,
    float baseline)
{
    SubpixelDisparityResult result;
    if (!left_img || !right_img || left_pitch <= 0 || right_pitch <= 0 ||
        !left_circle.valid || !right_circle.valid || max_disparity <= 0) {
        return result;
    }

    const float initial_disp = left_circle.cx - right_circle.cx;
    if (!std::isfinite(initial_disp) || initial_disp <= 0.5f ||
        initial_disp > static_cast<float>(max_disparity)) {
        return result;
    }

    const int patch_radius = std::clamp(dual_cfg.subpixel_patch_radius, 2, 12);
    const int search_radius = std::max(1, dual_cfg.subpixel_search_radius_px);
    const float max_delta = computeSubpixelDispDeltaGateCPU(
        initial_disp, focal, baseline,
        dual_cfg.subpixel_max_disp_delta_px,
        dual_cfg.subpixel_max_disp_delta_ratio,
        dual_cfg.subpixel_max_depth_delta_m);
    result.delta_gate_px = max_delta;

    const int x_left = static_cast<int>(std::lround(left_circle.cx));
    const int y_left = static_cast<int>(std::lround(left_circle.cy));
    const float expected_y_delta =
        dual_cfg.feature_y_offset_px + (left_circle.cy - right_circle.cy);
    const int y_center = static_cast<int>(std::lround(
        static_cast<float>(y_left) - expected_y_delta));
    const int y_radius = std::clamp(
        static_cast<int>(std::ceil(std::clamp(
            dual_cfg.feature_y_tolerance_px, 0.5f, 8.0f))),
        1,
        8);
    if (!patchInsideCPU(img_w, img_h, x_left, y_left,
                        patch_radius, dual_cfg.roi_denoise)) {
        return result;
    }

    const int d_start = std::max(1, static_cast<int>(std::floor(initial_disp)) - search_radius);
    const int d_end = std::min(max_disparity,
                               static_cast<int>(std::ceil(initial_disp)) + search_radius);
    if (d_start >= d_end) return result;

    auto score_at = [&](int disparity, int y_right) -> float {
        const int x_right = static_cast<int>(std::lround(
            static_cast<float>(x_left) - static_cast<float>(disparity)));
        if (!patchInsideCPU(img_w, img_h, x_right, y_right,
                            patch_radius, dual_cfg.roi_denoise)) {
            return -2.0f;
        }
        return znccPatchCPU(left_img, left_pitch, right_img, right_pitch,
                            x_left, y_left, x_right, y_right,
                            patch_radius, dual_cfg.roi_denoise);
    };

    float best_score = -2.0f;
    float second_score = -2.0f;
    int best_disp = -1;
    int best_y_right = y_center;
    for (int disp = d_start; disp <= d_end; ++disp) {
        for (int y_right = y_center - y_radius;
             y_right <= y_center + y_radius; ++y_right) {
            ++result.attempted;
            const float score = score_at(disp, y_right);
            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_disp = disp;
                best_y_right = y_right;
            } else if (score > second_score) {
                second_score = score;
            }
        }
    }
    const float min_score = std::max(0.10f, dual_cfg.subpixel_min_confidence * 0.60f);
    if (best_disp < 0 || best_score < min_score) {
        result.low_confidence = true;
        return result;
    }

    float sub_disp = static_cast<float>(best_disp);
    if (best_disp > d_start && best_disp < d_end) {
        const float s_minus = score_at(best_disp - 1, best_y_right);
        const float s_plus = score_at(best_disp + 1, best_y_right);
        const float denom = s_minus - 2.0f * best_score + s_plus;
        if (s_minus > -1.5f && s_plus > -1.5f && denom < -1e-5f) {
            const float delta = std::clamp(
                0.5f * (s_minus - s_plus) / denom,
                -1.0f, 1.0f);
            sub_disp += delta;
        }
    }

    const float uniqueness_margin =
        second_score > -1.5f ? best_score - second_score : 1.0f;
    if ((uniqueness_margin < 0.01f && best_score < 0.75f) ||
        std::abs(sub_disp - initial_disp) > max_delta ||
        sub_disp <= 0.5f ||
        sub_disp > static_cast<float>(max_disparity)) {
        result.low_confidence = true;
        return result;
    }

    result.valid = true;
    result.disparity = sub_disp;
    result.support = 1;
    result.stddev = 0.0f;
    result.confidence = std::clamp((best_score - 0.10f) / 0.80f, 0.0f, 1.0f);
    if (result.confidence < dual_cfg.subpixel_min_confidence) {
        result.valid = false;
        result.low_confidence = true;
    }
    return result;
}

SubpixelDisparityResult refineDisparityByROIMultiPointCPU(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    const CircleFit2D& left_circle,
    const CircleFit2D& right_circle,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity,
    float focal,
    float baseline)
{
    SubpixelDisparityResult result;
    if (!left_img || !right_img || left_pitch <= 0 || right_pitch <= 0 ||
        !left_circle.valid || !right_circle.valid || max_disparity <= 0) {
        return result;
    }

    const float initial_disp = left_circle.cx - right_circle.cx;
    if (!std::isfinite(initial_disp) || initial_disp <= 0.5f ||
        initial_disp > static_cast<float>(max_disparity)) {
        return result;
    }

    const int patch_radius = std::clamp(dual_cfg.subpixel_patch_radius, 2, 12);
    const int search_radius = std::max(1, dual_cfg.subpixel_search_radius_px);
    const int max_points = std::clamp(dual_cfg.subpixel_max_points, 1, 64);
    const int min_points = std::clamp(dual_cfg.subpixel_min_points, 1, max_points);
    const float max_delta = computeSubpixelDispDeltaGateCPU(
        initial_disp, focal, baseline,
        dual_cfg.subpixel_max_disp_delta_px,
        dual_cfg.subpixel_max_disp_delta_ratio,
        dual_cfg.subpixel_max_depth_delta_m);
    result.delta_gate_px = max_delta;
    const float max_stddev = std::max(0.05f, dual_cfg.subpixel_max_stddev_px);
    const float min_score = std::max(0.10f, dual_cfg.subpixel_min_confidence * 0.60f);
    const float sample_radius = std::min(left_circle.radius, right_circle.radius);
    const auto offsets = makeSubpixelSampleOffsetsCPU(sample_radius,
                                                      max_points,
                                                      patch_radius);
    const int y_radius = std::clamp(
        static_cast<int>(std::ceil(std::clamp(
            dual_cfg.feature_y_tolerance_px, 0.5f, 8.0f))),
        1,
        8);

    std::vector<float> disparities;
    std::vector<float> scores;
    disparities.reserve(offsets.size());
    scores.reserve(offsets.size());

    const int d_start = std::max(1, static_cast<int>(std::floor(initial_disp)) - search_radius);
    const int d_end = std::min(max_disparity,
                               static_cast<int>(std::ceil(initial_disp)) + search_radius);
    if (d_start >= d_end) return result;

    const auto score_at = [&](int x_left, int y_left,
                              int y_right,
                              int disparity) -> float {
        const int x_right = static_cast<int>(std::lround(
            static_cast<float>(x_left) - static_cast<float>(disparity)));
        if (!patchInsideCPU(img_w, img_h, x_right, y_right,
                            patch_radius, dual_cfg.roi_denoise)) {
            return -2.0f;
        }
        return znccPatchCPU(left_img, left_pitch, right_img, right_pitch,
                            x_left, y_left, x_right, y_right,
                            patch_radius, dual_cfg.roi_denoise);
    };

    for (const auto& offset : offsets) {
        const int x_left = static_cast<int>(std::lround(left_circle.cx + offset.dx));
        const int y_left = static_cast<int>(std::lround(left_circle.cy + offset.dy));
        const float expected_y_delta =
            dual_cfg.feature_y_offset_px +
            (left_circle.cy - right_circle.cy) +
            dual_cfg.feature_y_slope *
                (static_cast<float>(x_left) - left_circle.cx);
        const int y_center = static_cast<int>(std::lround(
            static_cast<float>(y_left) - expected_y_delta));

        if (!patchInsideCPU(img_w, img_h, x_left, y_left,
                            patch_radius, dual_cfg.roi_denoise) ||
            !patchInsideCPU(img_w, img_h,
                            static_cast<int>(std::lround(right_circle.cx + offset.dx)),
                            y_center, patch_radius, dual_cfg.roi_denoise)) {
            continue;
        }

        float best_score = -2.0f;
        float second_score = -2.0f;
        int best_disp = -1;
        int best_y_right = y_center;
        for (int disp = d_start; disp <= d_end; ++disp) {
            for (int y_right = y_center - y_radius;
                 y_right <= y_center + y_radius; ++y_right) {
                ++result.attempted;
                const float score = score_at(x_left, y_left, y_right, disp);
                if (score > best_score) {
                    second_score = best_score;
                    best_score = score;
                    best_disp = disp;
                    best_y_right = y_right;
                } else if (score > second_score) {
                    second_score = score;
                }
            }
        }

        if (best_disp < 0 || best_score < min_score) continue;

        float sub_disp = static_cast<float>(best_disp);
        if (best_disp > d_start && best_disp < d_end) {
            const float s_minus =
                score_at(x_left, y_left, best_y_right, best_disp - 1);
            const float s_plus =
                score_at(x_left, y_left, best_y_right, best_disp + 1);
            const float denom = s_minus - 2.0f * best_score + s_plus;
            if (s_minus > -1.5f && s_plus > -1.5f && denom < -1e-5f) {
                const float delta = std::clamp(
                    0.5f * (s_minus - s_plus) / denom,
                    -1.0f, 1.0f);
                sub_disp += delta;
            }
        }

        const float uniqueness_margin =
            second_score > -1.5f ? best_score - second_score : 1.0f;
        if (uniqueness_margin < 0.01f && best_score < 0.75f) continue;
        if (std::abs(sub_disp - initial_disp) > max_delta) continue;

        disparities.push_back(sub_disp);
        scores.push_back(best_score);
    }

    if (static_cast<int>(disparities.size()) < min_points) {
        result.low_confidence = true;
        return result;
    }

    std::vector<float> sorted = disparities;
    std::sort(sorted.begin(), sorted.end());
    const float median = medianOfSortedCPU(sorted);

    std::vector<float> abs_dev;
    abs_dev.reserve(sorted.size());
    for (float d : disparities) {
        abs_dev.push_back(std::abs(d - median));
    }
    std::sort(abs_dev.begin(), abs_dev.end());
    const float mad = medianOfSortedCPU(abs_dev);
    const float inlier_gate = std::max(0.60f, mad * 2.5f);

    double sum_disp = 0.0;
    double sum_score = 0.0;
    int inliers = 0;
    for (size_t i = 0; i < disparities.size(); ++i) {
        if (std::abs(disparities[i] - median) > inlier_gate) continue;
        sum_disp += disparities[i];
        sum_score += scores[i];
        ++inliers;
    }
    if (inliers < min_points) {
        result.low_confidence = true;
        return result;
    }

    const float refined_disp = static_cast<float>(sum_disp / static_cast<double>(inliers));
    double var = 0.0;
    for (float d : disparities) {
        if (std::abs(d - median) > inlier_gate) continue;
        const double diff = static_cast<double>(d - refined_disp);
        var += diff * diff;
    }
    result.stddev = static_cast<float>(
        std::sqrt(var / std::max(1.0, static_cast<double>(inliers))));
    result.support = inliers;
    result.disparity = refined_disp;

    if (result.stddev > max_stddev ||
        std::abs(result.disparity - initial_disp) > max_delta ||
        result.disparity <= 0.5f ||
        result.disparity > static_cast<float>(max_disparity)) {
        result.low_confidence = true;
        return result;
    }

    const float support_ratio = static_cast<float>(inliers) /
                                static_cast<float>(std::max(1, max_points));
    const float mean_score = static_cast<float>(sum_score / static_cast<double>(inliers));
    const float score_conf = std::clamp((mean_score - 0.10f) / 0.80f, 0.0f, 1.0f);
    const float consistency = std::clamp(1.0f / (1.0f + result.stddev),
                                         0.0f, 1.0f);
    const float delta_conf = 1.0f -
        std::min(1.0f, std::abs(result.disparity - initial_disp) / max_delta);
    result.confidence = std::clamp(0.35f * support_ratio +
                                   0.35f * score_conf +
                                   0.20f * consistency +
                                   0.10f * delta_conf,
                                   0.0f, 1.0f);
    if (result.confidence < dual_cfg.subpixel_min_confidence) {
        result.low_confidence = true;
        return result;
    }

    result.valid = true;
    return result;
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
