#include "roi_feature_match_cpu.h"

#include "roi_feature_match_common.h"
#include "roi_patch_match_cpu.h"
#include "roi_feature_match_cpu_helpers.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace stereo3d {

SparseFeatureDisparityResult matchSparseFeatureDisparityCPU(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    bool source_left,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    SparseFeatureMode mode)
{
    SparseFeatureDisparityResult result;
    if (!left_img || !right_img || left_pitch <= 0 || right_pitch <= 0 ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f ||
        initial_disp > static_cast<float>(max_disparity)) {
        return result;
    }

    const int patch_radius = std::clamp(cfg.subpixel_patch_radius, 2, 8);
    const int max_points = std::clamp(std::max(cfg.subpixel_max_points, 16), 4, 48);
    const int min_points = std::clamp(std::max(3, cfg.subpixel_min_points),
                                      3, max_points);
    const int search_radius = std::max(1, cfg.subpixel_search_radius_px);
    const int y_radius = std::clamp(
        static_cast<int>(std::ceil(strictFeatureYTolerance(cfg))),
        1, 3);
    const float max_delta = computeFeatureDeltaGate(initial_disp, focal, baseline, cfg);
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    const bool binary_mode = mode == SparseFeatureMode::BINARY;
    const float min_score = binary_mode
        ? std::max(0.58f, 0.50f + cfg.subpixel_min_confidence * 0.35f)
        : std::max(0.12f, cfg.subpixel_min_confidence * 0.60f);

    NormalizedROIPairCPU norm;
    if (buildNormalizedROIPairCPU(
            left_img, left_pitch, right_img, right_pitch,
            img_w, img_h, left_det, right_det, initial_disp, cfg,
            max_disparity, focal, patch_radius, search_radius, norm)) {
        ROIFeatureMatchConfig norm_cfg =
            makeNormalizedFeatureConfigCPU(cfg, norm.scale);
        const cv::Mat& norm_left =
            (mode == SparseFeatureMode::TEXTURE && !norm.left_edge.empty())
                ? norm.left_edge
                : norm.left_gray;
        const cv::Mat& norm_right =
            (mode == SparseFeatureMode::TEXTURE && !norm.right_edge.empty())
                ? norm.right_edge
                : norm.right_gray;
        SparseFeatureDisparityResult norm_result =
            matchSparseFeatureDisparityCPU(
                norm_left.data, static_cast<int>(norm_left.step[0]),
                norm_right.data, static_cast<int>(norm_right.step[0]),
                norm_left.cols, norm_left.rows,
                norm.left_det, norm.right_det,
                source_left,
                norm.initial_disp,
                norm_cfg,
                norm.max_disparity,
                norm.focal,
                baseline,
                mode);
        return mapNormalizedSparseResultCPU(norm_result, norm);
    }

    const Detection& source_det = source_left ? left_det : right_det;
    const auto points = detectSparseFeaturePointsInBBoxCPU(
        source_left ? left_img : right_img,
        source_left ? left_pitch : right_pitch,
        img_w, img_h, source_det, mode, patch_radius, max_points,
        cfg.roi_denoise, cfg.circle_max_roi_pixels);
    if (static_cast<int>(points.size()) < min_points) {
        result.low_confidence = true;
        return result;
    }

    const uint8_t* source_img = source_left ? left_img : right_img;
    const uint8_t* target_img = source_left ? right_img : left_img;
    const int source_pitch = source_left ? left_pitch : right_pitch;
    const int target_pitch = source_left ? right_pitch : left_pitch;
    const int d_start = std::max(1, static_cast<int>(std::floor(initial_disp)) - search_radius);
    const int d_end = std::min(max_disparity,
                               static_cast<int>(std::ceil(initial_disp)) + search_radius);
    if (d_start >= d_end) return result;

    std::vector<RobustMatchSample> samples;
    samples.reserve(points.size());

    auto score_at = [&](const SparseFeaturePoint& p, int disp, int dy) -> float {
        const int target_x = source_left ? (p.x - disp) : (p.x + disp);
        const int target_y = p.y + dy;
        if (!patchInsideCPU(img_w, img_h, target_x, target_y,
                            patch_radius, cfg.roi_denoise)) {
            return -2.0f;
        }
        if (binary_mode) {
            return censusPatchSimilarityCPU(source_img, source_pitch,
                                            target_img, target_pitch,
                                            p.x, p.y, target_x, target_y,
                                            patch_radius, cfg.roi_denoise);
        }
        return znccPatchCPU(source_img, source_pitch, target_img, target_pitch,
                            p.x, p.y, target_x, target_y,
                            patch_radius, cfg.roi_denoise);
    };

    for (const auto& p : points) {
        if (!patchInsideCPU(img_w, img_h, p.x, p.y,
                            patch_radius, cfg.roi_denoise)) {
            continue;
        }
        ++result.attempted;
        float best_score = -2.0f;
        float second_score = -2.0f;
        int best_disp = -1;
        int best_dy = 0;
        const float approx_left_x = source_left
            ? static_cast<float>(p.x)
            : static_cast<float>(p.x) + initial_disp;
        const float expected_y =
            expectedFeatureYDelta(approx_left_x, left_det, cfg);
        const int dy_center = static_cast<int>(std::lround(
            source_left ? -expected_y : expected_y));
        for (int dy = dy_center - y_radius; dy <= dy_center + y_radius; ++dy) {
            for (int disp = d_start; disp <= d_end; ++disp) {
                const float score = score_at(p, disp, dy);
                if (score > best_score) {
                    second_score = best_score;
                    best_score = score;
                    best_disp = disp;
                    best_dy = dy;
                } else if (score > second_score) {
                    second_score = score;
                }
            }
        }
        if (best_disp < 0 || best_score < min_score) continue;

        float sub_disp = static_cast<float>(best_disp);
        if (best_disp > d_start && best_disp < d_end) {
            const float s_minus = score_at(p, best_disp - 1, best_dy);
            const float s_plus = score_at(p, best_disp + 1, best_dy);
            const float denom = s_minus - 2.0f * best_score + s_plus;
            if (s_minus > -1.5f && s_plus > -1.5f && denom < -1e-5f) {
                sub_disp += std::clamp(
                    0.5f * (s_minus - s_plus) / denom,
                    -1.0f, 1.0f);
            }
        }

        const float uniqueness =
            second_score > -1.5f ? best_score - second_score : 1.0f;
        if ((uniqueness < 0.01f && best_score < 0.75f) ||
            std::abs(sub_disp - initial_disp) > max_delta ||
            sub_disp <= 0.5f ||
            sub_disp > static_cast<float>(max_disparity)) {
            continue;
        }

        RobustMatchSample sample;
        if (source_left) {
            sample.left_x = static_cast<float>(p.x);
            sample.left_y = static_cast<float>(p.y);
            sample.right_x = static_cast<float>(p.x) - sub_disp;
            sample.right_y = static_cast<float>(p.y + best_dy);
        } else {
            sample.left_x = static_cast<float>(p.x) + sub_disp;
            sample.left_y = static_cast<float>(p.y + best_dy);
            sample.right_x = static_cast<float>(p.x);
            sample.right_y = static_cast<float>(p.y);
        }
        sample.disparity = sub_disp;
        sample.score = best_score;

        if (std::abs(featureYResidual(sample, left_det, cfg)) >
                strictFeatureYTolerance(cfg) ||
            !passesFeatureOverlapGate(sample, left_det, right_det,
                                      initial_disp, cfg) ||
            !passesSphereRadiusGate(sample, left_det, initial_disp,
                                    focal, baseline, cfg)) {
            continue;
        }
        if (cfg.feature_reverse_check_px >= 0.0f) {
            const float reverse_err = reverseSparseMatchError(
                left_img, left_pitch, right_img, right_pitch,
                img_w, img_h, sample, patch_radius, d_start, d_end,
                y_radius, binary_mode, cfg.roi_denoise, left_det, cfg);
            if (reverse_err > std::max(0.25f, cfg.feature_reverse_check_px)) {
                continue;
            }
        }
        samples.push_back(sample);
    }

    if (static_cast<int>(samples.size()) < min_points) {
        result.low_confidence = true;
        return result;
    }

    const RobustAggregate robust = aggregateRobustMatches(
        samples, min_points, max_points, initial_disp, max_delta,
        max_stddev, cfg);
    if (!robust.valid) {
        result.low_confidence = true;
        return result;
    }

    result.disparity = robust.disparity;
    result.anchor_cx = robust.anchor_x;
    result.anchor_cy = robust.anchor_y;
    result.right_anchor_cx = robust.right_anchor_x;
    result.right_anchor_cy = robust.right_anchor_y;
    result.support = robust.support;
    copyDebugMatches(robust, result);
    result.stddev = robust.stddev;
    if (result.stddev > max_stddev ||
        std::abs(result.disparity - initial_disp) > max_delta ||
        result.disparity <= 0.5f ||
        result.disparity > static_cast<float>(max_disparity)) {
        result.low_confidence = true;
        return result;
    }

    const float support_ratio = static_cast<float>(robust.support) /
                                static_cast<float>(std::max(1, max_points));
    const float mean_score = robust.mean_score;
    const float score_conf = std::clamp((mean_score - min_score) /
                                        std::max(0.01f, 1.0f - min_score),
                                        0.0f, 1.0f);
    const float consistency = std::clamp(1.0f / (1.0f + result.stddev),
                                         0.0f, 1.0f);
    const float delta_conf = 1.0f -
        std::min(1.0f, std::abs(result.disparity - initial_disp) / max_delta);
    result.confidence = std::clamp(0.30f * support_ratio +
                                   0.35f * score_conf +
                                   0.25f * consistency +
                                   0.10f * delta_conf,
                                   0.0f, 1.0f);
    if (result.confidence < cfg.subpixel_min_confidence) {
        result.low_confidence = true;
        return result;
    }
    result.valid = true;
    return result;
}


}  // namespace stereo3d
