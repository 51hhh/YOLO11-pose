#include "roi_feature_match_cpu.h"

#include "roi_feature_match_common.h"
#include "roi_patch_match_cpu.h"
#include "../utils/logger.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>

#include "roi_feature_match_cpu_helpers.h"

namespace stereo3d {

std::string sparseFeatureModeName(SparseFeatureMode mode)
{
    switch (mode) {
    case SparseFeatureMode::CORNER: return "corner";
    case SparseFeatureMode::TEXTURE: return "texture";
    case SparseFeatureMode::BINARY: return "binary";
    }
    return "unknown";
}

const char* openCVFeatureModeName(OpenCVFeatureMode mode)
{
    switch (mode) {
    case OpenCVFeatureMode::ORB: return "ORB";
    case OpenCVFeatureMode::BRISK: return "BRISK";
    case OpenCVFeatureMode::AKAZE: return "AKAZE";
    case OpenCVFeatureMode::SIFT: return "SIFT";
    }
    return "UNKNOWN";
}

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

SparseFeatureDisparityResult matchOpenCVFeatureDisparityCPU(
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
    OpenCVFeatureMode mode)
{
    SparseFeatureDisparityResult result;
    if (!left_img || !right_img || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0 ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f ||
        initial_disp > static_cast<float>(max_disparity)) {
        return result;
    }

    const int patch_radius = std::clamp(cfg.subpixel_patch_radius, 2, 10);
    const int max_points = std::clamp(std::max(cfg.subpixel_max_points * 4, 48),
                                      16, 160);
    const int min_points = std::clamp(std::max(3, cfg.subpixel_min_points),
                                      3, max_points);
    const int search_radius = std::max(1, cfg.subpixel_search_radius_px);
    const float max_delta = computeFeatureDeltaGate(initial_disp, focal, baseline, cfg);
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    const int extra_margin = search_radius + static_cast<int>(std::ceil(max_delta)) + 2;
    const int border = std::max(2, patch_radius);

    NormalizedROIPairCPU norm;
    if (buildNormalizedROIPairCPU(
            left_img, left_pitch, right_img, right_pitch,
            img_w, img_h, left_det, right_det, initial_disp, cfg,
            max_disparity, focal, patch_radius, search_radius, norm)) {
        ROIFeatureMatchConfig norm_cfg =
            makeNormalizedFeatureConfigCPU(cfg, norm.scale);
        SparseFeatureDisparityResult norm_result =
            matchOpenCVFeatureDisparityCPU(
                norm.left_gray.data, static_cast<int>(norm.left_gray.step[0]),
                norm.right_gray.data, static_cast<int>(norm.right_gray.step[0]),
                norm.left_gray.cols, norm.left_gray.rows,
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
    const Detection& target_det = source_left ? right_det : left_det;
    const cv::Rect source_roi = featureROIFromDetectionCPU(
        source_det, img_w, img_h, border, 0.56f, 2);
    const cv::Rect target_roi = featureROIFromDetectionCPU(
        target_det, img_w, img_h, border, 0.62f, extra_margin);
    if (source_roi.empty() || target_roi.empty()) {
        result.low_confidence = true;
        return result;
    }

    try {
        cv::Mat left_full(img_h, img_w, CV_8UC1,
                          const_cast<uint8_t*>(left_img),
                          static_cast<size_t>(left_pitch));
        cv::Mat right_full(img_h, img_w, CV_8UC1,
                           const_cast<uint8_t*>(right_img),
                           static_cast<size_t>(right_pitch));
        const cv::Mat source_view = source_left
            ? left_full(source_roi)
            : right_full(source_roi);
        const cv::Mat target_view = source_left
            ? right_full(target_roi)
            : left_full(target_roi);

        cv::Mat source_proc = source_view;
        cv::Mat target_proc = target_view;
        cv::Mat source_denoised;
        cv::Mat target_denoised;
        if (cfg.roi_denoise) {
            cv::medianBlur(source_view, source_denoised, 3);
            cv::medianBlur(target_view, target_denoised, 3);
            source_proc = source_denoised;
            target_proc = target_denoised;
        }

        auto extractor = createOpenCVFeatureExtractorCPU(mode, max_points, patch_radius);
        if (!extractor) return result;

        std::vector<cv::KeyPoint> source_keypoints;
        std::vector<cv::KeyPoint> target_keypoints;
        cv::Mat source_descriptors;
        cv::Mat target_descriptors;
        detectAndDescribeOpenCVFeatureCPU(
            *extractor, source_proc, max_points, source_keypoints, source_descriptors);
        detectAndDescribeOpenCVFeatureCPU(
            *extractor, target_proc, max_points, target_keypoints, target_descriptors);
        if (source_keypoints.size() < static_cast<size_t>(min_points) ||
            target_keypoints.size() < static_cast<size_t>(min_points) ||
            source_descriptors.empty() || target_descriptors.empty() ||
            !descriptorDepthCompatible(mode, source_descriptors) ||
            !descriptorDepthCompatible(mode, target_descriptors)) {
            result.low_confidence = true;
            return result;
        }

        const float ratio_thresh = descriptorRatioThreshold(mode);
        cv::BFMatcher matcher(descriptorNormType(mode), false);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(source_descriptors, target_descriptors, knn_matches, 2);
        std::vector<std::vector<cv::DMatch>> reverse_knn_matches;
        matcher.knnMatch(target_descriptors, source_descriptors, reverse_knn_matches, 2);
        std::vector<int> reverse_best(target_keypoints.size(), -1);
        for (const auto& pair : reverse_knn_matches) {
            if (pair.empty()) continue;
            const cv::DMatch& best = pair[0];
            if (best.queryIdx < 0 || best.trainIdx < 0 ||
                best.queryIdx >= static_cast<int>(target_keypoints.size()) ||
                best.trainIdx >= static_cast<int>(source_keypoints.size())) {
                continue;
            }
            if (pair.size() > 1 && pair[1].distance > 0.0f &&
                best.distance > ratio_thresh * pair[1].distance) {
                continue;
            }
            reverse_best[best.queryIdx] = best.trainIdx;
        }
        result.attempted = static_cast<int>(knn_matches.size());
        if (result.attempted < min_points) {
            result.low_confidence = true;
            return result;
        }

        const float min_score = descriptorMinScore(mode, cfg);
        std::vector<RobustMatchSample> samples;
        samples.reserve(knn_matches.size());

        for (const auto& pair : knn_matches) {
            if (pair.empty()) continue;
            const cv::DMatch& best = pair[0];
            if (best.queryIdx < 0 || best.trainIdx < 0 ||
                best.queryIdx >= static_cast<int>(source_keypoints.size()) ||
                best.trainIdx >= static_cast<int>(target_keypoints.size())) {
                continue;
            }
            if (pair.size() > 1 && pair[1].distance > 0.0f &&
                best.distance > ratio_thresh * pair[1].distance) {
                continue;
            }
            if (cfg.feature_reverse_check_px >= 0.0f &&
                (best.trainIdx >= static_cast<int>(reverse_best.size()) ||
                 reverse_best[best.trainIdx] != best.queryIdx)) {
                continue;
            }

            const cv::KeyPoint& ks = source_keypoints[best.queryIdx];
            const cv::KeyPoint& kt = target_keypoints[best.trainIdx];
            const float source_x = static_cast<float>(source_roi.x) + ks.pt.x;
            const float source_y = static_cast<float>(source_roi.y) + ks.pt.y;
            const float target_x = static_cast<float>(target_roi.x) + kt.pt.x;
            const float target_y = static_cast<float>(target_roi.y) + kt.pt.y;
            const float disparity = source_left
                ? (source_x - target_x)
                : (target_x - source_x);
            if (disparity <= 0.5f ||
                disparity > static_cast<float>(max_disparity) ||
                std::abs(disparity - initial_disp) > max_delta) {
                continue;
            }

            const float score = descriptorMatchScore(
                mode, best.distance, source_descriptors.cols);
            if (score < min_score) continue;

            RobustMatchSample sample;
            if (source_left) {
                sample.left_x = source_x;
                sample.left_y = source_y;
                sample.right_x = target_x;
                sample.right_y = target_y;
            } else {
                sample.left_x = target_x;
                sample.left_y = target_y;
                sample.right_x = source_x;
                sample.right_y = source_y;
            }
            sample.disparity = disparity;
            sample.score = score;
            if (std::abs(featureYResidual(sample, left_det, cfg)) >
                    strictFeatureYTolerance(cfg) ||
                !passesFeatureOverlapGate(sample, left_det, right_det,
                                          initial_disp, cfg) ||
                !passesSphereRadiusGate(sample, left_det, initial_disp,
                                        focal, baseline, cfg)) {
                continue;
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
    } catch (const cv::Exception& e) {
        LOG_WARN("OpenCV %s ROI feature match failed: %s",
                 openCVFeatureModeName(mode), e.what());
        return result;
    }
}

DebugFeatureMatchResult makeDebugSparseFeatureMatchesCPU(
    const cv::Mat& left_gray,
    const cv::Mat& right_gray,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    SparseFeatureMode mode)
{
    DebugFeatureMatchResult out;
    out.name = sparseFeatureModeName(mode);
    if (left_gray.empty() || right_gray.empty() ||
        left_gray.type() != CV_8UC1 || right_gray.type() != CV_8UC1 ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f) {
        return out;
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
            left_gray.data, static_cast<int>(left_gray.step[0]),
            right_gray.data, static_cast<int>(right_gray.step[0]),
            left_gray.cols, left_gray.rows,
            left_det, right_det, initial_disp, cfg,
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
        DebugFeatureMatchResult norm_result = makeDebugSparseFeatureMatchesCPU(
            norm_left, norm_right,
            norm.left_det, norm.right_det,
            norm.initial_disp,
            norm_cfg,
            norm.max_disparity,
            norm.focal,
            baseline,
            mode);
        mapNormalizedDebugResultCPU(norm_result, norm);
        return norm_result;
    }

    const auto points = detectSparseFeaturePointsInBBoxCPU(
        left_gray.data, static_cast<int>(left_gray.step[0]),
        left_gray.cols, left_gray.rows, left_det, mode, patch_radius, max_points,
        cfg.roi_denoise, cfg.circle_max_roi_pixels);
    out.left_keypoints.reserve(points.size());
    for (const auto& p : points) {
        out.left_keypoints.emplace_back(cv::Point2f(static_cast<float>(p.x),
                                                    static_cast<float>(p.y)),
                                        static_cast<float>(patch_radius * 2 + 1));
    }
    if (static_cast<int>(points.size()) < min_points) return out;

    out.right_keypoints.reserve(points.size());
    std::vector<RobustMatchSample> samples;
    std::vector<cv::DMatch> candidates;
    samples.reserve(points.size());
    candidates.reserve(points.size());

    const int d_start = std::max(1, static_cast<int>(std::floor(initial_disp)) - search_radius);
    const int d_end = std::min(max_disparity,
                               static_cast<int>(std::ceil(initial_disp)) + search_radius);
    if (d_start >= d_end) return out;

    auto score_at = [&](const SparseFeaturePoint& p, int disp, int dy) -> float {
        const int xr = p.x - disp;
        const int yr = p.y + dy;
        if (!patchInsideCPU(left_gray.cols, left_gray.rows, xr, yr,
                            patch_radius, cfg.roi_denoise)) {
            return -2.0f;
        }
        if (binary_mode) {
            return censusPatchSimilarityCPU(left_gray.data, static_cast<int>(left_gray.step[0]),
                                            right_gray.data, static_cast<int>(right_gray.step[0]),
                                            p.x, p.y, xr, yr,
                                            patch_radius, cfg.roi_denoise);
        }
        return znccPatchCPU(left_gray.data, static_cast<int>(left_gray.step[0]),
                            right_gray.data, static_cast<int>(right_gray.step[0]),
                            p.x, p.y, xr, yr, patch_radius, cfg.roi_denoise);
    };

    for (size_t point_idx = 0; point_idx < points.size(); ++point_idx) {
        const auto& p = points[point_idx];
        if (!patchInsideCPU(left_gray.cols, left_gray.rows, p.x, p.y,
                            patch_radius, cfg.roi_denoise)) {
            continue;
        }
        float best_score = -2.0f;
        float second_score = -2.0f;
        int best_disp = -1;
        int best_dy = 0;
        const float expected_y = expectedFeatureYDelta(
            static_cast<float>(p.x), left_det, cfg);
        const int dy_center = static_cast<int>(std::lround(-expected_y));
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

        const int t = static_cast<int>(out.right_keypoints.size());
        const float xr = static_cast<float>(p.x) - sub_disp;
        const float yr = static_cast<float>(p.y + best_dy);
        RobustMatchSample sample;
        sample.left_x = static_cast<float>(p.x);
        sample.left_y = static_cast<float>(p.y);
        sample.right_x = xr;
        sample.right_y = yr;
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
                left_gray.data, static_cast<int>(left_gray.step[0]),
                right_gray.data, static_cast<int>(right_gray.step[0]),
                left_gray.cols, left_gray.rows, sample, patch_radius,
                d_start, d_end, y_radius, binary_mode, cfg.roi_denoise,
                left_det, cfg);
            if (reverse_err > std::max(0.25f, cfg.feature_reverse_check_px)) {
                continue;
            }
        }
        out.right_keypoints.emplace_back(cv::Point2f(xr, yr),
                                         static_cast<float>(patch_radius * 2 + 1));
        candidates.emplace_back(static_cast<int>(point_idx), t, 1.0f - best_score);
        samples.push_back(sample);
    }
    out.attempted_matches = static_cast<int>(candidates.size());

    if (static_cast<int>(samples.size()) < min_points) return out;

    const RobustAggregate robust = aggregateRobustMatches(
        samples, min_points, max_points, initial_disp, max_delta,
        max_stddev, cfg);
    if (!robust.valid) return out;

    const float display_gate = std::max(
        std::clamp(cfg.feature_ransac_gate_px, 0.25f, 3.0f),
        robust.stddev * std::max(2.0f, cfg.feature_mad_scale));
    for (size_t i = 0; i < samples.size() && i < candidates.size(); ++i) {
        if (std::abs(samples[i].disparity - robust.disparity) > display_gate) {
            continue;
        }
        out.matches.push_back(candidates[i]);
    }
    if (static_cast<int>(out.matches.size()) < min_points) {
        out.matches.clear();
        return out;
    }

    out.disparity = robust.disparity;
    out.stddev = robust.stddev;
    if (out.stddev > max_stddev ||
        std::abs(out.disparity - initial_disp) > max_delta) {
        out.matches.clear();
        return out;
    }
    out.confidence = std::clamp(
        0.35f * (static_cast<float>(robust.support) / static_cast<float>(std::max(1, max_points))) +
        0.35f * (robust.mean_score - min_score) /
            std::max(0.01f, 1.0f - min_score) +
        0.20f * std::clamp(1.0f / (1.0f + out.stddev), 0.0f, 1.0f),
        0.0f, 1.0f);
    return out;
}

DebugFeatureMatchResult makeDebugOpenCVFeatureMatchesCPU(
    const cv::Mat& left_gray,
    const cv::Mat& right_gray,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    OpenCVFeatureMode mode)
{
    DebugFeatureMatchResult out;
    out.name = openCVFeatureModeName(mode);
    std::transform(out.name.begin(), out.name.end(), out.name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (left_gray.empty() || right_gray.empty() ||
        left_gray.type() != CV_8UC1 || right_gray.type() != CV_8UC1 ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f) {
        return out;
    }

    const int patch_radius = std::clamp(cfg.subpixel_patch_radius, 2, 10);
    const int max_points = std::clamp(std::max(cfg.subpixel_max_points * 4, 48),
                                      16, 160);
    const int min_points = std::clamp(std::max(3, cfg.subpixel_min_points),
                                      3, max_points);
    const int search_radius = std::max(1, cfg.subpixel_search_radius_px);
    const float max_delta = computeFeatureDeltaGate(initial_disp, focal, baseline, cfg);
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    const int extra_margin = search_radius + static_cast<int>(std::ceil(max_delta)) + 2;
    const int border = std::max(2, patch_radius);

    NormalizedROIPairCPU norm;
    if (buildNormalizedROIPairCPU(
            left_gray.data, static_cast<int>(left_gray.step[0]),
            right_gray.data, static_cast<int>(right_gray.step[0]),
            left_gray.cols, left_gray.rows,
            left_det, right_det, initial_disp, cfg,
            max_disparity, focal, patch_radius, search_radius, norm)) {
        ROIFeatureMatchConfig norm_cfg =
            makeNormalizedFeatureConfigCPU(cfg, norm.scale);
        DebugFeatureMatchResult norm_result = makeDebugOpenCVFeatureMatchesCPU(
            norm.left_gray, norm.right_gray,
            norm.left_det, norm.right_det,
            norm.initial_disp,
            norm_cfg,
            norm.max_disparity,
            norm.focal,
            baseline,
            mode);
        mapNormalizedDebugResultCPU(norm_result, norm);
        return norm_result;
    }

    const cv::Rect left_roi = featureROIFromDetectionCPU(
        left_det, left_gray.cols, left_gray.rows, border, 0.56f, 2);
    const cv::Rect right_roi = featureROIFromDetectionCPU(
        right_det, right_gray.cols, right_gray.rows, border, 0.62f, extra_margin);
    if (left_roi.empty() || right_roi.empty()) return out;

    try {
        cv::Mat left_view = left_gray(left_roi);
        cv::Mat right_view = right_gray(right_roi);
        cv::Mat left_proc = left_view;
        cv::Mat right_proc = right_view;
        cv::Mat left_denoised;
        cv::Mat right_denoised;
        if (cfg.roi_denoise) {
            cv::medianBlur(left_view, left_denoised, 3);
            cv::medianBlur(right_view, right_denoised, 3);
            left_proc = left_denoised;
            right_proc = right_denoised;
        }

        auto extractor = createOpenCVFeatureExtractorCPU(mode, max_points, patch_radius);
        if (!extractor) return out;

        std::vector<cv::KeyPoint> left_local;
        std::vector<cv::KeyPoint> right_local;
        cv::Mat left_desc;
        cv::Mat right_desc;
        detectAndDescribeOpenCVFeatureCPU(*extractor, left_proc, max_points,
                                          left_local, left_desc);
        detectAndDescribeOpenCVFeatureCPU(*extractor, right_proc, max_points,
                                          right_local, right_desc);
        out.left_keypoints.reserve(left_local.size());
        for (const auto& kp : left_local) {
            cv::KeyPoint global_kp = kp;
            global_kp.pt.x += static_cast<float>(left_roi.x);
            global_kp.pt.y += static_cast<float>(left_roi.y);
            out.left_keypoints.push_back(global_kp);
        }
        out.right_keypoints.reserve(right_local.size());
        for (const auto& kp : right_local) {
            cv::KeyPoint global_kp = kp;
            global_kp.pt.x += static_cast<float>(right_roi.x);
            global_kp.pt.y += static_cast<float>(right_roi.y);
            out.right_keypoints.push_back(global_kp);
        }
        if (left_local.size() < static_cast<size_t>(min_points) ||
            right_local.size() < static_cast<size_t>(min_points) ||
            left_desc.empty() || right_desc.empty() ||
            !descriptorDepthCompatible(mode, left_desc) ||
            !descriptorDepthCompatible(mode, right_desc)) {
            return out;
        }

        const float ratio_thresh = descriptorRatioThreshold(mode);
        cv::BFMatcher matcher(descriptorNormType(mode), false);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(left_desc, right_desc, knn_matches, 2);
        std::vector<std::vector<cv::DMatch>> reverse_knn_matches;
        matcher.knnMatch(right_desc, left_desc, reverse_knn_matches, 2);
        std::vector<int> reverse_best(right_local.size(), -1);
        for (const auto& pair : reverse_knn_matches) {
            if (pair.empty()) continue;
            const cv::DMatch& best = pair[0];
            if (best.queryIdx < 0 || best.trainIdx < 0 ||
                best.queryIdx >= static_cast<int>(right_local.size()) ||
                best.trainIdx >= static_cast<int>(left_local.size())) {
                continue;
            }
            if (pair.size() > 1 && pair[1].distance > 0.0f &&
                best.distance > ratio_thresh * pair[1].distance) {
                continue;
            }
            reverse_best[best.queryIdx] = best.trainIdx;
        }
        const float min_score = descriptorMinScore(mode, cfg);

        std::vector<RobustMatchSample> samples;
        std::vector<cv::DMatch> candidates;
        samples.reserve(knn_matches.size());
        candidates.reserve(knn_matches.size());

        for (const auto& pair : knn_matches) {
            if (pair.empty()) continue;
            const cv::DMatch& best = pair[0];
            if (best.queryIdx < 0 || best.trainIdx < 0 ||
                best.queryIdx >= static_cast<int>(left_local.size()) ||
                best.trainIdx >= static_cast<int>(right_local.size())) {
                continue;
            }
            if (pair.size() > 1 && pair[1].distance > 0.0f &&
                best.distance > ratio_thresh * pair[1].distance) {
                continue;
            }
            if (cfg.feature_reverse_check_px >= 0.0f &&
                (best.trainIdx >= static_cast<int>(reverse_best.size()) ||
                 reverse_best[best.trainIdx] != best.queryIdx)) {
                continue;
            }

            const cv::KeyPoint& kl = left_local[best.queryIdx];
            const cv::KeyPoint& kr = right_local[best.trainIdx];
            const float lx = static_cast<float>(left_roi.x) + kl.pt.x;
            const float ly = static_cast<float>(left_roi.y) + kl.pt.y;
            const float rx = static_cast<float>(right_roi.x) + kr.pt.x;
            const float ry = static_cast<float>(right_roi.y) + kr.pt.y;
            const float disparity = lx - rx;
            if (disparity <= 0.5f ||
                disparity > static_cast<float>(max_disparity) ||
                std::abs(disparity - initial_disp) > max_delta) {
                continue;
            }

            const float score = descriptorMatchScore(mode, best.distance,
                                                     left_desc.cols);
            if (score < min_score) continue;

            RobustMatchSample sample;
            sample.left_x = lx;
            sample.left_y = ly;
            sample.right_x = rx;
            sample.right_y = ry;
            sample.disparity = disparity;
            sample.score = score;
            if (std::abs(featureYResidual(sample, left_det, cfg)) >
                    strictFeatureYTolerance(cfg) ||
                !passesFeatureOverlapGate(sample, left_det, right_det,
                                          initial_disp, cfg) ||
                !passesSphereRadiusGate(sample, left_det, initial_disp,
                                        focal, baseline, cfg)) {
                continue;
            }
            candidates.emplace_back(best.queryIdx, best.trainIdx, best.distance);
            samples.push_back(sample);
        }
        out.attempted_matches = static_cast<int>(candidates.size());

        if (static_cast<int>(samples.size()) < min_points) return out;
        const RobustAggregate robust = aggregateRobustMatches(
            samples, min_points, max_points, initial_disp, max_delta,
            max_stddev, cfg);
        if (!robust.valid) return out;

        const float display_gate = std::max(
            std::clamp(cfg.feature_ransac_gate_px, 0.25f, 3.0f),
            robust.stddev * std::max(2.0f, cfg.feature_mad_scale));
        for (size_t i = 0; i < samples.size() && i < candidates.size(); ++i) {
            if (std::abs(samples[i].disparity - robust.disparity) > display_gate) {
                continue;
            }
            out.matches.push_back(candidates[i]);
        }
        if (static_cast<int>(out.matches.size()) < min_points) {
            out.matches.clear();
            return out;
        }
        out.disparity = robust.disparity;
        out.stddev = robust.stddev;
        if (out.stddev > max_stddev ||
            std::abs(out.disparity - initial_disp) > max_delta) {
            out.matches.clear();
            return out;
        }
        out.confidence = std::clamp(
            0.30f * (static_cast<float>(robust.support) / static_cast<float>(std::max(1, max_points))) +
            0.35f * (robust.mean_score - min_score) /
                std::max(0.01f, 1.0f - min_score) +
            0.25f * std::clamp(1.0f / (1.0f + out.stddev), 0.0f, 1.0f),
            0.0f, 1.0f);
        return out;
    } catch (const cv::Exception& e) {
        LOG_WARN("Debug OpenCV %s match failed: %s",
                 openCVFeatureModeName(mode), e.what());
        return out;
    }
}

}  // namespace stereo3d
