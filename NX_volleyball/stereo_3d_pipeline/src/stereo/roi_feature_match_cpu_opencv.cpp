#include "roi_feature_match_cpu.h"

#include "roi_feature_match_common.h"
#include "roi_patch_match_cpu.h"
#include "../utils/logger.h"
#include "roi_feature_match_cpu_helpers.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace stereo3d {

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
    } catch (const cv::Exception& e) {
        LOG_WARN("OpenCV %s ROI feature match failed: %s",
                 openCVFeatureModeName(mode), e.what());
        return result;
    }
}


}  // namespace stereo3d
