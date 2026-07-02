#include "roi_feature_match_gpu.h"

#include "roi_feature_match_common.h"
#include "../utils/logger.h"

#ifdef HAS_OPENCV_CUDAFEATURES2D
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

namespace stereo3d {

SparseFeatureDisparityResult matchOpenCVORBDisparityGPU(
    const uint8_t* left_gpu, int left_pitch,
    const uint8_t* right_gpu, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    cudaStream_t stream)
{
    SparseFeatureDisparityResult result;
#ifndef HAS_OPENCV_CUDAFEATURES2D
    (void)left_gpu;
    (void)left_pitch;
    (void)right_gpu;
    (void)right_pitch;
    (void)img_w;
    (void)img_h;
    (void)left_det;
    (void)right_det;
    (void)initial_disp;
    (void)cfg;
    (void)max_disparity;
    (void)focal;
    (void)baseline;
    (void)stream;
    static std::once_flag warn_once;
    std::call_once(warn_once, [] {
        LOG_WARN("OpenCV CUDA ORB requested but cudafeatures2d was not available at build time");
    });
    result.low_confidence = true;
    return result;
#else
    if (!left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0 || !stream ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f) {
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
    const cv::Rect left_roi = featureROIFromDetectionCPU(
        left_det, img_w, img_h, border, 0.56f, 2);
    const cv::Rect right_roi = featureROIFromDetectionCPU(
        right_det, img_w, img_h, border, 0.62f, extra_margin);
    if (left_roi.empty() || right_roi.empty()) {
        result.low_confidence = true;
        return result;
    }

    try {
        cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
        cv::cuda::GpuMat left_full(img_h, img_w, CV_8UC1,
                                   const_cast<uint8_t*>(left_gpu),
                                   static_cast<size_t>(left_pitch));
        cv::cuda::GpuMat right_full(img_h, img_w, CV_8UC1,
                                    const_cast<uint8_t*>(right_gpu),
                                    static_cast<size_t>(right_pitch));
        cv::cuda::GpuMat left_view(left_full, left_roi);
        cv::cuda::GpuMat right_view(right_full, right_roi);
        cv::cuda::GpuMat left_proc;
        cv::cuda::GpuMat right_proc;
        left_view.copyTo(left_proc, cv_stream);
        right_view.copyTo(right_proc, cv_stream);

        const int patch_size = std::max(9, patch_radius * 2 + 1);
        const int edge_threshold = std::clamp(patch_radius + 3, 5, 16);
        cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(
            max_points, 1.2f, 3, edge_threshold, 0, 2,
            cv::ORB::HARRIS_SCORE, patch_size, 12, false);
        cv::cuda::GpuMat left_keypoints_gpu;
        cv::cuda::GpuMat right_keypoints_gpu;
        cv::cuda::GpuMat left_desc_gpu;
        cv::cuda::GpuMat right_desc_gpu;
        orb->detectAndComputeAsync(left_proc, cv::noArray(),
                                   left_keypoints_gpu, left_desc_gpu,
                                   false, cv_stream);
        orb->detectAndComputeAsync(right_proc, cv::noArray(),
                                   right_keypoints_gpu, right_desc_gpu,
                                   false, cv_stream);
        cv_stream.waitForCompletion();

        std::vector<cv::KeyPoint> left_keypoints;
        std::vector<cv::KeyPoint> right_keypoints;
        orb->convert(left_keypoints_gpu, left_keypoints);
        orb->convert(right_keypoints_gpu, right_keypoints);
        if (left_keypoints.size() < static_cast<size_t>(min_points) ||
            right_keypoints.size() < static_cast<size_t>(min_points) ||
            left_desc_gpu.empty() || right_desc_gpu.empty() ||
            left_desc_gpu.type() != CV_8U || right_desc_gpu.type() != CV_8U) {
            result.low_confidence = true;
            result.attempted = static_cast<int>(left_keypoints.size());
            return result;
        }

        cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
            cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        cv::cuda::GpuMat knn_gpu;
        cv::cuda::GpuMat reverse_knn_gpu;
        matcher->knnMatchAsync(left_desc_gpu, right_desc_gpu,
                               knn_gpu, 2, cv::noArray(), cv_stream);
        matcher->knnMatchAsync(right_desc_gpu, left_desc_gpu,
                               reverse_knn_gpu, 2, cv::noArray(), cv_stream);
        cv_stream.waitForCompletion();

        std::vector<std::vector<cv::DMatch>> knn_matches;
        std::vector<std::vector<cv::DMatch>> reverse_knn_matches;
        matcher->knnMatchConvert(knn_gpu, knn_matches);
        matcher->knnMatchConvert(reverse_knn_gpu, reverse_knn_matches);
        result.attempted = static_cast<int>(knn_matches.size());
        if (result.attempted < min_points) {
            result.low_confidence = true;
            return result;
        }

        const float ratio_thresh = 0.78f;
        std::vector<int> reverse_best(right_keypoints.size(), -1);
        for (const auto& pair : reverse_knn_matches) {
            if (pair.empty()) continue;
            const cv::DMatch& best = pair[0];
            if (best.queryIdx < 0 || best.trainIdx < 0 ||
                best.queryIdx >= static_cast<int>(right_keypoints.size()) ||
                best.trainIdx >= static_cast<int>(left_keypoints.size())) {
                continue;
            }
            if (pair.size() > 1 && pair[1].distance > 0.0f &&
                best.distance > ratio_thresh * pair[1].distance) {
                continue;
            }
            reverse_best[static_cast<size_t>(best.queryIdx)] = best.trainIdx;
        }

        const float max_hamming = static_cast<float>(
            std::max(1, left_desc_gpu.cols * 8));
        const float min_score = std::max(
            0.45f,
            0.35f + cfg.subpixel_min_confidence * 0.45f);
        std::vector<RobustMatchSample> samples;
        samples.reserve(knn_matches.size());

        for (const auto& pair : knn_matches) {
            if (pair.empty()) continue;
            const cv::DMatch& best = pair[0];
            if (best.queryIdx < 0 || best.trainIdx < 0 ||
                best.queryIdx >= static_cast<int>(left_keypoints.size()) ||
                best.trainIdx >= static_cast<int>(right_keypoints.size())) {
                continue;
            }
            if (pair.size() > 1 && pair[1].distance > 0.0f &&
                best.distance > ratio_thresh * pair[1].distance) {
                continue;
            }
            if (cfg.feature_reverse_check_px >= 0.0f &&
                (best.trainIdx >= static_cast<int>(reverse_best.size()) ||
                 reverse_best[static_cast<size_t>(best.trainIdx)] != best.queryIdx)) {
                continue;
            }

            const cv::KeyPoint& kl = left_keypoints[static_cast<size_t>(best.queryIdx)];
            const cv::KeyPoint& kr = right_keypoints[static_cast<size_t>(best.trainIdx)];
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

            const float score = 1.0f - std::min(1.0f, best.distance / max_hamming);
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
        const float score_conf = std::clamp((robust.mean_score - min_score) /
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
        LOG_WARN("OpenCV CUDA ORB ROI feature match failed: %s", e.what());
        return result;
    }
#endif
}

}  // namespace stereo3d
