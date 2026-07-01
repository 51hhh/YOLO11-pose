#include "roi_feature_match_cpu.h"

#include "roi_patch_match_cpu.h"
#include "../utils/logger.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>

namespace stereo3d {
namespace {

struct SparseFeaturePoint {
    int x = 0;
    int y = 0;
    float response = 0.0f;
};

std::vector<SparseFeaturePoint> detectSparseFeaturePointsInBBoxCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const Detection& det,
    SparseFeatureMode mode,
    int patch_radius,
    int max_points,
    bool denoise,
    int max_roi_pixels)
{
    std::vector<SparseFeaturePoint> selected;
    if (!img || pitch <= 0 || img_w <= 0 || img_h <= 0 ||
        det.width < 10.0f || det.height < 10.0f) {
        return selected;
    }

    max_points = std::clamp(max_points, 4, 64);
    patch_radius = std::clamp(patch_radius, 2, 12);
    const int border = patch_radius + (denoise ? 2 : 1);
    int x1 = static_cast<int>(std::floor(det.cx - det.width * 0.46f));
    int y1 = static_cast<int>(std::floor(det.cy - det.height * 0.46f));
    int x2 = static_cast<int>(std::ceil(det.cx + det.width * 0.46f));
    int y2 = static_cast<int>(std::ceil(det.cy + det.height * 0.46f));
    x1 = std::max(border, x1);
    y1 = std::max(border, y1);
    x2 = std::min(img_w - 1 - border, x2);
    y2 = std::min(img_h - 1 - border, y2);
    if (x2 - x1 < patch_radius * 2 + 4 ||
        y2 - y1 < patch_radius * 2 + 4) {
        return selected;
    }

    const int roi_w = x2 - x1 + 1;
    const int roi_h = y2 - y1 + 1;
    const int area = roi_w * roi_h;
    const int stride = std::max(
        1, static_cast<int>(std::ceil(std::sqrt(
               static_cast<float>(area) /
               static_cast<float>(std::max(256, max_roi_pixels))))));
    const float rx = std::max(1.0f, det.width * 0.46f);
    const float ry = std::max(1.0f, det.height * 0.46f);
    const float inner_gate = 0.92f;

    std::vector<SparseFeaturePoint> candidates;
    candidates.reserve(static_cast<size_t>(area / std::max(1, stride * stride)));
    float best_response = 0.0f;

    auto local_variance = [&](int x, int y) -> float {
        double sum = 0.0;
        double sum2 = 0.0;
        int n = 0;
        for (int yy = -1; yy <= 1; ++yy) {
            for (int xx = -1; xx <= 1; ++xx) {
                const double v = sampleGrayCPU(img, pitch, x + xx, y + yy, denoise);
                sum += v;
                sum2 += v * v;
                ++n;
            }
        }
        const double mean = sum / std::max(1, n);
        return static_cast<float>(std::max(0.0, sum2 / std::max(1, n) - mean * mean));
    };

    for (int y = y1; y <= y2; y += stride) {
        for (int x = x1; x <= x2; x += stride) {
            const float nx = (static_cast<float>(x) - det.cx) / rx;
            const float ny = (static_cast<float>(y) - det.cy) / ry;
            if (nx * nx + ny * ny > inner_gate * inner_gate) continue;
            if (!patchInsideCPU(img_w, img_h, x, y, patch_radius, denoise)) continue;

            float response = 0.0f;
            if (mode == SparseFeatureMode::CORNER ||
                mode == SparseFeatureMode::BINARY) {
                double sxx = 0.0;
                double syy = 0.0;
                double sxy = 0.0;
                for (int yy = -1; yy <= 1; ++yy) {
                    for (int xx = -1; xx <= 1; ++xx) {
                        const float gx = sampleGrayCPU(img, pitch, x + xx + 1, y + yy, denoise) -
                                         sampleGrayCPU(img, pitch, x + xx - 1, y + yy, denoise);
                        const float gy = sampleGrayCPU(img, pitch, x + xx, y + yy + 1, denoise) -
                                         sampleGrayCPU(img, pitch, x + xx, y + yy - 1, denoise);
                        sxx += static_cast<double>(gx) * gx;
                        syy += static_cast<double>(gy) * gy;
                        sxy += static_cast<double>(gx) * gy;
                    }
                }
                const double tr = sxx + syy;
                const double det_m = sxx * syy - sxy * sxy;
                if (tr > 1e-6 && det_m > 0.0) {
                    const double disc = std::max(0.0, tr * tr - 4.0 * det_m);
                    response = static_cast<float>(0.5 * (tr - std::sqrt(disc)));
                }
                if (mode == SparseFeatureMode::BINARY) {
                    const float gx = sampleGrayCPU(img, pitch, x + 1, y, denoise) -
                                     sampleGrayCPU(img, pitch, x - 1, y, denoise);
                    const float gy = sampleGrayCPU(img, pitch, x, y + 1, denoise) -
                                     sampleGrayCPU(img, pitch, x, y - 1, denoise);
                    const float mag = std::sqrt(gx * gx + gy * gy);
                    const float texture = mag * std::sqrt(std::max(0.0f, local_variance(x, y)));
                    response = 0.65f * response + 0.35f * texture;
                }
            } else {
                const float gx = sampleGrayCPU(img, pitch, x + 1, y, denoise) -
                                 sampleGrayCPU(img, pitch, x - 1, y, denoise);
                const float gy = sampleGrayCPU(img, pitch, x, y + 1, denoise) -
                                 sampleGrayCPU(img, pitch, x, y - 1, denoise);
                const float mag = std::sqrt(gx * gx + gy * gy);
                response = mag * std::sqrt(std::max(0.0f, local_variance(x, y)));
            }

            if (response <= 1e-3f) continue;
            candidates.push_back({x, y, response});
            best_response = std::max(best_response, response);
        }
    }

    if (candidates.empty() || best_response <= 0.0f) return selected;
    std::sort(candidates.begin(), candidates.end(),
              [](const SparseFeaturePoint& a, const SparseFeaturePoint& b) {
                  return a.response > b.response;
              });

    const float min_response = best_response * 0.12f;
    const float min_dist = std::max(3.0f, std::min(det.width, det.height) * 0.08f);
    const float min_dist2 = min_dist * min_dist;
    selected.reserve(static_cast<size_t>(max_points));
    for (const auto& p : candidates) {
        if (p.response < min_response) break;
        bool too_close = false;
        for (const auto& kept : selected) {
            const float dx = static_cast<float>(p.x - kept.x);
            const float dy = static_cast<float>(p.y - kept.y);
            if (dx * dx + dy * dy < min_dist2) {
                too_close = true;
                break;
            }
        }
        if (too_close) continue;
        selected.push_back(p);
        if (static_cast<int>(selected.size()) >= max_points) break;
    }
    return selected;
}

cv::Rect featureROIFromDetectionCPU(
    const Detection& det,
    int img_w,
    int img_h,
    int border,
    float scale,
    int extra_margin)
{
    if (img_w <= 0 || img_h <= 0 || det.width < 6.0f || det.height < 6.0f) {
        return {};
    }
    int x1 = static_cast<int>(std::floor(det.cx - det.width * scale)) - extra_margin;
    int y1 = static_cast<int>(std::floor(det.cy - det.height * scale)) - extra_margin;
    int x2 = static_cast<int>(std::ceil(det.cx + det.width * scale)) + extra_margin;
    int y2 = static_cast<int>(std::ceil(det.cy + det.height * scale)) + extra_margin;
    x1 = std::max(border, x1);
    y1 = std::max(border, y1);
    x2 = std::min(img_w - 1 - border, x2);
    y2 = std::min(img_h - 1 - border, y2);
    if (x2 <= x1 || y2 <= y1) return {};
    return cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
}

cv::Ptr<cv::Feature2D> createOpenCVFeatureExtractorCPU(
    OpenCVFeatureMode mode,
    int max_features,
    int patch_radius)
{
    max_features = std::clamp(max_features, 16, 256);
    patch_radius = std::clamp(patch_radius, 2, 12);
    const int patch_size = std::max(9, patch_radius * 2 + 1);
    const int edge_threshold = std::clamp(patch_radius + 3, 5, 16);

    switch (mode) {
    case OpenCVFeatureMode::ORB:
        return cv::ORB::create(max_features, 1.2f, 3, edge_threshold, 0, 2,
                               cv::ORB::HARRIS_SCORE, patch_size, 12);
    case OpenCVFeatureMode::BRISK:
        return cv::BRISK::create(18, 2, 1.0f);
    case OpenCVFeatureMode::AKAZE: {
        auto akaze = cv::AKAZE::create();
        akaze->setDescriptorType(cv::AKAZE::DESCRIPTOR_MLDB);
        akaze->setThreshold(0.0015);
        akaze->setNOctaves(2);
        akaze->setNOctaveLayers(2);
        return akaze;
    }
    }
    return {};
}

void detectAndDescribeOpenCVFeatureCPU(
    cv::Feature2D& extractor,
    const cv::Mat& image,
    int max_points,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors)
{
    keypoints.clear();
    descriptors.release();
    extractor.detect(image, keypoints);
    if (keypoints.empty()) return;

    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });
    if (static_cast<int>(keypoints.size()) > max_points) {
        keypoints.resize(static_cast<size_t>(max_points));
    }
    extractor.compute(image, keypoints, descriptors);
}

float computeFeatureDeltaGate(
    float initial_disp,
    float focal,
    float baseline,
    const ROIFeatureMatchConfig& cfg)
{
    return computeSubpixelDispDeltaGateCPU(
        initial_disp, focal, baseline,
        cfg.subpixel_max_disp_delta_px,
        cfg.subpixel_max_disp_delta_ratio,
        cfg.subpixel_max_depth_delta_m);
}

}  // namespace

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
        static_cast<int>(std::lround(cfg.epipolar_y_tolerance * 0.20f)),
        1, 3);
    const float max_delta = computeFeatureDeltaGate(initial_disp, focal, baseline, cfg);
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    const bool binary_mode = mode == SparseFeatureMode::BINARY;
    const float min_score = binary_mode
        ? std::max(0.58f, 0.50f + cfg.subpixel_min_confidence * 0.35f)
        : std::max(0.12f, cfg.subpixel_min_confidence * 0.60f);

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

    std::vector<float> disparities;
    std::vector<float> scores;
    std::vector<float> anchors_x;
    std::vector<float> anchors_y;
    disparities.reserve(points.size());
    scores.reserve(points.size());
    anchors_x.reserve(points.size());
    anchors_y.reserve(points.size());

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
        for (int dy = -y_radius; dy <= y_radius; ++dy) {
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

        disparities.push_back(sub_disp);
        scores.push_back(best_score);
        if (source_left) {
            anchors_x.push_back(static_cast<float>(p.x));
            anchors_y.push_back(static_cast<float>(p.y));
        } else {
            anchors_x.push_back(static_cast<float>(p.x) + sub_disp);
            anchors_y.push_back(static_cast<float>(p.y + best_dy));
        }
    }

    if (static_cast<int>(disparities.size()) < min_points) {
        result.low_confidence = true;
        return result;
    }

    std::vector<float> sorted = disparities;
    std::sort(sorted.begin(), sorted.end());
    const float median = medianOfSortedCPU(sorted);

    std::vector<float> abs_dev;
    abs_dev.reserve(disparities.size());
    for (float d : disparities) abs_dev.push_back(std::abs(d - median));
    std::sort(abs_dev.begin(), abs_dev.end());
    const float mad = medianOfSortedCPU(abs_dev);
    const float inlier_gate = std::max(0.60f, mad * 2.5f);

    double sum_disp = 0.0;
    double sum_score = 0.0;
    double sum_anchor_x = 0.0;
    double sum_anchor_y = 0.0;
    int inliers = 0;
    for (size_t i = 0; i < disparities.size(); ++i) {
        if (std::abs(disparities[i] - median) > inlier_gate) continue;
        const double w = std::max(0.05f, scores[i]);
        sum_disp += w * static_cast<double>(disparities[i]);
        sum_anchor_x += w * static_cast<double>(anchors_x[i]);
        sum_anchor_y += w * static_cast<double>(anchors_y[i]);
        sum_score += w;
        ++inliers;
    }
    if (inliers < min_points || sum_score <= 0.0) {
        result.low_confidence = true;
        return result;
    }

    result.disparity = static_cast<float>(sum_disp / sum_score);
    result.anchor_cx = static_cast<float>(sum_anchor_x / sum_score);
    result.anchor_cy = static_cast<float>(sum_anchor_y / sum_score);
    result.support = inliers;
    double var = 0.0;
    for (size_t i = 0; i < disparities.size(); ++i) {
        if (std::abs(disparities[i] - median) > inlier_gate) continue;
        const double diff = static_cast<double>(disparities[i] - result.disparity);
        var += diff * diff;
    }
    result.stddev = static_cast<float>(
        std::sqrt(var / std::max(1.0, static_cast<double>(inliers))));
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
    const int y_radius = std::clamp(
        static_cast<int>(std::lround(cfg.epipolar_y_tolerance * 0.30f)),
        1, 5);
    const float max_delta = computeFeatureDeltaGate(initial_disp, focal, baseline, cfg);
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    const int extra_margin = search_radius + static_cast<int>(std::ceil(max_delta)) + 2;
    const int border = std::max(2, patch_radius);

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
            source_descriptors.depth() != CV_8U ||
            target_descriptors.depth() != CV_8U) {
            result.low_confidence = true;
            return result;
        }

        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(source_descriptors, target_descriptors, knn_matches, 2);
        result.attempted = static_cast<int>(knn_matches.size());
        if (result.attempted < min_points) {
            result.low_confidence = true;
            return result;
        }

        const float ratio_thresh = 0.78f;
        const float max_hamming = static_cast<float>(
            std::max(1, source_descriptors.cols * 8));
        const float min_score = std::max(0.45f,
            0.35f + cfg.subpixel_min_confidence * 0.45f);
        std::vector<float> disparities;
        std::vector<float> scores;
        std::vector<float> anchors_x;
        std::vector<float> anchors_y;
        disparities.reserve(knn_matches.size());
        scores.reserve(knn_matches.size());
        anchors_x.reserve(knn_matches.size());
        anchors_y.reserve(knn_matches.size());

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

            const cv::KeyPoint& ks = source_keypoints[best.queryIdx];
            const cv::KeyPoint& kt = target_keypoints[best.trainIdx];
            const float source_x = static_cast<float>(source_roi.x) + ks.pt.x;
            const float source_y = static_cast<float>(source_roi.y) + ks.pt.y;
            const float target_x = static_cast<float>(target_roi.x) + kt.pt.x;
            const float target_y = static_cast<float>(target_roi.y) + kt.pt.y;
            const float disparity = source_left
                ? (source_x - target_x)
                : (target_x - source_x);
            if (std::abs(source_y - target_y) > static_cast<float>(y_radius) ||
                disparity <= 0.5f ||
                disparity > static_cast<float>(max_disparity) ||
                std::abs(disparity - initial_disp) > max_delta) {
                continue;
            }

            const float score = 1.0f - std::min(1.0f, best.distance / max_hamming);
            if (score < min_score) continue;

            disparities.push_back(disparity);
            scores.push_back(score);
            if (source_left) {
                anchors_x.push_back(source_x);
                anchors_y.push_back(source_y);
            } else {
                anchors_x.push_back(target_x);
                anchors_y.push_back(target_y);
            }
        }

        if (static_cast<int>(disparities.size()) < min_points) {
            result.low_confidence = true;
            return result;
        }

        std::vector<float> sorted = disparities;
        std::sort(sorted.begin(), sorted.end());
        const float median = medianOfSortedCPU(sorted);

        std::vector<float> abs_dev;
        abs_dev.reserve(disparities.size());
        for (float d : disparities) abs_dev.push_back(std::abs(d - median));
        std::sort(abs_dev.begin(), abs_dev.end());
        const float mad = medianOfSortedCPU(abs_dev);
        const float inlier_gate = std::max(0.60f, mad * 2.5f);

        double sum_disp = 0.0;
        double sum_score = 0.0;
        double sum_anchor_x = 0.0;
        double sum_anchor_y = 0.0;
        int inliers = 0;
        for (size_t i = 0; i < disparities.size(); ++i) {
            if (std::abs(disparities[i] - median) > inlier_gate) continue;
            const double w = std::max(0.05f, scores[i]);
            sum_disp += w * static_cast<double>(disparities[i]);
            sum_anchor_x += w * static_cast<double>(anchors_x[i]);
            sum_anchor_y += w * static_cast<double>(anchors_y[i]);
            sum_score += w;
            ++inliers;
        }
        if (inliers < min_points || sum_score <= 0.0) {
            result.low_confidence = true;
            return result;
        }

        result.disparity = static_cast<float>(sum_disp / sum_score);
        result.anchor_cx = static_cast<float>(sum_anchor_x / sum_score);
        result.anchor_cy = static_cast<float>(sum_anchor_y / sum_score);
        result.support = inliers;

        double var = 0.0;
        for (size_t i = 0; i < disparities.size(); ++i) {
            if (std::abs(disparities[i] - median) > inlier_gate) continue;
            const double diff = static_cast<double>(disparities[i] - result.disparity);
            var += diff * diff;
        }
        result.stddev = static_cast<float>(
            std::sqrt(var / std::max(1.0, static_cast<double>(inliers))));
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
        static_cast<int>(std::lround(cfg.epipolar_y_tolerance * 0.20f)),
        1, 3);
    const float max_delta = computeFeatureDeltaGate(initial_disp, focal, baseline, cfg);
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    const bool binary_mode = mode == SparseFeatureMode::BINARY;
    const float min_score = binary_mode
        ? std::max(0.58f, 0.50f + cfg.subpixel_min_confidence * 0.35f)
        : std::max(0.12f, cfg.subpixel_min_confidence * 0.60f);

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
    std::vector<float> disparities;
    std::vector<float> scores;
    std::vector<cv::DMatch> candidates;
    disparities.reserve(points.size());
    scores.reserve(points.size());
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
        for (int dy = -y_radius; dy <= y_radius; ++dy) {
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
        out.right_keypoints.emplace_back(cv::Point2f(xr, yr),
                                         static_cast<float>(patch_radius * 2 + 1));
        candidates.emplace_back(static_cast<int>(point_idx), t, 1.0f - best_score);
        disparities.push_back(sub_disp);
        scores.push_back(best_score);
    }
    out.attempted_matches = static_cast<int>(candidates.size());

    if (static_cast<int>(disparities.size()) < min_points) return out;

    std::vector<float> sorted = disparities;
    std::sort(sorted.begin(), sorted.end());
    const float median = medianOfSortedCPU(sorted);
    std::vector<float> abs_dev;
    abs_dev.reserve(disparities.size());
    for (float d : disparities) abs_dev.push_back(std::abs(d - median));
    std::sort(abs_dev.begin(), abs_dev.end());
    const float mad = medianOfSortedCPU(abs_dev);
    const float inlier_gate = std::max(0.60f, mad * 2.5f);

    double sum_disp = 0.0;
    double sum_score = 0.0;
    int inliers = 0;
    for (size_t i = 0; i < disparities.size(); ++i) {
        if (std::abs(disparities[i] - median) > inlier_gate) continue;
        const double w = std::max(0.05f, scores[i]);
        sum_disp += w * static_cast<double>(disparities[i]);
        sum_score += w;
        ++inliers;
        out.matches.push_back(candidates[i]);
    }
    if (inliers < min_points || sum_score <= 0.0) {
        out.matches.clear();
        return out;
    }

    out.disparity = static_cast<float>(sum_disp / sum_score);
    double var = 0.0;
    for (float d : disparities) {
        if (std::abs(d - median) > inlier_gate) continue;
        const double diff = static_cast<double>(d - out.disparity);
        var += diff * diff;
    }
    out.stddev = static_cast<float>(
        std::sqrt(var / std::max(1.0, static_cast<double>(inliers))));
    if (out.stddev > max_stddev ||
        std::abs(out.disparity - initial_disp) > max_delta) {
        out.matches.clear();
        return out;
    }
    out.confidence = std::clamp(
        0.35f * (static_cast<float>(inliers) / static_cast<float>(std::max(1, max_points))) +
        0.35f * (static_cast<float>(sum_score / static_cast<double>(inliers)) - min_score) /
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
    const int y_radius = std::clamp(
        static_cast<int>(std::lround(cfg.epipolar_y_tolerance * 0.30f)),
        1, 5);
    const float max_delta = computeFeatureDeltaGate(initial_disp, focal, baseline, cfg);
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    const int extra_margin = search_radius + static_cast<int>(std::ceil(max_delta)) + 2;
    const int border = std::max(2, patch_radius);

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
            left_desc.depth() != CV_8U || right_desc.depth() != CV_8U) {
            return out;
        }

        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(left_desc, right_desc, knn_matches, 2);
        const float ratio_thresh = 0.78f;
        const float max_hamming = static_cast<float>(std::max(1, left_desc.cols * 8));
        const float min_score = std::max(0.45f,
            0.35f + cfg.subpixel_min_confidence * 0.45f);

        std::vector<float> disparities;
        std::vector<float> scores;
        std::vector<cv::DMatch> candidates;
        disparities.reserve(knn_matches.size());
        scores.reserve(knn_matches.size());
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

            const cv::KeyPoint& kl = left_local[best.queryIdx];
            const cv::KeyPoint& kr = right_local[best.trainIdx];
            const float lx = static_cast<float>(left_roi.x) + kl.pt.x;
            const float ly = static_cast<float>(left_roi.y) + kl.pt.y;
            const float rx = static_cast<float>(right_roi.x) + kr.pt.x;
            const float ry = static_cast<float>(right_roi.y) + kr.pt.y;
            const float disparity = lx - rx;
            if (std::abs(ly - ry) > static_cast<float>(y_radius) ||
                disparity <= 0.5f ||
                disparity > static_cast<float>(max_disparity) ||
                std::abs(disparity - initial_disp) > max_delta) {
                continue;
            }

            const float score = 1.0f - std::min(1.0f, best.distance / max_hamming);
            if (score < min_score) continue;

            candidates.emplace_back(best.queryIdx, best.trainIdx, best.distance);
            disparities.push_back(disparity);
            scores.push_back(score);
        }
        out.attempted_matches = static_cast<int>(candidates.size());

        if (static_cast<int>(disparities.size()) < min_points) return out;
        std::vector<float> sorted = disparities;
        std::sort(sorted.begin(), sorted.end());
        const float median = medianOfSortedCPU(sorted);
        std::vector<float> abs_dev;
        abs_dev.reserve(disparities.size());
        for (float d : disparities) abs_dev.push_back(std::abs(d - median));
        std::sort(abs_dev.begin(), abs_dev.end());
        const float mad = medianOfSortedCPU(abs_dev);
        const float inlier_gate = std::max(0.60f, mad * 2.5f);

        double sum_disp = 0.0;
        double sum_score = 0.0;
        int inliers = 0;
        for (size_t i = 0; i < disparities.size(); ++i) {
            if (std::abs(disparities[i] - median) > inlier_gate) continue;
            const double w = std::max(0.05f, scores[i]);
            sum_disp += w * static_cast<double>(disparities[i]);
            sum_score += w;
            ++inliers;
            out.matches.push_back(candidates[i]);
        }
        if (inliers < min_points || sum_score <= 0.0) {
            out.matches.clear();
            return out;
        }
        out.disparity = static_cast<float>(sum_disp / sum_score);
        double var = 0.0;
        for (float d : disparities) {
            if (std::abs(d - median) > inlier_gate) continue;
            const double diff = static_cast<double>(d - out.disparity);
            var += diff * diff;
        }
        out.stddev = static_cast<float>(
            std::sqrt(var / std::max(1.0, static_cast<double>(inliers))));
        if (out.stddev > max_stddev ||
            std::abs(out.disparity - initial_disp) > max_delta) {
            out.matches.clear();
            return out;
        }
        out.confidence = std::clamp(
            0.30f * (static_cast<float>(inliers) / static_cast<float>(std::max(1, max_points))) +
            0.35f * (static_cast<float>(sum_score / static_cast<double>(inliers)) - min_score) /
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
