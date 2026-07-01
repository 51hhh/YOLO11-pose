#include "roi_feature_match_cpu.h"

#include "roi_patch_match_cpu.h"
#include "../utils/logger.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace stereo3d {
namespace {

struct SparseFeaturePoint {
    int x = 0;
    int y = 0;
    float response = 0.0f;
};

struct RobustMatchSample {
    float left_x = 0.0f;
    float left_y = 0.0f;
    float right_x = 0.0f;
    float right_y = 0.0f;
    float disparity = 0.0f;
    float score = 0.0f;
};

struct RobustAggregate {
    bool valid = false;
    float disparity = 0.0f;
    float anchor_x = 0.0f;
    float anchor_y = 0.0f;
    float stddev = 0.0f;
    float mean_score = 0.0f;
    int support = 0;
};

struct NormalizedROIPairCPU {
    bool valid = false;
    cv::Rect roi;
    float scale = 1.0f;
    cv::Mat left_gray;
    cv::Mat right_gray;
    cv::Mat left_edge;
    cv::Mat right_edge;
    cv::Mat left_label;
    cv::Mat right_label;
    Detection left_det;
    Detection right_det;
    float initial_disp = 0.0f;
    float focal = 0.0f;
    int max_disparity = 0;
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

float detectionDiameterCPU(const Detection& det)
{
    return std::max(det.width, det.height);
}

Detection transformDetectionToROI(
    const Detection& det,
    const cv::Rect& roi,
    float scale)
{
    Detection out = det;
    out.cx = (det.cx - static_cast<float>(roi.x)) * scale;
    out.cy = (det.cy - static_cast<float>(roi.y)) * scale;
    out.width = det.width * scale;
    out.height = det.height * scale;
    return out;
}

void buildEllipseLabelCPU(
    const Detection& det,
    const cv::Size& size,
    cv::Mat& label)
{
    label = cv::Mat::zeros(size, CV_8UC1);
    const cv::Point center(static_cast<int>(std::lround(det.cx)),
                           static_cast<int>(std::lround(det.cy)));
    const cv::Size axes(
        std::max(1, static_cast<int>(std::lround(det.width * 0.50f))),
        std::max(1, static_cast<int>(std::lround(det.height * 0.50f))));
    cv::ellipse(label, center, axes, 0.0, 0.0, 360.0, cv::Scalar(255), cv::FILLED);
}

void suppressOutsideLabelCPU(cv::Mat& gray, const cv::Mat& label)
{
    if (gray.empty() || label.empty() || gray.size() != label.size()) return;
    const cv::Scalar mean_scalar = cv::mean(gray, label);
    cv::Mat outside;
    cv::bitwise_not(label, outside);
    gray.setTo(mean_scalar, outside);
}

void buildEdgeMapCPU(const cv::Mat& gray, cv::Mat& edge)
{
    edge.release();
    if (gray.empty() || gray.type() != CV_8UC1) return;
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::Mat mag;
    cv::magnitude(gx, gy, mag);
    cv::normalize(mag, edge, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);
}

bool buildNormalizedROIPairCPU(
    const uint8_t* left_img,
    int left_pitch,
    const uint8_t* right_img,
    int right_pitch,
    int img_w,
    int img_h,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    int patch_radius,
    int search_radius,
    NormalizedROIPairCPU& out)
{
    out = NormalizedROIPairCPU{};
    if (!cfg.feature_normalize_large_roi ||
        cfg.feature_normalized_diameter_px <= 0 ||
        !left_img || !right_img || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0) {
        return false;
    }

    const float diameter = std::max(detectionDiameterCPU(left_det),
                                    detectionDiameterCPU(right_det));
    const float min_diameter = std::max(8.0f, cfg.feature_normalize_min_diameter_px);
    if (diameter < min_diameter) return false;

    const float target = static_cast<float>(
        std::clamp(cfg.feature_normalized_diameter_px, 32, 256));
    const float scale = std::min(1.0f, target / std::max(1.0f, diameter));
    if (scale > 0.98f) return false;

    const float margin_scale = std::clamp(cfg.feature_normalize_margin_scale,
                                          0.52f, 1.20f);
    const float pad = static_cast<float>(
        std::max(4, patch_radius + search_radius + 3));
    auto bounds_for = [&](const Detection& det,
                          float& x1, float& y1,
                          float& x2, float& y2) {
        x1 = det.cx - det.width * margin_scale - pad;
        y1 = det.cy - det.height * margin_scale - pad;
        x2 = det.cx + det.width * margin_scale + pad;
        y2 = det.cy + det.height * margin_scale + pad;
    };
    float lx1, ly1, lx2, ly2;
    float rx1, ry1, rx2, ry2;
    bounds_for(left_det, lx1, ly1, lx2, ly2);
    bounds_for(right_det, rx1, ry1, rx2, ry2);
    const int x1 = std::max(0, static_cast<int>(std::floor(std::min(lx1, rx1))));
    const int y1 = std::max(0, static_cast<int>(std::floor(std::min(ly1, ry1))));
    const int x2 = std::min(img_w - 1, static_cast<int>(std::ceil(std::max(lx2, rx2))));
    const int y2 = std::min(img_h - 1, static_cast<int>(std::ceil(std::max(ly2, ry2))));
    if (x2 <= x1 || y2 <= y1) return false;

    const cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
    const int scaled_w = std::max(8, static_cast<int>(std::lround(roi.width * scale)));
    const int scaled_h = std::max(8, static_cast<int>(std::lround(roi.height * scale)));
    if (scaled_w < patch_radius * 2 + 8 ||
        scaled_h < patch_radius * 2 + 8) {
        return false;
    }

    cv::Mat left_full(img_h, img_w, CV_8UC1,
                      const_cast<uint8_t*>(left_img),
                      static_cast<size_t>(left_pitch));
    cv::Mat right_full(img_h, img_w, CV_8UC1,
                       const_cast<uint8_t*>(right_img),
                       static_cast<size_t>(right_pitch));
    cv::resize(left_full(roi), out.left_gray,
               cv::Size(scaled_w, scaled_h), 0.0, 0.0, cv::INTER_AREA);
    cv::resize(right_full(roi), out.right_gray,
               cv::Size(scaled_w, scaled_h), 0.0, 0.0, cv::INTER_AREA);

    out.roi = roi;
    out.scale = scale;
    out.left_det = transformDetectionToROI(left_det, roi, scale);
    out.right_det = transformDetectionToROI(right_det, roi, scale);
    out.initial_disp = initial_disp * scale;
    out.focal = focal * scale;
    out.max_disparity = std::max(
        1, static_cast<int>(std::ceil(static_cast<float>(max_disparity) * scale)) + 2);

    if (cfg.feature_precompute_roi_maps) {
        buildEllipseLabelCPU(out.left_det, out.left_gray.size(), out.left_label);
        buildEllipseLabelCPU(out.right_det, out.right_gray.size(), out.right_label);
        suppressOutsideLabelCPU(out.left_gray, out.left_label);
        suppressOutsideLabelCPU(out.right_gray, out.right_label);
        buildEdgeMapCPU(out.left_gray, out.left_edge);
        buildEdgeMapCPU(out.right_gray, out.right_edge);
    }

    out.valid = !out.left_gray.empty() && !out.right_gray.empty() &&
                out.initial_disp > 0.5f;
    return out.valid;
}

SparseFeatureDisparityResult mapNormalizedSparseResultCPU(
    SparseFeatureDisparityResult result,
    const NormalizedROIPairCPU& norm)
{
    if (norm.scale <= 1e-6f) return result;
    if (result.disparity > 0.0f) result.disparity /= norm.scale;
    if (result.stddev > 0.0f) result.stddev /= norm.scale;
    result.anchor_cx = result.anchor_cx / norm.scale +
                       static_cast<float>(norm.roi.x);
    result.anchor_cy = result.anchor_cy / norm.scale +
                       static_cast<float>(norm.roi.y);
    return result;
}

ROIFeatureMatchConfig makeNormalizedFeatureConfigCPU(
    const ROIFeatureMatchConfig& cfg,
    float scale)
{
    ROIFeatureMatchConfig out = cfg;
    out.feature_normalize_large_roi = false;
    out.subpixel_search_radius_px =
        std::max(1, static_cast<int>(std::lround(
                        static_cast<float>(cfg.subpixel_search_radius_px) * scale)));
    out.subpixel_max_disp_delta_px =
        std::max(0.25f, cfg.subpixel_max_disp_delta_px * scale);
    out.subpixel_max_stddev_px =
        std::max(0.05f, cfg.subpixel_max_stddev_px * scale);
    out.feature_y_tolerance_px =
        std::max(0.5f, cfg.feature_y_tolerance_px * scale);
    out.feature_y_offset_px = cfg.feature_y_offset_px * scale;
    if (out.feature_reverse_check_px >= 0.0f) {
        out.feature_reverse_check_px =
            std::max(0.25f, cfg.feature_reverse_check_px * scale);
    }
    out.feature_ransac_gate_px =
        std::max(0.25f, cfg.feature_ransac_gate_px * scale);
    return out;
}

void mapNormalizedDebugResultCPU(
    DebugFeatureMatchResult& result,
    const NormalizedROIPairCPU& norm)
{
    if (norm.scale <= 1e-6f) return;
    for (auto& kp : result.left_keypoints) {
        kp.pt.x = kp.pt.x / norm.scale + static_cast<float>(norm.roi.x);
        kp.pt.y = kp.pt.y / norm.scale + static_cast<float>(norm.roi.y);
        kp.size /= norm.scale;
    }
    for (auto& kp : result.right_keypoints) {
        kp.pt.x = kp.pt.x / norm.scale + static_cast<float>(norm.roi.x);
        kp.pt.y = kp.pt.y / norm.scale + static_cast<float>(norm.roi.y);
        kp.size /= norm.scale;
    }
    if (result.disparity > 0.0f) result.disparity /= norm.scale;
    if (result.stddev > 0.0f) result.stddev /= norm.scale;
}

float strictFeatureYTolerance(const ROIFeatureMatchConfig& cfg)
{
    return std::clamp(cfg.feature_y_tolerance_px, 0.5f, 8.0f);
}

float expectedFeatureYDelta(
    float left_x,
    const Detection& left_det,
    const ROIFeatureMatchConfig& cfg)
{
    return cfg.feature_y_offset_px +
           cfg.feature_y_slope * (left_x - left_det.cx);
}

float featureYResidual(
    const RobustMatchSample& sample,
    const Detection& left_det,
    const ROIFeatureMatchConfig& cfg)
{
    const float expected = expectedFeatureYDelta(sample.left_x, left_det, cfg);
    return (sample.left_y - sample.right_y) - expected;
}

bool pointInsideDetectionEllipse(
    const Detection& det,
    float x,
    float y,
    float scale)
{
    if (det.width <= 1.0f || det.height <= 1.0f) return false;
    const float rx = std::max(1.0f, det.width * scale);
    const float ry = std::max(1.0f, det.height * scale);
    const float nx = (x - det.cx) / rx;
    const float ny = (y - det.cy) / ry;
    return nx * nx + ny * ny <= 1.0f;
}

bool passesFeatureOverlapGate(
    const RobustMatchSample& sample,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg)
{
    const float scale = std::clamp(cfg.feature_overlap_scale, 0.35f, 0.90f);
    const float projection_scale = std::min(0.98f, scale + 0.12f);
    if (!pointInsideDetectionEllipse(left_det, sample.left_x, sample.left_y, scale) ||
        !pointInsideDetectionEllipse(right_det, sample.right_x, sample.right_y, scale)) {
        return false;
    }
    // Both measured points must also lie in the other detection after the
    // current stereo prior is applied. This approximates the valid L/R ball
    // overlap mask without allocating a per-frame mask.
    if (!pointInsideDetectionEllipse(right_det,
                                     sample.left_x - initial_disp,
                                     sample.left_y,
                                     projection_scale)) {
        return false;
    }
    if (!pointInsideDetectionEllipse(left_det,
                                     sample.right_x + initial_disp,
                                     sample.right_y,
                                     projection_scale)) {
        return false;
    }
    return true;
}

bool passesSphereRadiusGate(
    const RobustMatchSample& sample,
    const Detection& left_det,
    float initial_disp,
    float focal,
    float baseline,
    const ROIFeatureMatchConfig& cfg)
{
    const float radius_m = cfg.feature_sphere_radius_m;
    if (radius_m <= 0.0f || focal <= 1e-3f || baseline <= 1e-6f ||
        initial_disp <= 0.5f || sample.disparity <= 0.5f) {
        return true;
    }
    const float fb = focal * baseline;
    const float center_z = fb / initial_disp;
    const float z = fb / sample.disparity;
    if (!std::isfinite(center_z) || !std::isfinite(z)) return false;

    const float dx = (sample.left_x - left_det.cx) * z / focal;
    const float dy = (sample.left_y - left_det.cy) * z / focal;
    const float dz = z - center_z;
    const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    const float max_distance =
        radius_m * std::max(1.0f, cfg.feature_sphere_radius_scale) +
        std::max(0.0f, cfg.feature_sphere_margin_m);
    return distance <= max_distance;
}

float weightedMedianDisparity(std::vector<RobustMatchSample> samples)
{
    if (samples.empty()) return 0.0f;
    std::sort(samples.begin(), samples.end(),
              [](const RobustMatchSample& a, const RobustMatchSample& b) {
                  return a.disparity < b.disparity;
              });
    double total = 0.0;
    for (const auto& s : samples) total += std::max(0.05f, s.score);
    const double half = total * 0.5;
    double accum = 0.0;
    for (const auto& s : samples) {
        accum += std::max(0.05f, s.score);
        if (accum >= half) return s.disparity;
    }
    return samples.back().disparity;
}

RobustAggregate aggregateRobustMatches(
    const std::vector<RobustMatchSample>& samples,
    int min_points,
    int max_points,
    float initial_disp,
    float max_delta,
    float max_stddev,
    const ROIFeatureMatchConfig& cfg)
{
    RobustAggregate out;
    if (static_cast<int>(samples.size()) < min_points) return out;

    std::vector<float> sorted;
    sorted.reserve(samples.size());
    for (const auto& s : samples) sorted.push_back(s.disparity);
    std::sort(sorted.begin(), sorted.end());
    const float median = medianOfSortedCPU(sorted);

    std::vector<float> abs_dev;
    abs_dev.reserve(samples.size());
    for (const auto& s : samples) abs_dev.push_back(std::abs(s.disparity - median));
    std::sort(abs_dev.begin(), abs_dev.end());
    const float mad = medianOfSortedCPU(abs_dev);
    const float robust_sigma = 1.4826f * mad;
    const float mad_gate = std::max(std::clamp(cfg.feature_ransac_gate_px, 0.25f, 3.0f),
                                    robust_sigma * std::max(1.0f, cfg.feature_mad_scale));
    const float gate = std::min(std::max(0.35f, max_delta), mad_gate);

    double best_weight = -1.0;
    float best_center = median;
    int best_support = 0;
    for (const auto& candidate : samples) {
        double support_weight = 0.0;
        int support = 0;
        for (const auto& s : samples) {
            if (std::abs(s.disparity - candidate.disparity) > gate) continue;
            support_weight += std::max(0.05f, s.score);
            ++support;
        }
        if (support > best_support ||
            (support == best_support && support_weight > best_weight)) {
            best_support = support;
            best_weight = support_weight;
            best_center = candidate.disparity;
        }
    }

    std::vector<RobustMatchSample> inliers;
    inliers.reserve(samples.size());
    const float median_gate = std::max(gate, std::clamp(cfg.feature_ransac_gate_px, 0.25f, 3.0f) * 1.5f);
    for (const auto& s : samples) {
        if (std::abs(s.disparity - best_center) <= gate &&
            std::abs(s.disparity - median) <= median_gate) {
            inliers.push_back(s);
        }
    }
    if (static_cast<int>(inliers.size()) < min_points) return out;

    out.disparity = weightedMedianDisparity(inliers);
    if (std::abs(out.disparity - initial_disp) > max_delta ||
        out.disparity <= 0.5f) {
        return out;
    }

    double sum_w = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_score = 0.0;
    double var = 0.0;
    for (const auto& s : inliers) {
        const double w = std::max(0.05f, s.score);
        sum_w += w;
        sum_x += w * static_cast<double>(s.left_x);
        sum_y += w * static_cast<double>(s.left_y);
        sum_score += w;
        const double diff = static_cast<double>(s.disparity - out.disparity);
        var += w * diff * diff;
    }
    if (sum_w <= 0.0) return out;

    out.anchor_x = static_cast<float>(sum_x / sum_w);
    out.anchor_y = static_cast<float>(sum_y / sum_w);
    out.stddev = static_cast<float>(std::sqrt(var / sum_w));
    out.mean_score = static_cast<float>(sum_score / static_cast<double>(inliers.size()));
    out.support = static_cast<int>(inliers.size());
    out.valid = out.stddev <= max_stddev &&
                out.support >= min_points &&
                out.support <= std::max(min_points, max_points);
    return out;
}

float reverseSparseMatchError(
    const uint8_t* left_img,
    int left_pitch,
    const uint8_t* right_img,
    int right_pitch,
    int img_w,
    int img_h,
    const RobustMatchSample& sample,
    int patch_radius,
    int d_start,
    int d_end,
    int y_radius,
    bool binary_mode,
    bool denoise,
    const Detection& left_det,
    const ROIFeatureMatchConfig& cfg)
{
    const int rx = static_cast<int>(std::lround(sample.right_x));
    const int ry = static_cast<int>(std::lround(sample.right_y));
    if (!patchInsideCPU(img_w, img_h, rx, ry, patch_radius, denoise)) {
        return std::numeric_limits<float>::infinity();
    }
    float best_score = -2.0f;
    int best_lx = -1;
    int best_ly = -1;
    for (int disp = d_start; disp <= d_end; ++disp) {
        const int lx = rx + disp;
        const int dy_center = static_cast<int>(std::lround(
            expectedFeatureYDelta(static_cast<float>(lx), left_det, cfg)));
        for (int dy_offset = -y_radius; dy_offset <= y_radius; ++dy_offset) {
            const int ly = ry + dy_center + dy_offset;
            if (!patchInsideCPU(img_w, img_h, lx, ly, patch_radius, denoise)) {
                continue;
            }
            const float score = binary_mode
                ? censusPatchSimilarityCPU(right_img, right_pitch,
                                           left_img, left_pitch,
                                           rx, ry, lx, ly,
                                           patch_radius, denoise)
                : znccPatchCPU(right_img, right_pitch,
                               left_img, left_pitch,
                               rx, ry, lx, ly,
                               patch_radius, denoise);
            if (score > best_score) {
                best_score = score;
                best_lx = lx;
                best_ly = ly;
            }
        }
    }
    if (best_lx < 0) return std::numeric_limits<float>::infinity();
    const float dx = static_cast<float>(best_lx) - sample.left_x;
    const float dy = static_cast<float>(best_ly) - sample.left_y;
    return std::sqrt(dx * dx + dy * dy);
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
            source_descriptors.depth() != CV_8U ||
            target_descriptors.depth() != CV_8U) {
            result.low_confidence = true;
            return result;
        }

        const float ratio_thresh = 0.78f;
        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
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

        const float max_hamming = static_cast<float>(
            std::max(1, source_descriptors.cols * 8));
        const float min_score = std::max(0.45f,
            0.35f + cfg.subpixel_min_confidence * 0.45f);
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

            const float score = 1.0f - std::min(1.0f, best.distance / max_hamming);
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
            left_desc.depth() != CV_8U || right_desc.depth() != CV_8U) {
            return out;
        }

        const float ratio_thresh = 0.78f;
        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
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
        const float max_hamming = static_cast<float>(std::max(1, left_desc.cols * 8));
        const float min_score = std::max(0.45f,
            0.35f + cfg.subpixel_min_confidence * 0.45f);

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
