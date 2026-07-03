#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_CPU_HELPERS_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_CPU_HELPERS_H_

#include "roi_feature_match_cpu.h"

#include "roi_feature_match_common.h"
#include "roi_feature_match_opencv_helpers.h"
#include "roi_patch_match_cpu.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace stereo3d {
namespace {

struct SparseFeaturePoint {
    int x = 0;
    int y = 0;
    float response = 0.0f;
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
    result.right_anchor_cx = result.right_anchor_cx / norm.scale +
                             static_cast<float>(norm.roi.x);
    result.right_anchor_cy = result.right_anchor_cy / norm.scale +
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
}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_CPU_HELPERS_H_
