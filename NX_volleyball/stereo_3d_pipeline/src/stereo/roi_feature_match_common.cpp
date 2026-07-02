#include "roi_feature_match_common.h"

#include <algorithm>
#include <cmath>

namespace stereo3d {
namespace {

float medianOfSorted(const std::vector<float>& values)
{
    if (values.empty()) return 0.0f;
    const size_t mid = values.size() / 2;
    if ((values.size() & 1U) != 0U) return values[mid];
    return 0.5f * (values[mid - 1] + values[mid]);
}

}  // namespace

float computeFeatureDeltaGate(
    float initial_disp,
    float focal,
    float baseline,
    const ROIFeatureMatchConfig& cfg)
{
    const float abs_gate = std::max(0.25f, cfg.subpixel_max_disp_delta_px);
    const float ratio_gate = std::max(
        0.25f,
        std::max(0.0f, cfg.subpixel_max_disp_delta_ratio) * initial_disp);
    float gate = std::max(abs_gate, ratio_gate);

    const float fb = focal * baseline;
    if (fb > 1e-3f && initial_disp > 0.5f &&
        cfg.subpixel_max_depth_delta_m > 0.01f) {
        const float z0 = fb / initial_disp;
        const float disp_far = fb / (z0 + cfg.subpixel_max_depth_delta_m);
        gate = std::max(gate, std::max(0.0f, initial_disp - disp_far));
        if (z0 > cfg.subpixel_max_depth_delta_m + 0.01f) {
            const float disp_near =
                fb / (z0 - cfg.subpixel_max_depth_delta_m);
            gate = std::max(gate, std::max(0.0f, disp_near - initial_disp));
        }
    }
    return std::max(0.25f, gate);
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
    const float median = medianOfSorted(sorted);

    std::vector<float> abs_dev;
    abs_dev.reserve(samples.size());
    for (const auto& s : samples) {
        abs_dev.push_back(std::abs(s.disparity - median));
    }
    std::sort(abs_dev.begin(), abs_dev.end());
    const float mad = medianOfSorted(abs_dev);
    const float robust_sigma = 1.4826f * mad;
    const float mad_gate = std::max(
        std::clamp(cfg.feature_ransac_gate_px, 0.25f, 3.0f),
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
    const float median_gate = std::max(
        gate,
        std::clamp(cfg.feature_ransac_gate_px, 0.25f, 3.0f) * 1.5f);
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
    double sum_rx = 0.0;
    double sum_ry = 0.0;
    double sum_score = 0.0;
    double var = 0.0;
    for (const auto& s : inliers) {
        const double w = std::max(0.05f, s.score);
        sum_w += w;
        sum_x += w * static_cast<double>(s.left_x);
        sum_y += w * static_cast<double>(s.left_y);
        sum_rx += w * static_cast<double>(s.right_x);
        sum_ry += w * static_cast<double>(s.right_y);
        sum_score += w;
        const double diff = static_cast<double>(s.disparity - out.disparity);
        var += w * diff * diff;
    }
    if (sum_w <= 0.0) return out;

    out.anchor_x = static_cast<float>(sum_x / sum_w);
    out.anchor_y = static_cast<float>(sum_y / sum_w);
    out.right_anchor_x = static_cast<float>(sum_rx / sum_w);
    out.right_anchor_y = static_cast<float>(sum_ry / sum_w);
    out.stddev = static_cast<float>(std::sqrt(var / sum_w));
    out.mean_score = static_cast<float>(sum_score / static_cast<double>(inliers.size()));
    out.support = static_cast<int>(inliers.size());
    out.valid = out.stddev <= max_stddev &&
                out.support >= min_points &&
                out.support <= std::max(min_points, max_points);
    return out;
}

}  // namespace stereo3d
