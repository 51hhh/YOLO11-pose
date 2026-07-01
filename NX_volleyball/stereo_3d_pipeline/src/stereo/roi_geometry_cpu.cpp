#include "roi_geometry_cpu.h"

#include "roi_patch_match_cpu.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace stereo3d {
namespace {

bool solve3x3CPU(
    double A00, double A01, double A02,
    double A10, double A11, double A12,
    double A20, double A21, double A22,
    double b0,  double b1,  double b2,
    double& x0, double& x1, double& x2)
{
    double det = A00 * (A11 * A22 - A12 * A21)
               - A01 * (A10 * A22 - A12 * A20)
               + A02 * (A10 * A21 - A11 * A20);
    if (std::abs(det) < 1e-12) return false;

    const double inv_det = 1.0 / det;
    x0 = (b0 * (A11 * A22 - A12 * A21)
        - A01 * (b1 * A22 - A12 * b2)
        + A02 * (b1 * A21 - A11 * b2)) * inv_det;
    x1 = (A00 * (b1 * A22 - A12 * b2)
        - b0 * (A10 * A22 - A12 * A20)
        + A02 * (A10 * b2 - b1 * A20)) * inv_det;
    x2 = (A00 * (A11 * b2 - b1 * A21)
        - A01 * (A10 * b2 - b1 * A20)
        + b0 * (A10 * A21 - A11 * A20)) * inv_det;
    return true;
}

bool solve2x2CPU(
    double A00, double A01,
    double A10, double A11,
    double b0, double b1,
    double& x0, double& x1)
{
    const double det = A00 * A11 - A01 * A10;
    if (std::abs(det) < 1e-12) return false;
    const double inv_det = 1.0 / det;
    x0 = (b0 * A11 - A01 * b1) * inv_det;
    x1 = (A00 * b1 - b0 * A10) * inv_det;
    return true;
}

}  // namespace

CircleFit2D fitCircleInRegionCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    int x1, int y1, int x2, int y2,
    float expected_cx, float expected_cy, float expected_radius,
    const CircleFitOptions& options)
{
    CircleFit2D out;
    if (!img || pitch <= 0 || img_w <= 0 || img_h <= 0 ||
        expected_radius < 4.0f) {
        return out;
    }

    const int border = options.denoise ? 2 : 1;
    x1 = std::max(border, x1);
    y1 = std::max(border, y1);
    x2 = std::min(img_w - 1 - border, x2);
    y2 = std::min(img_h - 1 - border, y2);
    if (x2 - x1 < 8 || y2 - y1 < 8) return out;

    const int roi_w = x2 - x1 + 1;
    const int roi_h = y2 - y1 + 1;
    const int area = roi_w * roi_h;
    const int max_pixels = std::max(256, options.max_roi_pixels);
    const int stride = std::max(
        1, static_cast<int>(std::ceil(std::sqrt(
               static_cast<float>(area) / static_cast<float>(max_pixels)))));

    float max_grad = 0.0f;
    for (int y = y1; y <= y2; y += stride) {
        for (int x = x1; x <= x2; x += stride) {
            const float gx = sampleGrayCPU(img, pitch, x + 1, y, options.denoise) -
                             sampleGrayCPU(img, pitch, x - 1, y, options.denoise);
            const float gy = sampleGrayCPU(img, pitch, x, y + 1, options.denoise) -
                             sampleGrayCPU(img, pitch, x, y - 1, options.denoise);
            max_grad = std::max(max_grad, std::sqrt(gx * gx + gy * gy));
        }
    }
    if (max_grad < 8.0f) return out;

    const float grad_thresh = std::max(10.0f, max_grad * 0.25f);
    double sw = 0.0, swx = 0.0, swy = 0.0;
    double swxx = 0.0, swyy = 0.0, swxy = 0.0;
    double swxz = 0.0, swyz = 0.0, swz = 0.0;
    int edge_count = 0;

    for (int y = y1; y <= y2; y += stride) {
        for (int x = x1; x <= x2; x += stride) {
            const float gx = sampleGrayCPU(img, pitch, x + 1, y, options.denoise) -
                             sampleGrayCPU(img, pitch, x - 1, y, options.denoise);
            const float gy = sampleGrayCPU(img, pitch, x, y + 1, options.denoise) -
                             sampleGrayCPU(img, pitch, x, y - 1, options.denoise);
            const float mag = std::sqrt(gx * gx + gy * gy);
            if (mag < grad_thresh) continue;

            const double w = static_cast<double>(mag);
            const double dx = static_cast<double>(x);
            const double dy = static_cast<double>(y);
            const double z = dx * dx + dy * dy;
            sw += w;
            swx += w * dx;
            swy += w * dy;
            swxx += w * dx * dx;
            swyy += w * dy * dy;
            swxy += w * dx * dy;
            swxz += w * dx * z;
            swyz += w * dy * z;
            swz += w * z;
            ++edge_count;
        }
    }

    if (edge_count < 8 || sw <= 0.0) return out;

    double a = 0.0, b = 0.0, c = 0.0;
    if (!solve3x3CPU(swxx, swxy, swx,
                     swxy, swyy, swy,
                     swx,  swy,  sw,
                     swxz, swyz, swz,
                     a, b, c)) {
        return out;
    }

    const float cx = static_cast<float>(a * 0.5);
    const float cy = static_cast<float>(b * 0.5);
    const float r2 = static_cast<float>(c + static_cast<double>(cx) * cx +
                                        static_cast<double>(cy) * cy);
    if (r2 <= 0.0f) return out;
    const float radius = std::sqrt(r2);

    const float min_r = std::max(4.0f, expected_radius * options.min_radius_ratio);
    const float max_r = std::max(min_r + 1.0f, expected_radius * options.max_radius_ratio);
    const float center_dist = std::sqrt((cx - expected_cx) * (cx - expected_cx) +
                                        (cy - expected_cy) * (cy - expected_cy));
    const float max_center_shift = options.max_center_shift > 0.0f
        ? options.max_center_shift
        : std::max(roi_w, roi_h) * 0.5f;
    if (radius < min_r || radius > max_r || center_dist > max_center_shift) return out;

    out.cx = cx;
    out.cy = cy;
    out.radius = radius;
    const float dense_edge_count = static_cast<float>(edge_count * stride * stride);
    const float edge_conf = std::min(1.0f, dense_edge_count / 80.0f);
    const float center_conf = std::max(0.2f, 1.0f - center_dist / std::max(1.0f, max_center_shift));
    out.confidence = edge_conf * center_conf;
    out.valid = true;
    return out;
}

CircleFit2D fitCircleInBBoxCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const Detection& det, bool denoise, int max_roi_pixels)
{
    if (det.width < 8.0f || det.height < 8.0f) return {};

    const int x1 = static_cast<int>(std::floor(det.cx - det.width * 0.55f));
    const int y1 = static_cast<int>(std::floor(det.cy - det.height * 0.55f));
    const int x2 = static_cast<int>(std::ceil(det.cx + det.width * 0.55f));
    const int y2 = static_cast<int>(std::ceil(det.cy + det.height * 0.55f));

    CircleFitOptions options;
    options.denoise = denoise;
    options.max_roi_pixels = max_roi_pixels;
    options.max_center_shift = std::max(det.width, det.height) * 0.65f;
    CircleFit2D circle = fitCircleInRegionCPU(img, pitch, img_w, img_h,
                                              x1, y1, x2, y2,
                                              det.cx, det.cy,
                                              std::max(det.width, det.height) * 0.5f,
                                              options);
    if (circle.valid) circle.source = kCircleSourceRoiFit;
    return circle;
}

PointMeasure2D edgeCentroidInBBoxCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const Detection& det, bool denoise, int max_roi_pixels)
{
    PointMeasure2D out;
    if (!img || pitch <= 0 || det.width < 8.0f || det.height < 8.0f) {
        return out;
    }

    const int border = denoise ? 2 : 1;
    int x1 = static_cast<int>(std::floor(det.cx - det.width * 0.55f));
    int y1 = static_cast<int>(std::floor(det.cy - det.height * 0.55f));
    int x2 = static_cast<int>(std::ceil(det.cx + det.width * 0.55f));
    int y2 = static_cast<int>(std::ceil(det.cy + det.height * 0.55f));
    x1 = std::max(border, x1);
    y1 = std::max(border, y1);
    x2 = std::min(img_w - 1 - border, x2);
    y2 = std::min(img_h - 1 - border, y2);
    if (x2 - x1 < 8 || y2 - y1 < 8) return out;

    const int roi_w = x2 - x1 + 1;
    const int roi_h = y2 - y1 + 1;
    const int area = roi_w * roi_h;
    const int max_pixels = std::max(256, max_roi_pixels);
    const int stride = std::max(
        1, static_cast<int>(std::ceil(std::sqrt(
               static_cast<float>(area) / static_cast<float>(max_pixels)))));

    float max_grad = 0.0f;
    for (int y = y1; y <= y2; y += stride) {
        for (int x = x1; x <= x2; x += stride) {
            const float gx = sampleGrayCPU(img, pitch, x + 1, y, denoise) -
                             sampleGrayCPU(img, pitch, x - 1, y, denoise);
            const float gy = sampleGrayCPU(img, pitch, x, y + 1, denoise) -
                             sampleGrayCPU(img, pitch, x, y - 1, denoise);
            max_grad = std::max(max_grad, std::sqrt(gx * gx + gy * gy));
        }
    }
    if (max_grad < 8.0f) return out;

    const float grad_thresh = std::max(10.0f, max_grad * 0.25f);
    double sw = 0.0;
    double swx = 0.0;
    double swy = 0.0;
    int edge_count = 0;
    for (int y = y1; y <= y2; y += stride) {
        for (int x = x1; x <= x2; x += stride) {
            const float gx = sampleGrayCPU(img, pitch, x + 1, y, denoise) -
                             sampleGrayCPU(img, pitch, x - 1, y, denoise);
            const float gy = sampleGrayCPU(img, pitch, x, y + 1, denoise) -
                             sampleGrayCPU(img, pitch, x, y - 1, denoise);
            const float mag = std::sqrt(gx * gx + gy * gy);
            if (mag < grad_thresh) continue;
            const double w = static_cast<double>(mag);
            sw += w;
            swx += w * static_cast<double>(x);
            swy += w * static_cast<double>(y);
            ++edge_count;
        }
    }
    if (edge_count < 8 || sw <= 0.0) return out;

    out.cx = static_cast<float>(swx / sw);
    out.cy = static_cast<float>(swy / sw);
    const float dense_edge_count = static_cast<float>(edge_count * stride * stride);
    out.confidence = std::min(1.0f, dense_edge_count / 80.0f);
    out.valid = true;
    return out;
}

PointMeasure2D radialCenterInBBoxCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const Detection& det, bool denoise, int max_roi_pixels)
{
    PointMeasure2D out;
    if (!img || pitch <= 0 || det.width < 8.0f || det.height < 8.0f) {
        return out;
    }

    const int border = denoise ? 2 : 1;
    int x1 = static_cast<int>(std::floor(det.cx - det.width * 0.55f));
    int y1 = static_cast<int>(std::floor(det.cy - det.height * 0.55f));
    int x2 = static_cast<int>(std::ceil(det.cx + det.width * 0.55f));
    int y2 = static_cast<int>(std::ceil(det.cy + det.height * 0.55f));
    x1 = std::max(border, x1);
    y1 = std::max(border, y1);
    x2 = std::min(img_w - 1 - border, x2);
    y2 = std::min(img_h - 1 - border, y2);
    if (x2 - x1 < 8 || y2 - y1 < 8) return out;

    const int roi_w = x2 - x1 + 1;
    const int roi_h = y2 - y1 + 1;
    const int area = roi_w * roi_h;
    const int max_pixels = std::max(256, max_roi_pixels);
    const int stride = std::max(
        1, static_cast<int>(std::ceil(std::sqrt(
               static_cast<float>(area) / static_cast<float>(max_pixels)))));

    float max_grad = 0.0f;
    for (int y = y1; y <= y2; y += stride) {
        for (int x = x1; x <= x2; x += stride) {
            const float gx = sampleGrayCPU(img, pitch, x + 1, y, denoise) -
                             sampleGrayCPU(img, pitch, x - 1, y, denoise);
            const float gy = sampleGrayCPU(img, pitch, x, y + 1, denoise) -
                             sampleGrayCPU(img, pitch, x, y - 1, denoise);
            max_grad = std::max(max_grad, std::sqrt(gx * gx + gy * gy));
        }
    }
    if (max_grad < 8.0f) return out;

    const float grad_thresh = std::max(10.0f, max_grad * 0.25f);
    double A00 = 0.0, A01 = 0.0, A11 = 0.0;
    double b0 = 0.0, b1 = 0.0;
    double sw = 0.0;
    int edge_count = 0;
    for (int y = y1; y <= y2; y += stride) {
        for (int x = x1; x <= x2; x += stride) {
            const float gx = sampleGrayCPU(img, pitch, x + 1, y, denoise) -
                             sampleGrayCPU(img, pitch, x - 1, y, denoise);
            const float gy = sampleGrayCPU(img, pitch, x, y + 1, denoise) -
                             sampleGrayCPU(img, pitch, x, y - 1, denoise);
            const float mag = std::sqrt(gx * gx + gy * gy);
            if (mag < grad_thresh) continue;

            const double inv_mag = 1.0 / std::max(1e-6, static_cast<double>(mag));
            const double ux = static_cast<double>(gx) * inv_mag;
            const double uy = static_cast<double>(gy) * inv_mag;
            const double m00 = 1.0 - ux * ux;
            const double m01 = -ux * uy;
            const double m11 = 1.0 - uy * uy;
            const double w = static_cast<double>(mag);
            const double px = static_cast<double>(x);
            const double py = static_cast<double>(y);

            A00 += w * m00;
            A01 += w * m01;
            A11 += w * m11;
            b0 += w * (m00 * px + m01 * py);
            b1 += w * (m01 * px + m11 * py);
            sw += w;
            ++edge_count;
        }
    }
    if (edge_count < 8 || sw <= 0.0) return out;

    double cx = 0.0;
    double cy = 0.0;
    if (!solve2x2CPU(A00, A01, A01, A11, b0, b1, cx, cy)) {
        return out;
    }

    const float center_shift = std::sqrt(
        static_cast<float>((cx - det.cx) * (cx - det.cx) +
                           (cy - det.cy) * (cy - det.cy)));
    const float max_shift = std::max(det.width, det.height) * 0.55f;
    if (cx < x1 || cx > x2 || cy < y1 || cy > y2 || center_shift > max_shift) {
        return out;
    }

    std::vector<float> radii;
    radii.reserve(static_cast<size_t>(edge_count));
    for (int y = y1; y <= y2; y += stride) {
        for (int x = x1; x <= x2; x += stride) {
            const float gx = sampleGrayCPU(img, pitch, x + 1, y, denoise) -
                             sampleGrayCPU(img, pitch, x - 1, y, denoise);
            const float gy = sampleGrayCPU(img, pitch, x, y + 1, denoise) -
                             sampleGrayCPU(img, pitch, x, y - 1, denoise);
            const float mag = std::sqrt(gx * gx + gy * gy);
            if (mag < grad_thresh) continue;
            radii.push_back(std::sqrt(
                static_cast<float>((static_cast<double>(x) - cx) *
                                   (static_cast<double>(x) - cx) +
                                   (static_cast<double>(y) - cy) *
                                   (static_cast<double>(y) - cy))));
        }
    }
    if (radii.size() < 8) return out;
    std::sort(radii.begin(), radii.end());
    const size_t mid = radii.size() / 2;
    const float radius = (radii.size() & 1U) != 0U
        ? radii[mid]
        : 0.5f * (radii[mid - 1] + radii[mid]);
    const float expected = std::max(det.width, det.height) * 0.5f;
    if (radius < std::max(4.0f, expected * 0.35f) || radius > expected * 1.75f) {
        return out;
    }

    const float dense_edge_count = static_cast<float>(edge_count * stride * stride);
    const float edge_conf = std::min(1.0f, dense_edge_count / 80.0f);
    const float shift_conf = std::max(0.2f, 1.0f - center_shift / std::max(1.0f, max_shift));
    out.cx = static_cast<float>(cx);
    out.cy = static_cast<float>(cy);
    out.confidence = edge_conf * shift_conf;
    out.valid = true;
    return out;
}

PointMeasure2D edgePairCenterInBBoxCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const Detection& det, bool denoise, int max_roi_pixels)
{
    PointMeasure2D out;
    if (!img || pitch <= 0 || det.width < 8.0f || det.height < 8.0f) {
        return out;
    }

    const int border = denoise ? 2 : 1;
    int x1 = static_cast<int>(std::floor(det.cx - det.width * 0.60f));
    int y1 = static_cast<int>(std::floor(det.cy - det.height * 0.45f));
    int x2 = static_cast<int>(std::ceil(det.cx + det.width * 0.60f));
    int y2 = static_cast<int>(std::ceil(det.cy + det.height * 0.45f));
    x1 = std::max(border, x1);
    y1 = std::max(border, y1);
    x2 = std::min(img_w - 1 - border, x2);
    y2 = std::min(img_h - 1 - border, y2);
    if (x2 - x1 < 8 || y2 - y1 < 8) return out;

    const int roi_w = x2 - x1 + 1;
    const int roi_h = y2 - y1 + 1;
    const int area = roi_w * roi_h;
    const int max_pixels = std::max(256, max_roi_pixels);
    const int row_stride = std::max(
        1, static_cast<int>(std::ceil(std::sqrt(
               static_cast<float>(area) / static_cast<float>(max_pixels)))));
    const int x_mid = static_cast<int>(std::lround(det.cx));
    const float min_width = std::max(6.0f, det.width * 0.35f);
    const float max_width = std::max(min_width + 1.0f, det.width * 1.45f);

    std::vector<float> centers_x;
    std::vector<float> centers_y;
    std::vector<float> weights;
    for (int y = y1; y <= y2; y += row_stride) {
        float best_left = -1.0f;
        float best_left_score = 0.0f;
        float best_right = -1.0f;
        float best_right_score = 0.0f;
        for (int x = x1 + 1; x <= x2 - 1; ++x) {
            const float score = std::abs(sampleGrayCPU(img, pitch, x + 1, y, denoise) -
                                         sampleGrayCPU(img, pitch, x - 1, y, denoise));
            if (x <= x_mid && score > best_left_score) {
                best_left_score = score;
                best_left = static_cast<float>(x);
            } else if (x > x_mid && score > best_right_score) {
                best_right_score = score;
                best_right = static_cast<float>(x);
            }
        }
        if (best_left < 0.0f || best_right < 0.0f) continue;
        const float width = best_right - best_left;
        const float score = std::min(best_left_score, best_right_score);
        if (width < min_width || width > max_width || score < 8.0f) continue;
        centers_x.push_back(0.5f * (best_left + best_right));
        centers_y.push_back(static_cast<float>(y));
        weights.push_back(score);
    }

    if (centers_x.size() < 3) return out;
    double sw = 0.0;
    double swx = 0.0;
    double swy = 0.0;
    for (size_t i = 0; i < centers_x.size(); ++i) {
        const double w = static_cast<double>(weights[i]);
        sw += w;
        swx += w * static_cast<double>(centers_x[i]);
        swy += w * static_cast<double>(centers_y[i]);
    }
    if (sw <= 0.0) return out;

    const float cx = static_cast<float>(swx / sw);
    const float cy = static_cast<float>(swy / sw);
    const float center_shift = std::sqrt((cx - det.cx) * (cx - det.cx) +
                                         (cy - det.cy) * (cy - det.cy));
    const float max_shift = std::max(det.width, det.height) * 0.45f;
    if (center_shift > max_shift) return out;

    const float support_conf = std::min(1.0f,
        static_cast<float>(centers_x.size() * row_stride) / std::max(8.0f, det.height * 0.45f));
    const float shift_conf = std::max(0.2f, 1.0f - center_shift / std::max(1.0f, max_shift));
    out.cx = cx;
    out.cy = cy;
    out.confidence = support_conf * shift_conf;
    out.valid = true;
    return out;
}

CircleFit2D circleFromDetectionCPU(const Detection& det)
{
    CircleFit2D circle;
    if (det.width < 2.0f || det.height < 2.0f) return circle;
    circle.cx = det.cx;
    circle.cy = det.cy;
    circle.radius = std::max(det.width, det.height) * 0.5f;
    circle.confidence = 1.0f;
    circle.source = kCircleSourceBboxProxy;
    circle.valid = true;
    return circle;
}

Detection detectionFromCircleCPU(const CircleFit2D& circle, const Detection& source)
{
    Detection det;
    det.cx = circle.cx;
    det.cy = circle.cy;
    det.width = std::max(2.0f, circle.radius * 2.0f);
    det.height = det.width;
    det.confidence = source.confidence * std::max(0.2f, circle.confidence);
    det.class_id = source.class_id;
    return det;
}

Detection detectionWithCircleCenterCPU(const CircleFit2D& circle, const Detection& source)
{
    Detection det = source;
    det.cx = circle.cx;
    det.cy = circle.cy;
    det.confidence = source.confidence * std::max(0.2f, circle.confidence);
    return det;
}

CircleFit2D searchCircleOnEpipolarCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const CircleFit2D& source_circle,
    float predicted_cx, float predicted_cy,
    float y_tolerance,
    const ROICircleSearchConfig& config)
{
    if (!img || pitch <= 0 || !source_circle.valid) return {};

    const float expected_radius = std::max(4.0f, source_circle.radius);
    const float max_width = std::max(32.0f, static_cast<float>(config.fallback_max_width_px));
    const float max_roi_half_x = max_width * 0.5f;
    const float radius_pad = expected_radius * 1.05f;
    const float margin = std::max(4.0f, static_cast<float>(config.fallback_search_margin_px));
    const float center_half_x = std::min(margin, std::max(4.0f, max_roi_half_x - radius_pad));
    const float roi_half_x = std::min(max_roi_half_x, center_half_x + radius_pad);
    const float roi_half_y = radius_pad + y_tolerance + 2.0f;

    const int x1 = static_cast<int>(std::floor(predicted_cx - roi_half_x));
    const int x2 = static_cast<int>(std::ceil(predicted_cx + roi_half_x));
    const int y1 = static_cast<int>(std::floor(predicted_cy - roi_half_y));
    const int y2 = static_cast<int>(std::ceil(predicted_cy + roi_half_y));

    CircleFitOptions options;
    options.denoise = config.denoise;
    options.max_roi_pixels = config.max_roi_pixels;
    options.min_radius_ratio = 0.45f;
    options.max_radius_ratio = 1.70f;
    options.max_center_shift = std::sqrt(center_half_x * center_half_x +
                                         y_tolerance * y_tolerance) + 2.0f;

    CircleFit2D circle = fitCircleInRegionCPU(
        img, pitch, img_w, img_h,
        x1, y1, x2, y2,
        predicted_cx, predicted_cy,
        expected_radius,
        options);
    if (!circle.valid) return circle;
    if (std::abs(circle.cx - predicted_cx) > center_half_x ||
        std::abs(circle.cy - predicted_cy) > y_tolerance) {
        return {};
    }
    circle.source = kCircleSourceEpipolarSearch;
    return circle;
}

}  // namespace stereo3d
