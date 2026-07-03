#include "dual_yolo_depth_gpu.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cfloat>
#include <cstdint>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxEdges = 1024;
constexpr int kMaxFeaturePoints = 64;
constexpr int kThreadsPerPoint = 4;
constexpr int kMaxParallelFeaturePoints = kThreads / kThreadsPerPoint;

__device__ __forceinline__ int clampInt(int v, int lo, int hi) {
    return max(lo, min(hi, v));
}

__device__ __forceinline__ float clampFloat(float v, float lo, float hi) {
    return fmaxf(lo, fminf(hi, v));
}

__device__ __forceinline__ float readGray(
    const uint8_t* img, int pitch, int x, int y) {
    return static_cast<float>(img[y * pitch + x]);
}

__device__ __forceinline__ void readBgr(
    const uint8_t* img, int pitch, int x, int y,
    float* b, float* g, float* r) {
    const uint8_t* px = img + y * pitch + x * 3;
    *b = static_cast<float>(px[0]);
    *g = static_cast<float>(px[1]);
    *r = static_cast<float>(px[2]);
}

__device__ __forceinline__ int colorLabel(float b, float g, float r) {
    const float hi = fmaxf(r, fmaxf(g, b));
    const float lo = fminf(r, fminf(g, b));
    const float luma = 0.114f * b + 0.587f * g + 0.299f * r;
    const float sat = hi > 1.0f ? (hi - lo) / hi : 0.0f;
    if (luma > 150.0f && sat < 0.25f) return 1;     // white panel
    if (r > 95.0f && g > 85.0f && b < 0.78f * fminf(r, g)) return 2; // yellow
    if (b > 80.0f && b > 1.18f * r && b > 1.08f * g) return 3;       // blue
    if (hi < 45.0f) return 4;                       // dark edge/shadow
    return 0;
}

__device__ __forceinline__ float colorBallLikelihood(float b, float g, float r) {
    const int label = colorLabel(b, g, r);
    if (label == 1 || label == 2 || label == 3) return 1.0f;
    const float hi = fmaxf(r, fmaxf(g, b));
    const float lo = fminf(r, fminf(g, b));
    const float sat = hi > 1.0f ? (hi - lo) / hi : 0.0f;
    const float luma = 0.114f * b + 0.587f * g + 0.299f * r;
    return clampFloat(0.35f * sat + 0.35f * (luma / 255.0f), 0.0f, 1.0f);
}

__device__ __forceinline__ float sobelMag(
    const uint8_t* img, int pitch, int x, int y) {
    const float gx =
        readGray(img, pitch, x + 1, y) - readGray(img, pitch, x - 1, y);
    const float gy =
        readGray(img, pitch, x, y + 1) - readGray(img, pitch, x, y - 1);
    return sqrtf(gx * gx + gy * gy);
}

__device__ __forceinline__ bool patchInside(
    int w, int h, int x, int y, int r) {
    return x >= r && y >= r && x < w - r && y < h - r;
}

__device__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ double blockReduceSum(double val) {
    __shared__ double warp_sums[4];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = lane < 4 ? warp_sums[lane] : 0.0;
        val = warpReduceSum(val);
    }
    __syncthreads();
    return val;
}

__device__ bool solve3x3(
    double A00, double A01, double A02,
    double A10, double A11, double A12,
    double A20, double A21, double A22,
    double b0, double b1, double b2,
    double& x0, double& x1, double& x2) {
    const double det = A00 * (A11 * A22 - A12 * A21) -
                       A01 * (A10 * A22 - A12 * A20) +
                       A02 * (A10 * A21 - A11 * A20);
    if (fabs(det) < 1e-12) return false;
    const double inv = 1.0 / det;
    x0 = (b0 * (A11 * A22 - A12 * A21) -
          A01 * (b1 * A22 - A12 * b2) +
          A02 * (b1 * A21 - A11 * b2)) * inv;
    x1 = (A00 * (b1 * A22 - A12 * b2) -
          b0 * (A10 * A22 - A12 * A20) +
          A02 * (A10 * b2 - b1 * A20)) * inv;
    x2 = (A00 * (A11 * b2 - b1 * A21) -
          A01 * (A10 * b2 - b1 * A20) +
          b0 * (A10 * A21 - A11 * A20)) * inv;
    return true;
}

__device__ __forceinline__ void clearCircle(stereo3d::DualYoloGpuCircle* c) {
    c->cx = 0.0f;
    c->cy = 0.0f;
    c->radius = 0.0f;
    c->confidence = 0.0f;
    c->source = 0;
    c->valid = 0;
}

__device__ __forceinline__ void clearPoint(stereo3d::DualYoloGpuPointMeasure* p) {
    p->cx = 0.0f;
    p->cy = 0.0f;
    p->confidence = 0.0f;
    p->valid = 0;
}

__device__ __forceinline__ void clearDisparity(stereo3d::DualYoloGpuDisparity* d) {
    d->disparity = -1.0f;
    d->confidence = 0.0f;
    d->stddev = -1.0f;
    d->delta_gate_px = 0.0f;
    d->anchor_cx = 0.0f;
    d->anchor_cy = 0.0f;
    d->support = 0;
    d->attempted = 0;
    d->low_confidence = 0;
    d->valid = 0;
}

__device__ float znccScore(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int x_left, int y_left, int x_right, int y_right,
    int patch_radius) {
    double sum_l = 0.0;
    double sum_r = 0.0;
    double sum_l2 = 0.0;
    double sum_r2 = 0.0;
    double sum_lr = 0.0;
    int n = 0;
    for (int yy = -patch_radius; yy <= patch_radius; ++yy) {
        const uint8_t* lrow = left_img + (y_left + yy) * left_pitch;
        const uint8_t* rrow = right_img + (y_right + yy) * right_pitch;
        for (int xx = -patch_radius; xx <= patch_radius; ++xx) {
            const double lv = static_cast<double>(lrow[x_left + xx]);
            const double rv = static_cast<double>(rrow[x_right + xx]);
            sum_l += lv;
            sum_r += rv;
            sum_l2 += lv * lv;
            sum_r2 += rv * rv;
            sum_lr += lv * rv;
            ++n;
        }
    }
    if (n <= 1) return -2.0f;
    const double inv_n = 1.0 / static_cast<double>(n);
    const double mean_l = sum_l * inv_n;
    const double mean_r = sum_r * inv_n;
    const double var_l = sum_l2 - static_cast<double>(n) * mean_l * mean_l;
    const double var_r = sum_r2 - static_cast<double>(n) * mean_r * mean_r;
    if (var_l <= 1e-6 || var_r <= 1e-6) return -2.0f;
    const double cov = sum_lr - static_cast<double>(n) * mean_l * mean_r;
    return static_cast<float>(cov / sqrt(var_l * var_r));
}

__device__ uint32_t census5x5(
    const uint8_t* img, int pitch, int x, int y) {
    const uint8_t center = img[y * pitch + x];
    uint32_t bits = 0u;
    int bit = 0;
    for (int yy = -2; yy <= 2; ++yy) {
        for (int xx = -2; xx <= 2; ++xx) {
            if (xx == 0 && yy == 0) continue;
            bits |= (img[(y + yy) * pitch + x + xx] > center ? 1u : 0u) << bit;
            ++bit;
        }
    }
    return bits;
}

__device__ float binaryPatchScore(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int x_left, int y_left, int x_right, int y_right,
    int patch_radius) {
    if (patch_radius < 2) return -2.0f;
    const uint32_t lc = census5x5(left_img, left_pitch, x_left, y_left);
    const uint32_t rc = census5x5(right_img, right_pitch, x_right, y_right);
    const float census = 1.0f - static_cast<float>(__popc(lc ^ rc)) / 24.0f;
    const float zncc = znccScore(left_img, left_pitch, right_img, right_pitch,
                                 x_left, y_left, x_right, y_right, patch_radius);
    return zncc > -1.5f ? 0.55f * census + 0.45f * (0.5f + 0.5f * zncc) : census;
}

__device__ float colorPatchScore(
    const uint8_t* left_gray, int left_pitch,
    const uint8_t* right_gray, int right_pitch,
    const uint8_t* left_bgr, int left_bgr_pitch,
    const uint8_t* right_bgr, int right_bgr_pitch,
    int x_left, int y_left, int x_right, int y_right,
    int patch_radius) {
    if (!left_bgr || !right_bgr || left_bgr_pitch <= 0 || right_bgr_pitch <= 0) {
        const float zncc = znccScore(left_gray, left_pitch, right_gray, right_pitch,
                                     x_left, y_left, x_right, y_right, patch_radius);
        return zncc > -1.5f ? 0.5f + 0.5f * zncc : -2.0f;
    }

    float color_diff = 0.0f;
    float label_same = 0.0f;
    float gray_sim = 0.0f;
    float color_support = 0.0f;
    int n = 0;
    for (int yy = -patch_radius; yy <= patch_radius; ++yy) {
        for (int xx = -patch_radius; xx <= patch_radius; ++xx) {
            float lb, lg, lr, rb, rg, rr;
            readBgr(left_bgr, left_bgr_pitch, x_left + xx, y_left + yy,
                    &lb, &lg, &lr);
            readBgr(right_bgr, right_bgr_pitch, x_right + xx, y_right + yy,
                    &rb, &rg, &rr);
            const float diff =
                (fabsf(lb - rb) + fabsf(lg - rg) + fabsf(lr - rr)) / (3.0f * 255.0f);
            color_diff += diff;
            const int ll = colorLabel(lb, lg, lr);
            const int rl = colorLabel(rb, rg, rr);
            label_same += (ll == rl && ll != 0) ? 1.0f : 0.0f;
            color_support += 0.5f * (colorBallLikelihood(lb, lg, lr) +
                                     colorBallLikelihood(rb, rg, rr));
            const float lv = readGray(left_gray, left_pitch, x_left + xx, y_left + yy);
            const float rv = readGray(right_gray, right_pitch, x_right + xx, y_right + yy);
            gray_sim += 1.0f - fminf(1.0f, fabsf(lv - rv) / 255.0f);
            ++n;
        }
    }
    if (n <= 0) return -2.0f;

    const float inv_n = 1.0f / static_cast<float>(n);
    const float color_similarity = clampFloat(1.0f - color_diff * inv_n * 2.4f,
                                              0.0f, 1.0f);
    const float label_iou = clampFloat(label_same * inv_n, 0.0f, 1.0f);
    const float support = clampFloat(color_support * inv_n, 0.0f, 1.0f);
    const float gray_consistency = clampFloat(gray_sim * inv_n, 0.0f, 1.0f);
    const float zncc = znccScore(left_gray, left_pitch, right_gray, right_pitch,
                                 x_left, y_left, x_right, y_right, patch_radius);
    const float zncc01 = zncc > -1.5f ? clampFloat(0.5f + 0.5f * zncc, 0.0f, 1.0f) : 0.0f;

    return clampFloat(0.30f * zncc01 +
                      0.25f * color_similarity +
                      0.25f * label_iou +
                      0.12f * gray_consistency +
                      0.08f * support,
                      0.0f, 1.0f);
}

__device__ float localVariance(
    const uint8_t* img, int pitch, int x, int y) {
    float sum = 0.0f;
    float sum2 = 0.0f;
    int n = 0;
    for (int yy = -1; yy <= 1; ++yy) {
        for (int xx = -1; xx <= 1; ++xx) {
            const float v = readGray(img, pitch, x + xx, y + yy);
            sum += v;
            sum2 += v * v;
            ++n;
        }
    }
    const float mean = sum / static_cast<float>(n);
    return fmaxf(0.0f, sum2 / static_cast<float>(n) - mean * mean);
}

__device__ float sparseResponse(
    const uint8_t* img, int pitch, int x, int y, int mode) {
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    for (int yy = -1; yy <= 1; ++yy) {
        for (int xx = -1; xx <= 1; ++xx) {
            const float gx = readGray(img, pitch, x + xx + 1, y + yy) -
                             readGray(img, pitch, x + xx - 1, y + yy);
            const float gy = readGray(img, pitch, x + xx, y + yy + 1) -
                             readGray(img, pitch, x + xx, y + yy - 1);
            sxx += static_cast<double>(gx) * gx;
            syy += static_cast<double>(gy) * gy;
            sxy += static_cast<double>(gx) * gy;
        }
    }
    const double tr = sxx + syy;
    const double det = sxx * syy - sxy * sxy;
    float corner = 0.0f;
    if (tr > 1e-6 && det > 0.0) {
        const double disc = fmax(0.0, tr * tr - 4.0 * det);
        corner = static_cast<float>(0.5 * (tr - sqrt(disc)));
    }
    const float gx = readGray(img, pitch, x + 1, y) - readGray(img, pitch, x - 1, y);
    const float gy = readGray(img, pitch, x, y + 1) - readGray(img, pitch, x, y - 1);
    const float texture = sqrtf(gx * gx + gy * gy) *
                          sqrtf(fmaxf(0.0f, localVariance(img, pitch, x, y)));
    if (mode == 0) return corner;
    if (mode == 1) return texture;
    return 0.65f * corner + 0.35f * texture;
}

__device__ float colorSparseResponse(
    const uint8_t* gray, int gray_pitch,
    const uint8_t* bgr, int bgr_pitch,
    int x, int y,
    int mode) {
    if (!bgr || bgr_pitch <= 0) return 0.0f;
    float b, g, r;
    readBgr(bgr, bgr_pitch, x, y, &b, &g, &r);
    const float support = colorBallLikelihood(b, g, r);
    const float edge = sobelMag(gray, gray_pitch, x, y);
    const float variance = sqrtf(fmaxf(0.0f, localVariance(gray, gray_pitch, x, y)));
    if (mode == 3) {
        return 64.0f * support + 0.18f * variance + 0.08f * edge;
    }
    return edge * (0.35f + 0.65f * support) + 0.10f * variance;
}

__device__ float disparityDeltaGate(
    float initial_disp,
    float focal,
    float baseline,
    float max_disp_delta_px,
    float max_disp_delta_ratio,
    float max_depth_delta_m) {
    float gate = fmaxf(max_disp_delta_px, fabsf(initial_disp) * max_disp_delta_ratio);
    const float fb = focal * baseline;
    if (fb > 0.0f && max_depth_delta_m > 0.0f && initial_disp > 0.5f) {
        const float z = fb / initial_disp;
        const float near_z = fmaxf(0.01f, z - max_depth_delta_m);
        const float far_z = z + max_depth_delta_m;
        gate = fmaxf(gate, fabsf(fb / near_z - initial_disp));
        gate = fmaxf(gate, fabsf(initial_disp - fb / far_z));
    }
    return fmaxf(gate, 0.5f);
}

__device__ __forceinline__ float sampleWeight(float score) {
    return fmaxf(0.05f, score);
}

__device__ float medianSortedValues(const float* values, int n) {
    if (n <= 0) return 0.0f;
    return (n & 1) ? values[n / 2]
                   : 0.5f * (values[n / 2 - 1] + values[n / 2]);
}

__device__ void sortSamplesByDisparity(
    float* sample_disp,
    float* sample_score,
    float* sample_x,
    float* sample_y,
    int n) {
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (sample_disp[j] <= sample_disp[j + 1]) continue;
            const float td = sample_disp[j];
            const float ts = sample_score[j];
            const float tx = sample_x[j];
            const float ty = sample_y[j];
            sample_disp[j] = sample_disp[j + 1];
            sample_score[j] = sample_score[j + 1];
            sample_x[j] = sample_x[j + 1];
            sample_y[j] = sample_y[j + 1];
            sample_disp[j + 1] = td;
            sample_score[j + 1] = ts;
            sample_x[j + 1] = tx;
            sample_y[j + 1] = ty;
        }
    }
}

__device__ void sortValues(float* values, int n) {
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (values[j] <= values[j + 1]) continue;
            const float tmp = values[j];
            values[j] = values[j + 1];
            values[j + 1] = tmp;
        }
    }
}

__device__ bool robustAggregateSamples(
    int n,
    int min_points,
    float initial_disp,
    float max_delta,
    float max_stddev,
    float feature_mad_scale,
    float feature_ransac_gate_px,
    float* sample_disp,
    float* sample_score,
    float* sample_x,
    float* sample_y,
    float* scratch,
    float* out_disp,
    float* out_anchor_x,
    float* out_anchor_y,
    float* out_stddev,
    float* out_avg_score,
    int* out_support) {
    if (n < min_points) return false;

    sortSamplesByDisparity(sample_disp, sample_score, sample_x, sample_y, n);
    const float median = medianSortedValues(sample_disp, n);

    for (int i = 0; i < n; ++i) {
        scratch[i] = fabsf(sample_disp[i] - median);
    }
    sortValues(scratch, n);
    const float mad = medianSortedValues(scratch, n);
    const float robust_sigma = 1.4826f * mad;
    const float min_gate = clampFloat(feature_ransac_gate_px, 0.25f, 3.0f);
    const float mad_gate =
        fmaxf(min_gate, robust_sigma * fmaxf(1.0f, feature_mad_scale));
    const float gate = fminf(fmaxf(0.35f, max_delta), mad_gate);

    float best_center = median;
    float best_weight = -1.0f;
    int best_support = 0;
    for (int i = 0; i < n; ++i) {
        float support_weight = 0.0f;
        int support = 0;
        for (int j = 0; j < n; ++j) {
            if (fabsf(sample_disp[j] - sample_disp[i]) > gate) continue;
            support_weight += sampleWeight(sample_score[j]);
            ++support;
        }
        if (support > best_support ||
            (support == best_support && support_weight > best_weight)) {
            best_support = support;
            best_weight = support_weight;
            best_center = sample_disp[i];
        }
    }

    const float median_gate = fmaxf(gate, min_gate * 1.5f);
    int inliers = 0;
    float total_w = 0.0f;
    for (int i = 0; i < n; ++i) {
        const bool keep =
            fabsf(sample_disp[i] - best_center) <= gate &&
            fabsf(sample_disp[i] - median) <= median_gate;
        scratch[i] = keep ? 1.0f : 0.0f;
        if (!keep) continue;
        total_w += sampleWeight(sample_score[i]);
        ++inliers;
    }
    if (inliers < min_points || total_w <= 0.0f) return false;

    const float half_w = total_w * 0.5f;
    float accum_w = 0.0f;
    float weighted_median = best_center;
    for (int i = 0; i < n; ++i) {
        if (scratch[i] <= 0.0f) continue;
        accum_w += sampleWeight(sample_score[i]);
        if (accum_w >= half_w) {
            weighted_median = sample_disp[i];
            break;
        }
    }
    if (weighted_median <= 0.5f ||
        fabsf(weighted_median - initial_disp) > max_delta) {
        return false;
    }

    float sum_w = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_score = 0.0f;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (scratch[i] <= 0.0f) continue;
        const float w = sampleWeight(sample_score[i]);
        const float diff = sample_disp[i] - weighted_median;
        sum_w += w;
        sum_x += w * sample_x[i];
        sum_y += w * sample_y[i];
        sum_score += w;
        var += w * diff * diff;
    }
    if (sum_w <= 0.0f) return false;

    const float stddev = sqrtf(var / sum_w);
    if (stddev > max_stddev) return false;

    *out_disp = weighted_median;
    *out_anchor_x = sum_x / sum_w;
    *out_anchor_y = sum_y / sum_w;
    *out_stddev = stddev;
    *out_avg_score = sum_score / static_cast<float>(max(1, inliers));
    *out_support = inliers;
    return true;
}

__device__ bool pointInsideDetectionEllipse(
    const stereo3d::DualYoloGpuDetection& det,
    float x,
    float y,
    float scale) {
    if (det.width <= 1.0f || det.height <= 1.0f) return false;
    const float rx = fmaxf(1.0f, det.width * scale);
    const float ry = fmaxf(1.0f, det.height * scale);
    const float nx = (x - det.cx) / rx;
    const float ny = (y - det.cy) / ry;
    return nx * nx + ny * ny <= 1.0f;
}

__device__ __forceinline__ float expectedFeatureYDelta(
    float left_x,
    const stereo3d::DualYoloGpuDetection& left_det,
    float feature_y_slope,
    float feature_y_offset_px) {
    return feature_y_offset_px + feature_y_slope * (left_x - left_det.cx);
}

__device__ __forceinline__ bool passesFeatureYGate(
    float left_x,
    float left_y,
    float right_y,
    const stereo3d::DualYoloGpuDetection& left_det,
    float feature_y_tolerance_px,
    float feature_y_slope,
    float feature_y_offset_px) {
    const float expected = expectedFeatureYDelta(
        left_x, left_det, feature_y_slope, feature_y_offset_px);
    const float residual = (left_y - right_y) - expected;
    return fabsf(residual) <= clampFloat(feature_y_tolerance_px, 0.5f, 8.0f);
}

__device__ float reverseSparseMatchError(
    const uint8_t* left_img,
    int left_pitch,
    const uint8_t* right_img,
    int right_pitch,
    const uint8_t* left_bgr,
    int left_bgr_pitch,
    const uint8_t* right_bgr,
    int right_bgr_pitch,
    int img_w,
    int img_h,
    float left_x,
    float left_y,
    float right_x,
    float right_y,
    int patch_radius,
    int d_start,
    int d_end,
    int y_radius,
    int mode,
    const stereo3d::DualYoloGpuDetection& left_det,
    float feature_y_slope,
    float feature_y_offset_px) {
    const int rx = static_cast<int>(rintf(right_x));
    const int ry = static_cast<int>(rintf(right_y));
    if (!patchInside(img_w, img_h, rx, ry, patch_radius)) return FLT_MAX;

    float best_score = -2.0f;
    int best_lx = -1;
    int best_ly = -1;
    for (int d = d_start; d <= d_end; ++d) {
        const int lx = rx + d;
        const float expected_y = expectedFeatureYDelta(
            static_cast<float>(lx), left_det,
            feature_y_slope, feature_y_offset_px);
        const int dy_center = static_cast<int>(rintf(expected_y));
        for (int dy = dy_center - y_radius; dy <= dy_center + y_radius; ++dy) {
            const int ly = ry + dy;
            if (!patchInside(img_w, img_h, lx, ly, patch_radius)) continue;
            const float score = mode == 2
                ? binaryPatchScore(right_img, right_pitch, left_img, left_pitch,
                                   rx, ry, lx, ly, patch_radius)
                : (mode >= 3
                    ? colorPatchScore(right_img, right_pitch,
                                      left_img, left_pitch,
                                      right_bgr, right_bgr_pitch,
                                      left_bgr, left_bgr_pitch,
                                      rx, ry, lx, ly, patch_radius)
                    : znccScore(right_img, right_pitch, left_img, left_pitch,
                                rx, ry, lx, ly, patch_radius));
            if (score > best_score) {
                best_score = score;
                best_lx = lx;
                best_ly = ly;
            }
        }
    }
    if (best_lx < 0) return FLT_MAX;
    const float dx = static_cast<float>(best_lx) - left_x;
    const float dy = static_cast<float>(best_ly) - left_y;
    return sqrtf(dx * dx + dy * dy);
}

__device__ bool passesFeatureOverlapGate(
    const stereo3d::DualYoloGpuDetection& left_det,
    const stereo3d::DualYoloGpuDetection& right_det,
    float left_x,
    float left_y,
    float right_x,
    float right_y,
    float initial_disp,
    float feature_overlap_scale) {
    const float scale = clampFloat(feature_overlap_scale, 0.35f, 0.90f);
    const float projection_scale = fminf(0.98f, scale + 0.12f);
    if (!pointInsideDetectionEllipse(left_det, left_x, left_y, scale) ||
        !pointInsideDetectionEllipse(right_det, right_x, right_y, scale)) {
        return false;
    }
    if (!pointInsideDetectionEllipse(right_det,
                                     left_x - initial_disp,
                                     left_y,
                                     projection_scale)) {
        return false;
    }
    if (!pointInsideDetectionEllipse(left_det,
                                     right_x + initial_disp,
                                     right_y,
                                     projection_scale)) {
        return false;
    }
    return true;
}

__device__ bool passesSphereRadiusGate(
    float left_x,
    float left_y,
    float ball_cx,
    float ball_cy,
    float disparity,
    float initial_disp,
    float focal,
    float baseline,
    float radius_m,
    float radius_scale,
    float margin_m) {
    if (radius_m <= 0.0f || focal <= 1e-3f || baseline <= 1e-6f ||
        initial_disp <= 0.5f || disparity <= 0.5f) {
        return true;
    }
    const float fb = focal * baseline;
    const float center_z = fb / initial_disp;
    const float z = fb / disparity;
    if (!isfinite(center_z) || !isfinite(z)) return false;
    const float dx = (left_x - ball_cx) * z / focal;
    const float dy = (left_y - ball_cy) * z / focal;
    const float dz = z - center_z;
    const float distance = sqrtf(dx * dx + dy * dy + dz * dz);
    const float max_distance =
        radius_m * fmaxf(1.0f, radius_scale) + fmaxf(0.0f, margin_m);
    return distance <= max_distance;
}

__device__ void fitGeometryInBBox(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const stereo3d::DualYoloGpuDetection& det,
    int max_roi_pixels,
    float* edge_x,
    float* edge_y,
    float* edge_w,
    int* edge_n,
    float* max_grad,
    stereo3d::DualYoloGpuCircle* circle,
    stereo3d::DualYoloGpuPointMeasure* edge_centroid,
    stereo3d::DualYoloGpuPointMeasure* radial_center,
    stereo3d::DualYoloGpuPointMeasure* edge_pair_center) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        *edge_n = 0;
        *max_grad = 0.0f;
        clearCircle(circle);
        clearPoint(edge_centroid);
        clearPoint(radial_center);
        clearPoint(edge_pair_center);
    }
    __syncthreads();

    int x1 = static_cast<int>(floorf(det.cx - det.width * 0.5f));
    int y1 = static_cast<int>(floorf(det.cy - det.height * 0.5f));
    int x2 = static_cast<int>(ceilf(det.cx + det.width * 0.5f));
    int y2 = static_cast<int>(ceilf(det.cy + det.height * 0.5f));
    x1 = clampInt(x1, 1, img_w - 2);
    y1 = clampInt(y1, 1, img_h - 2);
    x2 = clampInt(x2, 1, img_w - 2);
    y2 = clampInt(y2, 1, img_h - 2);
    const int roi_w = x2 - x1 + 1;
    const int roi_h = y2 - y1 + 1;
    if (roi_w < 12 || roi_h < 12) {
        __syncthreads();
        return;
    }

    int stride = 1;
    const int area = roi_w * roi_h;
    while (max_roi_pixels > 0 && area / (stride * stride) > max_roi_pixels) {
        ++stride;
    }
    const int inner_w = max(1, (roi_w - 2 + stride - 1) / stride);
    const int inner_h = max(1, (roi_h - 2 + stride - 1) / stride);
    const int total = inner_w * inner_h;

    float local_max = 0.0f;
    for (int i = tid; i < total; i += kThreads) {
        const int x = x1 + 1 + (i % inner_w) * stride;
        const int y = y1 + 1 + (i / inner_w) * stride;
        if (x <= 0 || x >= img_w - 1 || y <= 0 || y >= img_h - 1) continue;
        local_max = fmaxf(local_max, sobelMag(img, pitch, x, y));
    }
    atomicMax(reinterpret_cast<int*>(max_grad), __float_as_int(local_max));
    __syncthreads();

    const float threshold = fmaxf(10.0f, *max_grad * 0.25f);
    for (int i = tid; i < total; i += kThreads) {
        const int x = x1 + 1 + (i % inner_w) * stride;
        const int y = y1 + 1 + (i / inner_w) * stride;
        if (x <= 0 || x >= img_w - 1 || y <= 0 || y >= img_h - 1) continue;
        const float mag = sobelMag(img, pitch, x, y);
        if (mag <= threshold) continue;
        const int idx = atomicAdd(edge_n, 1);
        if (idx < kMaxEdges) {
            edge_x[idx] = static_cast<float>(x);
            edge_y[idx] = static_cast<float>(y);
            edge_w[idx] = mag;
        }
    }
    __syncthreads();

    const int n = min(*edge_n, kMaxEdges);
    if (n < 12) {
        __syncthreads();
        return;
    }

    double sw = 0.0, swx = 0.0, swy = 0.0;
    double swxx = 0.0, swyy = 0.0, swxy = 0.0;
    double swxz = 0.0, swyz = 0.0, swz = 0.0;
    int local_min_x = 0x3fffffff;
    int local_max_x = -0x3fffffff;
    int local_min_y = 0x3fffffff;
    int local_max_y = -0x3fffffff;

    for (int i = tid; i < n; i += kThreads) {
        const double w = static_cast<double>(edge_w[i]);
        const double x = static_cast<double>(edge_x[i]);
        const double y = static_cast<double>(edge_y[i]);
        const double z = x * x + y * y;
        sw += w;
        swx += w * x;
        swy += w * y;
        swxx += w * x * x;
        swyy += w * y * y;
        swxy += w * x * y;
        swxz += w * x * z;
        swyz += w * y * z;
        swz += w * z;
        local_min_x = min(local_min_x, static_cast<int>(edge_x[i]));
        local_max_x = max(local_max_x, static_cast<int>(edge_x[i]));
        local_min_y = min(local_min_y, static_cast<int>(edge_y[i]));
        local_max_y = max(local_max_y, static_cast<int>(edge_y[i]));
    }

    sw = blockReduceSum(sw);
    swx = blockReduceSum(swx);
    swy = blockReduceSum(swy);
    swxx = blockReduceSum(swxx);
    swyy = blockReduceSum(swyy);
    swxy = blockReduceSum(swxy);
    swxz = blockReduceSum(swxz);
    swyz = blockReduceSum(swyz);
    swz = blockReduceSum(swz);

    __shared__ int min_x, max_x, min_y, max_y;
    if (tid == 0) {
        min_x = 0x3fffffff;
        max_x = -0x3fffffff;
        min_y = 0x3fffffff;
        max_y = -0x3fffffff;
    }
    __syncthreads();
    atomicMin(&min_x, local_min_x);
    atomicMax(&max_x, local_max_x);
    atomicMin(&min_y, local_min_y);
    atomicMax(&max_y, local_max_y);
    __syncthreads();

    if (tid == 0) {
        const float edge_score = clampFloat(static_cast<float>(n) / 80.0f, 0.0f, 1.0f);
        if (sw > 1e-6) {
            edge_centroid->cx = static_cast<float>(swx / sw);
            edge_centroid->cy = static_cast<float>(swy / sw);
            edge_centroid->confidence = edge_score;
            edge_centroid->valid = 1;
        }
        if (min_x < max_x && min_y < max_y) {
            edge_pair_center->cx = 0.5f * static_cast<float>(min_x + max_x);
            edge_pair_center->cy = 0.5f * static_cast<float>(min_y + max_y);
            edge_pair_center->confidence = edge_score * 0.85f;
            edge_pair_center->valid = 1;
        }

        double a = 0.0, b = 0.0, c = 0.0;
        if (solve3x3(swxx, swxy, swx,
                     swxy, swyy, swy,
                     swx, swy, sw,
                     swxz, swyz, swz,
                     a, b, c)) {
            const double cx = a * 0.5;
            const double cy = b * 0.5;
            const double r2 = c + cx * cx + cy * cy;
            const float r = r2 > 0.0 ? static_cast<float>(sqrt(r2)) : 0.0f;
            const float bbox_cx = det.cx;
            const float bbox_cy = det.cy;
            const float max_r = fmaxf(det.width, det.height) * 0.75f;
            const float min_r = fmaxf(4.0f, fminf(det.width, det.height) * 0.20f);
            const float center_gate = fmaxf(det.width, det.height) * 0.55f;
            if (r >= min_r && r <= max_r &&
                fabsf(static_cast<float>(cx) - bbox_cx) <= center_gate &&
                fabsf(static_cast<float>(cy) - bbox_cy) <= center_gate) {
                circle->cx = static_cast<float>(cx);
                circle->cy = static_cast<float>(cy);
                circle->radius = r;
                circle->confidence = fmaxf(0.15f, edge_score);
                circle->source = 2;
                circle->valid = 1;
                radial_center->cx = circle->cx;
                radial_center->cy = circle->cy;
                radial_center->confidence = circle->confidence * 0.9f;
                radial_center->valid = 1;
            }
        }
    }
    __syncthreads();
}

__device__ void matchPatchAtPoint(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    float x_left_f, float y_left_f,
    float initial_disp,
    int patch_radius,
    int search_radius,
    int max_disparity,
    float min_confidence,
    float max_delta,
    float focal,
    float baseline,
    float min_depth,
    float max_depth,
    stereo3d::DualYoloGpuDisparity* out) {
    if (threadIdx.x == 0) {
        clearDisparity(out);
        out->attempted = 0;
    }
    __syncthreads();

    const int x_left = static_cast<int>(rintf(x_left_f));
    const int y_left = static_cast<int>(rintf(y_left_f));
    const int d0 = static_cast<int>(rintf(initial_disp));
    const int d_start = max(1, d0 - search_radius);
    const int d_end = min(max_disparity, d0 + search_radius);
    if (!patchInside(img_w, img_h, x_left, y_left, patch_radius) ||
        d_start >= d_end) {
        __syncthreads();
        return;
    }

    __shared__ float best_score_parts[kThreadsPerPoint];
    __shared__ int best_disp_parts[kThreadsPerPoint];
    __shared__ float second_score_parts[kThreadsPerPoint];
    if (threadIdx.x < kThreadsPerPoint) {
        best_score_parts[threadIdx.x] = -2.0f;
        best_disp_parts[threadIdx.x] = -1;
        second_score_parts[threadIdx.x] = -2.0f;
    }
    __syncthreads();

    if (threadIdx.x < kThreadsPerPoint) {
        const int range = d_end - d_start + 1;
        const int per = (range + kThreadsPerPoint - 1) / kThreadsPerPoint;
        const int begin = d_start + threadIdx.x * per;
        const int end = min(d_end, begin + per - 1);
        float best_score = -2.0f;
        float second_score = -2.0f;
        int best_disp = -1;
        for (int d = begin; d <= end; ++d) {
            const int xr = x_left - d;
            if (!patchInside(img_w, img_h, xr, y_left, patch_radius)) continue;
            const float score = znccScore(left_img, left_pitch, right_img, right_pitch,
                                          x_left, y_left, xr, y_left, patch_radius);
            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_disp = d;
            } else if (score > second_score) {
                second_score = score;
            }
        }
        best_score_parts[threadIdx.x] = best_score;
        best_disp_parts[threadIdx.x] = best_disp;
        second_score_parts[threadIdx.x] = second_score;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_score = best_score_parts[0];
        float second_score = second_score_parts[0];
        int best_disp = best_disp_parts[0];
        for (int i = 1; i < kThreadsPerPoint; ++i) {
            if (best_score_parts[i] > best_score) {
                second_score = fmaxf(second_score, fmaxf(best_score, second_score_parts[i]));
                best_score = best_score_parts[i];
                best_disp = best_disp_parts[i];
            } else {
                second_score = fmaxf(second_score, best_score_parts[i]);
            }
        }
        out->attempted = d_end - d_start + 1;
        bool accept = true;
        if (best_disp <= 0 || best_score < fmaxf(0.10f, min_confidence * 0.60f)) {
            out->low_confidence = 1;
            accept = false;
        }

        float sub_disp = static_cast<float>(best_disp);
        if (accept && best_disp > d_start && best_disp < d_end) {
            const int xr_m = x_left - (best_disp - 1);
            const int xr_p = x_left - (best_disp + 1);
            if (patchInside(img_w, img_h, xr_m, y_left, patch_radius) &&
                patchInside(img_w, img_h, xr_p, y_left, patch_radius)) {
                const float s_m = znccScore(left_img, left_pitch, right_img, right_pitch,
                                            x_left, y_left, xr_m, y_left, patch_radius);
                const float s_p = znccScore(left_img, left_pitch, right_img, right_pitch,
                                            x_left, y_left, xr_p, y_left, patch_radius);
                const float denom = s_m - 2.0f * best_score + s_p;
                if (s_m > -1.5f && s_p > -1.5f && denom < -1e-5f) {
                    sub_disp += clampFloat(0.5f * (s_m - s_p) / denom, -1.0f, 1.0f);
                }
            }
        }

        const float uniqueness = second_score > -1.5f ? best_score - second_score : 1.0f;
        const float z = focal * baseline / fmaxf(0.5f, sub_disp);
        if (accept &&
            ((uniqueness < 0.01f && best_score < 0.75f) ||
             fabsf(sub_disp - initial_disp) > max_delta ||
             sub_disp <= 0.5f || sub_disp > static_cast<float>(max_disparity) ||
             z < min_depth || z > max_depth)) {
            out->low_confidence = 1;
            accept = false;
        }
        if (accept) {
            out->disparity = sub_disp;
            out->confidence = clampFloat((best_score - 0.10f) / 0.80f, 0.0f, 1.0f);
            out->stddev = 0.0f;
            out->delta_gate_px = max_delta;
            out->anchor_cx = x_left_f;
            out->anchor_cy = y_left_f;
            out->support = 1;
            out->valid = out->confidence >= min_confidence ? 1 : 0;
            out->low_confidence = out->valid ? 0 : 1;
        }
    }
    __syncthreads();
}

__device__ void matchSparsePoints(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    const uint8_t* left_bgr, int left_bgr_pitch,
    const uint8_t* right_bgr, int right_bgr_pitch,
    int img_w, int img_h,
    const stereo3d::DualYoloGpuDetection& left_det,
    const stereo3d::DualYoloGpuDetection& right_det,
    float center_x, float center_y, float radius,
    float initial_disp,
    int mode,
    int patch_radius,
    int search_radius,
    int max_points,
    int min_points,
    int max_disparity,
    float min_confidence,
    float max_delta,
    float max_stddev,
    float focal,
    float baseline,
    float min_depth,
    float max_depth,
    float feature_y_tolerance_px,
    float feature_y_slope,
    float feature_y_offset_px,
    float feature_reverse_check_px,
    float feature_overlap_scale,
    float feature_mad_scale,
    float feature_ransac_gate_px,
    float feature_sphere_radius_m,
    float feature_sphere_radius_scale,
    float feature_sphere_margin_m,
    float* sample_disp,
    float* sample_score,
    float* sample_x,
    float* sample_y,
    float* point_x,
    float* point_y,
    float best_score_parts[kMaxFeaturePoints][kThreadsPerPoint],
    float best_disp_parts[kMaxFeaturePoints][kThreadsPerPoint],
    float best_dy_parts[kMaxFeaturePoints][kThreadsPerPoint],
    int* valid_count,
    stereo3d::DualYoloGpuDisparity* out) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        *valid_count = 0;
        clearDisparity(out);
    }
    for (int i = tid; i < kMaxFeaturePoints; i += kThreads) {
        sample_disp[i] = 0.0f;
        sample_score[i] = 0.0f;
        sample_x[i] = 0.0f;
        sample_y[i] = 0.0f;
        point_x[i] = 0.0f;
        point_y[i] = 0.0f;
        for (int j = 0; j < kThreadsPerPoint; ++j) {
            best_score_parts[i][j] = -2.0f;
            best_disp_parts[i][j] = -1.0f;
            best_dy_parts[i][j] = 0.0f;
        }
    }
    __syncthreads();

    max_points = clampInt(max_points, 1,
                          min(kMaxFeaturePoints, kMaxParallelFeaturePoints));
    min_points = clampInt(min_points, 1, max_points);
    patch_radius = clampInt(patch_radius, 2, 8);
    const int point_idx = tid / kThreadsPerPoint;
    const int sub_idx = tid % kThreadsPerPoint;
    if (point_idx < max_points) {
        const int grid = max(2, static_cast<int>(ceilf(sqrtf(static_cast<float>(max_points)))));
        const int gx = point_idx % grid;
        const int gy = point_idx / grid;
        float sample_cx = center_x;
        float sample_cy = center_y;
        float rx = fmaxf(4.0f, fminf(radius * 0.82f, left_det.width * 0.42f));
        float ry = fmaxf(4.0f, fminf(radius * 0.82f, left_det.height * 0.42f));
        if (mode >= 3) {
            const float projected_right_cx = right_det.cx + initial_disp;
            sample_cx = 0.5f * (left_det.cx + projected_right_cx);
            sample_cy = 0.5f * (left_det.cy + right_det.cy);
            rx = fmaxf(4.0f, fminf(radius * 0.90f,
                                   fminf(left_det.width, right_det.width) * 0.42f));
            ry = fmaxf(4.0f, fminf(radius * 0.90f,
                                   fminf(left_det.height, right_det.height) * 0.42f));
        }
        const float u = (static_cast<float>(gx) + 0.5f) / static_cast<float>(grid);
        const float v = (static_cast<float>(gy) + 0.5f) / static_cast<float>(grid);
        const float x_f = sample_cx + (u - 0.5f) * 2.0f * rx;
        const float y_f = sample_cy + (v - 0.5f) * 2.0f * ry;
        const float nx = (x_f - sample_cx) / fmaxf(rx, 1.0f);
        const float ny = (y_f - sample_cy) / fmaxf(ry, 1.0f);
        const int x = static_cast<int>(rintf(x_f));
        const int y = static_cast<int>(rintf(y_f));
        if (nx * nx + ny * ny <= 0.92f * 0.92f &&
            patchInside(img_w, img_h, x, y, patch_radius)) {
            const float response = mode >= 3
                ? colorSparseResponse(left_img, left_pitch,
                                      left_bgr, left_bgr_pitch, x, y, mode)
                : sparseResponse(left_img, left_pitch, x, y, mode);
            const float response_floor =
                mode == 3 ? 24.0f : (mode == 4 ? 12.0f : (mode == 1 ? 20.0f : 8.0f));
            if (mode >= 3 || response > response_floor) {
                const int d0 = static_cast<int>(rintf(initial_disp));
                const int d_start = max(1, d0 - search_radius);
                const int d_end = min(max_disparity, d0 + search_radius);
                if (d_start < d_end) {
                    const int range = d_end - d_start + 1;
                    const int per = (range + kThreadsPerPoint - 1) / kThreadsPerPoint;
                    const int begin = d_start + sub_idx * per;
                    const int end = min(d_end, begin + per - 1);
                    const float expected_y = expectedFeatureYDelta(
                        static_cast<float>(x), left_det,
                        feature_y_slope, feature_y_offset_px);
                    const int y_radius = mode >= 3
                        ? 1
                        : clampInt(static_cast<int>(ceilf(
                                       clampFloat(feature_y_tolerance_px, 0.5f, 8.0f))),
                                   1, 3);
                    const int dy_center = static_cast<int>(rintf(-expected_y));
                    float best_score = -2.0f;
                    float best_disp = -1.0f;
                    float best_dy = 0.0f;
                    for (int d = begin; d <= end; ++d) {
                        const int xr = x - d;
                        for (int dy = dy_center - y_radius;
                             dy <= dy_center + y_radius; ++dy) {
                            const int yr = y + dy;
                            if (!patchInside(img_w, img_h, xr, yr, patch_radius)) continue;
                            const float score = mode == 2
                                ? binaryPatchScore(left_img, left_pitch, right_img, right_pitch,
                                                   x, y, xr, yr, patch_radius)
                                : (mode >= 3
                                    ? colorPatchScore(left_img, left_pitch,
                                                      right_img, right_pitch,
                                                      left_bgr, left_bgr_pitch,
                                                      right_bgr, right_bgr_pitch,
                                                      x, y, xr, yr, patch_radius)
                                    : znccScore(left_img, left_pitch, right_img, right_pitch,
                                                x, y, xr, yr, patch_radius));
                            if (score > best_score) {
                                best_score = score;
                                best_disp = static_cast<float>(d);
                                best_dy = static_cast<float>(dy);
                            }
                        }
                    }
                    best_score_parts[point_idx][sub_idx] = best_score;
                    best_disp_parts[point_idx][sub_idx] = best_disp;
                    best_dy_parts[point_idx][sub_idx] = best_dy;
                    if (sub_idx == 0) {
                        point_x[point_idx] = static_cast<float>(x);
                        point_y[point_idx] = static_cast<float>(y);
                    }
                }
            }
        }
    }
    __syncthreads();

    if (point_idx < max_points && sub_idx == 0) {
        float best_score = best_score_parts[point_idx][0];
        float best_disp = best_disp_parts[point_idx][0];
        float best_dy = best_dy_parts[point_idx][0];
        for (int i = 1; i < kThreadsPerPoint; ++i) {
            if (best_score_parts[point_idx][i] > best_score) {
                best_score = best_score_parts[point_idx][i];
                best_disp = best_disp_parts[point_idx][i];
                best_dy = best_dy_parts[point_idx][i];
            }
        }
        const float min_score = mode == 2
            ? fmaxf(0.58f, 0.50f + min_confidence * 0.35f)
            : (mode >= 3
                ? fmaxf(0.40f, 0.36f + min_confidence * 0.30f)
                : fmaxf(0.12f, min_confidence * 0.60f));
        if (best_disp > 0.5f && best_score >= min_score &&
            fabsf(best_disp - initial_disp) <= max_delta) {
            const float z = focal * baseline / best_disp;
            const float left_x = point_x[point_idx];
            const float left_y = point_y[point_idx];
            const float right_x = left_x - best_disp;
            const float right_y = left_y + best_dy;
            const bool cheap_ok =
                z >= min_depth && z <= max_depth &&
                passesFeatureYGate(left_x, left_y, right_y, left_det,
                                   feature_y_tolerance_px,
                                   feature_y_slope,
                                   feature_y_offset_px) &&
                passesFeatureOverlapGate(left_det, right_det,
                                         left_x, left_y, right_x, right_y,
                                         initial_disp, feature_overlap_scale) &&
                passesSphereRadiusGate(left_x, left_y,
                                       center_x, center_y,
                                       best_disp, initial_disp,
                                       focal, baseline,
                                       feature_sphere_radius_m,
                                       feature_sphere_radius_scale,
                                       feature_sphere_margin_m);
            if (cheap_ok) {
                const int d0 = static_cast<int>(rintf(initial_disp));
                const int d_start = max(1, d0 - search_radius);
                const int d_end = min(max_disparity, d0 + search_radius);
                const int reverse_y_radius = mode >= 3
                    ? 1
                    : clampInt(static_cast<int>(ceilf(
                                   clampFloat(feature_y_tolerance_px, 0.5f, 8.0f))),
                               1, 3);
                const float reverse_err = feature_reverse_check_px >= 0.0f
                    ? reverseSparseMatchError(left_img, left_pitch,
                                              right_img, right_pitch,
                                              left_bgr, left_bgr_pitch,
                                              right_bgr, right_bgr_pitch,
                                              img_w, img_h,
                                              left_x, left_y,
                                              right_x, right_y,
                                              patch_radius,
                                              d_start, d_end,
                                              reverse_y_radius,
                                              mode,
                                              left_det,
                                              feature_y_slope,
                                              feature_y_offset_px)
                    : 0.0f;
                if (feature_reverse_check_px < 0.0f ||
                    reverse_err <= fmaxf(0.25f, feature_reverse_check_px)) {
                    const int idx = atomicAdd(valid_count, 1);
                    if (idx < kMaxFeaturePoints) {
                        sample_disp[idx] = best_disp;
                        sample_score[idx] = best_score;
                        sample_x[idx] = left_x;
                        sample_y[idx] = left_y;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        const int n = min(*valid_count, kMaxFeaturePoints);
        out->attempted = max_points;
        out->support = n;
        if (n < min_points) {
            out->low_confidence = 1;
        } else {
            float disparity = 0.0f;
            float anchor_x = 0.0f;
            float anchor_y = 0.0f;
            float stddev = 0.0f;
            float avg_score = 0.0f;
            int support = 0;
            if (!robustAggregateSamples(n, min_points, initial_disp, max_delta,
                                        max_stddev, feature_mad_scale,
                                        feature_ransac_gate_px,
                                        sample_disp, sample_score,
                                        sample_x, sample_y, point_x,
                                        &disparity, &anchor_x, &anchor_y,
                                        &stddev, &avg_score, &support)) {
                out->low_confidence = 1;
            } else {
                const float z = focal * baseline / fmaxf(disparity, 0.5f);
                if (disparity > static_cast<float>(max_disparity) ||
                    z < min_depth || z > max_depth) {
                    out->low_confidence = 1;
                } else {
                    const float min_score = mode == 2
                        ? fmaxf(0.58f, 0.50f + min_confidence * 0.35f)
                        : (mode >= 3
                            ? fmaxf(0.40f, 0.36f + min_confidence * 0.30f)
                            : fmaxf(0.12f, min_confidence * 0.60f));
                    const float score_conf =
                        clampFloat((avg_score - min_score) /
                                   fmaxf(0.01f, 1.0f - min_score),
                                   0.0f, 1.0f);
                    const float support_ratio =
                        static_cast<float>(support) /
                        static_cast<float>(max(1, max_points));
                    const float consistency =
                        clampFloat(1.0f / (1.0f + stddev), 0.0f, 1.0f);
                    const float delta_conf =
                        1.0f - fminf(1.0f, fabsf(disparity - initial_disp) / max_delta);
                    out->disparity = disparity;
                    out->confidence = clampFloat(0.30f * support_ratio +
                                                 0.35f * score_conf +
                                                 0.25f * consistency +
                                                 0.10f * delta_conf,
                                                 0.0f, 1.0f);
                    out->stddev = stddev;
                    out->delta_gate_px = max_delta;
                    out->anchor_cx = anchor_x;
                    out->anchor_cy = anchor_y;
                    out->support = support;
                    out->valid = out->confidence >= min_confidence ? 1 : 0;
                    out->low_confidence = out->valid ? 0 : 1;
                }
            }
        }
    }
    __syncthreads();
}

__device__ void matchMultiPointPatch(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    const stereo3d::DualYoloGpuDetection& left_det,
    const stereo3d::DualYoloGpuDetection& right_det,
    float center_x, float center_y, float radius,
    float initial_disp,
    int patch_radius,
    int search_radius,
    int max_points,
    int min_points,
    int max_disparity,
    float min_confidence,
    float max_delta,
    float max_stddev,
    float focal,
    float baseline,
    float min_depth,
    float max_depth,
    float feature_y_tolerance_px,
    float feature_y_slope,
    float feature_y_offset_px,
    float feature_reverse_check_px,
    float feature_overlap_scale,
    float feature_mad_scale,
    float feature_ransac_gate_px,
    float feature_sphere_radius_m,
    float feature_sphere_radius_scale,
    float feature_sphere_margin_m,
    float* sample_disp,
    float* sample_score,
    float* sample_x,
    float* sample_y,
    float* point_x,
    float* point_y,
    float best_score_parts[kMaxFeaturePoints][kThreadsPerPoint],
    float best_disp_parts[kMaxFeaturePoints][kThreadsPerPoint],
    float best_dy_parts[kMaxFeaturePoints][kThreadsPerPoint],
    int* valid_count,
    stereo3d::DualYoloGpuDisparity* out) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        *valid_count = 0;
        clearDisparity(out);
    }
    for (int i = tid; i < kMaxFeaturePoints; i += kThreads) {
        sample_disp[i] = 0.0f;
        sample_score[i] = 0.0f;
        sample_x[i] = 0.0f;
        sample_y[i] = 0.0f;
        point_x[i] = 0.0f;
        point_y[i] = 0.0f;
        for (int j = 0; j < kThreadsPerPoint; ++j) {
            best_score_parts[i][j] = -2.0f;
            best_disp_parts[i][j] = -1.0f;
            best_dy_parts[i][j] = 0.0f;
        }
    }
    __syncthreads();

    max_points = clampInt(max_points, 1,
                          min(kMaxFeaturePoints, kMaxParallelFeaturePoints));
    min_points = clampInt(min_points, 1, max_points);
    patch_radius = clampInt(patch_radius, 2, 8);
    const int point_idx = tid / kThreadsPerPoint;
    const int sub_idx = tid % kThreadsPerPoint;
    if (point_idx < max_points) {
        float x_f = center_x;
        float y_f = center_y;
        if (point_idx > 0) {
            const int ring_idx = point_idx - 1;
            const int angle_idx = ring_idx % 8;
            const int ring = ring_idx / 8;
            const float ring_frac = ring == 0 ? 0.28f : (ring == 1 ? 0.48f : 0.66f);
            const float angle = 0.78539816339f * static_cast<float>(angle_idx);
            const float rr = fmaxf(static_cast<float>(patch_radius + 2), radius * ring_frac);
            x_f += rr * cosf(angle);
            y_f += rr * sinf(angle);
        }
        const int x = static_cast<int>(rintf(x_f));
        const int y = static_cast<int>(rintf(y_f));
        if (patchInside(img_w, img_h, x, y, patch_radius)) {
            const int d0 = static_cast<int>(rintf(initial_disp));
            const int d_start = max(1, d0 - search_radius);
            const int d_end = min(max_disparity, d0 + search_radius);
            if (d_start < d_end) {
                const int range = d_end - d_start + 1;
                const int per = (range + kThreadsPerPoint - 1) / kThreadsPerPoint;
                const int begin = d_start + sub_idx * per;
                const int end = min(d_end, begin + per - 1);
                const float expected_y = expectedFeatureYDelta(
                    static_cast<float>(x), left_det,
                    feature_y_slope, feature_y_offset_px);
                const int y_radius = clampInt(
                    static_cast<int>(ceilf(
                        clampFloat(feature_y_tolerance_px, 0.5f, 8.0f))),
                    1, 3);
                const int dy_center = static_cast<int>(rintf(-expected_y));
                float best_score = -2.0f;
                float best_disp = -1.0f;
                float best_dy = 0.0f;
                for (int d = begin; d <= end; ++d) {
                    const int xr = x - d;
                    for (int dy = dy_center - y_radius;
                         dy <= dy_center + y_radius; ++dy) {
                        const int yr = y + dy;
                        if (!patchInside(img_w, img_h, xr, yr, patch_radius)) continue;
                        const float score = znccScore(left_img, left_pitch,
                                                      right_img, right_pitch,
                                                      x, y, xr, yr, patch_radius);
                        if (score > best_score) {
                            best_score = score;
                            best_disp = static_cast<float>(d);
                            best_dy = static_cast<float>(dy);
                        }
                    }
                }
                best_score_parts[point_idx][sub_idx] = best_score;
                best_disp_parts[point_idx][sub_idx] = best_disp;
                best_dy_parts[point_idx][sub_idx] = best_dy;
                if (sub_idx == 0) {
                    point_x[point_idx] = static_cast<float>(x);
                    point_y[point_idx] = static_cast<float>(y);
                }
            }
        }
    }
    __syncthreads();

    if (point_idx < max_points && sub_idx == 0) {
        float best_score = best_score_parts[point_idx][0];
        float best_disp = best_disp_parts[point_idx][0];
        float best_dy = best_dy_parts[point_idx][0];
        for (int i = 1; i < kThreadsPerPoint; ++i) {
            if (best_score_parts[point_idx][i] > best_score) {
                best_score = best_score_parts[point_idx][i];
                best_disp = best_disp_parts[point_idx][i];
                best_dy = best_dy_parts[point_idx][i];
            }
        }
        const float min_score = fmaxf(0.10f, min_confidence * 0.60f);
        if (best_disp > 0.5f && best_score >= min_score &&
            fabsf(best_disp - initial_disp) <= max_delta) {
            const float z = focal * baseline / best_disp;
            const float left_x = point_x[point_idx];
            const float left_y = point_y[point_idx];
            const float right_x = left_x - best_disp;
            const float right_y = left_y + best_dy;
            const bool cheap_ok =
                z >= min_depth && z <= max_depth &&
                passesFeatureYGate(left_x, left_y, right_y, left_det,
                                   feature_y_tolerance_px,
                                   feature_y_slope,
                                   feature_y_offset_px) &&
                passesFeatureOverlapGate(left_det, right_det,
                                         left_x, left_y, right_x, right_y,
                                         initial_disp, feature_overlap_scale) &&
                passesSphereRadiusGate(left_x, left_y,
                                       center_x, center_y,
                                       best_disp, initial_disp,
                                       focal, baseline,
                                       feature_sphere_radius_m,
                                       feature_sphere_radius_scale,
                                       feature_sphere_margin_m);
            if (cheap_ok) {
                const int d0 = static_cast<int>(rintf(initial_disp));
                const int d_start = max(1, d0 - search_radius);
                const int d_end = min(max_disparity, d0 + search_radius);
                const int reverse_y_radius = clampInt(
                    static_cast<int>(ceilf(
                        clampFloat(feature_y_tolerance_px, 0.5f, 8.0f))),
                    1, 3);
                const float reverse_err = feature_reverse_check_px >= 0.0f
                    ? reverseSparseMatchError(left_img, left_pitch,
                                              right_img, right_pitch,
                                              nullptr, 0,
                                              nullptr, 0,
                                              img_w, img_h,
                                              left_x, left_y,
                                              right_x, right_y,
                                              patch_radius,
                                              d_start, d_end,
                                              reverse_y_radius,
                                              1,
                                              left_det,
                                              feature_y_slope,
                                              feature_y_offset_px)
                    : 0.0f;
                if (feature_reverse_check_px < 0.0f ||
                    reverse_err <= fmaxf(0.25f, feature_reverse_check_px)) {
                    const int idx = atomicAdd(valid_count, 1);
                    if (idx < kMaxFeaturePoints) {
                        sample_disp[idx] = best_disp;
                        sample_score[idx] = best_score;
                        sample_x[idx] = left_x;
                        sample_y[idx] = left_y;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        const int n = min(*valid_count, kMaxFeaturePoints);
        out->attempted = max_points;
        out->support = n;
        if (n < min_points) {
            out->low_confidence = 1;
        } else {
            float disparity = 0.0f;
            float anchor_x = 0.0f;
            float anchor_y = 0.0f;
            float stddev = 0.0f;
            float avg_score = 0.0f;
            int support = 0;
            if (!robustAggregateSamples(n, min_points, initial_disp, max_delta,
                                        max_stddev, feature_mad_scale,
                                        feature_ransac_gate_px,
                                        sample_disp, sample_score,
                                        sample_x, sample_y, point_x,
                                        &disparity, &anchor_x, &anchor_y,
                                        &stddev, &avg_score, &support)) {
                out->low_confidence = 1;
            } else {
                const float z = focal * baseline / fmaxf(disparity, 0.5f);
                if (disparity > static_cast<float>(max_disparity) ||
                    z < min_depth || z > max_depth) {
                    out->low_confidence = 1;
                } else {
                    const float min_score = fmaxf(0.10f, min_confidence * 0.60f);
                    const float score_conf =
                        clampFloat((avg_score - min_score) /
                                   fmaxf(0.01f, 1.0f - min_score),
                                   0.0f, 1.0f);
                    const float support_ratio =
                        static_cast<float>(support) /
                        static_cast<float>(max(1, max_points));
                    const float consistency =
                        clampFloat(1.0f / (1.0f + stddev), 0.0f, 1.0f);
                    const float delta_conf =
                        1.0f - fminf(1.0f, fabsf(disparity - initial_disp) / max_delta);
                    out->disparity = disparity;
                    out->confidence = clampFloat(0.30f * support_ratio +
                                                 0.35f * score_conf +
                                                 0.25f * consistency +
                                                 0.10f * delta_conf,
                                                 0.0f, 1.0f);
                    out->stddev = stddev;
                    out->delta_gate_px = max_delta;
                    out->anchor_cx = anchor_x;
                    out->anchor_cy = anchor_y;
                    out->support = support;
                    out->valid = out->confidence >= min_confidence ? 1 : 0;
                    out->low_confidence = out->valid ? 0 : 1;
                }
            }
        }
    }
    __syncthreads();
}

__global__ void dualYoloDepthCandidatesKernel(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    const uint8_t* left_bgr, int left_bgr_pitch,
    const uint8_t* right_bgr, int right_bgr_pitch,
    int img_w, int img_h,
    const stereo3d::DualYoloGpuDetectionPair* pairs,
    int num_pairs,
    stereo3d::DualYoloGpuCandidate* results,
    int max_disparity,
    int patch_radius,
    int search_radius_px,
    int max_points,
    int min_points,
    int circle_max_roi_pixels,
    float min_confidence,
    float max_disp_delta_px,
    float max_disp_delta_ratio,
    float max_depth_delta_m,
    float max_stddev_px,
    float epipolar_y_tolerance,
    float feature_y_tolerance_px,
    float feature_y_slope,
    float feature_y_offset_px,
    float feature_reverse_check_px,
    float feature_overlap_scale,
    float feature_mad_scale,
    float feature_ransac_gate_px,
    float feature_sphere_radius_m,
    float feature_sphere_radius_scale,
    float feature_sphere_margin_m,
    int compute_geometry,
    int compute_center_patch,
    int compute_multi_point,
    int compute_corner_points,
    int compute_texture_points,
    int compute_binary_points,
    int compute_orb_points,
    int compute_brisk_points,
    int compute_akaze_points,
    int compute_sift_points,
    int compute_iou_region_color_patch,
    int compute_patch_iou_color_edge,
    float focal,
    float baseline,
    float min_depth,
    float max_depth) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    __shared__ float edge_x[kMaxEdges];
    __shared__ float edge_y[kMaxEdges];
    __shared__ float edge_w[kMaxEdges];
    __shared__ int edge_n;
    __shared__ float max_grad;
    __shared__ float sample_disp[kMaxFeaturePoints];
    __shared__ float sample_score[kMaxFeaturePoints];
    __shared__ float sample_x[kMaxFeaturePoints];
    __shared__ float sample_y[kMaxFeaturePoints];
    __shared__ float point_x[kMaxFeaturePoints];
    __shared__ float point_y[kMaxFeaturePoints];
    __shared__ float best_score_parts[kMaxFeaturePoints][kThreadsPerPoint];
    __shared__ float best_disp_parts[kMaxFeaturePoints][kThreadsPerPoint];
    __shared__ float best_dy_parts[kMaxFeaturePoints][kThreadsPerPoint];
    __shared__ int valid_count;

    stereo3d::DualYoloGpuCandidate* out = &results[pair_idx];
    const stereo3d::DualYoloGpuDetectionPair pair = pairs[pair_idx];
    if (threadIdx.x == 0) {
        out->left_index = pair.left_index;
        out->right_index = pair.right_index;
        clearCircle(&out->left_circle);
        clearCircle(&out->right_circle);
        clearPoint(&out->left_edge_centroid);
        clearPoint(&out->right_edge_centroid);
        clearPoint(&out->left_radial_center);
        clearPoint(&out->right_radial_center);
        clearPoint(&out->left_edge_pair_center);
        clearPoint(&out->right_edge_pair_center);
        clearDisparity(&out->center_patch);
        clearDisparity(&out->multi_point);
        clearDisparity(&out->corner_points);
        clearDisparity(&out->texture_points);
        clearDisparity(&out->binary_points);
        clearDisparity(&out->orb_points);
        clearDisparity(&out->brisk_points);
        clearDisparity(&out->akaze_points);
        clearDisparity(&out->sift_points);
        clearDisparity(&out->iou_region_color_patch);
        clearDisparity(&out->patch_iou_color_edge);
    }
    __syncthreads();

    if (compute_geometry) {
        fitGeometryInBBox(left_img, left_pitch, img_w, img_h, pair.left,
                          circle_max_roi_pixels,
                          edge_x, edge_y, edge_w, &edge_n, &max_grad,
                          &out->left_circle,
                          &out->left_edge_centroid,
                          &out->left_radial_center,
                          &out->left_edge_pair_center);
        fitGeometryInBBox(right_img, right_pitch, img_w, img_h, pair.right,
                          circle_max_roi_pixels,
                          edge_x, edge_y, edge_w, &edge_n, &max_grad,
                          &out->right_circle,
                          &out->right_edge_centroid,
                          &out->right_radial_center,
                          &out->right_edge_pair_center);
        __syncthreads();
    }

    const float left_cx = out->left_circle.valid ? out->left_circle.cx : pair.left.cx;
    const float left_cy = out->left_circle.valid ? out->left_circle.cy : pair.left.cy;
    const float left_r = out->left_circle.valid
        ? out->left_circle.radius
        : fmaxf(3.0f, 0.25f * (pair.left.width + pair.left.height));
    const float right_cx = out->right_circle.valid ? out->right_circle.cx : pair.right.cx;
    const float initial_disp = left_cx - right_cx;
    const float max_delta = disparityDeltaGate(initial_disp, focal, baseline,
                                               max_disp_delta_px,
                                               max_disp_delta_ratio,
                                               max_depth_delta_m);
    const int fast_points = clampInt(max_points, 4, 6);
    const int fast_min_points = clampInt(min_points, 2, fast_points);
    const int fast_search_radius = clampInt(search_radius_px, 2, 3);
    const int color_points = clampInt(max_points * 2, 16, 24);
    const int color_min_points = clampInt(max(min_points, 4), 3, color_points);
    const int color_search_radius = clampInt((search_radius_px + 2) / 3, 2, 3);
    if (initial_disp <= 0.5f ||
        initial_disp > static_cast<float>(max_disparity) ||
        fabsf(left_cy - (out->right_circle.valid ? out->right_circle.cy : pair.right.cy)) >
            fmaxf(1.0f, epipolar_y_tolerance)) {
        return;
    }

    if (compute_center_patch) {
        matchPatchAtPoint(left_img, left_pitch, right_img, right_pitch,
                          img_w, img_h,
                          left_cx, left_cy,
                          initial_disp,
                          patch_radius,
                          search_radius_px,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          &out->center_patch);
    }
    if (compute_multi_point) {
        matchMultiPointPatch(left_img, left_pitch, right_img, right_pitch,
                             img_w, img_h,
                             pair.left, pair.right,
                             left_cx, left_cy, left_r,
                             initial_disp,
                             patch_radius,
                             search_radius_px,
                             max_points,
                             min_points,
                             max_disparity,
                             min_confidence,
                             max_delta,
                             max_stddev_px,
                             focal,
                             baseline,
                             min_depth,
                             max_depth,
                             feature_y_tolerance_px,
                             feature_y_slope,
                             feature_y_offset_px,
                             feature_reverse_check_px,
                             feature_overlap_scale,
                             feature_mad_scale,
                             feature_ransac_gate_px,
                             feature_sphere_radius_m,
                             feature_sphere_radius_scale,
                             feature_sphere_margin_m,
                             sample_disp, sample_score, sample_x, sample_y,
                             point_x, point_y,
                             best_score_parts, best_disp_parts,
                             best_dy_parts, &valid_count,
                             &out->multi_point);
    }
    if (compute_corner_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          0,
                          min(patch_radius, 4),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->corner_points);
    }
    if (compute_texture_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          1,
                          min(patch_radius, 4),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->texture_points);
    }
    if (compute_binary_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          2,
                          min(patch_radius, 3),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->binary_points);
    }
    if (compute_orb_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          2,
                          min(patch_radius, 3),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->orb_points);
    }
    if (compute_brisk_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          2,
                          min(patch_radius, 3),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->brisk_points);
    }
    if (compute_akaze_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          1,
                          min(patch_radius, 4),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->akaze_points);
    }
    if (compute_sift_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          1,
                          min(patch_radius, 4),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->sift_points);
    }
    if (compute_iou_region_color_patch && left_bgr && right_bgr) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          left_bgr, left_bgr_pitch, right_bgr, right_bgr_pitch,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          3,
                          min(patch_radius, 2),
                          color_search_radius,
                          color_points,
                          color_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->iou_region_color_patch);
    }
    if (compute_patch_iou_color_edge && left_bgr && right_bgr) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          left_bgr, left_bgr_pitch, right_bgr, right_bgr_pitch,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          4,
                          min(patch_radius, 2),
                          color_search_radius,
                          color_points,
                          color_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->patch_iou_color_edge);
    }
}

}  // namespace

extern "C" void launchDualYoloDepthCandidatesGpu(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    const uint8_t* left_bgr, int left_bgr_pitch,
    const uint8_t* right_bgr, int right_bgr_pitch,
    int img_width, int img_height,
    const stereo3d::DualYoloGpuDetectionPair* pairs,
    int num_pairs,
    stereo3d::DualYoloGpuCandidate* results,
    int max_disparity,
    int patch_radius,
    int search_radius_px,
    int max_points,
    int min_points,
    int circle_max_roi_pixels,
    float min_confidence,
    float max_disp_delta_px,
    float max_disp_delta_ratio,
    float max_depth_delta_m,
    float max_stddev_px,
    float epipolar_y_tolerance,
    float feature_y_tolerance_px,
    float feature_y_slope,
    float feature_y_offset_px,
    float feature_reverse_check_px,
    float feature_overlap_scale,
    float feature_mad_scale,
    float feature_ransac_gate_px,
    float feature_sphere_radius_m,
    float feature_sphere_radius_scale,
    float feature_sphere_margin_m,
    int compute_geometry,
    int compute_center_patch,
    int compute_multi_point,
    int compute_corner_points,
    int compute_texture_points,
    int compute_binary_points,
    int compute_orb_points,
    int compute_brisk_points,
    int compute_akaze_points,
    int compute_sift_points,
    int compute_iou_region_color_patch,
    int compute_patch_iou_color_edge,
    float focal,
    float baseline,
    float min_depth,
    float max_depth,
    cudaStream_t stream) {
    if (num_pairs <= 0) return;
    const int blocks = min(num_pairs, 256);
    dualYoloDepthCandidatesKernel<<<blocks, kThreads, 0, stream>>>(
        left_img, left_pitch,
        right_img, right_pitch,
        left_bgr, left_bgr_pitch,
        right_bgr, right_bgr_pitch,
        img_width, img_height,
        pairs,
        num_pairs,
        results,
        max_disparity,
        patch_radius,
        search_radius_px,
        max_points,
        min_points,
        circle_max_roi_pixels,
        min_confidence,
        max_disp_delta_px,
        max_disp_delta_ratio,
        max_depth_delta_m,
        max_stddev_px,
        epipolar_y_tolerance,
        feature_y_tolerance_px,
        feature_y_slope,
        feature_y_offset_px,
        feature_reverse_check_px,
        feature_overlap_scale,
        feature_mad_scale,
        feature_ransac_gate_px,
        feature_sphere_radius_m,
        feature_sphere_radius_scale,
        feature_sphere_margin_m,
        compute_geometry,
        compute_center_patch,
        compute_multi_point,
        compute_corner_points,
        compute_texture_points,
        compute_binary_points,
        compute_orb_points,
        compute_brisk_points,
        compute_akaze_points,
        compute_sift_points,
        compute_iou_region_color_patch,
        compute_patch_iou_color_edge,
        focal,
        baseline,
        min_depth,
        max_depth);
}
