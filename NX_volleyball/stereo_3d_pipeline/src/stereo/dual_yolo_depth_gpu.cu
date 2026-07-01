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
    int img_w, int img_h,
    const stereo3d::DualYoloGpuDetection& left_det,
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
    float* sample_disp,
    float* sample_score,
    float* sample_x,
    float* sample_y,
    float* point_x,
    float* point_y,
    float best_score_parts[kMaxFeaturePoints][kThreadsPerPoint],
    float best_disp_parts[kMaxFeaturePoints][kThreadsPerPoint],
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
        }
    }
    __syncthreads();

    max_points = clampInt(max_points, 1, kMaxFeaturePoints);
    min_points = clampInt(min_points, 1, max_points);
    patch_radius = clampInt(patch_radius, 2, 8);
    const int point_idx = tid / kThreadsPerPoint;
    const int sub_idx = tid % kThreadsPerPoint;
    if (point_idx < max_points) {
        const int grid = max(2, static_cast<int>(ceilf(sqrtf(static_cast<float>(max_points)))));
        const int gx = point_idx % grid;
        const int gy = point_idx / grid;
        const float rx = fmaxf(4.0f, fminf(radius * 0.82f, left_det.width * 0.42f));
        const float ry = fmaxf(4.0f, fminf(radius * 0.82f, left_det.height * 0.42f));
        const float u = grid > 1 ? (static_cast<float>(gx) / static_cast<float>(grid - 1)) : 0.5f;
        const float v = grid > 1 ? (static_cast<float>(gy) / static_cast<float>(grid - 1)) : 0.5f;
        const float x_f = center_x + (u - 0.5f) * 2.0f * rx;
        const float y_f = center_y + (v - 0.5f) * 2.0f * ry;
        const float nx = (x_f - center_x) / fmaxf(rx, 1.0f);
        const float ny = (y_f - center_y) / fmaxf(ry, 1.0f);
        const int x = static_cast<int>(rintf(x_f));
        const int y = static_cast<int>(rintf(y_f));
        if (nx * nx + ny * ny <= 0.92f * 0.92f &&
            patchInside(img_w, img_h, x, y, patch_radius)) {
            const float response = sparseResponse(left_img, left_pitch, x, y, mode);
            const float response_floor = mode == 1 ? 20.0f : 8.0f;
            if (response > response_floor) {
                const int d0 = static_cast<int>(rintf(initial_disp));
                const int d_start = max(1, d0 - search_radius);
                const int d_end = min(max_disparity, d0 + search_radius);
                if (d_start < d_end) {
                    const int range = d_end - d_start + 1;
                    const int per = (range + kThreadsPerPoint - 1) / kThreadsPerPoint;
                    const int begin = d_start + sub_idx * per;
                    const int end = min(d_end, begin + per - 1);
                    float best_score = -2.0f;
                    float best_disp = -1.0f;
                    for (int d = begin; d <= end; ++d) {
                        const int xr = x - d;
                        if (!patchInside(img_w, img_h, xr, y, patch_radius)) continue;
                        const float score = mode == 2
                            ? binaryPatchScore(left_img, left_pitch, right_img, right_pitch,
                                               x, y, xr, y, patch_radius)
                            : znccScore(left_img, left_pitch, right_img, right_pitch,
                                        x, y, xr, y, patch_radius);
                        if (score > best_score) {
                            best_score = score;
                            best_disp = static_cast<float>(d);
                        }
                    }
                    best_score_parts[point_idx][sub_idx] = best_score;
                    best_disp_parts[point_idx][sub_idx] = best_disp;
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
        for (int i = 1; i < kThreadsPerPoint; ++i) {
            if (best_score_parts[point_idx][i] > best_score) {
                best_score = best_score_parts[point_idx][i];
                best_disp = best_disp_parts[point_idx][i];
            }
        }
        const float min_score = mode == 2
            ? fmaxf(0.58f, 0.50f + min_confidence * 0.35f)
            : fmaxf(0.12f, min_confidence * 0.60f);
        if (best_disp > 0.5f && best_score >= min_score &&
            fabsf(best_disp - initial_disp) <= max_delta) {
            const float z = focal * baseline / best_disp;
            if (z >= min_depth && z <= max_depth) {
                const int idx = atomicAdd(valid_count, 1);
                if (idx < kMaxFeaturePoints) {
                    sample_disp[idx] = best_disp;
                    sample_score[idx] = best_score;
                    sample_x[idx] = point_x[point_idx];
                    sample_y[idx] = point_y[point_idx];
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        const int n = min(*valid_count, kMaxFeaturePoints);
        out->attempted = max_points;
        if (n < min_points) {
            out->low_confidence = 1;
        } else {
            for (int i = 0; i < n - 1; ++i) {
                for (int j = 0; j < n - i - 1; ++j) {
                    if (sample_disp[j] > sample_disp[j + 1]) {
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
            int q1 = n / 4;
            int q3 = (3 * n) / 4;
            if (q1 == q3) {
                q1 = 0;
                q3 = n - 1;
            }
            float sum_d = 0.0f;
            float sum_s = 0.0f;
            float sum_x = 0.0f;
            float sum_y = 0.0f;
            int count = 0;
            for (int i = q1; i <= q3; ++i) {
                sum_d += sample_disp[i];
                sum_s += sample_score[i];
                sum_x += sample_x[i];
                sum_y += sample_y[i];
                ++count;
            }
            if (count <= 0) {
                out->low_confidence = 1;
            } else {
                const float mean = sum_d / static_cast<float>(count);
                float var = 0.0f;
                for (int i = q1; i <= q3; ++i) {
                    const float e = sample_disp[i] - mean;
                    var += e * e;
                }
                const float stddev = sqrtf(var / static_cast<float>(count));
                const float z = focal * baseline / fmaxf(mean, 0.5f);
                if (stddev > max_stddev ||
                    fabsf(mean - initial_disp) > max_delta ||
                    mean <= 0.5f || mean > static_cast<float>(max_disparity) ||
                    z < min_depth || z > max_depth) {
                    out->low_confidence = 1;
                } else {
                    const float avg_score = sum_s / static_cast<float>(count);
                    out->disparity = mean;
                    out->confidence = mode == 2
                        ? clampFloat((avg_score - 0.50f) / 0.45f, 0.0f, 1.0f)
                        : clampFloat((avg_score - 0.10f) / 0.80f, 0.0f, 1.0f);
                    out->stddev = stddev;
                    out->delta_gate_px = max_delta;
                    out->anchor_cx = sum_x / static_cast<float>(count);
                    out->anchor_cy = sum_y / static_cast<float>(count);
                    out->support = n;
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
    float* sample_disp,
    float* sample_score,
    float* sample_x,
    float* sample_y,
    float* point_x,
    float* point_y,
    float best_score_parts[kMaxFeaturePoints][kThreadsPerPoint],
    float best_disp_parts[kMaxFeaturePoints][kThreadsPerPoint],
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
        }
    }
    __syncthreads();

    max_points = clampInt(max_points, 1, kMaxFeaturePoints);
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
                float best_score = -2.0f;
                float best_disp = -1.0f;
                for (int d = begin; d <= end; ++d) {
                    const int xr = x - d;
                    if (!patchInside(img_w, img_h, xr, y, patch_radius)) continue;
                    const float score = znccScore(left_img, left_pitch, right_img, right_pitch,
                                                  x, y, xr, y, patch_radius);
                    if (score > best_score) {
                        best_score = score;
                        best_disp = static_cast<float>(d);
                    }
                }
                best_score_parts[point_idx][sub_idx] = best_score;
                best_disp_parts[point_idx][sub_idx] = best_disp;
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
        for (int i = 1; i < kThreadsPerPoint; ++i) {
            if (best_score_parts[point_idx][i] > best_score) {
                best_score = best_score_parts[point_idx][i];
                best_disp = best_disp_parts[point_idx][i];
            }
        }
        const float min_score = fmaxf(0.10f, min_confidence * 0.60f);
        if (best_disp > 0.5f && best_score >= min_score &&
            fabsf(best_disp - initial_disp) <= max_delta) {
            const float z = focal * baseline / best_disp;
            if (z >= min_depth && z <= max_depth) {
                const int idx = atomicAdd(valid_count, 1);
                if (idx < kMaxFeaturePoints) {
                    sample_disp[idx] = best_disp;
                    sample_score[idx] = best_score;
                    sample_x[idx] = point_x[point_idx];
                    sample_y[idx] = point_y[point_idx];
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        const int n = min(*valid_count, kMaxFeaturePoints);
        out->attempted = max_points;
        if (n < min_points) {
            out->low_confidence = 1;
        } else {
            for (int i = 0; i < n - 1; ++i) {
                for (int j = 0; j < n - i - 1; ++j) {
                    if (sample_disp[j] > sample_disp[j + 1]) {
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
            int q1 = n / 4;
            int q3 = (3 * n) / 4;
            if (q1 == q3) {
                q1 = 0;
                q3 = n - 1;
            }
            float sum_d = 0.0f;
            float sum_s = 0.0f;
            float sum_x = 0.0f;
            float sum_y = 0.0f;
            int count = 0;
            for (int i = q1; i <= q3; ++i) {
                sum_d += sample_disp[i];
                sum_s += sample_score[i];
                sum_x += sample_x[i];
                sum_y += sample_y[i];
                ++count;
            }
            const float mean = sum_d / static_cast<float>(max(1, count));
            float var = 0.0f;
            for (int i = q1; i <= q3; ++i) {
                const float e = sample_disp[i] - mean;
                var += e * e;
            }
            const float stddev = sqrtf(var / static_cast<float>(max(1, count)));
            const float z = focal * baseline / fmaxf(mean, 0.5f);
            if (stddev > max_stddev ||
                fabsf(mean - initial_disp) > max_delta ||
                mean <= 0.5f || mean > static_cast<float>(max_disparity) ||
                z < min_depth || z > max_depth) {
                out->low_confidence = 1;
            } else {
                const float avg_score = sum_s / static_cast<float>(max(1, count));
                out->disparity = mean;
                out->confidence = clampFloat((avg_score - 0.10f) / 0.80f, 0.0f, 1.0f);
                out->stddev = stddev;
                out->delta_gate_px = max_delta;
                out->anchor_cx = sum_x / static_cast<float>(max(1, count));
                out->anchor_cy = sum_y / static_cast<float>(max(1, count));
                out->support = n;
                out->valid = out->confidence >= min_confidence ? 1 : 0;
                out->low_confidence = out->valid ? 0 : 1;
            }
        }
    }
    __syncthreads();
}

__global__ void dualYoloDepthCandidatesKernel(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
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
    int compute_geometry,
    int compute_center_patch,
    int compute_multi_point,
    int compute_corner_points,
    int compute_texture_points,
    int compute_binary_points,
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
                             sample_disp, sample_score, sample_x, sample_y,
                             point_x, point_y,
                             best_score_parts, best_disp_parts, &valid_count,
                             &out->multi_point);
    }
    if (compute_corner_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          img_w, img_h,
                          pair.left,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          0,
                          patch_radius,
                          search_radius_px,
                          max(max_points, 16),
                          min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts, &valid_count,
                          &out->corner_points);
    }
    if (compute_texture_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          img_w, img_h,
                          pair.left,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          1,
                          patch_radius,
                          search_radius_px,
                          max(max_points, 16),
                          min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts, &valid_count,
                          &out->texture_points);
    }
    if (compute_binary_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          img_w, img_h,
                          pair.left,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          2,
                          patch_radius,
                          search_radius_px,
                          max(max_points, 16),
                          min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts, &valid_count,
                          &out->binary_points);
    }
}

}  // namespace

extern "C" void launchDualYoloDepthCandidatesGpu(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
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
    int compute_geometry,
    int compute_center_patch,
    int compute_multi_point,
    int compute_corner_points,
    int compute_texture_points,
    int compute_binary_points,
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
        compute_geometry,
        compute_center_patch,
        compute_multi_point,
        compute_corner_points,
        compute_texture_points,
        compute_binary_points,
        focal,
        baseline,
        min_depth,
        max_depth);
}
