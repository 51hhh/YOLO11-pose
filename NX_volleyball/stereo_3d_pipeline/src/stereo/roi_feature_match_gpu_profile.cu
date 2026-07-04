#include "roi_feature_match_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace stereo3d {
namespace {

constexpr int kProfileThreads = 128;
constexpr float kPi = 3.14159265358979323846f;

__device__ float sampleGrayBilinearDevice(
    const uint8_t* image,
    int pitch,
    int width,
    int height,
    float x,
    float y,
    int* ok) {
    if (!image || pitch <= 0 || x < 1.0f || y < 1.0f ||
        x > static_cast<float>(width - 2) ||
        y > static_cast<float>(height - 2)) {
        *ok = 0;
        return 0.0f;
    }
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const uint8_t* row0 = image + static_cast<size_t>(y0) *
                                      static_cast<size_t>(pitch);
    const uint8_t* row1 = image + static_cast<size_t>(y0 + 1) *
                                      static_cast<size_t>(pitch);
    const float v00 = static_cast<float>(row0[x0]);
    const float v01 = static_cast<float>(row0[x0 + 1]);
    const float v10 = static_cast<float>(row1[x0]);
    const float v11 = static_cast<float>(row1[x0 + 1]);
    *ok = 1;
    return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) +
           fy * ((1.0f - fx) * v10 + fx * v11);
}

__device__ bool insideDetectionEllipseDevice(
    const Detection& det,
    float x,
    float y,
    float scale) {
    const float rx = fmaxf(2.0f, det.width * 0.5f * scale);
    const float ry = fmaxf(2.0f, det.height * 0.5f * scale);
    const float dx = (x - det.cx) / rx;
    const float dy = (y - det.cy) / ry;
    return dx * dx + dy * dy <= 1.0f;
}

__host__ __device__ float profileRingScale(int ring_idx) {
    switch (ring_idx) {
    case 0: return 0.45f;
    case 1: return 0.66f;
    default: return 0.88f;
    }
}

__global__ void ringEdgeProfileCostKernel(
    const uint8_t* left,
    int left_pitch,
    const uint8_t* right,
    int right_pitch,
    int width,
    int height,
    Detection left_det,
    Detection right_det,
    int min_disparity,
    int candidate_count,
    int angle_count,
    int ring_count,
    float* costs,
    int* supports) {
    __shared__ float shared_cost[kProfileThreads];
    __shared__ int shared_support[kProfileThreads];

    const int tid = threadIdx.x;
    const int candidate = blockIdx.x;
    if (candidate >= candidate_count) {
        return;
    }

    const float disparity =
        static_cast<float>(min_disparity + candidate);
    const float rx = fmaxf(2.0f, left_det.width * 0.5f);
    const float ry = fmaxf(2.0f, left_det.height * 0.5f);
    const int sample_count = angle_count * ring_count;
    float local_cost = 0.0f;
    int local_support = 0;

    for (int sample_idx = tid; sample_idx < sample_count;
         sample_idx += blockDim.x) {
        const int angle_idx = sample_idx % angle_count;
        const int ring_idx = sample_idx / angle_count;
        const float theta =
            (static_cast<float>(angle_idx) + 0.5f) *
            (2.0f * kPi / static_cast<float>(angle_count));
        const float nx = cosf(theta);
        const float ny = sinf(theta);
        const float scale = profileRingScale(ring_idx);
        const float lx = left_det.cx + nx * rx * scale;
        const float ly = left_det.cy + ny * ry * scale;
        const float rr_x = lx - disparity;
        const float rr_y = ly;

        if (!insideDetectionEllipseDevice(left_det, lx, ly, 0.96f) ||
            !insideDetectionEllipseDevice(right_det, rr_x, rr_y, 1.05f)) {
            continue;
        }

        int ok_l = 0;
        int ok_r = 0;
        const float left_center = sampleGrayBilinearDevice(
            left, left_pitch, width, height, lx, ly, &ok_l);
        const float right_center = sampleGrayBilinearDevice(
            right, right_pitch, width, height, rr_x, rr_y, &ok_r);
        if (!ok_l || !ok_r) {
            continue;
        }

        const float edge_step = 1.5f;
        const float l_in_x = lx - nx * edge_step;
        const float l_in_y = ly - ny * edge_step;
        const float l_out_x = lx + nx * edge_step;
        const float l_out_y = ly + ny * edge_step;
        const float r_in_x = rr_x - nx * edge_step;
        const float r_in_y = rr_y - ny * edge_step;
        const float r_out_x = rr_x + nx * edge_step;
        const float r_out_y = rr_y + ny * edge_step;

        int ok_li = 0;
        int ok_lo = 0;
        int ok_ri = 0;
        int ok_ro = 0;
        const float left_in = sampleGrayBilinearDevice(
            left, left_pitch, width, height, l_in_x, l_in_y, &ok_li);
        const float left_out = sampleGrayBilinearDevice(
            left, left_pitch, width, height, l_out_x, l_out_y, &ok_lo);
        const float right_in = sampleGrayBilinearDevice(
            right, right_pitch, width, height, r_in_x, r_in_y, &ok_ri);
        const float right_out = sampleGrayBilinearDevice(
            right, right_pitch, width, height, r_out_x, r_out_y, &ok_ro);
        if (!ok_li || !ok_lo || !ok_ri || !ok_ro) {
            continue;
        }

        const float intensity_cost =
            fabsf(left_center - right_center) * (1.0f / 255.0f);
        const float left_edge = fabsf(left_out - left_in);
        const float right_edge = fabsf(right_out - right_in);
        const float edge_cost =
            fabsf(left_edge - right_edge) * (1.0f / 255.0f);
        const float ring_weight = ring_idx == ring_count - 1 ? 1.25f : 1.0f;
        local_cost += ring_weight *
                      (0.48f * intensity_cost + 0.52f * edge_cost);
        ++local_support;
    }

    shared_cost[tid] = local_cost;
    shared_support[tid] = local_support;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_cost[tid] += shared_cost[tid + stride];
            shared_support[tid] += shared_support[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int support = shared_support[0];
        supports[candidate] = support;
        costs[candidate] = support > 0
            ? shared_cost[0] / static_cast<float>(support)
            : 1.0e6f;
    }
}

struct RingEdgeProfileScratch {
    float* device_costs = nullptr;
    int* device_supports = nullptr;
    int capacity = 0;
    std::vector<float> host_costs;
    std::vector<int> host_supports;

    ~RingEdgeProfileScratch() {
        if (device_costs) {
            cudaFree(device_costs);
            device_costs = nullptr;
        }
        if (device_supports) {
            cudaFree(device_supports);
            device_supports = nullptr;
        }
    }

    bool ensure(int count) {
        if (count <= capacity) {
            host_costs.resize(static_cast<size_t>(count));
            host_supports.resize(static_cast<size_t>(count));
            return true;
        }
        if (device_costs) {
            cudaFree(device_costs);
            device_costs = nullptr;
        }
        if (device_supports) {
            cudaFree(device_supports);
            device_supports = nullptr;
        }
        capacity = 0;
        if (cudaMalloc(reinterpret_cast<void**>(&device_costs),
                       sizeof(float) * static_cast<size_t>(count)) !=
            cudaSuccess) {
            device_costs = nullptr;
            return false;
        }
        if (cudaMalloc(reinterpret_cast<void**>(&device_supports),
                       sizeof(int) * static_cast<size_t>(count)) !=
            cudaSuccess) {
            cudaFree(device_costs);
            device_costs = nullptr;
            device_supports = nullptr;
            return false;
        }
        capacity = count;
        host_costs.resize(static_cast<size_t>(count));
        host_supports.resize(static_cast<size_t>(count));
        return true;
    }
};

float profileDeltaGate(float initial_disp,
                       float focal,
                       float baseline,
                       const ROIFeatureMatchConfig& cfg) {
    float gate = std::max(1.0f, cfg.subpixel_max_disp_delta_px);
    if (std::isfinite(initial_disp) && initial_disp > 0.5f) {
        gate = std::max(gate,
                        std::abs(initial_disp) *
                            std::max(0.0f, cfg.subpixel_max_disp_delta_ratio));
    }
    if (std::isfinite(focal) && focal > 0.0f &&
        std::isfinite(baseline) && baseline > 0.0f &&
        std::isfinite(initial_disp) && initial_disp > 0.5f &&
        std::isfinite(cfg.subpixel_max_depth_delta_m) &&
        cfg.subpixel_max_depth_delta_m > 0.0f) {
        const float z = focal * baseline / initial_disp;
        const float near_z = std::max(0.05f, z - cfg.subpixel_max_depth_delta_m);
        const float far_z = z + cfg.subpixel_max_depth_delta_m;
        const float near_disp = focal * baseline / near_z;
        const float far_disp = focal * baseline / far_z;
        gate = std::max(gate, std::abs(near_disp - initial_disp));
        gate = std::max(gate, std::abs(far_disp - initial_disp));
    }
    return std::clamp(gate, 1.0f, 64.0f);
}

void setRingEdgeDebugMatches(SparseFeatureDisparityResult& result,
                             const Detection& left_det,
                             float disparity,
                             int angle_count,
                             int ring_count,
                             float score) {
    if (!std::isfinite(disparity) || disparity <= 0.5f ||
        angle_count <= 0 || ring_count <= 0) {
        return;
    }
    const int sample_count = angle_count * ring_count;
    const int debug_count = std::min(sample_count,
                                     kMaxSparseFeatureDebugMatches);
    const float rx = std::max(2.0f, left_det.width * 0.5f);
    const float ry = std::max(2.0f, left_det.height * 0.5f);
    result.debug_match_count = debug_count;
    for (int i = 0; i < debug_count; ++i) {
        const int sample_idx =
            static_cast<int>((static_cast<int64_t>(i) * sample_count) /
                             std::max(1, debug_count));
        const int angle_idx = sample_idx % angle_count;
        const int ring_idx = sample_idx / angle_count;
        const float theta =
            (static_cast<float>(angle_idx) + 0.5f) *
            (2.0f * kPi / static_cast<float>(angle_count));
        const float scale = profileRingScale(ring_idx);
        const float lx = left_det.cx + std::cos(theta) * rx * scale;
        const float ly = left_det.cy + std::sin(theta) * ry * scale;
        auto& dst = result.debug_matches[static_cast<size_t>(i)];
        dst.left_x = lx;
        dst.left_y = ly;
        dst.right_x = lx - disparity;
        dst.right_y = ly;
        dst.disparity = disparity;
        dst.score = score;
    }
}

}  // namespace

SparseFeatureDisparityResult matchCudaRingEdgeProfileDisparityGPU(
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
    cudaStream_t stream) {
    SparseFeatureDisparityResult result;
    if (!left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0 || !stream ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f ||
        max_disparity <= 1 ||
        left_det.width < 8.0f || left_det.height < 8.0f ||
        right_det.width < 8.0f || right_det.height < 8.0f) {
        return result;
    }

    const float max_delta = profileDeltaGate(initial_disp, focal, baseline, cfg);
    const int search_radius = std::clamp(
        std::max(cfg.subpixel_search_radius_px,
                 static_cast<int>(std::ceil(max_delta))),
        2,
        64);
    const int center_disp =
        static_cast<int>(std::round(initial_disp));
    const int min_disp =
        std::max(1, center_disp - search_radius);
    const int max_disp =
        std::min(max_disparity, center_disp + search_radius);
    const int candidate_count = max_disp - min_disp + 1;
    if (candidate_count <= 0 || candidate_count > 129) {
        result.low_confidence = true;
        return result;
    }

    const int angle_count = std::clamp(
        std::max(24, cfg.subpixel_max_points * 3), 24, 72);
    const int ring_count = 3;
    const int sample_count = angle_count * ring_count;
    const int min_support = std::clamp(
        std::max(cfg.subpixel_min_points * 4, sample_count / 4),
        8,
        sample_count);

    thread_local RingEdgeProfileScratch scratch;
    if (!scratch.ensure(candidate_count)) {
        result.low_confidence = true;
        return result;
    }

    ringEdgeProfileCostKernel<<<candidate_count, kProfileThreads, 0, stream>>>(
        left_gpu,
        left_pitch,
        right_gpu,
        right_pitch,
        img_w,
        img_h,
        left_det,
        right_det,
        min_disp,
        candidate_count,
        angle_count,
        ring_count,
        scratch.device_costs,
        scratch.device_supports);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(scratch.host_costs.data(),
                              scratch.device_costs,
                              sizeof(float) *
                                  static_cast<size_t>(candidate_count),
                              cudaMemcpyDeviceToHost,
                              stream);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(scratch.host_supports.data(),
                              scratch.device_supports,
                              sizeof(int) *
                                  static_cast<size_t>(candidate_count),
                              cudaMemcpyDeviceToHost,
                              stream);
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }
    if (err != cudaSuccess) {
        return result;
    }

    int best_idx = -1;
    int second_idx = -1;
    float best_cost = std::numeric_limits<float>::infinity();
    float second_cost = std::numeric_limits<float>::infinity();
    for (int i = 0; i < candidate_count; ++i) {
        const int support = scratch.host_supports[static_cast<size_t>(i)];
        const float cost = scratch.host_costs[static_cast<size_t>(i)];
        if (support < min_support || !std::isfinite(cost)) {
            continue;
        }
        if (cost < best_cost) {
            second_cost = best_cost;
            second_idx = best_idx;
            best_cost = cost;
            best_idx = i;
        } else if (cost < second_cost) {
            second_cost = cost;
            second_idx = i;
        }
    }

    result.attempted = candidate_count * sample_count;
    if (best_idx < 0) {
        result.low_confidence = true;
        return result;
    }

    float disparity = static_cast<float>(min_disp + best_idx);
    if (best_idx > 0 && best_idx + 1 < candidate_count) {
        const float c0 = scratch.host_costs[static_cast<size_t>(best_idx - 1)];
        const float c1 = scratch.host_costs[static_cast<size_t>(best_idx)];
        const float c2 = scratch.host_costs[static_cast<size_t>(best_idx + 1)];
        if (std::isfinite(c0) && std::isfinite(c1) && std::isfinite(c2)) {
            const float denom = c0 - 2.0f * c1 + c2;
            if (std::abs(denom) > 1.0e-6f) {
                const float offset =
                    std::clamp(0.5f * (c0 - c2) / denom, -0.5f, 0.5f);
                disparity += offset;
            }
        }
    }
    result.disparity = disparity;
    result.support = scratch.host_supports[static_cast<size_t>(best_idx)];
    result.anchor_cx = left_det.cx;
    result.anchor_cy = left_det.cy;
    result.right_anchor_cx = left_det.cx - disparity;
    result.right_anchor_cy = left_det.cy;
    if (cfg.debug_patch_enabled) {
        const float debug_score = std::clamp(1.0f - best_cost * 2.5f,
                                             0.0f, 1.0f);
        setRingEdgeDebugMatches(result, left_det, disparity,
                                angle_count, ring_count, debug_score);
    }
    if (std::abs(disparity - initial_disp) > max_delta ||
        disparity <= 0.5f ||
        disparity > static_cast<float>(max_disparity)) {
        result.low_confidence = true;
        return result;
    }

    const float second_gap =
        (second_idx >= 0 && std::isfinite(second_cost))
            ? std::clamp((second_cost - best_cost) /
                             std::max(0.02f, second_cost),
                         0.0f,
                         1.0f)
            : 0.25f;
    const float support_ratio =
        static_cast<float>(scratch.host_supports[static_cast<size_t>(best_idx)]) /
        static_cast<float>(std::max(1, sample_count));
    const float cost_conf = std::clamp(1.0f - best_cost * 2.5f, 0.0f, 1.0f);
    const float delta_conf = 1.0f -
        std::min(1.0f, std::abs(disparity - initial_disp) /
                           std::max(1.0f, max_delta));
    result.confidence = std::clamp(
        0.45f * cost_conf +
        0.25f * support_ratio +
        0.20f * second_gap +
        0.10f * delta_conf,
        0.0f,
        1.0f);
    if (result.confidence < cfg.subpixel_min_confidence) {
        result.low_confidence = true;
        return result;
    }

    float mean = 0.0f;
    float weight_sum = 0.0f;
    for (int i = 0; i < candidate_count; ++i) {
        const int support = scratch.host_supports[static_cast<size_t>(i)];
        const float cost = scratch.host_costs[static_cast<size_t>(i)];
        if (support < min_support || !std::isfinite(cost)) {
            continue;
        }
        const float rel = std::max(0.0f, cost - best_cost);
        if (rel > 0.12f) {
            continue;
        }
        const float weight =
            static_cast<float>(support) / std::max(0.02f, 0.02f + rel);
        mean += weight * static_cast<float>(min_disp + i);
        weight_sum += weight;
    }
    mean = weight_sum > 0.0f ? mean / weight_sum : disparity;
    float variance = 0.0f;
    if (weight_sum > 0.0f) {
        for (int i = 0; i < candidate_count; ++i) {
            const int support = scratch.host_supports[static_cast<size_t>(i)];
            const float cost = scratch.host_costs[static_cast<size_t>(i)];
            if (support < min_support || !std::isfinite(cost)) {
                continue;
            }
            const float rel = std::max(0.0f, cost - best_cost);
            if (rel > 0.12f) {
                continue;
            }
            const float d = static_cast<float>(min_disp + i) - mean;
            const float weight =
                static_cast<float>(support) / std::max(0.02f, 0.02f + rel);
            variance += weight * d * d;
        }
        variance /= weight_sum;
    }

    const float stddev = std::sqrt(std::max(0.0f, variance));
    if (stddev > std::max(0.05f, cfg.subpixel_max_stddev_px)) {
        result.low_confidence = true;
        return result;
    }

    result.valid = true;
    result.stddev = stddev;
    return result;
}

}  // namespace stereo3d
