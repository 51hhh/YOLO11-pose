#include "neural_feature_direct_gpu_postprocess.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace stereo3d {
namespace {

constexpr float kDirectContext = 1.20f;
constexpr float kNegInf = -1.0e30f;

bool cudaOk(cudaError_t err) {
    return err == cudaSuccess;
}

__device__ float keypointCoord(const float* keypoints,
                               int count,
                               DirectFeatureKeypointLayout layout,
                               int index,
                               int coord) {
    if (!keypoints || index < 0 || index >= count) return 0.0f;
    if (layout == DIRECT_KPTS_K2) {
        return keypoints[static_cast<size_t>(index) * 2u +
                         static_cast<size_t>(coord)];
    }
    return keypoints[static_cast<size_t>(coord) *
                         static_cast<size_t>(count) +
                     static_cast<size_t>(index)];
}

__device__ float descriptorValue(const float* descriptors,
                                 int count,
                                 int desc_dim,
                                 DirectFeatureDescriptorLayout layout,
                                 int index,
                                 int channel) {
    if (!descriptors || index < 0 || index >= count ||
        channel < 0 || channel >= desc_dim) {
        return 0.0f;
    }
    if (layout == DIRECT_DESC_KD) {
        return descriptors[static_cast<size_t>(index) *
                               static_cast<size_t>(desc_dim) +
                           static_cast<size_t>(channel)];
    }
    return descriptors[static_cast<size_t>(channel) *
                           static_cast<size_t>(count) +
                       static_cast<size_t>(index)];
}

__device__ bool validScore(const float* scores, int score_count, int index) {
    if (index < 0 || index >= score_count) return false;
    if (!scores) return true;
    const float score = scores[index];
    return isfinite(score) && score > 0.0f;
}

__device__ float dotDescriptor(const float* query_desc,
                               const float* train_desc,
                               int query_count,
                               int train_count,
                               int desc_dim,
                               DirectFeatureDescriptorLayout layout,
                               int qi,
                               int ti) {
    float dot = 0.0f;
    float qn = 0.0f;
    float tn = 0.0f;
    for (int c = 0; c < desc_dim; ++c) {
        const float q =
            descriptorValue(query_desc, query_count, desc_dim, layout, qi, c);
        const float t =
            descriptorValue(train_desc, train_count, desc_dim, layout, ti, c);
        dot += q * t;
        qn += q * q;
        tn += t * t;
    }
    if (qn <= 1e-12f || tn <= 1e-12f) return kNegInf;
    return dot * rsqrtf(qn * tn);
}

__global__ void bestDirectMatchKernel(
    const float* query_desc,
    const float* train_desc,
    const float* query_scores,
    const float* train_scores,
    int query_count,
    int train_count,
    int query_score_count,
    int train_score_count,
    int desc_dim,
    DirectFeatureDescriptorLayout desc_layout,
    int* best_index,
    float* best_score,
    float* second_score) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= query_count) return;

    int best = -1;
    float best_s = kNegInf;
    float second_s = kNegInf;
    if (validScore(query_scores, query_score_count, i)) {
        for (int j = 0; j < train_count; ++j) {
            if (!validScore(train_scores, train_score_count, j)) continue;
            const float s = dotDescriptor(query_desc,
                                          train_desc,
                                          query_count,
                                          train_count,
                                          desc_dim,
                                          desc_layout,
                                          i,
                                          j);
            if (s > best_s) {
                second_s = best_s;
                best_s = s;
                best = j;
            } else if (s > second_s) {
                second_s = s;
            }
        }
    }

    best_index[i] = best;
    best_score[i] = best_s;
    if (second_score) second_score[i] = second_s;
}

__global__ void collectDirectMatchesKernel(
    const float* left_keypoints,
    const float* right_keypoints,
    const float* left_scores,
    const float* right_scores,
    const int* left_best,
    const int* right_best,
    const float* left_best_score,
    const float* left_second_score,
    int left_count,
    int right_count,
    int left_score_count,
    int right_score_count,
    DirectFeatureKeypointLayout keypoint_layout,
    int roi_size,
    float min_score,
    float match_margin,
    float max_y_error_px,
    float max_disp_delta_px,
    float initial_disparity,
    int max_disparity,
    float left_cx,
    float left_cy,
    float left_w,
    float left_h,
    float right_cx,
    float right_cy,
    float right_w,
    float right_h,
    DirectFeatureGpuMatch* matches,
    int* match_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= left_count || !validScore(left_scores, left_score_count, i)) {
        return;
    }
    const int j = left_best[i];
    if (j < 0 || j >= right_count ||
        !validScore(right_scores, right_score_count, j) ||
        right_best[j] != i) {
        return;
    }
    const float score = left_best_score[i];
    if (score < min_score) return;
    if (match_margin > 0.0f &&
        isfinite(left_second_score[i]) &&
        score - left_second_score[i] < match_margin) {
        return;
    }

    const float left_s =
        sqrtf(fmaxf(1.0f, left_w * kDirectContext * left_h * kDirectContext));
    const float right_s =
        sqrtf(fmaxf(1.0f, right_w * kDirectContext * right_h * kDirectContext));
    const float left_x0 = left_cx - 0.5f * left_s;
    const float left_y0 = left_cy - 0.5f * left_s;
    const float right_x0 = right_cx - 0.5f * right_s;
    const float right_y0 = right_cy - 0.5f * right_s;

    const float lkx = keypointCoord(left_keypoints, left_count,
                                    keypoint_layout, i, 0);
    const float lky = keypointCoord(left_keypoints, left_count,
                                    keypoint_layout, i, 1);
    const float rkx = keypointCoord(right_keypoints, right_count,
                                    keypoint_layout, j, 0);
    const float rky = keypointCoord(right_keypoints, right_count,
                                    keypoint_layout, j, 1);
    if (!isfinite(lkx) || !isfinite(lky) ||
        !isfinite(rkx) || !isfinite(rky) ||
        lkx < 0.0f || lky < 0.0f ||
        rkx < 0.0f || rky < 0.0f ||
        lkx >= static_cast<float>(roi_size) ||
        lky >= static_cast<float>(roi_size) ||
        rkx >= static_cast<float>(roi_size) ||
        rky >= static_cast<float>(roi_size)) {
        return;
    }

    const float lx = left_x0 +
                     (lkx + 0.5f) * left_s /
                         static_cast<float>(roi_size) -
                     0.5f;
    const float ly = left_y0 +
                     (lky + 0.5f) * left_s /
                         static_cast<float>(roi_size) -
                     0.5f;
    const float rx = right_x0 +
                     (rkx + 0.5f) * right_s /
                         static_cast<float>(roi_size) -
                     0.5f;
    const float ry = right_y0 +
                     (rky + 0.5f) * right_s /
                         static_cast<float>(roi_size) -
                     0.5f;
    const float disp = lx - rx;
    if (disp <= 0.5f ||
        disp > static_cast<float>(max_disparity) ||
        fabsf(ly - ry) > max_y_error_px ||
        fabsf(disp - initial_disparity) > max_disp_delta_px) {
        return;
    }

    const int out_idx = atomicAdd(match_count, 1);
    if (out_idx >= left_count) return;
    DirectFeatureGpuMatch m;
    m.left_x = lx;
    m.left_y = ly;
    m.right_x = rx;
    m.right_y = ry;
    m.disparity = disp;
    m.score = score;
    matches[out_idx] = m;
}

}  // namespace

void releaseDirectFeatureGpuWorkspace(DirectFeatureGpuWorkspace& w) {
    cudaFree(w.left_best);
    cudaFree(w.right_best);
    cudaFree(w.left_best_score);
    cudaFree(w.right_best_score);
    cudaFree(w.left_second_score);
    cudaFree(w.matches);
    cudaFree(w.match_count);
    cudaFreeHost(w.host_matches);
    cudaFreeHost(w.host_match_count);
    w = DirectFeatureGpuWorkspace{};
}

bool ensureDirectFeatureGpuWorkspace(DirectFeatureGpuWorkspace& w,
                                     int top_k,
                                     int desc_dim) {
    if (top_k <= 0 || desc_dim <= 0) return false;
    if (w.left_best && w.top_k == top_k && w.desc_dim == desc_dim) {
        return true;
    }
    releaseDirectFeatureGpuWorkspace(w);
    w.top_k = top_k;
    w.desc_dim = desc_dim;

    const size_t int_bytes = static_cast<size_t>(top_k) * sizeof(int);
    const size_t score_bytes = static_cast<size_t>(top_k) * sizeof(float);
    const size_t match_bytes =
        static_cast<size_t>(top_k) * sizeof(DirectFeatureGpuMatch);

    bool ok = true;
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_best),
                                 int_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.right_best),
                                 int_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_best_score),
                                 score_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.right_best_score),
                                 score_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_second_score),
                                 score_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.matches),
                                 match_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.match_count),
                                 sizeof(int)));
    ok = ok && cudaOk(cudaHostAlloc(reinterpret_cast<void**>(&w.host_matches),
                                    match_bytes, cudaHostAllocDefault));
    ok = ok && cudaOk(cudaHostAlloc(reinterpret_cast<void**>(&w.host_match_count),
                                    sizeof(int), cudaHostAllocDefault));
    if (!ok) {
        releaseDirectFeatureGpuWorkspace(w);
        return false;
    }
    return true;
}

bool runDirectFeatureGpuPostprocess(
    DirectFeatureGpuWorkspace& w,
    const float* left_keypoints,
    const float* right_keypoints,
    const float* left_descriptors,
    const float* right_descriptors,
    const float* left_scores,
    const float* right_scores,
    int keypoint_count,
    int descriptor_count,
    int score_count,
    int desc_dim,
    DirectFeatureKeypointLayout keypoint_layout,
    DirectFeatureDescriptorLayout descriptor_layout,
    int roi_size,
    int min_matches,
    float min_score,
    float match_margin,
    float max_y_error_px,
    float max_disp_delta_px,
    float initial_disparity,
    int max_disparity,
    float left_cx,
    float left_cy,
    float left_w,
    float left_h,
    float right_cx,
    float right_cy,
    float right_w,
    float right_h,
    cudaStream_t stream,
    std::vector<DirectFeatureGpuMatch>* matches) {
    if (!matches || !left_keypoints || !right_keypoints ||
        !left_descriptors || !right_descriptors ||
        keypoint_count <= 0 || descriptor_count <= 0 ||
        score_count <= 0 || desc_dim <= 0 || roi_size <= 0 ||
        min_matches <= 0) {
        return false;
    }
    const int count = std::min({keypoint_count, descriptor_count, score_count,
                                w.top_k});
    if (count <= 0) return false;
    matches->clear();

    const int threads = 128;
    const int blocks = (count + threads - 1) / threads;
    bestDirectMatchKernel<<<blocks, threads, 0, stream>>>(
        left_descriptors,
        right_descriptors,
        left_scores,
        right_scores,
        count,
        count,
        count,
        count,
        desc_dim,
        descriptor_layout,
        w.left_best,
        w.left_best_score,
        w.left_second_score);
    bestDirectMatchKernel<<<blocks, threads, 0, stream>>>(
        right_descriptors,
        left_descriptors,
        right_scores,
        left_scores,
        count,
        count,
        count,
        count,
        desc_dim,
        descriptor_layout,
        w.right_best,
        w.right_best_score,
        nullptr);
    if (!cudaOk(cudaGetLastError())) return false;
    if (!cudaOk(cudaMemsetAsync(w.match_count, 0, sizeof(int), stream))) {
        return false;
    }
    collectDirectMatchesKernel<<<blocks, threads, 0, stream>>>(
        left_keypoints,
        right_keypoints,
        left_scores,
        right_scores,
        w.left_best,
        w.right_best,
        w.left_best_score,
        w.left_second_score,
        count,
        count,
        count,
        count,
        keypoint_layout,
        roi_size,
        min_score,
        match_margin,
        max_y_error_px,
        max_disp_delta_px,
        initial_disparity,
        max_disparity,
        left_cx,
        left_cy,
        left_w,
        left_h,
        right_cx,
        right_cy,
        right_w,
        right_h,
        w.matches,
        w.match_count);
    if (!cudaOk(cudaGetLastError())) return false;
    *w.host_match_count = 0;
    if (!cudaOk(cudaMemcpyAsync(w.host_match_count,
                                w.match_count,
                                sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream)) ||
        !cudaOk(cudaMemcpyAsync(w.host_matches,
                                w.matches,
                                static_cast<size_t>(count) *
                                    sizeof(DirectFeatureGpuMatch),
                                cudaMemcpyDeviceToHost,
                                stream))) {
        matches->clear();
        return false;
    }
    if (!cudaOk(cudaStreamSynchronize(stream))) {
        matches->clear();
        return false;
    }
    const int host_count = std::clamp(*w.host_match_count, 0, count);
    matches->assign(w.host_matches, w.host_matches + host_count);
    return true;
}

}  // namespace stereo3d
