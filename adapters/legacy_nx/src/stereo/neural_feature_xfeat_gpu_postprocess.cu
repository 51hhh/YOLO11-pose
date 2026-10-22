#include "neural_feature_xfeat_gpu_postprocess.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace stereo3d {
namespace {

constexpr float kXFeatContext = 1.20f;
constexpr float kNegInf = -1.0e30f;

__device__ float sampleChw(const float* data,
                           int channels,
                           int height,
                           int width,
                           int channel,
                           float x,
                           float y) {
    if (!data || channels <= 0 || height <= 0 || width <= 0 ||
        channel < 0 || channel >= channels) {
        return 0.0f;
    }
    x = fminf(fmaxf(x, 0.0f), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0f), static_cast<float>(height - 1));
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const auto at = [&](int xx, int yy) {
        const int idx = (channel * height + yy) * width + xx;
        return data[idx];
    };
    const float v00 = at(x0, y0);
    const float v10 = at(x1, y0);
    const float v01 = at(x0, y1);
    const float v11 = at(x1, y1);
    return v00 * (1.0f - fx) * (1.0f - fy) +
           v10 * fx * (1.0f - fy) +
           v01 * (1.0f - fx) * fy +
           v11 * fx * fy;
}

__global__ void decodeCandidatesKernel(const float* keypoints,
                                       const float* heatmap,
                                       int feat_h,
                                       int feat_w,
                                       int roi_size,
                                       XFeatGpuCandidate* candidates,
                                       float* scores,
                                       int* indices) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cells = feat_h * feat_w;
    if (cell >= cells) return;

    const int yy = cell / feat_w;
    const int xx = cell - yy * feat_w;
    float max_logit = kNegInf;
    for (int c = 0; c < 65; ++c) {
        const int idx = (c * feat_h + yy) * feat_w + xx;
        max_logit = fmaxf(max_logit, keypoints[idx]);
    }

    float denom = 0.0f;
    float best_prob = 0.0f;
    int best_bin = -1;
    for (int c = 0; c < 65; ++c) {
        const int idx = (c * feat_h + yy) * feat_w + xx;
        const float e = expf(keypoints[idx] - max_logit);
        if (c < 64 && e > best_prob) {
            best_prob = e;
            best_bin = c;
        }
        denom += e;
    }

    XFeatGpuCandidate cand;
    float score = -1.0f;
    if (denom > 0.0f && best_bin >= 0) {
        const float prob = best_prob / denom;
        const int ox = best_bin & 7;
        const int oy = best_bin >> 3;
        const int x = xx * 8 + ox;
        const int y = yy * 8 + oy;
        if (x < roi_size && y < roi_size && prob > 0.05f) {
            cand.x = static_cast<float>(x);
            cand.y = static_cast<float>(y);
            cand.feat_x = static_cast<float>(feat_w) *
                          static_cast<float>(x) /
                          static_cast<float>(max(1, roi_size - 1)) - 0.5f;
            cand.feat_y = static_cast<float>(feat_h) *
                          static_cast<float>(y) /
                          static_cast<float>(max(1, roi_size - 1)) - 0.5f;
            const float reliability =
                sampleChw(heatmap, 1, feat_h, feat_w, 0,
                          cand.feat_x, cand.feat_y);
            score = prob * reliability;
            cand.score = score;
        }
    }

    candidates[cell] = cand;
    scores[cell] = score;
    indices[cell] = cell;
}

__global__ void selectTopKKernel(const float* scores,
                                 int cells,
                                 int top_k,
                                 int* top_indices) {
    extern __shared__ unsigned char smem[];
    float* shared_scores = reinterpret_cast<float*>(smem);
    int* shared_indices =
        reinterpret_cast<int*>(shared_scores + blockDim.x);
    const int tid = threadIdx.x;
    for (int rank = 0; rank < top_k; ++rank) {
        float best_score = -1.0e30f;
        int best_idx = -1;
        for (int cell = tid; cell < cells; cell += blockDim.x) {
            const float score = scores[cell];
            bool used = false;
            for (int prev = 0; prev < rank; ++prev) {
                if (top_indices[prev] == cell) {
                    used = true;
                    break;
                }
            }
            if (!used && score > best_score) {
                best_score = score;
                best_idx = cell;
            }
        }
        shared_scores[tid] = best_score;
        shared_indices[tid] = best_idx;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_score = shared_scores[tid + stride];
                const int other_idx = shared_indices[tid + stride];
                if (other_score > shared_scores[tid]) {
                    shared_scores[tid] = other_score;
                    shared_indices[tid] = other_idx;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            top_indices[rank] =
                shared_scores[0] > 0.0f ? shared_indices[0] : -1;
        }
        __syncthreads();
    }
}

__global__ void buildFeaturesKernel(const float* feats,
                                    const XFeatGpuCandidate* candidates,
                                    const int* sorted_indices,
                                    int feat_h,
                                    int feat_w,
                                    int desc_dim,
                                    int top_k,
                                    XFeatGpuFeature* features,
                                    float* descriptors) {
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= top_k) return;

    XFeatGpuFeature feature;
    const int cell = sorted_indices[rank];
    if (cell < 0) {
        features[rank] = feature;
        float* dst = descriptors + static_cast<size_t>(rank) * desc_dim;
        for (int c = 0; c < desc_dim; ++c) dst[c] = 0.0f;
        return;
    }
    const XFeatGpuCandidate cand = candidates[cell];
    feature.x = cand.x;
    feature.y = cand.y;
    feature.score = cand.score;

    float norm2 = 0.0f;
    float* dst = descriptors + static_cast<size_t>(rank) * desc_dim;
    if (cand.score > 0.0f) {
        for (int c = 0; c < desc_dim; ++c) {
            const float d =
                sampleChw(feats, desc_dim, feat_h, feat_w, c,
                          cand.feat_x, cand.feat_y);
            dst[c] = d;
            norm2 += d * d;
        }
    } else {
        for (int c = 0; c < desc_dim; ++c) dst[c] = 0.0f;
    }

    const float inv_norm = norm2 > 1e-12f ? rsqrtf(norm2) : 0.0f;
    if (inv_norm > 0.0f) {
        for (int c = 0; c < desc_dim; ++c) dst[c] *= inv_norm;
    } else {
        feature.score = -1.0f;
    }
    features[rank] = feature;
}

__global__ void bestMatchKernel(const XFeatGpuFeature* query_features,
                                const XFeatGpuFeature* train_features,
                                const float* query_desc,
                                const float* train_desc,
                                int top_k,
                                int desc_dim,
                                int* best_index,
                                float* best_score,
                                float* second_score) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= top_k) return;

    int best = -1;
    float best_s = kNegInf;
    float second_s = kNegInf;
    if (query_features[i].score > 0.0f) {
        const float* q = query_desc + static_cast<size_t>(i) * desc_dim;
        for (int j = 0; j < top_k; ++j) {
            if (train_features[j].score <= 0.0f) continue;
            const float* t = train_desc + static_cast<size_t>(j) * desc_dim;
            float s = 0.0f;
            for (int c = 0; c < desc_dim; ++c) {
                s += q[c] * t[c];
            }
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
    second_score[i] = second_s;
}

__global__ void collectMatchesKernel(const XFeatGpuFeature* left_features,
                                     const XFeatGpuFeature* right_features,
                                     const int* left_best,
                                     const int* right_best,
                                     const float* left_best_score,
                                     const float* left_second_score,
                                     int top_k,
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
                                     XFeatGpuMatch* matches,
                                     int* match_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= top_k || left_features[i].score <= 0.0f) return;
    const int j = left_best[i];
    if (j < 0 || j >= top_k || right_features[j].score <= 0.0f ||
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

    const float left_s = sqrtf(fmaxf(1.0f, left_w * kXFeatContext *
                                           left_h * kXFeatContext));
    const float right_s = sqrtf(fmaxf(1.0f, right_w * kXFeatContext *
                                            right_h * kXFeatContext));
    const float left_x0 = left_cx - 0.5f * left_s;
    const float left_y0 = left_cy - 0.5f * left_s;
    const float right_x0 = right_cx - 0.5f * right_s;
    const float right_y0 = right_cy - 0.5f * right_s;

    const float lx = left_x0 +
                     (left_features[i].x + 0.5f) * left_s /
                         static_cast<float>(roi_size) -
                     0.5f;
    const float ly = left_y0 +
                     (left_features[i].y + 0.5f) * left_s /
                         static_cast<float>(roi_size) -
                     0.5f;
    const float rx = right_x0 +
                     (right_features[j].x + 0.5f) * right_s /
                         static_cast<float>(roi_size) -
                     0.5f;
    const float ry = right_y0 +
                     (right_features[j].y + 0.5f) * right_s /
                         static_cast<float>(roi_size) -
                     0.5f;
    const float disp = lx - rx;
    if (!isfinite(score) ||
        !isfinite(lx) || !isfinite(ly) ||
        !isfinite(rx) || !isfinite(ry) ||
        !isfinite(disp) ||
        disp <= 0.5f ||
        disp > static_cast<float>(max_disparity) ||
        fabsf(ly - ry) > max_y_error_px ||
        fabsf(disp - initial_disparity) > max_disp_delta_px) {
        return;
    }

    const int out_idx = atomicAdd(match_count, 1);
    if (out_idx >= top_k) return;
    XFeatGpuMatch m;
    m.left_x = lx;
    m.left_y = ly;
    m.right_x = rx;
    m.right_y = ry;
    m.disparity = disp;
    m.score = score;
    matches[out_idx] = m;
}

bool cudaOk(cudaError_t err) {
    return err == cudaSuccess;
}

}  // namespace

void releaseXFeatGpuWorkspace(XFeatGpuWorkspace& w) {
    cudaFree(w.left_feats);
    cudaFree(w.left_keypoints);
    cudaFree(w.left_heatmap);
    cudaFree(w.left_candidates);
    cudaFree(w.right_candidates);
    cudaFree(w.left_scores);
    cudaFree(w.right_scores);
    cudaFree(w.left_indices);
    cudaFree(w.right_indices);
    cudaFree(w.left_features);
    cudaFree(w.right_features);
    cudaFree(w.left_desc);
    cudaFree(w.right_desc);
    cudaFree(w.left_best);
    cudaFree(w.right_best);
    cudaFree(w.left_best_score);
    cudaFree(w.right_best_score);
    cudaFree(w.left_second_score);
    cudaFree(w.right_second_score);
    cudaFree(w.matches);
    cudaFree(w.match_count);
    cudaFreeHost(w.host_matches);
    cudaFreeHost(w.host_match_count);
    w = XFeatGpuWorkspace{};
}

bool ensureXFeatGpuWorkspace(XFeatGpuWorkspace& w,
                             int feat_h,
                             int feat_w,
                             int desc_dim,
                             int top_k) {
    if (feat_h <= 0 || feat_w <= 0 || desc_dim <= 0 || top_k <= 0) {
        return false;
    }
    const int cells = feat_h * feat_w;
    top_k = std::min(top_k, cells);
    if (w.left_feats && w.feat_h == feat_h && w.feat_w == feat_w &&
        w.desc_dim == desc_dim && w.top_k == top_k) {
        return true;
    }

    releaseXFeatGpuWorkspace(w);
    w.feat_h = feat_h;
    w.feat_w = feat_w;
    w.cells = cells;
    w.desc_dim = desc_dim;
    w.top_k = top_k;

    const size_t feat_bytes =
        static_cast<size_t>(desc_dim) * cells * sizeof(float);
    const size_t keypoint_bytes = static_cast<size_t>(65) * cells * sizeof(float);
    const size_t heatmap_bytes = static_cast<size_t>(cells) * sizeof(float);
    const size_t candidate_bytes =
        static_cast<size_t>(cells) * sizeof(XFeatGpuCandidate);
    const size_t score_bytes = static_cast<size_t>(cells) * sizeof(float);
    const size_t index_bytes = static_cast<size_t>(cells) * sizeof(int);
    const size_t feature_bytes =
        static_cast<size_t>(top_k) * sizeof(XFeatGpuFeature);
    const size_t desc_bytes =
        static_cast<size_t>(top_k) * desc_dim * sizeof(float);
    const size_t best_int_bytes = static_cast<size_t>(top_k) * sizeof(int);
    const size_t best_float_bytes = static_cast<size_t>(top_k) * sizeof(float);
    const size_t match_bytes = static_cast<size_t>(top_k) * sizeof(XFeatGpuMatch);

    bool ok = true;
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_feats), feat_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_keypoints), keypoint_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_heatmap), heatmap_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_candidates), candidate_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.right_candidates), candidate_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_scores), score_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.right_scores), score_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_indices), index_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.right_indices), index_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_features), feature_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.right_features), feature_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_desc), desc_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.right_desc), desc_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_best), best_int_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.right_best), best_int_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_best_score), best_float_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.right_best_score), best_float_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.left_second_score), best_float_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.right_second_score), best_float_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.matches), match_bytes));
    ok = ok && cudaOk(cudaMalloc(reinterpret_cast<void**>(&w.match_count), sizeof(int)));
    ok = ok && cudaOk(cudaHostAlloc(reinterpret_cast<void**>(&w.host_matches),
                                    match_bytes, cudaHostAllocDefault));
    ok = ok && cudaOk(cudaHostAlloc(reinterpret_cast<void**>(&w.host_match_count),
                                    sizeof(int), cudaHostAllocDefault));
    if (!ok) {
        releaseXFeatGpuWorkspace(w);
        return false;
    }
    return true;
}

bool runXFeatGpuPostprocess(XFeatGpuWorkspace& w,
                            const float* left_feats,
                            const float* left_keypoints,
                            const float* left_heatmap,
                            const float* right_feats,
                            const float* right_keypoints,
                            const float* right_heatmap,
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
                            std::vector<XFeatGpuMatch>* matches) {
    if (!matches || !left_feats || !left_keypoints || !left_heatmap ||
        !right_feats || !right_keypoints || !right_heatmap || !stream ||
        w.cells <= 0 || w.top_k <= 0 || min_matches <= 0) {
        return false;
    }
    matches->clear();

    const int threads = 128;
    const int cell_blocks = (w.cells + threads - 1) / threads;
    decodeCandidatesKernel<<<cell_blocks, threads, 0, stream>>>(
        left_keypoints, left_heatmap, w.feat_h, w.feat_w, roi_size,
        w.left_candidates, w.left_scores, w.left_indices);
    decodeCandidatesKernel<<<cell_blocks, threads, 0, stream>>>(
        right_keypoints, right_heatmap, w.feat_h, w.feat_w, roi_size,
        w.right_candidates, w.right_scores, w.right_indices);
    if (!cudaOk(cudaGetLastError())) return false;

    const int top_k = std::min(w.top_k, w.cells);
    const int topk_threads = 256;
    const size_t topk_smem =
        static_cast<size_t>(topk_threads) * (sizeof(float) + sizeof(int));
    selectTopKKernel<<<1, topk_threads, topk_smem, stream>>>(
        w.left_scores, w.cells, top_k, w.left_indices);
    selectTopKKernel<<<1, topk_threads, topk_smem, stream>>>(
        w.right_scores, w.cells, top_k, w.right_indices);
    const int top_blocks = (top_k + threads - 1) / threads;
    buildFeaturesKernel<<<top_blocks, threads, 0, stream>>>(
        left_feats, w.left_candidates, w.left_indices,
        w.feat_h, w.feat_w, w.desc_dim, top_k,
        w.left_features, w.left_desc);
    buildFeaturesKernel<<<top_blocks, threads, 0, stream>>>(
        right_feats, w.right_candidates, w.right_indices,
        w.feat_h, w.feat_w, w.desc_dim, top_k,
        w.right_features, w.right_desc);
    bestMatchKernel<<<top_blocks, threads, 0, stream>>>(
        w.left_features, w.right_features,
        w.left_desc, w.right_desc, top_k, w.desc_dim,
        w.left_best, w.left_best_score, w.left_second_score);
    bestMatchKernel<<<top_blocks, threads, 0, stream>>>(
        w.right_features, w.left_features,
        w.right_desc, w.left_desc, top_k, w.desc_dim,
        w.right_best, w.right_best_score, w.right_second_score);
    if (!cudaOk(cudaMemsetAsync(w.match_count, 0, sizeof(int), stream))) {
        return false;
    }
    collectMatchesKernel<<<top_blocks, threads, 0, stream>>>(
        w.left_features, w.right_features,
        w.left_best, w.right_best,
        w.left_best_score, w.left_second_score,
        top_k, roi_size, min_score, match_margin,
        max_y_error_px, max_disp_delta_px, initial_disparity,
        max_disparity,
        left_cx, left_cy, left_w, left_h,
        right_cx, right_cy, right_w, right_h,
        w.matches, w.match_count);
    if (!cudaOk(cudaGetLastError())) return false;

    if (!w.host_match_count || !w.host_matches) {
        return false;
    }
    *w.host_match_count = 0;
    if (!cudaOk(cudaMemcpyAsync(w.host_match_count, w.match_count, sizeof(int),
                                cudaMemcpyDeviceToHost, stream)) ||
        !cudaOk(cudaMemcpyAsync(w.host_matches, w.matches,
                                static_cast<size_t>(top_k) *
                                    sizeof(XFeatGpuMatch),
                                cudaMemcpyDeviceToHost, stream))) {
        matches->clear();
        return false;
    }
    if (!cudaOk(cudaStreamSynchronize(stream))) {
        matches->clear();
        return false;
    }
    const int host_count = std::clamp(*w.host_match_count, 0, top_k);
    matches->assign(w.host_matches, w.host_matches + host_count);
    return true;
}

}  // namespace stereo3d
