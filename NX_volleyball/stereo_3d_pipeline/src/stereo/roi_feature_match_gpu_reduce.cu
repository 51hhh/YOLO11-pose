#include "roi_feature_match_gpu_reduce.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <math_constants.h>

namespace stereo3d {

namespace {

constexpr int kTemplateScoreReduceThreads = 256;
constexpr int kTemplateCcoeffThreads = 128;

__global__ void computeTemplateCcoeffNormedKernel(
    const uint8_t* left,
    size_t left_pitch_bytes,
    const uint8_t* right,
    size_t right_pitch_bytes,
    int templ_x,
    int templ_y,
    int search_x,
    int search_y,
    int patch_size,
    float* score,
    size_t score_pitch_bytes,
    int score_width,
    int score_height) {
    const int ox = blockIdx.x;
    const int oy = blockIdx.y;
    if (ox >= score_width || oy >= score_height) {
        return;
    }

    __shared__ float sum_t[kTemplateCcoeffThreads];
    __shared__ float sum_i[kTemplateCcoeffThreads];
    __shared__ float sum_tt[kTemplateCcoeffThreads];
    __shared__ float sum_ii[kTemplateCcoeffThreads];
    __shared__ float sum_ti[kTemplateCcoeffThreads];
    __shared__ float sum_sse[kTemplateCcoeffThreads];

    const int tid = threadIdx.x;
    float local_t = 0.0f;
    float local_i = 0.0f;
    float local_tt = 0.0f;
    float local_ii = 0.0f;
    float local_ti = 0.0f;
    float local_sse = 0.0f;
    const int total = patch_size * patch_size;
    const int right_x0 = search_x + ox;
    const int right_y0 = search_y + oy;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        const int py = idx / patch_size;
        const int px = idx - py * patch_size;
        const auto* left_row = left +
            static_cast<size_t>(templ_y + py) * left_pitch_bytes;
        const auto* right_row = right +
            static_cast<size_t>(right_y0 + py) * right_pitch_bytes;
        const float tv = static_cast<float>(left_row[templ_x + px]);
        const float iv = static_cast<float>(right_row[right_x0 + px]);
        local_t += tv;
        local_i += iv;
        local_tt += tv * tv;
        local_ii += iv * iv;
        local_ti += tv * iv;
        const float diff = tv - iv;
        local_sse += diff * diff;
    }

    sum_t[tid] = local_t;
    sum_i[tid] = local_i;
    sum_tt[tid] = local_tt;
    sum_ii[tid] = local_ii;
    sum_ti[tid] = local_ti;
    sum_sse[tid] = local_sse;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_t[tid] += sum_t[tid + stride];
            sum_i[tid] += sum_i[tid + stride];
            sum_tt[tid] += sum_tt[tid + stride];
            sum_ii[tid] += sum_ii[tid + stride];
            sum_ti[tid] += sum_ti[tid + stride];
            sum_sse[tid] += sum_sse[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float n = static_cast<float>(total);
        const float numerator = sum_ti[0] - (sum_t[0] * sum_i[0]) / n;
        const float denom_t = sum_tt[0] - (sum_t[0] * sum_t[0]) / n;
        const float denom_i = sum_ii[0] - (sum_i[0] * sum_i[0]) / n;
        float value = -1.0f;
        const float denom = denom_t * denom_i;
        if (denom > 1e-6f) {
            value = numerator / sqrtf(denom);
            value = fminf(1.0f, fmaxf(-1.0f, value));
        } else {
            const float rms = sqrtf(fmaxf(0.0f, sum_sse[0] / n));
            value = 1.0f - fminf(1.0f, rms * (1.0f / 255.0f));
        }
        const float cx = 0.5f * static_cast<float>(score_width - 1);
        const float cy = 0.5f * static_cast<float>(score_height - 1);
        const float dx = static_cast<float>(ox) - cx;
        const float dy = static_cast<float>(oy) - cy;
        value -= 1e-5f * (dx * dx + dy * dy);
        auto* row = reinterpret_cast<float*>(
            reinterpret_cast<char*>(score) +
            static_cast<size_t>(oy) * score_pitch_bytes);
        row[ox] = value;
    }
}

__global__ void reduceTemplateScorePeakKernel(
    const float* score,
    size_t pitch_bytes,
    int width,
    int height,
    CudaTemplateScorePeak* out) {
    __shared__ float best_values[kTemplateScoreReduceThreads];
    __shared__ int best_indices[kTemplateScoreReduceThreads];

    const int tid = threadIdx.x;
    float best_value = -CUDART_INF_F;
    int best_index = -1;
    const int total = width * height;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        const int y = idx / width;
        const int x = idx - y * width;
        const auto* row = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(score) +
            static_cast<size_t>(y) * pitch_bytes);
        const float value = row[x];
        if (!isfinite(value)) {
            continue;
        }
        if (value > best_value ||
            (value == best_value &&
             (best_index < 0 || idx < best_index))) {
            best_value = value;
            best_index = idx;
        }
    }

    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_value = best_values[tid + stride];
            const int other_index = best_indices[tid + stride];
            const int self_index = best_indices[tid];
            if (other_index >= 0 &&
                (self_index < 0 ||
                 other_value > best_values[tid] ||
                 (other_value == best_values[tid] &&
                  other_index < self_index))) {
                best_values[tid] = other_value;
                best_indices[tid] = other_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int idx = best_indices[0];
        if (idx < 0) {
            out->value = -1.0f;
            out->x = -1;
            out->y = -1;
            out->valid = 0;
        } else {
            out->value = best_values[0];
            out->x = idx % width;
            out->y = idx / width;
            out->valid = 1;
        }
    }
}

__global__ void findTemplateCcoeffNormedPeakKernel(
    const uint8_t* left,
    size_t left_pitch_bytes,
    const uint8_t* right,
    size_t right_pitch_bytes,
    int templ_x,
    int templ_y,
    int search_x,
    int search_y,
    int patch_size,
    int score_width,
    int score_height,
    CudaTemplateScorePeak* out) {
    __shared__ float best_values[kTemplateScoreReduceThreads];
    __shared__ int best_indices[kTemplateScoreReduceThreads];

    const int tid = threadIdx.x;
    const int total_scores = score_width * score_height;
    const int patch_pixels = patch_size * patch_size;
    float best_value = -CUDART_INF_F;
    int best_index = -1;

    for (int idx = tid; idx < total_scores; idx += blockDim.x) {
        const int oy = idx / score_width;
        const int ox = idx - oy * score_width;
        const int right_x0 = search_x + ox;
        const int right_y0 = search_y + oy;
        float sum_t = 0.0f;
        float sum_i = 0.0f;
        float sum_tt = 0.0f;
        float sum_ii = 0.0f;
        float sum_ti = 0.0f;
        float sum_sse = 0.0f;

        for (int py = 0; py < patch_size; ++py) {
            const auto* left_row = left +
                static_cast<size_t>(templ_y + py) * left_pitch_bytes;
            const auto* right_row = right +
                static_cast<size_t>(right_y0 + py) * right_pitch_bytes;
            for (int px = 0; px < patch_size; ++px) {
                const float tv = static_cast<float>(left_row[templ_x + px]);
                const float iv = static_cast<float>(right_row[right_x0 + px]);
                sum_t += tv;
                sum_i += iv;
                sum_tt += tv * tv;
                sum_ii += iv * iv;
                sum_ti += tv * iv;
                const float diff = tv - iv;
                sum_sse += diff * diff;
            }
        }

        const float n = static_cast<float>(patch_pixels);
        const float numerator = sum_ti - (sum_t * sum_i) / n;
        const float denom_t = sum_tt - (sum_t * sum_t) / n;
        const float denom_i = sum_ii - (sum_i * sum_i) / n;
        float value = -1.0f;
        const float denom = denom_t * denom_i;
        if (denom > 1e-6f) {
            value = numerator / sqrtf(denom);
            value = fminf(1.0f, fmaxf(-1.0f, value));
        } else {
            const float rms = sqrtf(fmaxf(0.0f, sum_sse / n));
            value = 1.0f - fminf(1.0f, rms * (1.0f / 255.0f));
        }
        const float cx = 0.5f * static_cast<float>(score_width - 1);
        const float cy = 0.5f * static_cast<float>(score_height - 1);
        const float dx = static_cast<float>(ox) - cx;
        const float dy = static_cast<float>(oy) - cy;
        value -= 1e-5f * (dx * dx + dy * dy);

        if (isfinite(value) &&
            (value > best_value ||
             (value == best_value &&
              (best_index < 0 || idx < best_index)))) {
            best_value = value;
            best_index = idx;
        }
    }

    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_value = best_values[tid + stride];
            const int other_index = best_indices[tid + stride];
            const int self_index = best_indices[tid];
            if (other_index >= 0 &&
                (self_index < 0 ||
                 other_value > best_values[tid] ||
                 (other_value == best_values[tid] &&
                  other_index < self_index))) {
                best_values[tid] = other_value;
                best_indices[tid] = other_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int idx = best_indices[0];
        if (idx < 0) {
            out->value = -1.0f;
            out->x = -1;
            out->y = -1;
            out->valid = 0;
        } else {
            out->value = best_values[0];
            out->x = idx % score_width;
            out->y = idx / score_width;
            out->valid = 1;
        }
    }
}

}  // namespace

cudaError_t computeCudaTemplateCcoeffNormedScoreMap(
    const uint8_t* left_gpu,
    size_t left_pitch_bytes,
    const uint8_t* right_gpu,
    size_t right_pitch_bytes,
    int templ_x,
    int templ_y,
    int search_x,
    int search_y,
    int patch_size,
    float* score_gpu,
    size_t score_pitch_bytes,
    int score_width,
    int score_height,
    cudaStream_t stream) {
    if (!left_gpu || !right_gpu || left_pitch_bytes == 0 ||
        right_pitch_bytes == 0 || templ_x < 0 || templ_y < 0 ||
        search_x < 0 || search_y < 0 || patch_size <= 0 ||
        !score_gpu || score_pitch_bytes == 0 ||
        score_width <= 0 || score_height <= 0 || !stream) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(kTemplateCcoeffThreads, 1, 1);
    const dim3 grid(score_width, score_height, 1);
    computeTemplateCcoeffNormedKernel<<<grid, block, 0, stream>>>(
        left_gpu,
        left_pitch_bytes,
        right_gpu,
        right_pitch_bytes,
        templ_x,
        templ_y,
        search_x,
        search_y,
        patch_size,
        score_gpu,
        score_pitch_bytes,
        score_width,
        score_height);
    return cudaGetLastError();
}

cudaError_t findCudaTemplateScorePeak(
    const float* score_gpu,
    size_t score_pitch_bytes,
    int width,
    int height,
    CudaTemplateScorePeak* device_result,
    CudaTemplateScorePeak* host_result,
    cudaStream_t stream) {
    if (!score_gpu || score_pitch_bytes == 0 || width <= 0 || height <= 0 ||
        !device_result || !host_result || !stream) {
        return cudaErrorInvalidValue;
    }
    reduceTemplateScorePeakKernel<<<1, kTemplateScoreReduceThreads, 0, stream>>>(
        score_gpu, score_pitch_bytes, width, height, device_result);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    return cudaMemcpyAsync(host_result,
                           device_result,
                           sizeof(CudaTemplateScorePeak),
                           cudaMemcpyDeviceToHost,
                           stream);
}

cudaError_t findCudaTemplateCcoeffNormedPeak(
    const uint8_t* left_gpu,
    size_t left_pitch_bytes,
    const uint8_t* right_gpu,
    size_t right_pitch_bytes,
    int templ_x,
    int templ_y,
    int search_x,
    int search_y,
    int patch_size,
    int score_width,
    int score_height,
    CudaTemplateScorePeak* device_result,
    CudaTemplateScorePeak* host_result,
    cudaStream_t stream) {
    if (!left_gpu || !right_gpu || left_pitch_bytes == 0 ||
        right_pitch_bytes == 0 || templ_x < 0 || templ_y < 0 ||
        search_x < 0 || search_y < 0 || patch_size <= 0 ||
        score_width <= 0 || score_height <= 0 ||
        !device_result || !host_result || !stream) {
        return cudaErrorInvalidValue;
    }
    findTemplateCcoeffNormedPeakKernel<<<1, kTemplateScoreReduceThreads, 0,
                                         stream>>>(
        left_gpu,
        left_pitch_bytes,
        right_gpu,
        right_pitch_bytes,
        templ_x,
        templ_y,
        search_x,
        search_y,
        patch_size,
        score_width,
        score_height,
        device_result);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    return cudaMemcpyAsync(host_result,
                           device_result,
                           sizeof(CudaTemplateScorePeak),
                           cudaMemcpyDeviceToHost,
                           stream);
}

}  // namespace stereo3d
