#include "roi_feature_match_gpu_reduce.h"

#include <cuda_runtime.h>

#include <math_constants.h>

namespace stereo3d {

namespace {

constexpr int kTemplateScoreReduceThreads = 256;

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

}  // namespace

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

}  // namespace stereo3d
