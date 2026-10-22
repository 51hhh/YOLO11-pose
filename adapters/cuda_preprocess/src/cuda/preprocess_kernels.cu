/**
 * @file preprocess_kernels.cu
 * @brief Bayer RGGB 转换及双目标定浮点映射 CUDA kernel。
 */

#include "volleyball/adapters/cuda_preprocess/kernels.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void bayerToBGR8Kernel(
    const unsigned char* __restrict__ bayer,
    unsigned char* __restrict__ bgr,
    int width,
    int height,
    std::size_t bayer_pitch,
    std::size_t bgr_pitch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

#define BGET(gy, gx) \
    bayer[min(max((gy), 0), height - 1) * bayer_pitch + \
          min(max((gx), 0), width - 1)]

    float red;
    float green;
    float blue;
    const int x_parity = x & 1;
    const int y_parity = y & 1;

    if (y_parity == 0 && x_parity == 0) {
        red = BGET(y, x);
        green = (static_cast<float>(BGET(y, x - 1)) + BGET(y, x + 1) +
                 BGET(y - 1, x) + BGET(y + 1, x)) * 0.25F;
        blue = (static_cast<float>(BGET(y - 1, x - 1)) +
                BGET(y - 1, x + 1) + BGET(y + 1, x - 1) +
                BGET(y + 1, x + 1)) * 0.25F;
    } else if (y_parity == 0) {
        red = (static_cast<float>(BGET(y, x - 1)) + BGET(y, x + 1)) * 0.5F;
        green = BGET(y, x);
        blue = (static_cast<float>(BGET(y - 1, x)) + BGET(y + 1, x)) * 0.5F;
    } else if (x_parity == 0) {
        red = (static_cast<float>(BGET(y - 1, x)) + BGET(y + 1, x)) * 0.5F;
        green = BGET(y, x);
        blue = (static_cast<float>(BGET(y, x - 1)) + BGET(y, x + 1)) * 0.5F;
    } else {
        red = (static_cast<float>(BGET(y - 1, x - 1)) +
               BGET(y - 1, x + 1) + BGET(y + 1, x - 1) +
               BGET(y + 1, x + 1)) * 0.25F;
        green = (static_cast<float>(BGET(y, x - 1)) + BGET(y, x + 1) +
                 BGET(y - 1, x) + BGET(y + 1, x)) * 0.25F;
        blue = BGET(y, x);
    }

#undef BGET

    const std::size_t output =
        static_cast<std::size_t>(y) * bgr_pitch +
        static_cast<std::size_t>(x) * 3U;
    bgr[output] = static_cast<unsigned char>(
        __float2uint_rn(fminf(255.0F, fmaxf(0.0F, blue))));
    bgr[output + 1] = static_cast<unsigned char>(
        __float2uint_rn(fminf(255.0F, fmaxf(0.0F, green))));
    bgr[output + 2] = static_cast<unsigned char>(
        __float2uint_rn(fminf(255.0F, fmaxf(0.0F, red))));
}

__device__ float sampleBgr(
    const std::uint8_t* source,
    std::size_t pitch,
    int width,
    int height,
    int x,
    int y,
    int channel) {
    if (x < 0 || y < 0 || x >= width || y >= height) {
        return 0.0F;
    }
    return static_cast<float>(
        source[static_cast<std::size_t>(y) * pitch +
               static_cast<std::size_t>(x) * 3U +
               static_cast<std::size_t>(channel)]);
}

__device__ std::uint8_t toU8(float value) {
    return static_cast<std::uint8_t>(
        __float2uint_rn(fminf(255.0F, fmaxf(0.0F, value))));
}

__global__ void remapBgrToBgrGray8Kernel(
    const std::uint8_t* __restrict__ source,
    std::size_t source_pitch,
    int source_width,
    int source_height,
    const float* __restrict__ map_x,
    const float* __restrict__ map_y,
    std::size_t map_pitch,
    std::uint8_t* __restrict__ output_bgr,
    std::size_t output_bgr_pitch,
    std::uint8_t* __restrict__ output_gray,
    std::size_t output_gray_pitch,
    int output_width,
    int output_height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= output_width || y >= output_height) {
        return;
    }

    const auto* map_x_row = reinterpret_cast<const float*>(
        reinterpret_cast<const std::uint8_t*>(map_x) +
        static_cast<std::size_t>(y) * map_pitch);
    const auto* map_y_row = reinterpret_cast<const float*>(
        reinterpret_cast<const std::uint8_t*>(map_y) +
        static_cast<std::size_t>(y) * map_pitch);
    const float source_x = map_x_row[x];
    const float source_y = map_y_row[x];
    const std::size_t gray_offset =
        static_cast<std::size_t>(y) * output_gray_pitch +
        static_cast<std::size_t>(x);
    const std::size_t bgr_offset =
        static_cast<std::size_t>(y) * output_bgr_pitch +
        static_cast<std::size_t>(x) * 3U;
    if (!isfinite(source_x) || !isfinite(source_y)) {
        output_bgr[bgr_offset] = 0;
        output_bgr[bgr_offset + 1U] = 0;
        output_bgr[bgr_offset + 2U] = 0;
        output_gray[gray_offset] = 0;
        return;
    }

    const int x0 = static_cast<int>(floorf(source_x));
    const int y0 = static_cast<int>(floorf(source_y));
    const float dx = source_x - static_cast<float>(x0);
    const float dy = source_y - static_cast<float>(y0);
    float channels[3];
    for (int channel = 0; channel < 3; ++channel) {
        const float top =
            sampleBgr(
                source, source_pitch, source_width, source_height,
                x0, y0, channel) * (1.0F - dx) +
            sampleBgr(
                source, source_pitch, source_width, source_height,
                x0 + 1, y0, channel) * dx;
        const float bottom =
            sampleBgr(
                source, source_pitch, source_width, source_height,
                x0, y0 + 1, channel) * (1.0F - dx) +
            sampleBgr(
                source, source_pitch, source_width, source_height,
                x0 + 1, y0 + 1, channel) * dx;
        channels[channel] = top * (1.0F - dy) + bottom * dy;
        output_bgr[bgr_offset + static_cast<std::size_t>(channel)] =
            toU8(channels[channel]);
    }
    output_gray[gray_offset] = toU8(
        channels[0] * 0.114F + channels[1] * 0.587F +
        channels[2] * 0.299F);
}

}  // namespace

namespace volleyball::adapters::cuda_preprocess {

cudaError_t launchBayerRggbToBgr8(
    const std::uint8_t* bayer,
    std::size_t bayer_pitch_bytes,
    std::uint8_t* bgr,
    std::size_t bgr_pitch_bytes,
    int width,
    int height,
    cudaStream_t stream) noexcept {
    if (bayer == nullptr || bgr == nullptr || width <= 0 || height <= 0 ||
        bayer_pitch_bytes < static_cast<std::size_t>(width) ||
        bgr_pitch_bytes < static_cast<std::size_t>(width) * 3U) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(32, 8);
    const dim3 grid(
        (static_cast<unsigned int>(width) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(height) + block.y - 1U) / block.y);
    bayerToBGR8Kernel<<<grid, block, 0, stream>>>(
        bayer, bgr, width, height, bayer_pitch_bytes, bgr_pitch_bytes);
    return cudaPeekAtLastError();
}

cudaError_t launchRemapBgrToBgrGray8(
    const std::uint8_t* source_bgr,
    std::size_t source_pitch_bytes,
    int source_width,
    int source_height,
    const float* map_x,
    const float* map_y,
    std::size_t map_pitch_bytes,
    std::uint8_t* rectified_bgr,
    std::size_t rectified_bgr_pitch_bytes,
    std::uint8_t* rectified_gray,
    std::size_t rectified_gray_pitch_bytes,
    int output_width,
    int output_height,
    cudaStream_t stream) noexcept {
    if (source_bgr == nullptr || map_x == nullptr || map_y == nullptr ||
        rectified_bgr == nullptr || rectified_gray == nullptr ||
        source_width <= 0 || source_height <= 0 ||
        output_width <= 0 || output_height <= 0 ||
        source_pitch_bytes < static_cast<std::size_t>(source_width) * 3U ||
        map_pitch_bytes < static_cast<std::size_t>(output_width) * sizeof(float) ||
        rectified_bgr_pitch_bytes <
            static_cast<std::size_t>(output_width) * 3U ||
        rectified_gray_pitch_bytes < static_cast<std::size_t>(output_width)) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(32, 8);
    const dim3 grid(
        (static_cast<unsigned int>(output_width) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(output_height) + block.y - 1U) / block.y);
    remapBgrToBgrGray8Kernel<<<grid, block, 0, stream>>>(
        source_bgr,
        source_pitch_bytes,
        source_width,
        source_height,
        map_x,
        map_y,
        map_pitch_bytes,
        rectified_bgr,
        rectified_bgr_pitch_bytes,
        rectified_gray,
        rectified_gray_pitch_bytes,
        output_width,
        output_height);
    return cudaPeekAtLastError();
}

}  // namespace volleyball::adapters::cuda_preprocess
