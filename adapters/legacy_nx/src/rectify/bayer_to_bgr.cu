/**
 * @file bayer_to_bgr.cu
 * @brief 校正前使用的全分辨率 Bayer RGGB 到 BGR8 CUDA 转换。
 */

#include "bayer_to_bgr.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {

__global__ void bayerToBGR8Kernel(
    const unsigned char* __restrict__ bayer,
    unsigned char* __restrict__ bgr,
    int width,
    int height,
    int bayer_pitch,
    int bgr_pitch) {
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

    const int output = y * bgr_pitch + x * 3;
    bgr[output] = static_cast<unsigned char>(
        __float2uint_rn(fminf(255.0F, fmaxf(0.0F, blue))));
    bgr[output + 1] = static_cast<unsigned char>(
        __float2uint_rn(fminf(255.0F, fmaxf(0.0F, green))));
    bgr[output + 2] = static_cast<unsigned char>(
        __float2uint_rn(fminf(255.0F, fmaxf(0.0F, red))));
}

}  // namespace

extern "C" void launchBayerToBGR8(
    const unsigned char* bayer,
    unsigned char* bgr,
    int width,
    int height,
    int bayer_pitch,
    int bgr_pitch,
    cudaStream_t stream) {
    const dim3 block(32, 32);
    const dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);
    bayerToBGR8Kernel<<<grid, block, 0, stream>>>(
        bayer, bgr, width, height, bayer_pitch, bgr_pitch);
}
