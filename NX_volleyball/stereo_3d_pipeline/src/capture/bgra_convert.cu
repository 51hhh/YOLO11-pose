#include <cstdint>
#include <cuda_runtime.h>

__global__ void bgra2grayKernel(
    const uint8_t* __restrict__ src, int srcPitch,
    uint8_t* __restrict__ dst, int dstPitch,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint8_t* pixel = src + y * srcPitch + x * 4;
    float gray = 0.114f * pixel[0] + 0.587f * pixel[1] + 0.299f * pixel[2];
    dst[y * dstPitch + x] = (uint8_t)fminf(255.0f, gray + 0.5f);
}

extern "C" void launchBGRA2Gray(
    const uint8_t* src, int srcPitch,
    uint8_t* dst, int dstPitch,
    int width, int height,
    cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    bgra2grayKernel<<<grid, block, 0, stream>>>(src, srcPitch, dst, dstPitch, width, height);
}
