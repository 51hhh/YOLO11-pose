/**
 * @file detect_preprocess.cu
 * @brief 检测预处理 CUDA Kernels
 *
 * 两个核心 Kernel:
 *   1. GrayToRGB: 灰度 U8 → float32 RGB CHW (带 Bilinear Resize + 归一化)
 *   2. BayerToRGB: Bayer RG8 → float32 RGB CHW (去马赛克 + Resize + 归一化)
 *
 * 这些 Kernel 运行在 DLA CUDA Stream 上。
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ==================== 灰度 U8 → RGB float32 CHW ====================
// 灰度值复制到 R=G=B 三通道
__global__ void grayToRGBKernel(const unsigned char* __restrict__ gray,
                                 float* __restrict__ dst,
                                 int srcW, int srcH, int srcPitch,
                                 int dstW, int dstH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    float scaleX = (float)srcW / dstW;
    float scaleY = (float)srcH / dstH;

    // Bilinear 插值坐标
    float sx = x * scaleX;
    float sy = y * scaleY;
    int x0 = (int)sx;
    int y0 = (int)sy;
    int x1 = min(x0 + 1, srcW - 1);
    int y1 = min(y0 + 1, srcH - 1);
    float dx = sx - x0;
    float dy = sy - y0;

    // 使用 pitch 计算源地址 (VPI Image 可能有 padding)
    float v00 = gray[y0 * srcPitch + x0];
    float v01 = gray[y0 * srcPitch + x1];
    float v10 = gray[y1 * srcPitch + x0];
    float v11 = gray[y1 * srcPitch + x1];

    float val = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v01
              + (1 - dx) * dy * v10 + dx * dy * v11;
    val *= (1.0f / 255.0f);

    // CHW 格式: R = G = B = val
    int planeSize = dstW * dstH;
    int idx = y * dstW + x;
    dst[0 * planeSize + idx] = val;  // R
    dst[1 * planeSize + idx] = val;  // G
    dst[2 * planeSize + idx] = val;  // B
}

// ==================== Bayer RG8 → RGB float32 CHW ====================
// 预留: 当相机切换为 Bayer 输出模式时启用此路径
// (借用已有的 Bayer kernel 逻辑, 适配 pitch-based 输入)
__global__ void bayerToRGBKernel(const unsigned char* __restrict__ bayer,
                                  float* __restrict__ dst,
                                  int srcW, int srcH, int srcPitch,
                                  int dstW, int dstH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    float scaleX = (float)srcW / dstW;
    float scaleY = (float)srcH / dstH;

    float srcX = x * scaleX;
    float srcY = y * scaleY;
    int x0 = (int)srcX;
    int y0 = (int)srcY;

    // 对齐到偶数位置 (Bayer 2x2 block)
    x0 = (x0 >> 1) << 1;
    y0 = (y0 >> 1) << 1;
    int x1 = min(x0 + 1, srcW - 1);
    int y1 = min(y0 + 1, srcH - 1);

    // RGGB 模式
    float r = (float)bayer[y0 * srcPitch + x0];
    float g = ((float)bayer[y0 * srcPitch + x1] + (float)bayer[y1 * srcPitch + x0]) * 0.5f;
    float b = (float)bayer[y1 * srcPitch + x1];

    r *= (1.0f / 255.0f);
    g *= (1.0f / 255.0f);
    b *= (1.0f / 255.0f);

    int planeSize = dstW * dstH;
    int idx = y * dstW + x;
    dst[0 * planeSize + idx] = r;
    dst[1 * planeSize + idx] = g;
    dst[2 * planeSize + idx] = b;
}

// ==================== C 接口 ====================

extern "C" void launchGrayToRGBKernel(const unsigned char* gray, float* dst,
                                       int srcW, int srcH, int srcPitch,
                                       int dstW, int dstH,
                                       cudaStream_t stream)
{
    dim3 block(32, 32);
    dim3 grid((dstW + block.x - 1) / block.x,
              (dstH + block.y - 1) / block.y);
    grayToRGBKernel<<<grid, block, 0, stream>>>(gray, dst,
                                                  srcW, srcH, srcPitch,
                                                  dstW, dstH);
}

extern "C" void launchBayerToRGBKernel(const unsigned char* bayer, float* dst,
                                        int srcW, int srcH, int srcPitch,
                                        int dstW, int dstH,
                                        cudaStream_t stream)
{
    dim3 block(32, 32);
    dim3 grid((dstW + block.x - 1) / block.x,
              (dstH + block.y - 1) / block.y);
    bayerToRGBKernel<<<grid, block, 0, stream>>>(bayer, dst,
                                                   srcW, srcH, srcPitch,
                                                   dstW, dstH);
}
