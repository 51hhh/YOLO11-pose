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

// ==================== Letterbox: Bayer RG8 → RGB float32 CHW ====================
// 输入为 Bayer 单通道马赛克图, 输出 RGB CHW, 维持 YOLO letterbox 预处理一致性
__global__ void bayerToRGBLetterboxKernel(const unsigned char* __restrict__ bayer,
                                           float* __restrict__ dst,
                                           int srcW, int srcH, int srcPitch,
                                           int dstW, int dstH,
                                           int newW, int newH,
                                           int padX, int padY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    int planeSize = dstW * dstH;
    int idx = y * dstW + x;

    int cx = x - padX;
    int cy = y - padY;

    if (cx < 0 || cx >= newW || cy < 0 || cy >= newH) {
        float padVal = 114.0f / 255.0f;
        dst[0 * planeSize + idx] = padVal;
        dst[1 * planeSize + idx] = padVal;
        dst[2 * planeSize + idx] = padVal;
        return;
    }

    float scaleX = (float)srcW / newW;
    float scaleY = (float)srcH / newH;

    int sx = (int)(cx * scaleX);
    int sy = (int)(cy * scaleY);

    // 对齐到 Bayer 2x2 block 左上角
    sx = (sx >> 1) << 1;
    sy = (sy >> 1) << 1;
    int sx1 = min(sx + 1, srcW - 1);
    int sy1 = min(sy + 1, srcH - 1);

    // RGGB block
    float r = (float)bayer[sy * srcPitch + sx];
    float g = ((float)bayer[sy * srcPitch + sx1] + (float)bayer[sy1 * srcPitch + sx]) * 0.5f;
    float b = (float)bayer[sy1 * srcPitch + sx1];

    r *= (1.0f / 255.0f);
    g *= (1.0f / 255.0f);
    b *= (1.0f / 255.0f);

    dst[0 * planeSize + idx] = r;
    dst[1 * planeSize + idx] = g;
    dst[2 * planeSize + idx] = b;
}

// ==================== Letterbox: Gray U8 → RGB float32 CHW ====================
// Uniform scale + padding (114/255 gray), matches YOLO training preprocessing
__global__ void grayToRGBLetterboxKernel(const unsigned char* __restrict__ gray,
                                          float* __restrict__ dst,
                                          int srcW, int srcH, int srcPitch,
                                          int dstW, int dstH,
                                          int newW, int newH,
                                          int padX, int padY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    int planeSize = dstW * dstH;
    int idx = y * dstW + x;

    int cx = x - padX;
    int cy = y - padY;

    if (cx < 0 || cx >= newW || cy < 0 || cy >= newH) {
        float padVal = 114.0f / 255.0f;
        dst[0 * planeSize + idx] = padVal;
        dst[1 * planeSize + idx] = padVal;
        dst[2 * planeSize + idx] = padVal;
        return;
    }

    float scaleX = (float)srcW / newW;
    float scaleY = (float)srcH / newH;

    float sx = cx * scaleX;
    float sy = cy * scaleY;
    int x0 = (int)sx;
    int y0 = (int)sy;
    int x1 = min(x0 + 1, srcW - 1);
    int y1 = min(y0 + 1, srcH - 1);
    float dx = sx - x0;
    float dy = sy - y0;

    float v00 = gray[y0 * srcPitch + x0];
    float v01 = gray[y0 * srcPitch + x1];
    float v10 = gray[y1 * srcPitch + x0];
    float v11 = gray[y1 * srcPitch + x1];

    float val = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v01
              + (1 - dx) * dy * v10 + dx * dy * v11;
    val *= (1.0f / 255.0f);

    dst[0 * planeSize + idx] = val;
    dst[1 * planeSize + idx] = val;
    dst[2 * planeSize + idx] = val;
}

// ==================== BGR U8 → RGB float32 CHW (Letterbox) ====================
// 输入: 3通道 BGR U8 (来自 Bayer 去马赛克后的校正图像)
// 输出: 3通道 RGB float32 CHW (YOLO 模型输入格式)
__global__ void bgrToRGBLetterboxKernel(const unsigned char* __restrict__ bgr,
                                         float* __restrict__ dst,
                                         int srcW, int srcH, int srcStep,
                                         int dstW, int dstH,
                                         int newW, int newH,
                                         int padX, int padY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    int planeSize = dstW * dstH;
    int idx = y * dstW + x;

    int cx = x - padX;
    int cy = y - padY;

    if (cx < 0 || cx >= newW || cy < 0 || cy >= newH) {
        float padVal = 114.0f / 255.0f;
        dst[0 * planeSize + idx] = padVal;
        dst[1 * planeSize + idx] = padVal;
        dst[2 * planeSize + idx] = padVal;
        return;
    }

    float scaleX = (float)srcW / newW;
    float scaleY = (float)srcH / newH;

    float sx = cx * scaleX;
    float sy = cy * scaleY;
    int x0 = (int)sx;
    int y0 = (int)sy;
    int x1 = min(x0 + 1, srcW - 1);
    int y1 = min(y0 + 1, srcH - 1);
    float dx = sx - x0;
    float dy = sy - y0;

    // BGR interleaved: pixel at (x,y) = bgr[y * srcStep + x * 3 + c]
    const unsigned char* p00 = bgr + y0 * srcStep + x0 * 3;
    const unsigned char* p01 = bgr + y0 * srcStep + x1 * 3;
    const unsigned char* p10 = bgr + y1 * srcStep + x0 * 3;
    const unsigned char* p11 = bgr + y1 * srcStep + x1 * 3;

    // Bilinear interpolation per channel, BGR → RGB reorder
    float b = ((1-dx)*(1-dy)*p00[0] + dx*(1-dy)*p01[0] + (1-dx)*dy*p10[0] + dx*dy*p11[0]) / 255.0f;
    float g = ((1-dx)*(1-dy)*p00[1] + dx*(1-dy)*p01[1] + (1-dx)*dy*p10[1] + dx*dy*p11[1]) / 255.0f;
    float r = ((1-dx)*(1-dy)*p00[2] + dx*(1-dy)*p01[2] + (1-dx)*dy*p10[2] + dx*dy*p11[2]) / 255.0f;

    // RGB CHW output
    dst[0 * planeSize + idx] = r;
    dst[1 * planeSize + idx] = g;
    dst[2 * planeSize + idx] = b;
}

// ==================== Bayer RGGB → BGR8 (全分辨率, bilinear) ====================
// 用于 color pipeline: 产生 BGR8 interleaved 输出, 与 VPI remap 配合
// 质量: bilinear 插值 (4邻域加权), 比 nearest-neighbor 好很多
__global__ void bayerToBGR8_kernel(const unsigned char* __restrict__ bayer,
                                    unsigned char* __restrict__ bgr,
                                    int width, int height,
                                    int bayer_pitch, int bgr_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Clamp-read helper (inline lambda not supported in old CUDA, use macro)
    #define BGET(gy, gx) bayer[min(max(gy,0),height-1) * bayer_pitch + min(max(gx,0),width-1)]

    float r, g, b;
    int px = x & 1;
    int py = y & 1;

    if (py == 0 && px == 0) {
        // R site: R direct, G = avg of 4 cardinal, B = avg of 4 diagonal
        r = BGET(y, x);
        g = ((float)BGET(y,x-1) + BGET(y,x+1) + BGET(y-1,x) + BGET(y+1,x)) * 0.25f;
        b = ((float)BGET(y-1,x-1) + BGET(y-1,x+1) + BGET(y+1,x-1) + BGET(y+1,x+1)) * 0.25f;
    } else if (py == 0 && px == 1) {
        // Gr (green on R row): R from left/right, G direct, B from top/bottom
        r = ((float)BGET(y,x-1) + BGET(y,x+1)) * 0.5f;
        g = BGET(y, x);
        b = ((float)BGET(y-1,x) + BGET(y+1,x)) * 0.5f;
    } else if (py == 1 && px == 0) {
        // Gb (green on B row): R from top/bottom, G direct, B from left/right
        r = ((float)BGET(y-1,x) + BGET(y+1,x)) * 0.5f;
        g = BGET(y, x);
        b = ((float)BGET(y,x-1) + BGET(y,x+1)) * 0.5f;
    } else {
        // B site: R = avg of 4 diagonal, G = avg of 4 cardinal, B direct
        r = ((float)BGET(y-1,x-1) + BGET(y-1,x+1) + BGET(y+1,x-1) + BGET(y+1,x+1)) * 0.25f;
        g = ((float)BGET(y,x-1) + BGET(y,x+1) + BGET(y-1,x) + BGET(y+1,x)) * 0.25f;
        b = BGET(y, x);
    }

    #undef BGET

    int out = y * bgr_pitch + x * 3;
    bgr[out + 0] = (unsigned char)__float2uint_rn(fminf(255.0f, fmaxf(0.0f, b)));
    bgr[out + 1] = (unsigned char)__float2uint_rn(fminf(255.0f, fmaxf(0.0f, g)));
    bgr[out + 2] = (unsigned char)__float2uint_rn(fminf(255.0f, fmaxf(0.0f, r)));
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

extern "C" void launchGrayToRGBLetterboxKernel(const unsigned char* gray, float* dst,
                                                int srcW, int srcH, int srcPitch,
                                                int dstW, int dstH,
                                                int newW, int newH,
                                                int padX, int padY,
                                                cudaStream_t stream)
{
    dim3 block(32, 32);
    dim3 grid((dstW + block.x - 1) / block.x,
              (dstH + block.y - 1) / block.y);
    grayToRGBLetterboxKernel<<<grid, block, 0, stream>>>(gray, dst,
                                                          srcW, srcH, srcPitch,
                                                          dstW, dstH,
                                                          newW, newH, padX, padY);
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

extern "C" void launchBayerToRGBLetterboxKernel(const unsigned char* bayer, float* dst,
                                                   int srcW, int srcH, int srcPitch,
                                                   int dstW, int dstH,
                                                   int newW, int newH,
                                                   int padX, int padY,
                                                   cudaStream_t stream)
{
    dim3 block(32, 32);
    dim3 grid((dstW + block.x - 1) / block.x,
              (dstH + block.y - 1) / block.y);
    bayerToRGBLetterboxKernel<<<grid, block, 0, stream>>>(bayer, dst,
                                                           srcW, srcH, srcPitch,
                                                           dstW, dstH,
                                                           newW, newH, padX, padY);
}

extern "C" void launchBGRToRGBLetterboxKernel(const unsigned char* bgr, float* dst,
                                               int srcW, int srcH, int srcStep,
                                               int dstW, int dstH,
                                               int newW, int newH,
                                               int padX, int padY,
                                               cudaStream_t stream)
{
    dim3 block(32, 32);
    dim3 grid((dstW + block.x - 1) / block.x,
              (dstH + block.y - 1) / block.y);
    bgrToRGBLetterboxKernel<<<grid, block, 0, stream>>>(bgr, dst,
                                                         srcW, srcH, srcStep,
                                                         dstW, dstH,
                                                         newW, newH, padX, padY);
}

extern "C" void launchBayerToBGR8(const unsigned char* bayer, unsigned char* bgr,
                                   int width, int height,
                                   int bayer_pitch, int bgr_pitch,
                                   cudaStream_t stream)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    bayerToBGR8_kernel<<<grid, block, 0, stream>>>(bayer, bgr,
                                                     width, height,
                                                     bayer_pitch, bgr_pitch);
}
