/**
 * @file crop_resize.cu
 * @brief CUDA kernel: 从 U8 灰度全帧裁剪+缩放正方形 ROI patch (SOT tracker 预处理)
 *
 * 双线性插值, 边界 clamp, 输出归一化 float [0,1]
 * 自动将 (w,h) + context_factor 转换为正方形搜索区域 s = sqrt((w*ctx)*(h*ctx))
 */

#include <cuda_runtime.h>
#include <cstdint>

// kernel 定义 (匿名 namespace 隐藏符号)
namespace {

__global__ void cropResizeBilinearKernel(
    const uint8_t* __restrict__ src, int src_pitch, int src_w, int src_h,
    float* __restrict__ dst, int dst_size,
    float roi_x, float roi_y, float roi_w, float roi_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_size || dy >= dst_size) return;

    float sx = roi_x + (dx + 0.5f) * roi_w / dst_size - 0.5f;
    float sy = roi_y + (dy + 0.5f) * roi_h / dst_size - 0.5f;

    sx = fmaxf(0.0f, fminf(sx, (float)(src_w - 1)));
    sy = fmaxf(0.0f, fminf(sy, (float)(src_h - 1)));

    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;

    float v00 = src[y0 * src_pitch + x0];
    float v10 = src[y0 * src_pitch + x1];
    float v01 = src[y1 * src_pitch + x0];
    float v11 = src[y1 * src_pitch + x1];

    float val = v00 * (1-fx)*(1-fy) + v10 * fx*(1-fy)
              + v01 * (1-fx)*fy     + v11 * fx*fy;

    dst[dy * dst_size + dx] = val / 255.0f;
}

}  // anonymous namespace

extern "C" void cropResizeGPU(
    const uint8_t* src, int src_pitch, int src_w, int src_h,
    float* dst, int dst_size,
    float cx, float cy, float w, float h,
    float context_factor,
    cudaStream_t stream)
{
    // 正方形搜索区域: s = sqrt((w*ctx)*(h*ctx)), 避免宽高比畸变
    float s = sqrtf((w * context_factor) * (h * context_factor));
    float roi_w = s;
    float roi_h = s;
    float roi_x = cx - roi_w * 0.5f;
    float roi_y = cy - roi_h * 0.5f;

    dim3 block(16, 16);
    dim3 grid((dst_size + 15) / 16, (dst_size + 15) / 16);
    cropResizeBilinearKernel<<<grid, block, 0, stream>>>(
        src, src_pitch, src_w, src_h,
        dst, dst_size,
        roi_x, roi_y, roi_w, roi_h);
}

/**
 * @brief 3ch 版本: 灰度 → 3ch CHW float [3, dst_size, dst_size]
 * 将同一灰度值 repeat 到 R/G/B 三通道 (CHW layout)
 * 用于 3ch backbone 模型输入
 */
extern "C" void cropResizeGPU_3ch(
    const uint8_t* src, int src_pitch, int src_w, int src_h,
    float* dst, int dst_size,
    float cx, float cy, float w, float h,
    float context_factor,
    cudaStream_t stream)
{
    float s = sqrtf((w * context_factor) * (h * context_factor));
    float roi_w = s;
    float roi_h = s;
    float roi_x = cx - roi_w * 0.5f;
    float roi_y = cy - roi_h * 0.5f;

    int spatial = dst_size * dst_size;
    // Crop into first channel plane
    dim3 block(16, 16);
    dim3 grid((dst_size + 15) / 16, (dst_size + 15) / 16);
    cropResizeBilinearKernel<<<grid, block, 0, stream>>>(
        src, src_pitch, src_w, src_h,
        dst, dst_size,
        roi_x, roi_y, roi_w, roi_h);
    // Copy channel 0 → channel 1 and channel 2
    cudaMemcpyAsync(dst + spatial, dst, spatial * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dst + 2 * spatial, dst, spatial * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
}
