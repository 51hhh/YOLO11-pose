#pragma once
#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief CUDA kernel: 从 U8 灰度图像中裁剪 ROI 并 bilinear resize 到正方形目标尺寸
 *
 * 自动将 (cx,cy,w,h) + context_factor 转换为正方形搜索区域，
 * 避免非方形 ROI 导致的宽高比畸变。
 */
extern "C" void cropResizeGPU(
    const uint8_t* src, int src_pitch, int src_w, int src_h,
    float* dst, int dst_size,
    float cx, float cy, float box_w, float box_h,
    float context_factor, cudaStream_t stream);
