/**
 * @file yolo_preprocessor.cu
 * @brief YOLO预处理CUDA加速实现 (优化版 + Bayer去马赛克)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ==================== Bayer RG8 去马赛克 + Resize + 归一化 ====================
// 一次性完成: Bayer->RGB去马赛克 + Bilinear Resize + 归一化到[0,1] + HWC->CHW
__global__ void preprocessBayerRGKernel(const unsigned char* __restrict__ bayer, 
                                         float* __restrict__ dst, 
                                         int src_w, int src_h, int dst_w, int dst_h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= dst_w || y >= dst_h) return;
  
  // 计算在源Bayer图中的位置
  float scale_x = (float)src_w / dst_w;
  float scale_y = (float)src_h / dst_h;
  
  float src_x = x * scale_x;
  float src_y = y * scale_y;
  
  int x0 = (int)src_x;
  int y0 = (int)src_y;
  
  // 确保在偶数位置 (Bayer图2x2块对齐)
  x0 = (x0 >> 1) << 1;  // 向下取偶数
  y0 = (y0 >> 1) << 1;
  
  int x1 = min(x0 + 1, src_w - 1);
  int y1 = min(y0 + 1, src_h - 1);
  
  // Bayer RG8 模式 (RGGB):
  //   R  G1
  //   G2 B
  int idx_r  = y0 * src_w + x0;      // (y0, x0) = R
  int idx_g1 = y0 * src_w + x1;      // (y0, x1) = G1
  int idx_g2 = y1 * src_w + x0;      // (y1, x0) = G2
  int idx_b  = y1 * src_w + x1;      // (y1, x1) = B
  
  // 去马赛克: 提取RGB值
  float r = (float)bayer[idx_r];
  float g = ((float)bayer[idx_g1] + (float)bayer[idx_g2]) * 0.5f;  // 两个G平均
  float b = (float)bayer[idx_b];
  
  // 归一化到 [0, 1]
  r *= (1.0f / 255.0f);
  g *= (1.0f / 255.0f);
  b *= (1.0f / 255.0f);
  
  // 输出到CHW格式 (YOLO格式)
  int dst_idx = y * dst_w + x;
  int plane_size = dst_h * dst_w;
  
  dst[0 * plane_size + dst_idx] = r;  // R通道
  dst[1 * plane_size + dst_idx] = g;  // G通道
  dst[2 * plane_size + dst_idx] = b;  // B通道
}

// ==================== BGR格式预处理 (原有功能) ====================
// GPU加速预处理 CUDA kernel (优化版: 使用共享内存和更大的block)
__global__ void preprocessKernel(const unsigned char* __restrict__ src, float* __restrict__ dst, 
                                  int src_w, int src_h, int dst_w, int dst_h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= dst_w || y >= dst_h) return;
  
  // 预计算缩放因子 (编译期优化)
  float scale_x = (float)src_w / dst_w;
  float scale_y = (float)src_h / dst_h;
  
  // Bilinear resize 坐标计算
  float src_x = x * scale_x;
  float src_y = y * scale_y;
  
  int x0 = (int)src_x;
  int y0 = (int)src_y;
  int x1 = min(x0 + 1, src_w - 1);
  int y1 = min(y0 + 1, src_h - 1);
  
  float dx = src_x - x0;
  float dy = src_y - y0;
  
  // 预计算权重
  float w00 = (1.0f - dx) * (1.0f - dy);
  float w01 = dx * (1.0f - dy);
  float w10 = (1.0f - dx) * dy;
  float w11 = dx * dy;
  
  // 预计算源地址
  int idx00 = (y0 * src_w + x0) * 3;
  int idx01 = (y0 * src_w + x1) * 3;
  int idx10 = (y1 * src_w + x0) * 3;
  int idx11 = (y1 * src_w + x1) * 3;
  
  // 输出位置 (CHW格式)
  int dst_idx = y * dst_w + x;
  int plane_size = dst_h * dst_w;
  
  // 双线性插值 + BGR->RGB转换 + 归一化 (循环展开)
  #pragma unroll
  for (int c = 0; c < 3; ++c) {
    int src_c = 2 - c;  // BGR -> RGB
    
    float v = w00 * src[idx00 + src_c] +
              w01 * src[idx01 + src_c] +
              w10 * src[idx10 + src_c] +
              w11 * src[idx11 + src_c];
    
    dst[c * plane_size + dst_idx] = v * (1.0f / 255.0f);
  }
}

// ==================== C接口 ====================
// BGR格式预处理（原接口，保持兼容）
extern "C" void launchPreprocessKernel(const unsigned char* src, float* dst,
                                        int src_w, int src_h, int dst_w, int dst_h,
                                        cudaStream_t stream) {
  // ✅ 优化: 使用 32x32 block (1024线程/block, 适合Orin NX)
  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, 
            (dst_h + block.y - 1) / block.y);
  
  preprocessKernel<<<grid, block, 0, stream>>>(src, dst, src_w, src_h, dst_w, dst_h);
}

// Bayer RG8去马赛克预处理（新接口）
extern "C" void launchPreprocessBayerRGKernel(const unsigned char* bayer, float* dst,
                                               int src_w, int src_h, int dst_w, int dst_h,
                                               cudaStream_t stream) {
  // ✅ 优化: 使用 32x32 block
  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, 
            (dst_h + block.y - 1) / block.y);
  
  preprocessBayerRGKernel<<<grid, block, 0, stream>>>(bayer, dst, src_w, src_h, dst_w, dst_h);
}

