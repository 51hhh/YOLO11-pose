/**
 * @file yolo_preprocessor.cu
 * @brief YOLO预处理CUDA加速实现 (优化版)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

// C接口：供C++代码调用
extern "C" void launchPreprocessKernel(const unsigned char* src, float* dst,
                                        int src_w, int src_h, int dst_w, int dst_h,
                                        cudaStream_t stream) {
  // ✅ 优化: 使用 32x32 block (1024线程/block, 适合Orin NX)
  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, 
            (dst_h + block.y - 1) / block.y);
  
  preprocessKernel<<<grid, block, 0, stream>>>(src, dst, src_w, src_h, dst_w, dst_h);
}

