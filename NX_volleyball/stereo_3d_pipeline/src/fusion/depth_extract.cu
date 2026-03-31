/**
 * @file depth_extract.cu
 * @brief CUDA Kernel: 从检测框区域提取深度 (直方图峰值法)
 *
 * 方案文档设计:
 *   每个 CUDA Block 处理一个 BBox
 *   使用共享内存直方图统计框内视差值
 *   取直方图峰值作为代表性视差 (比中值更鲁棒)
 *
 * VPI 视差图格式: S16, Q8.8 定点 → 实际视差 = pixel / 256.0
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 直方图 bin 数量 (视差范围 0~255)
#define HIST_BINS 256
// 每个 block 的线程数
#define THREADS_PER_BLOCK 256

/**
 * @brief 直方图法提取 BBox 内峰值视差
 *
 * @param disparity S16 视差图 (Q8.8 格式)
 * @param dispPitch 视差图行跨度 (bytes)
 * @param imgWidth 图像宽度
 * @param bboxes [x1, y1, x2, y2] * numBoxes
 * @param numBoxes 检测框数量
 * @param depths 输出: 每个框的峰值视差 (像素, 已解码)
 */
__global__ void depthExtractKernel(
    const int16_t* __restrict__ disparity,
    int dispPitch,
    int imgWidth,  // 保留: BBox 越界检查可用
    const int* __restrict__ bboxes,
    int numBoxes,
    float* __restrict__ depths)
{
    int boxIdx = blockIdx.x;
    if (boxIdx >= numBoxes) return;

    // 读取 BBox
    int x1 = bboxes[boxIdx * 4 + 0];
    int y1 = bboxes[boxIdx * 4 + 1];
    int x2 = bboxes[boxIdx * 4 + 2];
    int y2 = bboxes[boxIdx * 4 + 3];

    int boxW = x2 - x1;
    int boxH = y2 - y1;
    if (boxW <= 0 || boxH <= 0) {
        if (threadIdx.x == 0) depths[boxIdx] = 0.0f;
        return;
    }

    // 共享内存直方图
    __shared__ int histogram[HIST_BINS];

    // 1. 清零直方图
    if (threadIdx.x < HIST_BINS) {
        histogram[threadIdx.x] = 0;
    }
    __syncthreads();

    // 2. 统计框内视差值
    // dispPitch 是字节跨度, S16 每元素 2 字节
    int pixelsPerRow = dispPitch / 2;
    int totalPixels = boxW * boxH;

    for (int i = threadIdx.x; i < totalPixels; i += blockDim.x) {
        int dy = i / boxW;
        int dx = i % boxW;
        int px = x1 + dx;
        int py = y1 + dy;

        int16_t raw_disp = disparity[py * pixelsPerRow + px];

        // Q8.8 → 实际视差 (整数部分用于直方图 bin)
        // raw_disp > 0 才有效
        if (raw_disp > 0) {
            int disp_int = raw_disp >> 8;  // 取整数部分
            if (disp_int >= 0 && disp_int < HIST_BINS) {
                atomicAdd(&histogram[disp_int], 1);
            }
        }
    }
    __syncthreads();

    // 3. 找直方图峰值
    // 使用并行规约找最大值
    __shared__ int maxCount;
    __shared__ int maxBin;

    if (threadIdx.x == 0) {
        maxCount = 0;
        maxBin = 0;
    }
    __syncthreads();

    // 简单方法: 每个线程检查一个 bin
    if (threadIdx.x < HIST_BINS) {
        int count = histogram[threadIdx.x];
        if (count > 0) {
            atomicMax(&maxCount, count);
        }
    }
    __syncthreads();

    // 找到具有最大 count 的 bin
    if (threadIdx.x < HIST_BINS) {
        if (histogram[threadIdx.x] == maxCount && maxCount > 0) {
            atomicMax(&maxBin, threadIdx.x);  // 取最大视差 (保守估计, 近处)
        }
    }
    __syncthreads();

    // 4. 输出: 使用峰值 bin 附近的加权平均提高精度
    if (threadIdx.x == 0) {
        if (maxCount == 0) {
            depths[boxIdx] = 0.0f;
        } else {
            // 在峰值 bin ±2 范围内做加权平均
            int lo = max(0, maxBin - 2);
            int hi = min(HIST_BINS - 1, maxBin + 2);
            float sum = 0.0f;
            float weight = 0.0f;
            for (int b = lo; b <= hi; ++b) {
                sum += (float)b * (float)histogram[b];
                weight += (float)histogram[b];
            }
            float disp_peak = (weight > 0) ? (sum / weight) : 0.0f;

            // 加上小数部分的大致估计 (Q8.8 亚像素)
            // 精确做法可以在 host 端用更复杂的插值
            depths[boxIdx] = disp_peak + 0.5f;  // 约 0.5 pixel 亚像素补偿
        }
    }
}

// ==================== C 接口 ====================

extern "C" void launchDepthExtractKernel(
    const int16_t* disparity, int dispPitch, int imgWidth,
    const int* bboxes, int numBoxes,
    float* depths,
    cudaStream_t stream)
{
    if (numBoxes <= 0) return;

    depthExtractKernel<<<numBoxes, THREADS_PER_BLOCK, 0, stream>>>(
        disparity, dispPitch, imgWidth,
        bboxes, numBoxes, depths);
}
