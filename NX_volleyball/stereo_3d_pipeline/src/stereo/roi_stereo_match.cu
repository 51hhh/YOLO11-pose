/**
 * @file roi_stereo_match.cu
 * @brief CUDA Kernel: ROI 多点 SAD 立体匹配 + 三角测距
 *
 * 取代全帧 VPI SGM 视差计算，仅在检测框内采样若干点做块匹配:
 *   1. 在 BBox 内布 5x5 网格采样点
 *   2. 每个采样点做 NxN SAD 块匹配 (沿极线水平搜索)
 *   3. 亚像素抛物线拟合精化
 *   4. 取中值视差作为目标深度
 *   5. 三角测距: Z = f*B/d, X = (cx-cx0)*Z/f, Y = (cy-cy0)*Z/f
 *
 * 性能: 10 个检测目标 < 0.5ms (NX GPU)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cfloat>

// ==================== 常量 ====================
#define MAX_SAMPLE_POINTS  25   // 5x5 network
#define MAX_GRID_SIZE       5
#define MAX_DETECTIONS     32
#define BLOCK_THREADS     128   // threads per block
#define THREADS_PER_POINT   5   // 协作视差搜索: 每点5线程

/**
 * @brief ROI 多点 SAD 立体匹配 + Median + 三角测距
 *
 * 一个 CUDA Block 处理一个检测目标。
 * threadIdx.x < numSamplePoints 的线程各负责一个采样点的匹配。
 *
 * @param leftImg       校正后左图 GPU 指针 (U8)
 * @param leftPitch     左图行字节跨度
 * @param rightImg      校正后右图 GPU 指针 (U8)
 * @param rightPitch    右图行字节跨度
 * @param imgWidth      图像宽度
 * @param imgHeight     图像高度
 * @param bboxes        检测框 [x1,y1,x2,y2] * numBoxes (int)
 * @param detCx         检测框中心 x (float, 用于 3D 投影)
 * @param detCy         检测框中心 y (float)
 * @param numBoxes      检测框数量
 * @param results       输出: [X, Y, Z, disparity, confidence] * numBoxes
 * @param maxDisparity  最大搜索视差
 * @param patchRadius   匹配块半径 (patch = (2r+1)^2)
 * @param focal         焦距 (pixels)
 * @param baseline      基线 (meters)
 * @param cx0           主点 x
 * @param cy0           主点 y
 * @param minDepth      最小有效深度 (m)
 * @param maxDepth      最大有效深度 (m)
 */
__global__ void roiMultiPointMatchKernel(
    const uint8_t* __restrict__ leftImg,   int leftPitch,
    const uint8_t* __restrict__ rightImg,  int rightPitch,
    int imgWidth, int imgHeight,
    const int* __restrict__ bboxes,
    const float* __restrict__ detCx,
    const float* __restrict__ detCy,
    int numBoxes,
    float* __restrict__ results,   // 5 floats per detection
    int maxDisparity,
    int patchRadius,
    float focal, float baseline,
    float cx0, float cy0,
    float minDepth, float maxDepth)
{
    int boxIdx = blockIdx.x;
    if (boxIdx >= numBoxes) return;

    // ---- 读取 BBox ----
    int x1 = bboxes[boxIdx * 4 + 0];
    int y1 = bboxes[boxIdx * 4 + 1];
    int x2 = bboxes[boxIdx * 4 + 2];
    int y2 = bboxes[boxIdx * 4 + 3];

    int boxW = x2 - x1;
    int boxH = y2 - y1;
    int patchSize = 2 * patchRadius + 1;

    // BBox 太小无法匹配
    if (boxW < patchSize || boxH < patchSize) {
        if (threadIdx.x == 0) {
            results[boxIdx * 5 + 0] = 0.0f;  // X
            results[boxIdx * 5 + 1] = 0.0f;  // Y
            results[boxIdx * 5 + 2] = 0.0f;  // Z
            results[boxIdx * 5 + 3] = 0.0f;  // disparity
            results[boxIdx * 5 + 4] = 0.0f;  // confidence
        }
        return;
    }

    // ---- 自适应网格大小 ----
    // 小目标用 3x3, 大目标用 5x5
    int gridSize;
    if (boxW < 30 || boxH < 30)
        gridSize = 3;
    else if (boxW < 60 || boxH < 60)
        gridSize = 4;
    else
        gridSize = 5;

    int numSamples = gridSize * gridSize;

    // ---- 共享内存: 协作视差搜索 + 采样点视差收集 ----
    __shared__ int   s_bestSAD [MAX_SAMPLE_POINTS][THREADS_PER_POINT];
    __shared__ int   s_bestDisp[MAX_SAMPLE_POINTS][THREADS_PER_POINT];
    __shared__ int   s_2ndSAD  [MAX_SAMPLE_POINTS][THREADS_PER_POINT];
    __shared__ float sampleDisp[MAX_SAMPLE_POINTS];
    __shared__ int   validCount;

    if (threadIdx.x == 0) validCount = 0;
    __syncthreads();

    // ---- 协作视差搜索: 每点 THREADS_PER_POINT 个线程分段搜索 ----
    int pointIdx = threadIdx.x / THREADS_PER_POINT;
    int subIdx   = threadIdx.x % THREADS_PER_POINT;

    // 采样点坐标 (同组线程计算相同值)
    int px = 0, py = 0, maxSearch = 0;

    if (pointIdx < numSamples) {
        int gx = pointIdx % gridSize;
        int gy = pointIdx / gridSize;

        int innerX1 = x1 + patchRadius;
        int innerY1 = y1 + patchRadius;
        int innerX2 = x2 - patchRadius;
        int innerY2 = y2 - patchRadius;

        if (gridSize > 1) {
            px = innerX1 + (innerX2 - innerX1) * gx / (gridSize - 1);
            py = innerY1 + (innerY2 - innerY1) * gy / (gridSize - 1);
        } else {
            px = (innerX1 + innerX2) / 2;
            py = (innerY1 + innerY2) / 2;
        }

        px = max(patchRadius, min(imgWidth - patchRadius - 1, px));
        py = max(patchRadius, min(imgHeight - patchRadius - 1, py));

        // ---- SAD 块匹配: 每线程搜索一段视差范围 ----
        maxSearch = min(maxDisparity, px - patchRadius);
        if (maxSearch < 1) maxSearch = 1;

        int rangeLen = maxSearch + 1;  // total disparities [0, maxSearch]
        int perThread = (rangeLen + THREADS_PER_POINT - 1) / THREADS_PER_POINT;
        int dStart = subIdx * perThread;
        int dEnd   = min(dStart + perThread, rangeLen);

        int bestDisp = 0;
        int bestSAD = 0x7FFFFFFF;
        int secondBestSAD = 0x7FFFFFFF;

        for (int d = dStart; d < dEnd; d++) {
            int sad = 0;
            for (int wy = -patchRadius; wy <= patchRadius; wy++) {
                const uint8_t* leftRow  = leftImg  + (py + wy) * leftPitch;
                const uint8_t* rightRow = rightImg + (py + wy) * rightPitch;
                for (int wx = -patchRadius; wx <= patchRadius; wx++) {
                    int lv = leftRow[px + wx];
                    int rv = rightRow[px + wx - d];
                    int diff = lv - rv;
                    sad += (diff >= 0) ? diff : -diff;
                }
            }
            if (sad < bestSAD) {
                secondBestSAD = bestSAD;
                bestSAD = sad;
                bestDisp = d;
            } else if (sad < secondBestSAD) {
                secondBestSAD = sad;
            }
        }

        s_bestSAD [pointIdx][subIdx] = bestSAD;
        s_bestDisp[pointIdx][subIdx] = bestDisp;
        s_2ndSAD  [pointIdx][subIdx] = secondBestSAD;
    }
    __syncthreads();

    // ---- Leader 线程归约 + 亚像素 + 有效性过滤 ----
    if (pointIdx < numSamples && subIdx == 0) {
        int gBestSAD  = s_bestSAD [pointIdx][0];
        int gBestDisp = s_bestDisp[pointIdx][0];
        int g2ndSAD   = s_2ndSAD  [pointIdx][0];

        for (int t = 1; t < THREADS_PER_POINT; t++) {
            int tSAD  = s_bestSAD [pointIdx][t];
            int tDisp = s_bestDisp[pointIdx][t];
            int t2nd  = s_2ndSAD  [pointIdx][t];

            if (tSAD < gBestSAD) {
                // 旧全局最优降级为 second-best 候选
                g2ndSAD  = min(g2ndSAD, min(gBestSAD, t2nd));
                gBestSAD  = tSAD;
                gBestDisp = tDisp;
            } else {
                g2ndSAD = min(g2ndSAD, tSAD);
            }
        }

        bool unique = (g2ndSAD - gBestSAD) > (gBestSAD / 4);

        // ---- 亚像素抛物线拟合 ----
        float subDisp = (float)gBestDisp;
        if (gBestDisp > 0 && gBestDisp < maxSearch) {
            int sadMinus = 0, sadPlus = 0;
            for (int wy = -patchRadius; wy <= patchRadius; wy++) {
                const uint8_t* leftRow  = leftImg  + (py + wy) * leftPitch;
                const uint8_t* rightRow = rightImg + (py + wy) * rightPitch;
                for (int wx = -patchRadius; wx <= patchRadius; wx++) {
                    int lv = leftRow[px + wx];
                    int rvM = rightRow[px + wx - (gBestDisp - 1)];
                    int rvP = rightRow[px + wx - (gBestDisp + 1)];
                    int diffM = lv - rvM;
                    int diffP = lv - rvP;
                    sadMinus += (diffM >= 0) ? diffM : -diffM;
                    sadPlus  += (diffP >= 0) ? diffP : -diffP;
                }
            }
            float denom = (float)(sadMinus - 2 * gBestSAD + sadPlus);
            if (denom > 0.1f) {
                float offset = (float)(sadMinus - sadPlus) / (2.0f * denom);
                offset = fmaxf(-0.5f, fminf(0.5f, offset));
                subDisp = (float)gBestDisp + offset;
            }
        }

        if (gBestDisp > 0 && unique && subDisp > 0.5f) {
            int idx = atomicAdd(&validCount, 1);
            sampleDisp[idx] = subDisp;
        }
    }
    __syncthreads();

    // ---- Thread 0: 中值 + 三角测距 ----
    if (threadIdx.x == 0) {
        int n = validCount;
        if (n < 2) {
            // 有效点不足, 无法可靠测距
            results[boxIdx * 5 + 0] = 0.0f;
            results[boxIdx * 5 + 1] = 0.0f;
            results[boxIdx * 5 + 2] = 0.0f;
            results[boxIdx * 5 + 3] = 0.0f;
            results[boxIdx * 5 + 4] = 0.0f;
            return;
        }

        // 简单排序 (n <= 25, 冒泡即可)
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (sampleDisp[j] > sampleDisp[j + 1]) {
                    float tmp = sampleDisp[j];
                    sampleDisp[j] = sampleDisp[j + 1];
                    sampleDisp[j + 1] = tmp;
                }
            }
        }

        // 中值 (排除离群值: 取 IQR 内的平均值)
        int q1 = n / 4;
        int q3 = (3 * n) / 4;
        if (q1 == q3) { q1 = 0; q3 = n - 1; }

        float sum = 0.0f;
        int count = 0;
        for (int i = q1; i <= q3; i++) {
            sum += sampleDisp[i];
            count++;
        }
        float medianDisp = (count > 0) ? (sum / (float)count) : 0.0f;

        // 三角测距
        float Z = 0.0f, X = 0.0f, Y = 0.0f;
        float conf = 0.0f;

        if (medianDisp > 0.5f) {
            Z = (focal * baseline) / medianDisp;

            if (Z >= minDepth && Z <= maxDepth) {
                float center_x = detCx[boxIdx];
                float center_y = detCy[boxIdx];
                X = (center_x - cx0) * Z / focal;
                Y = (center_y - cy0) * Z / focal;

                // 置信度: 有效点比例 * 一致性 (IQR 范围越窄越好)
                float validRatio = (float)n / (float)(gridSize * gridSize);
                float iqr = sampleDisp[q3] - sampleDisp[q1];
                float consistency = (iqr < 1.0f) ? 1.0f : 1.0f / iqr;
                conf = validRatio * fminf(1.0f, consistency);
            }
        }

        results[boxIdx * 5 + 0] = X;
        results[boxIdx * 5 + 1] = Y;
        results[boxIdx * 5 + 2] = Z;
        results[boxIdx * 5 + 3] = medianDisp;
        results[boxIdx * 5 + 4] = conf;
    }
}

// ==================== C 接口 ====================

extern "C" void launchROIMultiPointMatch(
    const uint8_t* leftImg,  int leftPitch,
    const uint8_t* rightImg, int rightPitch,
    int imgWidth, int imgHeight,
    const int* bboxes,
    const float* detCx, const float* detCy,
    int numBoxes,
    float* results,
    int maxDisparity, int patchRadius,
    float focal, float baseline,
    float cx0, float cy0,
    float minDepth, float maxDepth,
    cudaStream_t stream)
{
    if (numBoxes <= 0) return;
    int blocks = min(numBoxes, MAX_DETECTIONS);
    roiMultiPointMatchKernel<<<blocks, BLOCK_THREADS, 0, stream>>>(
        leftImg, leftPitch, rightImg, rightPitch,
        imgWidth, imgHeight,
        bboxes, detCx, detCy, numBoxes,
        results, maxDisparity, patchRadius,
        focal, baseline, cx0, cy0,
        minDepth, maxDepth);
}
