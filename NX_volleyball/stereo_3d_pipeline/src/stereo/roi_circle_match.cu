/**
 * @file roi_circle_match.cu
 * @brief CUDA Kernel: Sobel edge + circle fit stereo matching
 *
 * For textureless spherical objects (e.g., volleyball):
 *   1. Left image: Sobel gradient -> edge extraction -> Kasa circle fit
 *   2. Right image: narrow search band -> same pipeline -> circle fit
 *   3. Disparity = cx_left - cx_right -> triangulation
 *
 * Advantages over SAD grid matching:
 *   - Does NOT require surface texture
 *   - Uses ball contour (strong, stable feature)
 *   - Sub-pixel precision via statistical averaging of edge points
 *   - ~0.05ms per detection (faster than SAD grid)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

#define CIRCLE_MAX_EDGES    1024
#define CIRCLE_BLOCK_THREADS 128
#define CIRCLE_MAX_DETECTIONS 32

// ==================== Device helpers ====================

// Warp-level sum reduction
__device__ __forceinline__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Block-level sum reduction (128 threads = 4 warps)
__device__ double blockReduceSum(double val) {
    __shared__ double warpSums[4];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) warpSums[wid] = val;
    __syncthreads();

    // First warp reduces the 4 partial sums
    if (wid == 0) {
        val = (lane < 4) ? warpSums[lane] : 0.0;
        val = warpReduceSum(val);
    }
    return val;
}

// Cramer's rule for 3x3 system: A * x = b
__device__ bool solve3x3(
    double A00, double A01, double A02,
    double A10, double A11, double A12,
    double A20, double A21, double A22,
    double b0,  double b1,  double b2,
    double &x0, double &x1, double &x2)
{
    double det = A00 * (A11*A22 - A12*A21)
               - A01 * (A10*A22 - A12*A20)
               + A02 * (A10*A21 - A11*A20);
    if (fabs(det) < 1e-12) return false;
    double invDet = 1.0 / det;

    x0 = (b0*(A11*A22-A12*A21) - A01*(b1*A22-A12*b2) + A02*(b1*A21-A11*b2)) * invDet;
    x1 = (A00*(b1*A22-A12*b2) - b0*(A10*A22-A12*A20) + A02*(A10*b2-b1*A20)) * invDet;
    x2 = (A00*(A11*b2-b1*A21) - A01*(A10*b2-b1*A20) + b0*(A10*A21-A11*A20)) * invDet;
    return true;
}

// ==================== Main Kernel ====================

/**
 * @brief Circle-fit stereo matching kernel
 *
 * One block per detection. 128 threads cooperate on:
 *   Phase 1: Sobel + edge extraction (left ROI)
 *   Phase 2: Kasa weighted circle fit (left)
 *   Phase 3: Sobel + edge extraction (right search band)
 *   Phase 4: Constrained circle fit (right)
 *   Phase 5: Disparity + triangulation
 */
__global__ void roiCircleFitMatchKernel(
    const uint8_t* __restrict__ leftImg,   int leftPitch,
    const uint8_t* __restrict__ rightImg,  int rightPitch,
    int imgWidth, int imgHeight,
    const int*   __restrict__ bboxes,      // [x1,y1,x2,y2] * numBoxes
    const float* __restrict__ detCx,       // detection center x
    const float* __restrict__ detCy,       // detection center y
    int numBoxes,
    float* __restrict__ results,           // [X, Y, Z, disp, conf] * numBoxes
    int maxDisparity,
    float focal, float baseline,
    float cx0, float cy0,
    float minDepth, float maxDepth,
    float objectDiameter)                  // known ball diameter (m)
{
    int boxIdx = blockIdx.x;
    if (boxIdx >= numBoxes) return;
    int tid = threadIdx.x;

    // Read bbox
    int x1 = bboxes[boxIdx * 4 + 0];
    int y1 = bboxes[boxIdx * 4 + 1];
    int x2 = bboxes[boxIdx * 4 + 2];
    int y2 = bboxes[boxIdx * 4 + 3];
    int boxW = x2 - x1;
    int boxH = y2 - y1;

    // Too small for circle fit
    if (boxW < 15 || boxH < 15) {
        if (tid == 0) {
            results[boxIdx*5+0] = 0; results[boxIdx*5+1] = 0;
            results[boxIdx*5+2] = 0; results[boxIdx*5+3] = 0;
            results[boxIdx*5+4] = 0;
        }
        return;
    }

    // Shared memory for edge points
    __shared__ float edgeX[CIRCLE_MAX_EDGES];
    __shared__ float edgeY[CIRCLE_MAX_EDGES];
    __shared__ float edgeW[CIRCLE_MAX_EDGES];
    __shared__ int   edgeN;
    __shared__ float maxGrad;

    // ============ Phase 1: Left ROI Sobel + edge extraction ============

    // Pass 1a: find max gradient for adaptive threshold
    if (tid == 0) { maxGrad = 0.0f; edgeN = 0; }
    __syncthreads();

    int innerW = boxW - 2;  // skip 1-pixel border for Sobel
    int innerH = boxH - 2;
    int totalInner = innerW * innerH;
    float localMax = 0.0f;

    for (int i = tid; i < totalInner; i += CIRCLE_BLOCK_THREADS) {
        int lx = x1 + 1 + (i % innerW);
        int ly = y1 + 1 + (i / innerW);
        if (lx <= 0 || lx >= imgWidth - 1 || ly <= 0 || ly >= imgHeight - 1) continue;

        float gx = (float)leftImg[ly * leftPitch + lx + 1]
                  - (float)leftImg[ly * leftPitch + lx - 1];
        float gy = (float)leftImg[(ly+1) * leftPitch + lx]
                  - (float)leftImg[(ly-1) * leftPitch + lx];
        float mag = sqrtf(gx * gx + gy * gy);
        if (mag > localMax) localMax = mag;
    }

    // Reduce max across block via atomicMax (float→int reinterpret preserves order for positives)
    int localMaxInt = __float_as_int(localMax);
    atomicMax((int*)&maxGrad, localMaxInt);
    __syncthreads();

    float gradThresh = maxGrad * 0.25f;
    if (gradThresh < 10.0f) gradThresh = 10.0f;  // absolute minimum

    // Pass 1b: extract edges above threshold
    for (int i = tid; i < totalInner; i += CIRCLE_BLOCK_THREADS) {
        int lx = x1 + 1 + (i % innerW);
        int ly = y1 + 1 + (i / innerW);
        if (lx <= 0 || lx >= imgWidth - 1 || ly <= 0 || ly >= imgHeight - 1) continue;

        float gx = (float)leftImg[ly * leftPitch + lx + 1]
                  - (float)leftImg[ly * leftPitch + lx - 1];
        float gy = (float)leftImg[(ly+1) * leftPitch + lx]
                  - (float)leftImg[(ly-1) * leftPitch + lx];
        float mag = sqrtf(gx * gx + gy * gy);

        if (mag > gradThresh) {
            int idx = atomicAdd(&edgeN, 1);
            if (idx < CIRCLE_MAX_EDGES) {
                edgeX[idx] = (float)lx;
                edgeY[idx] = (float)ly;
                edgeW[idx] = mag;
            }
        }
    }
    __syncthreads();

    int nL = min(edgeN, CIRCLE_MAX_EDGES);

    // Need minimum edges for reliable fit
    if (nL < 15) {
        if (tid == 0) {
            results[boxIdx*5+0] = 0; results[boxIdx*5+1] = 0;
            results[boxIdx*5+2] = 0; results[boxIdx*5+3] = 0;
            results[boxIdx*5+4] = 0;
        }
        return;
    }

    // ============ Phase 2: Left circle fit (Kasa method) ============
    // Minimize: sum_i w_i * ((x_i - cx)^2 + (y_i - cy)^2 - r^2)^2
    // Linear form: x^2 + y^2 = a*x + b*y + c  where a=2cx, b=2cy, c=r^2-cx^2-cy^2

    double s_w = 0, s_wx = 0, s_wy = 0;
    double s_wxx = 0, s_wyy = 0, s_wxy = 0;
    double s_wxz = 0, s_wyz = 0, s_wz = 0;  // z = x^2 + y^2

    for (int i = tid; i < nL; i += CIRCLE_BLOCK_THREADS) {
        double w = edgeW[i];
        double ex = edgeX[i];
        double ey = edgeY[i];
        double ez = ex * ex + ey * ey;

        s_w   += w;
        s_wx  += w * ex;
        s_wy  += w * ey;
        s_wxx += w * ex * ex;
        s_wyy += w * ey * ey;
        s_wxy += w * ex * ey;
        s_wxz += w * ex * ez;
        s_wyz += w * ey * ez;
        s_wz  += w * ez;
    }

    // Block-level reduction of 9 accumulators
    s_w   = blockReduceSum(s_w);
    s_wx  = blockReduceSum(s_wx);
    s_wy  = blockReduceSum(s_wy);
    s_wxx = blockReduceSum(s_wxx);
    s_wyy = blockReduceSum(s_wyy);
    s_wxy = blockReduceSum(s_wxy);
    s_wxz = blockReduceSum(s_wxz);
    s_wyz = blockReduceSum(s_wyz);
    s_wz  = blockReduceSum(s_wz);

    __shared__ float cxL, cyL, rL;
    __shared__ int fitOkL;

    if (tid == 0) {
        fitOkL = 0;
        double a, b, c;
        if (solve3x3(s_wxx, s_wxy, s_wx,
                      s_wxy, s_wyy, s_wy,
                      s_wx,  s_wy,  s_w,
                      s_wxz, s_wyz, s_wz,
                      a, b, c)) {
            double cx_fit = a / 2.0;
            double cy_fit = b / 2.0;
            double r2 = c + cx_fit * cx_fit + cy_fit * cy_fit;
            if (r2 > 0) {
                cxL = (float)cx_fit;
                cyL = (float)cy_fit;
                rL  = (float)sqrt(r2);
                // Sanity: circle center should be near bbox center
                float bboxCx = (x1 + x2) * 0.5f;
                float bboxCy = (y1 + y2) * 0.5f;
                float maxR = fmaxf((float)boxW, (float)boxH) * 0.6f;
                if (fabsf(cxL - bboxCx) < maxR && fabsf(cyL - bboxCy) < maxR
                    && rL > 5.0f && rL < maxR) {
                    fitOkL = 1;
                }
            }
        }
    }
    __syncthreads();

    if (!fitOkL) {
        if (tid == 0) {
            results[boxIdx*5+0] = 0; results[boxIdx*5+1] = 0;
            results[boxIdx*5+2] = 0; results[boxIdx*5+3] = 0;
            results[boxIdx*5+4] = 0;
        }
        return;
    }

    // ============ Phase 3: Right image search band Sobel + edges ============

    // Estimate disparity from mono depth (ball diameter prior)
    float Z_mono = focal * objectDiameter / (float)boxW;
    float d_est = (Z_mono > 0.3f) ? (focal * baseline / Z_mono) : ((float)maxDisparity * 0.5f);
    int searchMargin = 40;  // +-40px around expected position

    // Right search band: centered at (cxL - d_est, cyL), width = 2*searchMargin + boxW*0.3
    int searchW = 2 * searchMargin + (int)(boxW * 0.3f);
    int rSearchX1 = max(1, (int)(cxL - d_est) - searchW / 2);
    int rSearchX2 = min(imgWidth - 2, rSearchX1 + searchW);
    int rSearchY1 = max(1, y1);
    int rSearchY2 = min(imgHeight - 2, y2);

    // Reuse shared memory for right edges
    if (tid == 0) { edgeN = 0; maxGrad = 0.0f; }
    __syncthreads();

    int rInnerW = rSearchX2 - rSearchX1 - 1;
    int rInnerH = rSearchY2 - rSearchY1 - 1;
    int rTotal = rInnerW * rInnerH;
    if (rTotal <= 0) {
        if (tid == 0) {
            results[boxIdx*5+0] = 0; results[boxIdx*5+1] = 0;
            results[boxIdx*5+2] = 0; results[boxIdx*5+3] = 0;
            results[boxIdx*5+4] = 0;
        }
        return;
    }

    // Pass 3a: find max gradient in right search band
    localMax = 0.0f;
    for (int i = tid; i < rTotal; i += CIRCLE_BLOCK_THREADS) {
        int rx = rSearchX1 + 1 + (i % rInnerW);
        int ry = rSearchY1 + 1 + (i / rInnerW);
        if (rx <= 0 || rx >= imgWidth - 1 || ry <= 0 || ry >= imgHeight - 1) continue;

        float gx = (float)rightImg[ry * rightPitch + rx + 1]
                  - (float)rightImg[ry * rightPitch + rx - 1];
        float gy = (float)rightImg[(ry+1) * rightPitch + rx]
                  - (float)rightImg[(ry-1) * rightPitch + rx];
        float mag = sqrtf(gx * gx + gy * gy);
        if (mag > localMax) localMax = mag;
    }
    localMaxInt = __float_as_int(localMax);
    atomicMax((int*)&maxGrad, localMaxInt);
    __syncthreads();

    float rGradThresh = maxGrad * 0.25f;
    if (rGradThresh < 10.0f) rGradThresh = 10.0f;

    // Pass 3b: extract right edges
    for (int i = tid; i < rTotal; i += CIRCLE_BLOCK_THREADS) {
        int rx = rSearchX1 + 1 + (i % rInnerW);
        int ry = rSearchY1 + 1 + (i / rInnerW);
        if (rx <= 0 || rx >= imgWidth - 1 || ry <= 0 || ry >= imgHeight - 1) continue;

        float gx = (float)rightImg[ry * rightPitch + rx + 1]
                  - (float)rightImg[ry * rightPitch + rx - 1];
        float gy = (float)rightImg[(ry+1) * rightPitch + rx]
                  - (float)rightImg[(ry-1) * rightPitch + rx];
        float mag = sqrtf(gx * gx + gy * gy);

        if (mag > rGradThresh) {
            int idx = atomicAdd(&edgeN, 1);
            if (idx < CIRCLE_MAX_EDGES) {
                edgeX[idx] = (float)rx;
                edgeY[idx] = (float)ry;
                edgeW[idx] = mag;
            }
        }
    }
    __syncthreads();

    int nR = min(edgeN, CIRCLE_MAX_EDGES);

    if (nR < 15) {
        if (tid == 0) {
            results[boxIdx*5+0] = 0; results[boxIdx*5+1] = 0;
            results[boxIdx*5+2] = 0; results[boxIdx*5+3] = 0;
            results[boxIdx*5+4] = 0;
        }
        return;
    }

    // ============ Phase 4: Right circle fit (radius-constrained) ============

    // Filter edge points: keep only those within ring [0.7*rL, 1.3*rL] from expected center
    // Expected right center: (cxL - d_est, cyL)
    float expCxR = cxL - d_est;
    float expCyR = cyL;
    float rMin = rL * 0.7f;
    float rMax = rL * 1.3f;

    s_w = 0; s_wx = 0; s_wy = 0;
    s_wxx = 0; s_wyy = 0; s_wxy = 0;
    s_wxz = 0; s_wyz = 0; s_wz = 0;

    int validR = 0;
    for (int i = tid; i < nR; i += CIRCLE_BLOCK_THREADS) {
        float ex = edgeX[i];
        float ey = edgeY[i];
        float dx = ex - expCxR;
        float dy = ey - expCyR;
        float dist = sqrtf(dx * dx + dy * dy);

        // Keep points within expected radius range
        if (dist >= rMin && dist <= rMax) {
            double w = edgeW[i];
            double ez = (double)ex * ex + (double)ey * ey;
            s_w   += w;
            s_wx  += w * ex;
            s_wy  += w * ey;
            s_wxx += w * ex * ex;
            s_wyy += w * ey * ey;
            s_wxy += w * ex * ey;
            s_wxz += w * ex * ez;
            s_wyz += w * ey * ez;
            s_wz  += w * ez;
            validR++;
        }
    }

    s_w   = blockReduceSum(s_w);
    s_wx  = blockReduceSum(s_wx);
    s_wy  = blockReduceSum(s_wy);
    s_wxx = blockReduceSum(s_wxx);
    s_wyy = blockReduceSum(s_wyy);
    s_wxy = blockReduceSum(s_wxy);
    s_wxz = blockReduceSum(s_wxz);
    s_wyz = blockReduceSum(s_wyz);
    s_wz  = blockReduceSum(s_wz);

    // Also reduce validR count
    double dValidR = (double)validR;
    dValidR = blockReduceSum(dValidR);

    // ============ Phase 5: Compute disparity + depth ============
    if (tid == 0) {
        int totalValidR = (int)dValidR;
        float X = 0, Y = 0, Z = 0, disp = 0, conf = 0;

        if (totalValidR >= 10) {
            double a, b, c;
            if (solve3x3(s_wxx, s_wxy, s_wx,
                          s_wxy, s_wyy, s_wy,
                          s_wx,  s_wy,  s_w,
                          s_wxz, s_wyz, s_wz,
                          a, b, c)) {
                float cxR = (float)(a / 2.0);
                float cyR = (float)(b / 2.0);
                float rR  = (float)sqrt(fmax(0.0, c + (double)cxR*cxR + (double)cyR*cyR));

                // Validate: radius consistency
                float rRatio = rR / rL;
                if (rRatio > 0.8f && rRatio < 1.2f && rR > 5.0f) {
                    // Compute disparity
                    disp = cxL - cxR;

                    if (disp > 0.5f && disp < (float)maxDisparity) {
                        Z = focal * baseline / disp;

                        if (Z >= minDepth && Z <= maxDepth) {
                            float center_x = detCx[boxIdx];
                            float center_y = detCy[boxIdx];
                            X = (center_x - cx0) * Z / focal;
                            Y = (center_y - cy0) * Z / focal;

                            // Confidence based on:
                            //   - edge point counts (both images)
                            //   - radius consistency
                            //   - fit quality
                            float edgeScore  = fminf(1.0f, (float)nL / 100.0f)
                                             * fminf(1.0f, (float)totalValidR / 80.0f);
                            float radiusConf = 1.0f - fabsf(rRatio - 1.0f) * 5.0f;  // penalize deviation
                            radiusConf = fmaxf(0.0f, fminf(1.0f, radiusConf));
                            conf = edgeScore * radiusConf;
                        }
                    }
                }
            }
        }

        results[boxIdx * 5 + 0] = X;
        results[boxIdx * 5 + 1] = Y;
        results[boxIdx * 5 + 2] = Z;
        results[boxIdx * 5 + 3] = disp;
        results[boxIdx * 5 + 4] = conf;
    }
}

// ==================== C interface ====================

extern "C" void launchROICircleFitMatch(
    const uint8_t* leftImg,  int leftPitch,
    const uint8_t* rightImg, int rightPitch,
    int imgWidth, int imgHeight,
    const int* bboxes,
    const float* detCx, const float* detCy,
    int numBoxes,
    float* results,
    int maxDisparity,
    float focal, float baseline,
    float cx0, float cy0,
    float minDepth, float maxDepth,
    float objectDiameter,
    cudaStream_t stream)
{
    if (numBoxes <= 0) return;
    int gridDim = min(numBoxes, CIRCLE_MAX_DETECTIONS);
    roiCircleFitMatchKernel<<<gridDim, CIRCLE_BLOCK_THREADS, 0, stream>>>(
        leftImg, leftPitch, rightImg, rightPitch,
        imgWidth, imgHeight,
        bboxes, detCx, detCy, numBoxes,
        results,
        maxDisparity,
        focal, baseline, cx0, cy0,
        minDepth, maxDepth,
        objectDiameter);
}
