/**
 * @file coordinate_3d.h
 * @brief BBox + 视差 → 3D 坐标 融合
 *
 * 对每个检测框:
 *   1. CUDA kernel 提取框内视差值的直方图峰值 (鲁棒中值)
 *   2. Z = focal * baseline / d_peak
 *   3. X = (cx - cx0) * Z / focal
 *   4. Y = (cy - cy0) * Z / focal
 */

#ifndef STEREO_3D_PIPELINE_COORDINATE_3D_H_
#define STEREO_3D_PIPELINE_COORDINATE_3D_H_

#include "../pipeline/frame_slot.h"
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

namespace stereo3d {

class Coordinate3D {
public:
    Coordinate3D() = default;
    ~Coordinate3D();

    Coordinate3D(const Coordinate3D&) = delete;
    Coordinate3D& operator=(const Coordinate3D&) = delete;

    /**
     * @brief 初始化
     * @param P1 左目投影矩阵 (3x4)
     * @param baseline 基线距离 (m)
     * @param minDepth 最小有效深度 (m)
     * @param maxDepth 最大有效深度 (m)
     */
    void init(const cv::Mat& P1, float baseline, float minDepth, float maxDepth);

    /**
     * @brief 对单个检测框计算 3D 坐标
     * @param det 检测结果 (2D BBox)
     * @param disparityGPU GPU 视差图指针 (S16, Q10.5)
     * @param dispPitch 视差图行跨度 (bytes)
     * @param imgWidth 图像宽度
     * @param imgHeight 图像高度
     * @param stream CUDA Stream
     * @return 3D 坐标 (confidence=0 表示无效)
     */
    Object3D compute(const Detection& det,
                     const int16_t* disparityGPU, int dispPitch,
                     int imgWidth, int imgHeight,
                     cudaStream_t stream,
                     float disparityScale = 1.0f);

    /**
     * @brief 批量计算
     * @param disparityScale 视差缩放係数 (半分辨率时传 2.0f)
     */
    std::vector<Object3D> computeBatch(const std::vector<Detection>& dets,
                                       const int16_t* disparityGPU, int dispPitch,
                                       int imgWidth, int imgHeight,
                                       cudaStream_t stream,
                                       float disparityScale = 1.0f);

private:
    float focal_    = 0.0f;    ///< 焦距 (pixels)
    float baseline_ = 0.0f;    ///< 基线 (m)
    float cx_       = 0.0f;    ///< 主点 x
    float cy_       = 0.0f;    ///< 主点 y
    float minDepth_ = 0.3f;
    float maxDepth_ = 15.0f;

    // GPU 辅助缓冲
    float* depthResults_device_ = nullptr;  ///< GPU 上的深度输出
    float* depthResults_host_   = nullptr;  ///< CPU 上的深度输出 (pinned)
    int*   bboxes_device_       = nullptr;  ///< GPU 上的 BBox 数据
    int maxBoxes_ = 32;                     ///< 最大检测框数

    bool allocateGPUBuffers();
    void freeGPUBuffers();
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_COORDINATE_3D_H_
