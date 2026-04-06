/**
 * @file roi_stereo_matcher.h
 * @brief ROI 多点立体匹配器 — 取代全帧视差计算
 *
 * 核心思路:
 *   全帧 VPI SGM (~10ms) 只用了检测框内的像素 → 大量浪费。
 *   改为: 仅在检测框 ROI 内采样 5x5 网格点做 SAD 块匹配,
 *   取中值视差 → 三角测距 → 3D 坐标。
 *
 * 性能: 10 个目标 < 0.5ms (NX GPU), 比全帧 SGM 快 20 倍。
 */

#ifndef STEREO_3D_PIPELINE_ROI_STEREO_MATCHER_H_
#define STEREO_3D_PIPELINE_ROI_STEREO_MATCHER_H_

#include "../pipeline/frame_slot.h"
#include <cuda_runtime.h>
#include <vector>

namespace stereo3d {

struct ROIMatchConfig {
    int   maxDisparity = 256;     ///< 最大搜索视差 (pixels)
    int   patchRadius  = 5;       ///< 匹配块半径 (patch = 11x11)
    float minDepth     = 0.3f;    ///< 最小有效深度 (m)
    float maxDepth     = 15.0f;   ///< 最大有效深度 (m)
    float objectDiameter = 0.215f; ///< 目标直径 (m), 用于圆拟合搜索范围估计
    bool  useCircleFit = true;    ///< 启用圆拟合匹配 (适合光滑球体)
};

class ROIStereoMatcher {
public:
    ROIStereoMatcher() = default;
    ~ROIStereoMatcher();

    ROIStereoMatcher(const ROIStereoMatcher&) = delete;
    ROIStereoMatcher& operator=(const ROIStereoMatcher&) = delete;

    /**
     * @brief 初始化
     * @param focal    焦距 (pixels, 从 P1[0,0])
     * @param baseline 基线 (meters)
     * @param cx       主点 x (从 P1[0,2])
     * @param cy       主点 y (从 P1[1,2])
     * @param config   匹配参数
     */
    void init(float focal, float baseline, float cx, float cy,
              const ROIMatchConfig& config = ROIMatchConfig());

    /**
     * @brief 对检测结果做 ROI 立体匹配 + 3D 测距
     *
     * @param leftGPU    校正后左图 GPU 指针 (U8)
     * @param leftPitch  左图行字节跨度
     * @param rightGPU   校正后右图 GPU 指针 (U8)
     * @param rightPitch 右图行字节跨度
     * @param imgWidth   图像宽度
     * @param imgHeight  图像高度
     * @param detections YOLO 检测结果
     * @param stream     CUDA Stream
     * @return 3D 定位结果 (仅包含有效深度的目标)
     */
    std::vector<Object3D> match(
        const uint8_t* leftGPU,  int leftPitch,
        const uint8_t* rightGPU, int rightPitch,
        int imgWidth, int imgHeight,
        const std::vector<Detection>& detections,
        cudaStream_t stream);

    float getFocal()    const { return focal_; }
    float getBaseline() const { return baseline_; }

private:
    float focal_    = 0.0f;
    float baseline_ = 0.0f;
    float cx_       = 0.0f;
    float cy_       = 0.0f;
    ROIMatchConfig config_;

    // GPU 缓冲区
    static constexpr int kMaxBoxes = 32;
    int*   bboxes_device_  = nullptr;   ///< [x1,y1,x2,y2] * N
    float* detCx_device_   = nullptr;   ///< 检测中心 x
    float* detCy_device_   = nullptr;   ///< 检测中心 y
    float* results_device_ = nullptr;   ///< [X,Y,Z,disp,conf] * N
    float* results_host_   = nullptr;   ///< pinned host mirror

    bool allocateBuffers();
    void freeBuffers();
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_STEREO_MATCHER_H_
