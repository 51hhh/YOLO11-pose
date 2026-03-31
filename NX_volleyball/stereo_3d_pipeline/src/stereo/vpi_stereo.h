/**
 * @file vpi_stereo.h
 * @brief VPI Stereo Disparity 硬件加速视差计算
 *
 * 使用 VPI 的 StereoDisparityEstimator:
 *   - GPU (CUDA) backend: 全帧高质量视差
 *   - PVA backend: 更快但窗口尺寸限制
 *
 * 输出: S16 视差图 (Q8.8 定点, 实际视差 = value / 256.0)
 */

#ifndef STEREO_3D_PIPELINE_VPI_STEREO_H_
#define STEREO_3D_PIPELINE_VPI_STEREO_H_

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/StereoDisparity.h>

namespace stereo3d {

class VPIStereo {
public:
    VPIStereo() = default;
    ~VPIStereo();

    VPIStereo(const VPIStereo&) = delete;
    VPIStereo& operator=(const VPIStereo&) = delete;

    /**
     * @brief 初始化视差计算器
     * @param maxDisparity 最大视差 (32/64/128/256)
     * @param windowSize 匹配窗口大小 (奇数, 推荐 5)
     * @param width 图像宽度
     * @param height 图像高度
     * @return true 初始化成功
     */
    bool init(int maxDisparity, int windowSize, int width, int height);

    /**
     * @brief 全帧视差计算 (GPU CUDA backend)
     * @param stream VPI Stream (CUDA backend)
     * @param rectL 校正后左图
     * @param rectR 校正后右图
     * @param disparity 视差图输出 (S16, Q8.8)
     * @param confidence 置信度图输出 (U16)
     */
    void compute(VPIStream stream,
                 VPIImage rectL, VPIImage rectR,
                 VPIImage disparity, VPIImage confidence);

    /**
     * @brief 半分辨率视差 (策略 B: 降采样 → 视差 → 还原)
     *
     * 内部自动降采样到 width/2 x height/2,
     * 视差值 ×2 还原到原始分辨率尺度。
     */
    void computeHalfRes(VPIStream stream,
                        VPIImage rectL, VPIImage rectR,
                        VPIImage disparity, VPIImage confidence);

    int getMaxDisparity() const { return maxDisparity_; }

private:
    VPIPayload stereoPayload_ = nullptr;  ///< Stereo 算法 payload
    VPIPayload stereoPayloadHalf_ = nullptr;  ///< 半分辨率 payload

    // 半分辨率临时图像
    VPIImage halfL_    = nullptr;
    VPIImage halfR_    = nullptr;
    VPIImage halfDisp_ = nullptr;
    VPIImage halfConf_ = nullptr;

    int maxDisparity_ = 128;
    int windowSize_   = 5;
    int width_  = 0;
    int height_ = 0;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_VPI_STEREO_H_
