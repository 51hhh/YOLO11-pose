/**
 * @file vpi_rectifier.h
 * @brief VPI Remap 硬件加速校正 (VIC/CUDA backend)
 *
 * 使用 VPI 的 vpiSubmitRemap 替代 OpenCV cv2.remap,
 * VPI Remap 在 VPI 3.x 上支持 CUDA/VIC backend。
 * VIC (Video Image Compositor) 是 Jetson 专用硬件, 不占用 GPU SM。
 */

#ifndef STEREO_3D_PIPELINE_VPI_RECTIFIER_H_
#define STEREO_3D_PIPELINE_VPI_RECTIFIER_H_

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/Remap.h>
#include <vpi/WarpMap.h>
#include <cstdint>

namespace stereo3d {

// 前向声明
class StereoCalibration;

class VPIRectifier {
public:
    VPIRectifier() = default;
    ~VPIRectifier();

    VPIRectifier(const VPIRectifier&) = delete;
    VPIRectifier& operator=(const VPIRectifier&) = delete;

    /**
     * @brief 初始化 VPI Remap
     * @param calib 标定参数 (用于生成 LUT)
     * @param width 图像宽度
     * @param height 图像高度
     * @param backend VPI backend flags (VPI_BACKEND_VIC 或 VPI_BACKEND_CUDA)
     * @return true 初始化成功
     */
    bool init(const StereoCalibration& calib, int width, int height,
              uint64_t backend = VPI_BACKEND_VIC);

    /**
     * @brief 异步提交校正任务
     * @param stream VPI Stream
     * @param rawL 左原始图
     * @param rawR 右原始图
     * @param rectL 校正后左图 (输出)
     * @param rectR 校正后右图 (输出)
     */
    void submit(VPIStream stream,
                VPIImage rawL, VPIImage rawR,
                VPIImage rectL, VPIImage rectR);

private:
    VPIPayload remapL_ = nullptr;   ///< 左目 Remap payload
    VPIPayload remapR_ = nullptr;   ///< 右目 Remap payload
    VPIWarpMap warpMapL_ = {};      ///< 左目 Warp Map (LUT)
    VPIWarpMap warpMapR_ = {};      ///< 右目 Warp Map (LUT)
    uint64_t backend_ = VPI_BACKEND_VIC;  ///< Remap 执行后端
    int width_  = 0;
    int height_ = 0;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_VPI_RECTIFIER_H_
