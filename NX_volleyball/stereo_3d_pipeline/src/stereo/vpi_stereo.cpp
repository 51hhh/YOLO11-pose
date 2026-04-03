/**
 * @file vpi_stereo.cpp
 * @brief VPI Stereo Disparity 实现
 *
 * CUDA backend 全帧视差 + 半分辨率降级策略。
 *
 * VPI 输出 S16 格式 (Q10.5 定点数):
 *   实际视差 = output_pixel / 32.0f
 */

#include "vpi_stereo.h"
#include "../utils/logger.h"

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/StereoDisparity.h>
#include <vpi/algo/Rescale.h>

namespace stereo3d {

VPIStereo::~VPIStereo() {
    if (stereoPayload_)     vpiPayloadDestroy(stereoPayload_);
    if (stereoPayloadHalf_) vpiPayloadDestroy(stereoPayloadHalf_);
    if (halfL_)    vpiImageDestroy(halfL_);
    if (halfR_)    vpiImageDestroy(halfR_);
    if (halfDisp_) vpiImageDestroy(halfDisp_);
    if (halfConf_) vpiImageDestroy(halfConf_);
}

bool VPIStereo::init(int maxDisparity, int windowSize, int width, int height) {
    maxDisparity_ = maxDisparity;
    windowSize_   = windowSize;
    width_  = width;
    height_ = height;

    VPIStatus err;

    // 1. 全帧视差 payload
    VPIStereoDisparityEstimatorCreationParams params;
    vpiInitStereoDisparityEstimatorCreationParams(&params);
    params.maxDisparity = maxDisparity;

    err = vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, width, height,
                                             VPI_IMAGE_FORMAT_U8, &params, &stereoPayload_);
    if (err != VPI_SUCCESS) {
        LOG_ERROR("Failed to create full-res stereo estimator: %d", err);
        return false;
    }

    // 2. 半分辨率 payload + 临时图像
    int halfW = width / 2;
    int halfH = height / 2;

    VPIStereoDisparityEstimatorCreationParams halfParams;
    vpiInitStereoDisparityEstimatorCreationParams(&halfParams);
    halfParams.maxDisparity = maxDisparity / 2;  // 半分辨率视差范围减半

    err = vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, halfW, halfH,
                                             VPI_IMAGE_FORMAT_U8, &halfParams, &stereoPayloadHalf_);
    if (err != VPI_SUCCESS) {
        LOG_ERROR("Failed to create half-res stereo estimator: %d", err);
        return false;
    }

    uint64_t cudaFlags = VPI_BACKEND_CUDA;
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_U8, cudaFlags, &halfL_);
    if (err != VPI_SUCCESS) { LOG_ERROR("halfL create failed"); return false; }
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_U8, cudaFlags, &halfR_);
    if (err != VPI_SUCCESS) { LOG_ERROR("halfR create failed"); return false; }
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_S16, cudaFlags, &halfDisp_);
    if (err != VPI_SUCCESS) { LOG_ERROR("halfDisp create failed"); return false; }
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_U16, cudaFlags, &halfConf_);
    if (err != VPI_SUCCESS) { LOG_ERROR("halfConf create failed"); return false; }

    LOG_INFO("VPI Stereo initialized: %dx%d, maxDisp=%d, win=%d",
             width, height, maxDisparity, windowSize);
    LOG_INFO("  Half-res: %dx%d, maxDisp=%d", halfW, halfH, maxDisparity / 2);

    return true;
}

void VPIStereo::compute(VPIStream stream,
                        VPIImage rectL, VPIImage rectR,
                        VPIImage disparity, VPIImage confidence)
{
    VPIStereoDisparityEstimatorParams submitParams;
    vpiInitStereoDisparityEstimatorParams(&submitParams);
    submitParams.windowSize   = windowSize_;
    submitParams.maxDisparity = maxDisparity_;

    vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA,
                                      stereoPayload_, rectL, rectR,
                                      disparity, confidence, &submitParams);
}

void VPIStereo::computeHalfRes(VPIStream stream,
                                VPIImage rectL, VPIImage rectR,
                                VPIImage disparity, VPIImage confidence)
{
    // 1. 降采样到半分辨率
    vpiSubmitRescale(stream, VPI_BACKEND_CUDA, rectL, halfL_,
                     VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiSubmitRescale(stream, VPI_BACKEND_CUDA, rectR, halfR_,
                     VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);

    // 2. 半分辨率视差计算
    VPIStereoDisparityEstimatorParams halfSubmitParams;
    vpiInitStereoDisparityEstimatorParams(&halfSubmitParams);
    halfSubmitParams.windowSize   = windowSize_;
    halfSubmitParams.maxDisparity = maxDisparity_ / 2;

    vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA,
                                      stereoPayloadHalf_, halfL_, halfR_,
                                      halfDisp_, halfConf_, &halfSubmitParams);

    // 3. 上采样视差图回原始分辨率
    // 注意: 视差值需要 ×2 (因为降了一半分辨率)
    // VPI Rescale 只做尺寸缩放，视差数值由 S16 Q10.5 格式自动继承
    // 后续使用时需要手动 ×2 补偿
    vpiSubmitRescale(stream, VPI_BACKEND_CUDA, halfDisp_, disparity,
                     VPI_INTERP_NEAREST, VPI_BORDER_ZERO, 0);
    vpiSubmitRescale(stream, VPI_BACKEND_CUDA, halfConf_, confidence,
                     VPI_INTERP_NEAREST, VPI_BORDER_ZERO, 0);
}

}  // namespace stereo3d
