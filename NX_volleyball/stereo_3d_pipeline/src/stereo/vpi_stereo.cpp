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

namespace {

bool checkVpiStereo(VPIStatus status, const char* operation) {
    if (status == VPI_SUCCESS) return true;
    LOG_ERROR("[VPI Stereo] %s failed: status=%d", operation, (int)status);
    return false;
}

}  // namespace

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
    if (!checkVpiStereo(
            vpiInitStereoDisparityEstimatorCreationParams(&params),
            "initialize full-resolution creation parameters")) {
        return false;
    }
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
    if (!checkVpiStereo(
            vpiInitStereoDisparityEstimatorCreationParams(&halfParams),
            "initialize half-resolution creation parameters")) {
        return false;
    }
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

bool VPIStereo::compute(VPIStream stream,
                        VPIImage rectL, VPIImage rectR,
                        VPIImage disparity, VPIImage confidence)
{
    VPIStereoDisparityEstimatorParams submitParams;
    if (!checkVpiStereo(vpiInitStereoDisparityEstimatorParams(&submitParams),
                        "initialize full-resolution submit parameters")) {
        return false;
    }
    submitParams.windowSize   = windowSize_;
    submitParams.maxDisparity = maxDisparity_;

    return checkVpiStereo(
        vpiSubmitStereoDisparityEstimator(
            stream, VPI_BACKEND_CUDA, stereoPayload_, rectL, rectR,
            disparity, confidence, &submitParams),
        "submit full-resolution disparity");
}

bool VPIStereo::computeHalfRes(VPIStream stream,
                               VPIImage rectL, VPIImage rectR,
                               VPIImage disparity, VPIImage confidence)
{
    // 1. 降采样到半分辨率
    if (!checkVpiStereo(
            vpiSubmitRescale(stream, VPI_BACKEND_CUDA, rectL, halfL_,
                             VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0),
            "submit left half-resolution rescale") ||
        !checkVpiStereo(
            vpiSubmitRescale(stream, VPI_BACKEND_CUDA, rectR, halfR_,
                             VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0),
            "submit right half-resolution rescale")) {
        return false;
    }

    // 2. 半分辨率视差计算
    VPIStereoDisparityEstimatorParams halfSubmitParams;
    if (!checkVpiStereo(
            vpiInitStereoDisparityEstimatorParams(&halfSubmitParams),
            "initialize half-resolution submit parameters")) {
        return false;
    }
    halfSubmitParams.windowSize   = windowSize_;
    halfSubmitParams.maxDisparity = maxDisparity_ / 2;

    if (!checkVpiStereo(
            vpiSubmitStereoDisparityEstimator(
                stream, VPI_BACKEND_CUDA, stereoPayloadHalf_, halfL_, halfR_,
                halfDisp_, halfConf_, &halfSubmitParams),
            "submit half-resolution disparity")) {
        return false;
    }

    // 3. 上采样视差图回原始分辨率
    // 注意: 视差值需要 ×2 (因为降了一半分辨率)
    // VPI Rescale 只做尺寸缩放，视差数值由 S16 Q10.5 格式自动继承
    // 后续使用时需要手动 ×2 补偿
    if (!checkVpiStereo(
            vpiSubmitRescale(stream, VPI_BACKEND_CUDA, halfDisp_, disparity,
                             VPI_INTERP_NEAREST, VPI_BORDER_ZERO, 0),
            "submit disparity upsample") ||
        !checkVpiStereo(
            vpiSubmitRescale(stream, VPI_BACKEND_CUDA, halfConf_, confidence,
                             VPI_INTERP_NEAREST, VPI_BORDER_ZERO, 0),
            "submit confidence upsample")) {
        return false;
    }
    return true;
}

}  // namespace stereo3d
