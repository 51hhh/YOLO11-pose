#pragma once

#include "calibration/stereo_calibration.h"
#include "stereo_depth_viewer_depth_utils.h"
#include "utils/logger.h"

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/Remap.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>
#include <vpi/WarpMap.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>

// ============================================================
//  VPI 资源封装
// ============================================================
struct VPIResources {
    VPIStream stream = nullptr;

    // 校正
    VPIPayload remapL = nullptr;
    VPIPayload remapR = nullptr;

    // 原始/校正 图像
    VPIImage rawL = nullptr, rawR = nullptr;
    VPIImage rectL = nullptr, rectR = nullptr;

    // 全帧视差
    VPIPayload stereoFull = nullptr;
    VPIImage dispFull = nullptr, confFull = nullptr;

    // 半帧视差
    VPIPayload stereoHalf = nullptr;
    VPIImage halfL = nullptr, halfR = nullptr;
    VPIImage dispHalf = nullptr, confHalf = nullptr;
    VPIImage dispUpscaled = nullptr, confUpscaled = nullptr;

    int width = 0, height = 0;
    int maxDisp = 256;
    int winSize = 5;

    void destroy() {
        auto safeDestroy = [](VPIImage& img) { if (img) { vpiImageDestroy(img); img = nullptr; } };
        auto safePayload = [](VPIPayload& p) { if (p) { vpiPayloadDestroy(p); p = nullptr; } };

        safeDestroy(rawL); safeDestroy(rawR);
        safeDestroy(rectL); safeDestroy(rectR);
        safeDestroy(dispFull); safeDestroy(confFull);
        safeDestroy(halfL); safeDestroy(halfR);
        safeDestroy(dispHalf); safeDestroy(confHalf);
        safeDestroy(dispUpscaled); safeDestroy(confUpscaled);

        safePayload(remapL); safePayload(remapR);
        safePayload(stereoFull); safePayload(stereoHalf);

        if (stream) { vpiStreamDestroy(stream); stream = nullptr; }
    }
};

static bool initVPI(VPIResources& vpi, const stereo3d::StereoCalibration& calib,
                    int width, int height, int maxDisp, int winSize)
{
    vpi.width = width;
    vpi.height = height;
    vpi.maxDisp = maxDisp;
    vpi.winSize = winSize;

    VPIStatus err;
    // CPU flag 必须包含, 否则 vpiImageLockData HOST_PITCH_LINEAR 会失败
    uint64_t flags = VPI_BACKEND_CUDA | VPI_BACKEND_CPU;

    // Stream
    err = vpiStreamCreate(VPI_BACKEND_CUDA, &vpi.stream);
    if (err != VPI_SUCCESS) { LOG_ERROR("vpiStreamCreate failed: %d", err); return false; }

    // 原始图像 (U8, Bayer 输入) — 需要 host 上传
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, flags, &vpi.rawL);
    if (err != VPI_SUCCESS) { LOG_ERROR("rawL create failed"); return false; }
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, flags, &vpi.rawR);
    if (err != VPI_SUCCESS) { LOG_ERROR("rawR create failed"); return false; }

    // 校正后图像 (U8) — 需要 host 下载给 OpenCV 算法
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, flags, &vpi.rectL);
    if (err != VPI_SUCCESS) { LOG_ERROR("rectL create failed"); return false; }
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, flags, &vpi.rectR);
    if (err != VPI_SUCCESS) { LOG_ERROR("rectR create failed"); return false; }

    // ---- 视差输出 (全帧) — 需要 host 下载 ----
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_S16, flags, &vpi.dispFull);
    if (err != VPI_SUCCESS) { LOG_ERROR("dispFull create failed"); return false; }
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U16, flags, &vpi.confFull);
    if (err != VPI_SUCCESS) { LOG_ERROR("confFull create failed"); return false; }

    // ---- 全帧 Stereo Payload ----
    {
        VPIStereoDisparityEstimatorCreationParams params;
        vpiInitStereoDisparityEstimatorCreationParams(&params);
        params.maxDisparity = maxDisp;
        err = vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, width, height,
                                                 VPI_IMAGE_FORMAT_U8, &params, &vpi.stereoFull);
        if (err != VPI_SUCCESS) { LOG_ERROR("stereoFull create failed: %d", err); return false; }
    }

    // ---- 半帧 (中间结果不需要 host 访问, 但 upscaled 需要下载) ----
    int halfW = width / 2, halfH = height / 2;
    uint64_t gpuOnly = VPI_BACKEND_CUDA;
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_U8, gpuOnly, &vpi.halfL);
    if (err != VPI_SUCCESS) { LOG_ERROR("halfL create failed"); return false; }
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_U8, gpuOnly, &vpi.halfR);
    if (err != VPI_SUCCESS) { LOG_ERROR("halfR create failed"); return false; }
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_S16, gpuOnly, &vpi.dispHalf);
    if (err != VPI_SUCCESS) { LOG_ERROR("dispHalf create failed"); return false; }
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_U16, gpuOnly, &vpi.confHalf);
    if (err != VPI_SUCCESS) { LOG_ERROR("confHalf create failed"); return false; }
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_S16, flags, &vpi.dispUpscaled);
    if (err != VPI_SUCCESS) { LOG_ERROR("dispUpscaled create failed"); return false; }
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U16, flags, &vpi.confUpscaled);
    if (err != VPI_SUCCESS) { LOG_ERROR("confUpscaled create failed"); return false; }

    {
        VPIStereoDisparityEstimatorCreationParams hparams;
        vpiInitStereoDisparityEstimatorCreationParams(&hparams);
        hparams.maxDisparity = maxDisp / 2;
        err = vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, halfW, halfH,
                                                 VPI_IMAGE_FORMAT_U8, &hparams, &vpi.stereoHalf);
        if (err != VPI_SUCCESS) { LOG_ERROR("stereoHalf create failed: %d", err); return false; }
    }

    // ---- 校正 Remap ----
    cv::Mat map1L, map2L, map1R, map2R;
    calib.buildRemapMaps(map1L, map2L, map1R, map2R, width, height);

    auto buildRemap = [&](const cv::Mat& mapX, const cv::Mat& mapY,
                          VPIPayload& payload, const char* name) -> bool {
        VPIWarpMap wm;
        memset(&wm, 0, sizeof(wm));
        wm.grid.numHorizRegions  = 1;
        wm.grid.numVertRegions   = 1;
        wm.grid.regionWidth[0]   = width;
        wm.grid.regionHeight[0]  = height;
        wm.grid.horizInterval[0] = 1;
        wm.grid.vertInterval[0]  = 1;

        VPIStatus e = vpiWarpMapAllocData(&wm);
        if (e != VPI_SUCCESS) { LOG_ERROR("WarpMap alloc %s", name); return false; }

        vpiWarpMapGenerateIdentity(&wm);

        for (int y = 0; y < height; ++y) {
            auto* row = reinterpret_cast<VPIKeypointF32*>(
                reinterpret_cast<uint8_t*>(wm.keypoints) + y * wm.pitchBytes);
            const float* mx = mapX.ptr<float>(y);
            const float* my = mapY.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                row[x].x = mx[x];
                row[x].y = my[x];
            }
        }

        e = vpiCreateRemap(VPI_BACKEND_CUDA, &wm, &payload);
        vpiWarpMapFreeData(&wm);
        if (e != VPI_SUCCESS) { LOG_ERROR("CreateRemap %s", name); return false; }
        return true;
    };

    if (!buildRemap(map1L, map2L, vpi.remapL, "Left")) return false;
    if (!buildRemap(map1R, map2R, vpi.remapR, "Right")) return false;

    LOG_INFO("VPI resources initialized: %dx%d, maxDisp=%d", width, height, maxDisp);
    return true;
}

// ============================================================
//  VPI 辅助: 上传 cv::Mat → VPIImage
// ============================================================
static bool uploadToVPI(VPIImage vpiImg, const cv::Mat& mat) {
    VPIImageData data;
    memset(&data, 0, sizeof(data));
    VPIStatus err = vpiImageLockData(vpiImg, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data);
    if (err != VPI_SUCCESS) {
        LOG_ERROR("uploadToVPI: vpiImageLockData failed: %d", err);
        return false;
    }
    int h = mat.rows;
    int w = mat.cols;
    int dstPitch = data.buffer.pitch.planes[0].pitchBytes;
    uint8_t* dst = reinterpret_cast<uint8_t*>(data.buffer.pitch.planes[0].data);
    for (int y = 0; y < h; ++y) {
        memcpy(dst + y * dstPitch, mat.ptr(y), w * mat.elemSize());
    }
    vpiImageUnlock(vpiImg);
    return true;
}

// ============================================================
//  VPI 辅助: 下载 VPIImage → cv::Mat
// ============================================================
static cv::Mat downloadFromVPI(VPIImage vpiImg, int cvType) {
    VPIImageData data;
    memset(&data, 0, sizeof(data));
    VPIStatus err = vpiImageLockData(vpiImg, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data);
    if (err != VPI_SUCCESS) {
        LOG_ERROR("downloadFromVPI: vpiImageLockData failed: %d", err);
        return cv::Mat();
    }
    int w = data.buffer.pitch.planes[0].width;
    int h = data.buffer.pitch.planes[0].height;
    int pitch = data.buffer.pitch.planes[0].pitchBytes;
    uint8_t* src = reinterpret_cast<uint8_t*>(data.buffer.pitch.planes[0].data);

    cv::Mat mat(h, w, cvType);
    for (int y = 0; y < h; ++y) {
        memcpy(mat.ptr(y), src + y * pitch, w * mat.elemSize());
    }
    vpiImageUnlock(vpiImg);
    return mat;
}

// ============================================================
//  VPI SGM 提交参数 (CUDA backend 优化)
// ============================================================
static void fillVPIParams(VPIStereoDisparityEstimatorParams& p, int maxDisp) {
    vpiInitStereoDisparityEstimatorParams(&p);
    p.maxDisparity = maxDisp;
    // CUDA backend: 9×7 census 固定窗口
    // SGM P1/P2: P1小→允许倾斜面; P2大→抑制深度突变
    p.p1 = 5;                       // 增大P1(默认3): 更好的倾斜面连续性
    p.p2 = 96;                      // 增大P2(之前72): 更强平滑约束
    p.confidenceThreshold = 49152;  // VPI 内部过滤 (越高越严格), 配合 applyConfidenceMask 二次过滤
    p.uniqueness = 0.97f;           // 唯一性比 (越高越严格), 实际由 confMask 控制有效像素数
    p.confidenceType = VPI_STEREO_CONFIDENCE_ABSOLUTE;
}

// ============================================================
//  VPI 视差 → 去除低置信度像素
// ============================================================
static void applyConfidenceMask(cv::Mat& dispF, const cv::Mat& confU16,
                                int threshold = 2000) {
    for (int y = 0; y < dispF.rows; ++y) {
        float* dp = dispF.ptr<float>(y);
        const uint16_t* cp = confU16.ptr<uint16_t>(y);
        for (int x = 0; x < dispF.cols; ++x) {
            if (cp[x] < (uint16_t)threshold) {
                dp[x] = 0.0f;  // 低置信度 → 无效
            }
        }
    }
}

// ============================================================
//  VPI 视差计算 (全帧)
// ============================================================
static cv::Mat computeVPICudaFull(VPIResources& vpi, cv::Mat* outConf = nullptr) {
    VPIStereoDisparityEstimatorParams p;
    fillVPIParams(p, vpi.maxDisp);

    vpiSubmitStereoDisparityEstimator(vpi.stream, VPI_BACKEND_CUDA,
        vpi.stereoFull, vpi.rectL, vpi.rectR,
        vpi.dispFull, vpi.confFull, &p);
    vpiStreamSync(vpi.stream);

    cv::Mat dispS16 = downloadFromVPI(vpi.dispFull, CV_16S);
    cv::Mat dispF;
    dispS16.convertTo(dispF, CV_32F, 1.0 / 32.0);  // Q10.5 (NVIDIA 官方格式)

    // 置信度过滤 (threshold=2000 经验证给出 64.7% 有效像素)
    cv::Mat confU16 = downloadFromVPI(vpi.confFull, CV_16U);
    if (!confU16.empty()) {
        applyConfidenceMask(dispF, confU16);
        if (outConf) *outConf = confU16;
    }

    return dispF;
}

// ============================================================
//  VPI 视差计算 (半帧)
// ============================================================
static cv::Mat computeVPICudaHalf(VPIResources& vpi) {
    // 降采样
    vpiSubmitRescale(vpi.stream, VPI_BACKEND_CUDA, vpi.rectL, vpi.halfL,
                     VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiSubmitRescale(vpi.stream, VPI_BACKEND_CUDA, vpi.rectR, vpi.halfR,
                     VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);

    VPIStereoDisparityEstimatorParams p;
    fillVPIParams(p, vpi.maxDisp / 2);

    vpiSubmitStereoDisparityEstimator(vpi.stream, VPI_BACKEND_CUDA,
        vpi.stereoHalf, vpi.halfL, vpi.halfR,
        vpi.dispHalf, vpi.confHalf, &p);

    // 上采样回原始分辨率
    vpiSubmitRescale(vpi.stream, VPI_BACKEND_CUDA, vpi.dispHalf, vpi.dispUpscaled,
                     VPI_INTERP_NEAREST, VPI_BORDER_ZERO, 0);
    vpiStreamSync(vpi.stream);

    cv::Mat dispS16 = downloadFromVPI(vpi.dispUpscaled, CV_16S);
    cv::Mat dispF;
    // 半分辨率视差 ×2 补偿, Q10.5 格式
    dispS16.convertTo(dispF, CV_32F, 2.0 / 32.0);  // Q10.5 × 2

    // 置信度过滤
    vpiSubmitRescale(vpi.stream, VPI_BACKEND_CUDA, vpi.confHalf, vpi.confUpscaled,
                     VPI_INTERP_NEAREST, VPI_BORDER_ZERO, 0);
    vpiStreamSync(vpi.stream);
    cv::Mat confU16 = downloadFromVPI(vpi.confUpscaled, CV_16U);
    if (!confU16.empty()) {
        applyConfidenceMask(dispF, confU16);
    }

    return dispF;
}
