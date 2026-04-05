/**
 * @file vpi_rectifier.cpp
 * @brief VPI Remap 硬件加速校正实现
 *
 * 将 OpenCV undistortRectifyMap 生成的 LUT 导入 VPI WarpMap,
 * 通过 VIC/CUDA backend 执行异步 Remap。
 * VIC backend 不占用 GPU SM, 推荐用于最大化推理吞吐量。
 */

#include "vpi_rectifier.h"
#include "../calibration/stereo_calibration.h"
#include "../utils/logger.h"

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/Remap.h>
#include <vpi/WarpMap.h>
#include <opencv2/opencv.hpp>
#include <cstring>

namespace stereo3d {

VPIRectifier::~VPIRectifier() {
    if (remapL_) vpiPayloadDestroy(remapL_);
    if (remapR_) vpiPayloadDestroy(remapR_);
    vpiWarpMapFreeData(&warpMapL_);
    vpiWarpMapFreeData(&warpMapR_);
}

bool VPIRectifier::init(const StereoCalibration& calib, int width, int height,
                        uint64_t backend) {
    width_   = width;
    height_  = height;
    backend_ = backend;

    const char* backendName = (backend & VPI_BACKEND_VIC) ? "VIC" : "CUDA";

    // 1. 用 OpenCV 生成 Remap LUT
    cv::Mat map1L, map2L, map1R, map2R;
    calib.buildRemapMaps(map1L, map2L, map1R, map2R, width, height);

    // 2. 初始化 VPI WarpMap
    VPIStatus err;
    auto initWarpMap = [&](VPIWarpMap& wm, const cv::Mat& mapX, const cv::Mat& mapY,
                           VPIPayload& payload, const char* name) -> bool {
        memset(&wm, 0, sizeof(wm));
        wm.grid.numHorizRegions  = 1;
        wm.grid.numVertRegions   = 1;
        wm.grid.regionWidth[0]   = width;
        wm.grid.regionHeight[0]  = height;
        wm.grid.horizInterval[0] = 1;
        wm.grid.vertInterval[0]  = 1;

        err = vpiWarpMapAllocData(&wm);
        if (err != VPI_SUCCESS) {
            LOG_ERROR("vpiWarpMapAllocData failed for %s", name);
            return false;
        }

        // 将 OpenCV 的 map1 (x) 和 map2 (y) 写入 VPI WarpMap
        err = vpiWarpMapGenerateIdentity(&wm);
        if (err != VPI_SUCCESS) {
            LOG_ERROR("vpiWarpMapGenerateIdentity failed for %s", name);
            return false;
        }

        // 逐像素写入映射关系
        for (int y = 0; y < height; ++y) {
            VPIKeypointF32* row = reinterpret_cast<VPIKeypointF32*>(
                reinterpret_cast<uint8_t*>(wm.keypoints) + y * wm.pitchBytes);
            const float* mx = mapX.ptr<float>(y);
            const float* my = mapY.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                row[x].x = mx[x];
                row[x].y = my[x];
            }
        }

        // 创建 Remap Payload (VIC 或 CUDA backend)
        err = vpiCreateRemap(backend, &wm, &payload);
        if (err != VPI_SUCCESS) {
            LOG_ERROR("vpiCreateRemap (%s) failed for %s", backendName, name);
            return false;
        }

        LOG_INFO("VPI Remap %s initialized (%dx%d, backend=%s)", name, width, height, backendName);
        return true;
    };

    if (!initWarpMap(warpMapL_, map1L, map2L, remapL_, "Left")) return false;
    if (!initWarpMap(warpMapR_, map1R, map2R, remapR_, "Right")) return false;

    return true;
}

void VPIRectifier::submit(VPIStream stream,
                          VPIImage rawL, VPIImage rawR,
                          VPIImage rectL, VPIImage rectR)
{
    // VIC/CUDA backend 异步提交 Remap
    vpiSubmitRemap(stream, backend_, remapL_, rawL, rectL,
                   VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiSubmitRemap(stream, backend_, remapR_, rawR, rectR,
                   VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
}

void VPIRectifier::submitBGR(VPIStream stream,
                              VPIImage bgrL, VPIImage bgrR,
                              VPIImage rectBGR_L, VPIImage rectBGR_R)
{
    // BGR 三通道 remap 使用与灰度相同的 LUT (坐标映射与通道数无关)
    // 必须 CUDA backend: payload 在 init() 中以 backend_ 创建
    VPIStatus stL = vpiSubmitRemap(stream, backend_, remapL_, bgrL, rectBGR_L,
                   VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    VPIStatus stR = vpiSubmitRemap(stream, backend_, remapR_, bgrR, rectBGR_R,
                   VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    if (stL != VPI_SUCCESS || stR != VPI_SUCCESS) {
        LOG_ERROR("submitBGR remap failed: L=%d R=%d", (int)stL, (int)stR);
    }
}

}  // namespace stereo3d
