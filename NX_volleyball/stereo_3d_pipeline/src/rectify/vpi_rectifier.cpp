/**
 * @file vpi_rectifier.cpp
 * @brief VPI Remap 硬件加速校正实现
 *
 * 将 OpenCV undistortRectifyMap 生成的 LUT 导入 VPI WarpMap,
 * 然后通过 PVA backend 执行异步 Remap。
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

bool VPIRectifier::init(const StereoCalibration& calib, int width, int height) {
    width_  = width;
    height_ = height;

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

        // 创建 Remap Payload (指定 PVA backend 优先, 回退 CUDA)
        err = vpiCreateRemap(VPI_BACKEND_PVA | VPI_BACKEND_CUDA, &wm, &payload);
        if (err != VPI_SUCCESS) {
            LOG_ERROR("vpiCreateRemap failed for %s", name);
            return false;
        }

        LOG_INFO("VPI Remap %s initialized (%dx%d)", name, width, height);
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
    // PVA backend 异步提交 Remap
    // VPI_INTERP_LINEAR = 双线性插值
    // VPI_BORDER_ZERO   = 边界填零
    vpiSubmitRemap(stream, VPI_BACKEND_PVA, remapL_, rawL, rectL,
                   VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiSubmitRemap(stream, VPI_BACKEND_PVA, remapR_, rawR, rectR,
                   VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
}

}  // namespace stereo3d
