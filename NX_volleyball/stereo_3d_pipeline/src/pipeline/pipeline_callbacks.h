/**
 * @file pipeline_callbacks.h
 * @brief Pipeline public callback type declarations.
 */

#ifndef STEREO_3D_PIPELINE_PIPELINE_CALLBACKS_H_
#define STEREO_3D_PIPELINE_PIPELINE_CALLBACKS_H_

#include "frame_slot.h"

#include <functional>
#include <vector>

namespace stereo3d {

/**
 * @brief 结果回调
 */
using ResultCallback = std::function<void(
    int frame_id,
    const std::vector<Object3D>& results,
    const FrameMetadata& metadata)>;

/**
 * @brief 帧回调视图。
 *
 * 回调同步执行, 这些 VPIImage 和 vector 引用只在回调期间有效。
 */
struct FrameCallbackData {
    int frame_id;
    VPIImage rect_gray_left;
    VPIImage rect_gray_right;
    VPIImage rect_bgr_left;
    VPIImage rect_bgr_right;
    VPIImage raw_left;
    VPIImage raw_right;
    const std::vector<Detection>& detections_left;
    const std::vector<Detection>& detections_right;
    const std::vector<Object3D>& results;
    FrameMetadata metadata;
    float fps;
};

using FrameCallback = std::function<void(const FrameCallbackData& frame)>;

/**
 * @brief 诊断回调 (深度图 + 检测框 + 3D结果)
 */
using DiagnosticCallback = std::function<void(
    int frame_id, const float* depth_gpu, int depth_pitch,
    int depth_w, int depth_h,
    const std::vector<Detection>& detections,
    const std::vector<Object3D>& results)>;

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PIPELINE_CALLBACKS_H_
