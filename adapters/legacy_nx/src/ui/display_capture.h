#pragma once

#include "display_types.h"
#include "../utils/logger.h"
#include "../utils/vpi_image_copy.h"

#include <opencv2/opencv.hpp>

#include <utility>

inline std::pair<int, int> computeDisplaySize(
    const stereo3d::PipelineConfig& cfg)
{
    int disp_w = cfg.rect_width;
    int disp_h = cfg.rect_height;
    if (cfg.camera.width > 0 && cfg.camera.height > 0) {
        disp_w = cfg.rect_height * cfg.camera.width / cfg.camera.height;
    }
    return {disp_w, disp_h};
}

inline bool captureDisplayFrame(
    const stereo3d::FrameCallbackData& frame_data,
    bool use_bgr,
    cv::Mat& frame)
{
    frame.release();
    if (use_bgr) {
        if (stereo3d::copyVpiBgr8ToHost(
                frame_data.rect_bgr_left, frame, "display BGR copy")) {
            return true;
        }
    }

    cv::Mat gray;
    if (!stereo3d::copyVpiGray8ToHost(
            frame_data.rect_gray_left, gray, "display gray copy")) {
        return false;
    }
    try {
        cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
    } catch (const cv::Exception& error) {
        LOG_ERROR("display gray-to-BGR conversion failed: %s", error.what());
        frame.release();
        return false;
    }
    return !frame.empty();
}
