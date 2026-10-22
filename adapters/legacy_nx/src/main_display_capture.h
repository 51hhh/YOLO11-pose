#pragma once

#include "main_display_types.h"
#include "utils/logger.h"
#include "utils/vpi_image_copy.h"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <exception>
#include <utility>

inline void buildVisualizationColorRemap(
    const stereo3d::PipelineConfig& cfg,
    cv::Mat& vis_map1,
    cv::Mat& vis_map2,
    bool& has_color_remap)
{
    has_color_remap = false;
    try {
        cv::FileStorage fs(cfg.calibration_file, cv::FileStorage::READ);
        if (!fs.isOpened()) return;

        cv::Mat K1, D1, R1, P1;
        fs["camera_matrix_left"] >> K1;
        fs["distortion_coefficients_left"] >> D1;
        fs["rectification_left"] >> R1;
        fs["projection_left"] >> P1;

        int cal_w = 0;
        int cal_h = 0;
        fs["image_width"] >> cal_w;
        fs["image_height"] >> cal_h;
        if (cal_w > 0 && cal_h > 0 &&
            (cfg.rect_width != cal_w || cfg.rect_height != cal_h)) {
            if (static_cast<int64_t>(cfg.rect_width) * cal_h !=
                static_cast<int64_t>(cfg.rect_height) * cal_w) {
                LOG_WARN("Skip color remap: calibration=%dx%d, requested=%dx%d "
                         "would require non-uniform scaling",
                         cal_w, cal_h, cfg.rect_width, cfg.rect_height);
                return;
            }
            const double sx = static_cast<double>(cfg.rect_width) / cal_w;
            const double sy = static_cast<double>(cfg.rect_height) / cal_h;
            P1 = P1.clone();
            P1.at<double>(0, 0) *= sx;
            P1.at<double>(1, 1) *= sy;
            P1.at<double>(0, 2) *= sx;
            P1.at<double>(1, 2) *= sy;
            P1.at<double>(0, 3) *= sx;
        }

        cv::initUndistortRectifyMap(K1, D1, R1, P1,
            cv::Size(cfg.rect_width, cfg.rect_height),
            CV_16SC2, vis_map1, vis_map2);
        has_color_remap = true;
        LOG_INFO("Color remap built for visualization (%dx%d -> %dx%d)",
                 cfg.camera.width, cfg.camera.height,
                 cfg.rect_width, cfg.rect_height);
    } catch (const std::exception& e) {
        LOG_WARN("Failed to build color remap: %s (visualization will be grayscale)",
                 e.what());
    }
}

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
