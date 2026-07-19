#pragma once

#include "main_display_types.h"
#include "utils/logger.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vpi/Image.h>

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
        VPIImage rectL = frame_data.rect_bgr_left;
        VPIImageData imgData;
        VPIStatus st = vpiImageLockData(rectL, VPI_LOCK_READ,
            VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData);
        if (st == VPI_SUCCESS) {
            const int h = imgData.buffer.pitch.planes[0].height;
            const int w = imgData.buffer.pitch.planes[0].width;
            const int gpuPitch = imgData.buffer.pitch.planes[0].pitchBytes;
            const void* gpuPtr = imgData.buffer.pitch.planes[0].data;
            frame.create(h, w, CV_8UC3);
            cudaMemcpy2D(frame.data, frame.step[0],
                         gpuPtr, gpuPitch,
                         w * 3, h,
                         cudaMemcpyDeviceToHost);
            vpiImageUnlock(rectL);
        }
    }

    if (!frame.empty()) return true;

    VPIImage rectL = frame_data.rect_gray_left;
    VPIImageData imgData;
    if (vpiImageLockData(rectL, VPI_LOCK_READ,
        VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData) != VPI_SUCCESS) {
        return false;
    }
    const int h = imgData.buffer.pitch.planes[0].height;
    const int w = imgData.buffer.pitch.planes[0].width;
    const int pitch = imgData.buffer.pitch.planes[0].pitchBytes;
    cv::Mat gray(h, w, CV_8UC1,
                 imgData.buffer.pitch.planes[0].data, pitch);
    cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
    vpiImageUnlock(rectL);
    return true;
}
