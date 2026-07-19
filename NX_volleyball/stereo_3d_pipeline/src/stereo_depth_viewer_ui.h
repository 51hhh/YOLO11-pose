/**
 * @file stereo_depth_viewer_ui.h
 * @brief View mode and overlay helpers for stereo_depth_viewer.
 */

#ifndef STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_UI_H_
#define STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_UI_H_

#include <opencv2/opencv.hpp>

enum class ViewMode {
    RAW_STEREO = 0,
    VPI_CUDA_FULL,
    VPI_CUDA_HALF,
    VPI_CUDA_BILATERAL,
    OPENCV_CUDA_SGM,
    OPENCV_CUDA_BM,
    OPENCV_CUDA_BP,
    OPENCV_CUDA_CSBP,
    OPENCV_SGBM_CPU,
    OPENCV_SGBM_WLS,
    OPENCV_SGBM_CENSUS,
    ONNX_CRESTEREO,
    ONNX_HITNET,
    MODE_COUNT
};

const char* viewModeName(ViewMode mode);

struct DepthViewerState {
    cv::Mat depth_map;
    float baseline_mm = 0.0f;
    float focal_px = 0.0f;
    int click_x = -1;
    int click_y = -1;
    float click_depth_mm = 0.0f;
    float depth_min_mm = 0.0f;
    float depth_max_mm = 0.0f;
};

void onDepthViewerMouse(int event, int x, int y, int flags, void* userdata);

void drawOSD(cv::Mat& frame,
             ViewMode mode,
             float fps,
             const DepthViewerState& state);

#endif  // STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_UI_H_
