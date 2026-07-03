/**
 * @file stereo_depth_viewer_ui.cpp
 * @brief View mode and overlay helpers for stereo_depth_viewer.
 */

#include "stereo_depth_viewer_ui.h"

#include <algorithm>
#include <cstdio>

const char* viewModeName(ViewMode m) {
    switch (m) {
        case ViewMode::RAW_STEREO: return "Raw Stereo (L|R)";
        case ViewMode::VPI_CUDA_FULL: return "VPI CUDA SGM Full";
        case ViewMode::VPI_CUDA_HALF: return "VPI CUDA SGM Half";
        case ViewMode::VPI_CUDA_BILATERAL: return "VPI CUDA SGM + Bilateral";
        case ViewMode::OPENCV_CUDA_SGM: return "OpenCV CUDA SGM";
        case ViewMode::OPENCV_CUDA_BM: return "OpenCV CUDA BM";
        case ViewMode::OPENCV_CUDA_BP: return "OpenCV CUDA BP";
        case ViewMode::OPENCV_CUDA_CSBP: return "OpenCV CUDA CSBP";
        case ViewMode::OPENCV_SGBM_CPU: return "OpenCV SGBM CPU";
        case ViewMode::OPENCV_SGBM_WLS: return "OpenCV SGBM+WLS";
        case ViewMode::OPENCV_SGBM_CENSUS: return "OpenCV SGBM Census";
        case ViewMode::ONNX_CRESTEREO: return "CREStereo ONNX DL";
        case ViewMode::ONNX_HITNET: return "HITNet ONNX DL";
        default: return "Unknown";
    }
}

void onDepthViewerMouse(int event, int x, int y, int /*flags*/, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* state = reinterpret_cast<DepthViewerState*>(userdata);
    if (state->depth_map.empty()) return;

    x = std::clamp(x, 0, state->depth_map.cols - 1);
    y = std::clamp(y, 0, state->depth_map.rows - 1);

    state->click_x = x;
    state->click_y = y;
    state->click_depth_mm = state->depth_map.at<float>(y, x);

    if (state->click_depth_mm > 0) {
        printf("[Click] (%d, %d) -> depth = %.1f mm (%.2f m)\n",
               x, y, state->click_depth_mm, state->click_depth_mm / 1000.0f);
    } else {
        printf("[Click] (%d, %d) -> invalid depth\n", x, y);
    }
}

void drawOSD(cv::Mat& frame, ViewMode mode, float fps,
             const DepthViewerState& state) {
    char buf[256];
    snprintf(buf, sizeof(buf), "[%d/%d] %s | FPS: %.1f | 't':switch 'q':quit",
             static_cast<int>(mode), static_cast<int>(ViewMode::MODE_COUNT) - 1,
             viewModeName(mode), fps);
    cv::putText(frame, buf, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    if (mode != ViewMode::RAW_STEREO) {
        if (state.depth_max_mm > 0) {
            snprintf(buf, sizeof(buf), "Depth: %.1f~%.1fm",
                     state.depth_min_mm / 1000.0f,
                     state.depth_max_mm / 1000.0f);
            cv::putText(frame, buf, cv::Point(10, frame.rows - 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 255), 2);
        }
        cv::putText(frame, "RED=near  BLUE=far  BLACK=invalid",
                    cv::Point(10, frame.rows - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(200, 200, 200), 1);
    }

    if (state.click_x >= 0 && state.click_depth_mm > 0 &&
        mode != ViewMode::RAW_STEREO) {
        snprintf(buf, sizeof(buf), "Depth(%d,%d): %.0fmm (%.2fm)",
                 state.click_x, state.click_y,
                 state.click_depth_mm, state.click_depth_mm / 1000.0f);
        cv::putText(frame, buf, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 255), 2);

        cv::circle(frame, cv::Point(state.click_x, state.click_y), 6,
                   cv::Scalar(0, 0, 255), 2);
        cv::drawMarker(frame, cv::Point(state.click_x, state.click_y),
                       cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 12, 2);
    }
}
