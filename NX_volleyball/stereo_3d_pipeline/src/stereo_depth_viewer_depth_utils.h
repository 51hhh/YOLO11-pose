#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdint>

// ============================================================
//  视差 → 深度图 (float, mm)
// ============================================================
static cv::Mat disparityToDepth(const cv::Mat& disp_float,
                                float baseline_mm,
                                float focal_px) {
    cv::Mat depth(disp_float.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < disp_float.rows; ++y) {
        const float* dp = disp_float.ptr<float>(y);
        float* dd = depth.ptr<float>(y);
        for (int x = 0; x < disp_float.cols; ++x) {
            if (dp[x] > 0.5f) {
                dd[x] = baseline_mm * focal_px / dp[x];
            }
        }
    }
    return depth;
}

// ============================================================
//  视差图着色 → 灰度深度显示
//  近 = 亮白, 远 = 暗黑, 无效 = 黑
// ============================================================
static cv::Mat depthToGray(const cv::Mat& depth_mm,
                           float minDepth_mm = 400.0f,
                           float maxDepth_mm = 15000.0f) {
    cv::Mat gray(depth_mm.size(), CV_8U, cv::Scalar(0));
    float range = maxDepth_mm - minDepth_mm;
    if (range <= 0) range = 1.0f;

    for (int y = 0; y < depth_mm.rows; ++y) {
        const float* dp = depth_mm.ptr<float>(y);
        uint8_t* gp = gray.ptr<uint8_t>(y);
        for (int x = 0; x < depth_mm.cols; ++x) {
            float d = dp[x];
            if (d > minDepth_mm && d < maxDepth_mm) {
                // 近->255(亮白), 远->0(黑)
                float norm = 1.0f - (d - minDepth_mm) / range;
                gp[x] = static_cast<uint8_t>(std::clamp(norm * 255.0f, 0.0f, 255.0f));
            }
        }
    }
    return gray;
}

// ============================================================
//  双边滤波平滑视差图 (CPU, CV_32F 无量化损失)
// ============================================================
static cv::Mat bilateralFilterDisparity(const cv::Mat& dispF, int d = 9,
                                         double sigmaColor = 25.0,
                                         double sigmaSpace = 25.0) {
    double maxVal;
    cv::minMaxLoc(dispF, nullptr, &maxVal);
    if (maxVal < 1.0) return dispF;

    cv::Mat mask = (dispF > 0.5f);

    cv::Mat filtered;
    cv::bilateralFilter(dispF, filtered, d, sigmaColor, sigmaSpace);

    filtered.setTo(0.0f, ~mask);
    return filtered;
}

// ============================================================
//  去斑点滤波 (去除小连通区域噪声)
// ============================================================
static void removeSpeckles(cv::Mat& dispF, int maxSpeckleSize = 400,
                           float maxDiff = 1.5f) {
    cv::Mat disp16;
    dispF.convertTo(disp16, CV_16S, 16.0);
    cv::filterSpeckles(disp16, 0, maxSpeckleSize, static_cast<int>(maxDiff * 16.0));
    disp16.convertTo(dispF, CV_32F, 1.0 / 16.0);
}

// ============================================================
//  中值滤波去噪 (5x5)
// ============================================================
static void medianFilterDisparity(cv::Mat& dispF) {
    double maxVal;
    cv::minMaxLoc(dispF, nullptr, &maxVal);
    if (maxVal < 1.0) return;

    cv::Mat origMask = (dispF > 0);

    cv::Mat disp8;
    dispF.convertTo(disp8, CV_8U, 255.0 / maxVal);
    cv::medianBlur(disp8, disp8, 5);

    disp8.convertTo(dispF, CV_32F, maxVal / 255.0);
    dispF.setTo(0.0f, ~origMask);
}
