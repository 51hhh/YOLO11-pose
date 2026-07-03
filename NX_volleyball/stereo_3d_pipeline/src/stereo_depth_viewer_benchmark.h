/**
 * @file stereo_depth_viewer_benchmark.h
 * @brief Headless benchmark report helpers for stereo_depth_viewer.
 */

#ifndef STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_BENCHMARK_H_
#define STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_BENCHMARK_H_

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

struct BenchmarkEntry {
    std::string name;
    double avgMs = 0.0;
    double fps = 0.0;
    int validPixels = 0;
    int totalPixels = 0;
    double validRatio = 0.0;
    float depthMin = 0.0f;
    float depthMax = 0.0f;
    float depthMean = 0.0f;
};

void writeHeadlessBenchmarkReport(
    int width,
    int height,
    int max_disparity,
    int frames_per_mode,
    const std::vector<BenchmarkEntry>& bench_results,
    const std::vector<std::pair<std::string, cv::Mat>>& saved_frames);

#endif  // STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_BENCHMARK_H_
