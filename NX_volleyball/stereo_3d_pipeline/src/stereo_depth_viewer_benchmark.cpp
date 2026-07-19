/**
 * @file stereo_depth_viewer_benchmark.cpp
 * @brief Headless benchmark report helpers for stereo_depth_viewer.
 */

#include "stereo_depth_viewer_benchmark.h"

#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <fstream>
#include <iomanip>

void writeHeadlessBenchmarkReport(
    int width,
    int height,
    int max_disparity,
    int frames_per_mode,
    const std::vector<BenchmarkEntry>& bench_results,
    const std::vector<std::pair<std::string, cv::Mat>>& saved_frames) {
    if (bench_results.empty()) return;

    {
        std::ofstream jf("diagnose_output/benchmark_report.json");
        if (jf.is_open()) {
            jf << "{\n  \"platform\": \"Jetson Orin NX 16GB\",\n";
            jf << "  \"resolution\": \"" << width << "x" << height << "\",\n";
            jf << "  \"max_disparity\": " << max_disparity << ",\n";
            jf << "  \"frames_per_mode\": " << frames_per_mode << ",\n";
            jf << "  \"algorithms\": [\n";
            for (size_t i = 0; i < bench_results.size(); ++i) {
                const auto& e = bench_results[i];
                jf << "    {\n";
                jf << "      \"name\": \"" << e.name << "\",\n";
                jf << "      \"avg_ms\": " << std::fixed << std::setprecision(2) << e.avgMs << ",\n";
                jf << "      \"fps\": " << std::fixed << std::setprecision(1) << e.fps << ",\n";
                jf << "      \"valid_pixels\": " << e.validPixels << ",\n";
                jf << "      \"valid_ratio\": " << std::fixed << std::setprecision(1) << e.validRatio << ",\n";
                jf << "      \"depth_min_mm\": " << std::fixed << std::setprecision(0) << e.depthMin << ",\n";
                jf << "      \"depth_max_mm\": " << std::fixed << std::setprecision(0) << e.depthMax << ",\n";
                jf << "      \"depth_mean_mm\": " << std::fixed << std::setprecision(0) << e.depthMean << "\n";
                jf << "    }" << (i + 1 < bench_results.size() ? "," : "") << "\n";
            }
            jf << "  ]\n}\n";
            jf.close();
            LOG_INFO("基准报告已保存: diagnose_output/benchmark_report.json");
        }
    }

    if (!saved_frames.empty()) {
        int thumbW = 480, thumbH = 360;
        int cols = 4;
        int rows = (static_cast<int>(saved_frames.size()) + cols - 1) / cols;
        cv::Mat grid(rows * (thumbH + 30), cols * thumbW,
                     CV_8UC3, cv::Scalar(30, 30, 30));

        for (size_t i = 0; i < saved_frames.size(); ++i) {
            int r = static_cast<int>(i) / cols;
            int c = static_cast<int>(i) % cols;
            int x0 = c * thumbW;
            int y0 = r * (thumbH + 30);

            cv::Mat thumb;
            cv::resize(saved_frames[i].second, thumb, cv::Size(thumbW, thumbH));
            if (thumb.channels() == 1)
                cv::cvtColor(thumb, thumb, cv::COLOR_GRAY2BGR);
            thumb.copyTo(grid(cv::Rect(x0, y0, thumbW, thumbH)));

            cv::putText(grid, saved_frames[i].first,
                        cv::Point(x0 + 5, y0 + thumbH + 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(200, 200, 200), 1);
        }
        cv::imwrite("diagnose_output/comparison_grid.png", grid);
        LOG_INFO("对比网格已保存: diagnose_output/comparison_grid.png");
    }

    printf("\n");
    printf("============ 基准测试汇总 ============\n");
    printf("%-24s %8s %8s %8s %10s\n", "算法", "ms/帧", "FPS", "有效%", "平均深度mm");
    printf("--------------------------------------------------------------\n");
    for (const auto& e : bench_results) {
        printf("%-24s %8.1f %8.1f %7.1f%% %10.0f\n",
               e.name.c_str(), e.avgMs, e.fps, e.validRatio, e.depthMean);
    }
    printf("============================================\n\n");
}
