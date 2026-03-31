/**
 * @file profiler.h
 * @brief Pipeline 性能计时器 (兼容 Nsight Systems NVTX)
 *
 * 使用方法:
 *   ScopedTimer t("Stage0_Grab");  // 作用域退出时自动记录耗时
 *   NVTX_RANGE("MyLabel");        // nsys 可视化标记
 */

#ifndef STEREO_3D_PIPELINE_PROFILER_H_
#define STEREO_3D_PIPELINE_PROFILER_H_

#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <mutex>
#include <cmath>

// NVTX 标记 (仅在安装 nsys 时启用)
#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_RANGE(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
#define NVTX_RANGE(name) ((void)0)
#define NVTX_RANGE_POP() ((void)0)
#endif

namespace stereo3d {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

/**
 * @brief 作用域计时器
 */
class ScopedTimer {
public:
    explicit ScopedTimer(const char* label, bool print = false)
        : label_(label), print_(print), start_(Clock::now()) {
        NVTX_RANGE(label);
    }

    ~ScopedTimer() {
        NVTX_RANGE_POP();
        auto end = Clock::now();
        elapsed_ms_ = std::chrono::duration<double, std::milli>(end - start_).count();
        if (print_) {
            fprintf(stderr, "[Timer] %s: %.2f ms\n", label_, elapsed_ms_);
        }
    }

    double elapsedMs() const {
        auto now = Clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    const char* label_;
    bool print_;
    TimePoint start_;
    double elapsed_ms_ = 0.0;
};

/**
 * @brief 性能统计聚合器
 *
 * 线程安全，可从多个 Stage 同时上报。
 */
class PerfAggregator {
public:
    struct Stats {
        double sum_ms    = 0.0;
        double max_ms    = 0.0;
        double min_ms    = 1e9;
        int count        = 0;

        double avgMs() const { return count > 0 ? sum_ms / count : 0.0; }
    };

    void record(const std::string& name, double ms) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& s = stats_[name];
        s.sum_ms += ms;
        s.count++;
        if (ms > s.max_ms) s.max_ms = ms;
        if (ms < s.min_ms) s.min_ms = ms;
    }

    void printReport() const {
        std::lock_guard<std::mutex> lock(mu_);
        fprintf(stderr, "\n===== Pipeline Performance Report =====\n");
        fprintf(stderr, "%-25s %8s %8s %8s %8s\n", "Stage", "Avg(ms)", "Min(ms)", "Max(ms)", "Count");
        fprintf(stderr, "--------------------------------------------------------------\n");
        for (const auto& [name, s] : stats_) {
            fprintf(stderr, "%-25s %8.2f %8.2f %8.2f %8d\n",
                    name.c_str(), s.avgMs(), s.min_ms, s.max_ms, s.count);
        }
        fprintf(stderr, "==============================================\n\n");
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        stats_.clear();
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, Stats> stats_;
};

// 全局性能收集器
inline PerfAggregator& globalPerf() {
    static PerfAggregator instance;
    return instance;
}

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PROFILER_H_
