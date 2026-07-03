#pragma once

#include "pipeline/pipeline.h"

#include <vpi/Image.h>
#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ==================== Realtime debug dump ====================

struct RealtimeDebugDumpConfig {
    bool enabled = false;
    std::string output_dir = "debug_realtime_dumps";
    int stride = 100;
    int max_frames = 0;
    int max_queue = 4;
    bool dump_fallback = true;
};

struct RealtimeDebugDumpJob {
    int frame_id = 0;
    float fps = 0.0f;
    cv::Mat left_gray;
    cv::Mat right_gray;
    std::vector<stereo3d::Detection> left_detections;
    std::vector<stereo3d::Detection> right_detections;
    std::vector<stereo3d::Object3D> results;
    stereo3d::FrameMetadata metadata;
};

class RealtimeDebugDumper {
public:
    ~RealtimeDebugDumper();

    void init(const RealtimeDebugDumpConfig& cfg);
    bool enabled() const;
    void record(const stereo3d::FrameCallbackData& frame);
    void close();

private:
    bool reserveSlot();
    void releaseReservedSlot(bool enqueue, RealtimeDebugDumpJob* job);
    static bool copyGray(VPIImage img, cv::Mat& out);
    void writeJob(const RealtimeDebugDumpJob& job);
    void writerLoop();

    RealtimeDebugDumpConfig cfg_;
    std::atomic<bool> running_{false};
    std::atomic<int> captured_count_{0};
    std::atomic<int> saved_count_{0};
    std::atomic<int> dropped_count_{0};
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<RealtimeDebugDumpJob> queue_;
    int reserved_count_ = 0;
    std::thread writer_;
};
