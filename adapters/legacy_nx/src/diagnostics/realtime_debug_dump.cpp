#include "realtime_debug_dump.h"

#include "../utils/logger.h"
#include "../utils/vpi_image_copy.h"

#include <filesystem>
#include <utility>

RealtimeDebugDumper::~RealtimeDebugDumper() {
    close();
}

void RealtimeDebugDumper::init(const RealtimeDebugDumpConfig& cfg) {
    close();
    cfg_ = cfg;
    if (!cfg_.enabled) return;
    captured_count_ = 0;
    saved_count_ = 0;
    dropped_count_ = 0;
    reserved_count_ = 0;
    queue_.clear();
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(cfg_.output_dir, ec);
    if (ec) {
        LOG_WARN("RealtimeDebugDumper: failed to create %s: %s",
                 cfg_.output_dir.c_str(), ec.message().c_str());
        cfg_.enabled = false;
        return;
    }
    running_ = true;
    try {
        writer_ = std::thread(&RealtimeDebugDumper::writerLoop, this);
    } catch (const std::exception& error) {
        running_ = false;
        cfg_.enabled = false;
        LOG_WARN("RealtimeDebugDumper: failed to start writer: %s",
                 error.what());
        return;
    } catch (...) {
        running_ = false;
        cfg_.enabled = false;
        LOG_WARN("RealtimeDebugDumper: unknown failure starting writer");
        return;
    }
    LOG_INFO("RealtimeDebugDumper: output=%s stride=%d max_frames=%d",
             cfg_.output_dir.c_str(), cfg_.stride, cfg_.max_frames);
}

bool RealtimeDebugDumper::enabled() const {
    return cfg_.enabled && running_;
}

void RealtimeDebugDumper::record(const stereo3d::FrameCallbackData& frame) {
    if (!enabled()) return;
    if (cfg_.max_frames > 0 && captured_count_.load() >= cfg_.max_frames) {
        return;
    }

    bool has_fallback = false;
    for (const auto& obj : frame.results) {
        if (obj.stereo_match_source == 2 || obj.stereo_match_source == 3) {
            has_fallback = true;
            break;
        }
    }
    const bool stride_hit =
        cfg_.stride > 0 && (frame.frame_id % cfg_.stride) == 0;
    if (!stride_hit && !(cfg_.dump_fallback && has_fallback)) {
        return;
    }
    if (!reserveSlot()) {
        return;
    }

    RealtimeDebugDumpJob job;
    job.frame_id = frame.frame_id;
    job.fps = frame.fps;
    job.left_detections = frame.detections_left;
    job.right_detections = frame.detections_right;
    job.results = frame.results;
    job.metadata = frame.metadata;
    if (!copyGray(frame.rect_gray_left, job.left_gray) ||
        !copyGray(frame.rect_gray_right, job.right_gray)) {
        ++dropped_count_;
        LOG_WARN("RealtimeDebugDumper: image copy failed at frame=%d",
                 frame.frame_id);
        releaseReservedSlot(false, nullptr);
        return;
    }
    releaseReservedSlot(true, &job);
    cv_.notify_one();
}

void RealtimeDebugDumper::close() {
    if (!cfg_.enabled && !writer_.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        running_ = false;
    }
    cv_.notify_all();
    if (writer_.joinable()) writer_.join();
    if (cfg_.enabled) {
        LOG_INFO("RealtimeDebugDumper: saved=%d dropped=%d",
                 saved_count_.load(), dropped_count_.load());
    }
    cfg_.enabled = false;
}

bool RealtimeDebugDumper::reserveSlot() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!running_) return false;
    const int reserved = reserved_count_;
    if (cfg_.max_frames > 0 &&
        captured_count_.load() + reserved >= cfg_.max_frames) {
        return false;
    }
    if (cfg_.max_queue > 0 &&
        queue_.size() + static_cast<size_t>(reserved) >=
            static_cast<size_t>(cfg_.max_queue)) {
        ++dropped_count_;
        return false;
    }
    ++reserved_count_;
    return true;
}

void RealtimeDebugDumper::releaseReservedSlot(bool enqueue,
                                              RealtimeDebugDumpJob* job) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (reserved_count_ > 0) --reserved_count_;
    if (enqueue && job && running_) {
        queue_.push_back(std::move(*job));
        ++captured_count_;
    } else if (enqueue && job) {
        ++dropped_count_;
    }
}

bool RealtimeDebugDumper::copyGray(VPIImage img, cv::Mat& out) {
    return stereo3d::copyVpiGray8ToHost(
        img, out, "realtime debug gray copy");
}
