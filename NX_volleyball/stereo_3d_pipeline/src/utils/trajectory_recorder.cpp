/**
 * @file trajectory_recorder.cpp
 * @brief CSV trajectory data recorder — async queue + background writer
 */

#include "trajectory_recorder.h"
#include "logger.h"
#include <filesystem>

namespace stereo3d {

namespace {

std::string deriveFrameSummaryPath(const std::string& output_path) {
    const std::string suffix = ".csv";
    if (output_path.size() >= suffix.size() &&
        output_path.compare(output_path.size() - suffix.size(),
                            suffix.size(),
                            suffix) == 0) {
        return output_path.substr(0, output_path.size() - suffix.size()) +
               ".frames.csv";
    }
    return output_path + ".frames.csv";
}

bool ensureParentDirectory(const std::string& output_path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path parent = fs::path(output_path).parent_path();
    if (parent.empty()) {
        return true;
    }
    fs::create_directories(parent, ec);
    if (ec) {
        LOG_WARN("TrajectoryRecorder: failed to create directory %s: %s",
                 parent.string().c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

}  // namespace

void TrajectoryRecorder::init(const TrajectoryRecorderConfig& config) {
    cfg_ = config;
    if (!cfg_.enabled) return;

    ensureParentDirectory(cfg_.output_path);
    file_.open(cfg_.output_path, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        LOG_WARN("TrajectoryRecorder: failed to open %s", cfg_.output_path.c_str());
        cfg_.enabled = false;
        return;
    }

    writeHeader();
    if (cfg_.frame_summary_enabled) {
        const std::string frame_path = cfg_.frame_summary_path.empty()
            ? deriveFrameSummaryPath(cfg_.output_path)
            : cfg_.frame_summary_path;
        ensureParentDirectory(frame_path);
        frame_file_.open(frame_path, std::ios::out | std::ios::trunc);
        if (!frame_file_.is_open()) {
            LOG_WARN("TrajectoryRecorder: failed to open frame summary %s",
                     frame_path.c_str());
        } else {
            writeFrameSummaryHeader();
            LOG_INFO("TrajectoryRecorder: frame summary to %s",
                     frame_path.c_str());
        }
    }
    frame_count_ = 0;
    dropped_frame_count_ = 0;
    running_ = true;
    writer_thread_ = std::thread(&TrajectoryRecorder::writerLoop, this);
    LOG_INFO("TrajectoryRecorder: recording to %s (max_queue_frames=%zu)",
             cfg_.output_path.c_str(), cfg_.max_queue_frames);
}

void TrajectoryRecorder::record(
    int frame_id, double timestamp,
    const std::vector<Object3D>& results,
    const std::vector<LandingPrediction>& preds) {

    if (!cfg_.enabled || !running_) return;

    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        if (cfg_.max_queue_frames > 0 &&
            queue_.size() >= cfg_.max_queue_frames) {
            const int dropped = ++dropped_frame_count_;
            if (dropped <= 3 || dropped % 100 == 0) {
                LOG_WARN("TrajectoryRecorder: queue full, dropping frame=%d dropped=%d",
                         frame_id, dropped);
            }
            return;
        }
        queue_.push_back({frame_id, timestamp, results, preds});
    }
    queue_cv_.notify_one();
    frame_count_++;
}

void TrajectoryRecorder::close() {
    if (running_) {
        running_ = false;
        queue_cv_.notify_one();
        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
    }
    if (file_.is_open()) {
        file_.flush();
        file_.close();
        if (frame_count_.load() > 0) {
            LOG_INFO("TrajectoryRecorder: saved %d frames (dropped=%d)",
                     frame_count_.load(), dropped_frame_count_.load());
        }
    }
    if (frame_file_.is_open()) {
        frame_file_.flush();
        frame_file_.close();
    }
}

}  // namespace stereo3d
