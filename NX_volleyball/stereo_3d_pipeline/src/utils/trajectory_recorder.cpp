/**
 * @file trajectory_recorder.cpp
 * @brief CSV trajectory data recorder — async queue + background writer
 */

#include "trajectory_recorder.h"
#include "logger.h"
#include <iomanip>

namespace stereo3d {

void TrajectoryRecorder::init(const TrajectoryRecorderConfig& config) {
    cfg_ = config;
    if (!cfg_.enabled) return;

    file_.open(cfg_.output_path, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        LOG_WARN("TrajectoryRecorder: failed to open %s", cfg_.output_path.c_str());
        cfg_.enabled = false;
        return;
    }

    writeHeader();
    frame_count_ = 0;
    running_ = true;
    writer_thread_ = std::thread(&TrajectoryRecorder::writerLoop, this);
    LOG_INFO("TrajectoryRecorder: recording to %s", cfg_.output_path.c_str());
}

void TrajectoryRecorder::writeHeader() {
    file_ << "frame_id,timestamp,track_id,"
          << "x,y,z,vx,vy,vz,ax,ay,az,"
          << "confidence,method,"
          << "landing_x,landing_y,landing_t\n";
    header_written_ = true;
}

void TrajectoryRecorder::record(
    int frame_id, double timestamp,
    const std::vector<Object3D>& results,
    const std::vector<LandingPrediction>& preds) {

    if (!cfg_.enabled || !running_) return;

    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        queue_.push_back({frame_id, timestamp, results, preds});
    }
    queue_cv_.notify_one();
    frame_count_++;
}

void TrajectoryRecorder::writerLoop() {
    std::deque<RecordEntry> batch;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            queue_cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
            if (!running_ && queue_.empty()) break;
            batch.swap(queue_);
        }
        for (const auto& entry : batch) {
            writeEntry(entry);
        }
        batch.clear();
        file_.flush();
    }
}

void TrajectoryRecorder::writeEntry(const RecordEntry& entry) {
    for (size_t i = 0; i < entry.results.size(); ++i) {
        const auto& r = entry.results[i];
        if (r.track_id < 0) continue;

        file_ << entry.frame_id << ","
              << std::fixed << std::setprecision(6) << entry.timestamp << ","
              << r.track_id << ","
              << std::setprecision(4)
              << r.x << "," << r.y << "," << r.z << ","
              << r.vx << "," << r.vy << "," << r.vz << ","
              << r.ax << "," << r.ay << "," << r.az << ","
              << r.confidence << ",";

        if (i < entry.preds.size() && entry.preds[i].valid) {
            file_ << entry.preds[i].method << ","
                  << entry.preds[i].x << "," << entry.preds[i].y << ","
                  << entry.preds[i].time_to_land;
        } else {
            file_ << "-1,0,0,0";
        }
        file_ << "\n";
    }
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
            LOG_INFO("TrajectoryRecorder: saved %d frames", frame_count_.load());
        }
    }
}

}  // namespace stereo3d
