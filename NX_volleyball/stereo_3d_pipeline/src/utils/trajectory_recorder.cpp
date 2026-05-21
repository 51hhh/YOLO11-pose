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
    if (cfg_.raw_mode) {
        file_ << "frame_id,timestamp,has_detection,"
              << "bbox_cx,bbox_cy,bbox_w,bbox_h,det_confidence,"
              << "z_mono,z_stereo,disparity,stereo_conf,depth_method,"
              << "obs_x,obs_y,obs_z\n";
    } else {
        file_ << "frame_id,timestamp,track_id,"
              << "x,y,z,vx,vy,vz,ax,ay,az,"
              << "z_mono,z_stereo,depth_method,"
              << "confidence,"
              << "landing_x,landing_y,landing_t\n";
    }
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
    if (cfg_.raw_mode) {
        // Raw mode: 每帧一行，丢失时输出空值
        if (entry.results.empty()) {
            // 没有检测到球 — 输出空行标记
            file_ << entry.frame_id << ","
                  << std::fixed << std::setprecision(6) << entry.timestamp << ","
                  << "0,"  // has_detection = false
                  << ",,,,,,,,,,,,,\n";
        } else {
            for (size_t i = 0; i < entry.results.size(); ++i) {
                const auto& r = entry.results[i];
                file_ << entry.frame_id << ","
                      << std::fixed << std::setprecision(6) << entry.timestamp << ","
                      << "1,"  // has_detection = true
                      << std::setprecision(2)
                      << r.bbox_cx << "," << r.bbox_cy << ","
                      << r.bbox_w << "," << r.bbox_h << ","
                      << std::setprecision(4) << r.confidence << ","
                      << r.z_mono << "," << r.z_stereo << ","
                      << r.disparity << "," << r.stereo_conf << ","
                      << r.depth_method << ","
                      << r.obs_x << "," << r.obs_y << "," << r.obs_z
                      << "\n";
            }
        }
    } else {
        // Original mode: Kalman-filtered trajectory
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
                  << r.z_mono << "," << r.z_stereo << "," << r.depth_method << ","
                  << r.confidence << ",";

            if (i < entry.preds.size() && entry.preds[i].valid) {
                file_ << entry.preds[i].x << "," << entry.preds[i].y << ","
                      << entry.preds[i].time_to_land;
            } else {
                file_ << "0,0,0";
            }
            file_ << "\n";
        }
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
