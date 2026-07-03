/**
 * @file trajectory_recorder.h
 * @brief CSV trajectory data recorder for offline visualization
 *
 * Records per-frame tracking results and predictions to CSV file.
 * Uses async queue + background writer thread to avoid IO blocking in pipeline.
 * Format matches scripts/visualize_trajectory.py expectations.
 */

#ifndef STEREO_3D_PIPELINE_TRAJECTORY_RECORDER_H_
#define STEREO_3D_PIPELINE_TRAJECTORY_RECORDER_H_

#include "../pipeline/frame_slot.h"
#include "../fusion/trajectory_predictor.h"
#include <string>
#include <fstream>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <deque>

namespace stereo3d {

enum class TrajectoryRecordDetail {
    LEGACY = 0,
    DEPTH_CANDIDATES = 1,
    EXTENDED = 2,
};

struct TrajectoryRecorderConfig {
    std::string output_path = "trajectory_data.csv";
    bool enabled = true;
    bool raw_mode = false; ///< true=写未滤波观测, false=写 Kalman 后轨迹
    TrajectoryRecordDetail detail_level = TrajectoryRecordDetail::LEGACY;
    size_t max_queue_frames = 1000; ///< 0=无限队列；实时路径建议保留上限，避免 IO 慢拖垮内存
    bool frame_summary_enabled = true; ///< 每帧 sidecar CSV, 用于统计无输出/误匹配退化
    std::string frame_summary_path; ///< 空=从 output_path 自动派生 *.frames.csv

    bool recordDepthCandidates() const {
        return static_cast<int>(detail_level) >=
               static_cast<int>(TrajectoryRecordDetail::DEPTH_CANDIDATES);
    }
    bool recordExtendedGeometry() const {
        return static_cast<int>(detail_level) >=
               static_cast<int>(TrajectoryRecordDetail::EXTENDED);
    }
};

class TrajectoryRecorder {
public:
    TrajectoryRecorder() = default;
    ~TrajectoryRecorder() { close(); }

    void init(const TrajectoryRecorderConfig& config = TrajectoryRecorderConfig());

    /**
     * @brief Record one frame's results (non-blocking, pushes to queue)
     */
    void record(int frame_id, double timestamp,
                const std::vector<Object3D>& results,
                const std::vector<LandingPrediction>& preds);

    void close();

    int frameCount() const { return frame_count_.load(); }
    int droppedFrameCount() const { return dropped_frame_count_.load(); }

private:
    struct RecordEntry {
        int frame_id;
        double timestamp;
        std::vector<Object3D> results;
        std::vector<LandingPrediction> preds;
    };

    TrajectoryRecorderConfig cfg_;
    std::ofstream file_;
    std::ofstream frame_file_;
    std::atomic<int> frame_count_{0};
    std::atomic<int> dropped_frame_count_{0};
    bool header_written_ = false;

    // Async write queue
    std::thread writer_thread_;
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::deque<RecordEntry> queue_;
    std::atomic<bool> running_{false};

    void writeHeader();
    void writeFrameSummaryHeader();
    void writerLoop();
    void writeEntry(const RecordEntry& entry);
    void writeFrameSummary(const RecordEntry& entry);
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_TRAJECTORY_RECORDER_H_
