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

struct TrajectoryRecorderConfig {
    std::string output_path = "trajectory_data.csv";
    bool enabled = true;
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

private:
    struct RecordEntry {
        int frame_id;
        double timestamp;
        std::vector<Object3D> results;
        std::vector<LandingPrediction> preds;
    };

    TrajectoryRecorderConfig cfg_;
    std::ofstream file_;
    std::atomic<int> frame_count_{0};
    bool header_written_ = false;

    // Async write queue
    std::thread writer_thread_;
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::deque<RecordEntry> queue_;
    std::atomic<bool> running_{false};

    void writeHeader();
    void writerLoop();
    void writeEntry(const RecordEntry& entry);
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_TRAJECTORY_RECORDER_H_
