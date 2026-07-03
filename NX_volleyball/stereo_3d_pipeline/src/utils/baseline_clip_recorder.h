/**
 * @file baseline_clip_recorder.h
 * @brief Fixed-length stereo image sequence recorder for feature-matching baselines.
 */

#ifndef STEREO_3D_PIPELINE_BASELINE_CLIP_RECORDER_H_
#define STEREO_3D_PIPELINE_BASELINE_CLIP_RECORDER_H_

#include "../pipeline/frame_slot.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <thread>
#include <vector>

namespace stereo3d {

struct BaselineClipRecorderConfig {
    bool enabled = false;
    std::string output_dir = "baseline_clips";
    double duration_sec = 3.0;
    int frame_limit = 0;
    int trigger_hz = 100;
    int clip_count = 1;
    double clip_gap_sec = 0.0;
    int clip_gap_frames = 0;
    bool require_left_detection = true;
    bool require_right_detection = true;
    bool require_pair_gate = false;
    float min_confidence = 0.0f;
    float pair_y_tolerance_px = 30.0f;
    float pair_max_size_ratio = 2.5f;
    float pair_min_disparity_px = 0.0f;
    std::string image_format = "png";
    std::string image_mode = "gray"; ///< gray|bgr; both is kept for debug compatibility.
    int png_compression = 1;
    bool write_after_capture = true; ///< Buffer frames in memory first, then write files after capture.
    bool stop_after_clip = true;
    size_t max_queue_frames = 0;  ///< 0 = unlimited; set to drop instead of growing memory.
};

class BaselineClipRecorder {
public:
    BaselineClipRecorder() = default;
    ~BaselineClipRecorder() { close(); }

    void init(const BaselineClipRecorderConfig& config);

    void record(int frame_id,
                VPIImage rect_gray_left,
                VPIImage rect_gray_right,
                VPIImage rect_bgr_left,
                VPIImage rect_bgr_right,
                const std::vector<Detection>& detections_left,
                const std::vector<Detection>& detections_right,
                const FrameMetadata& metadata,
                float fps);

    void close();

    bool enabled() const { return cfg_.enabled; }
    bool active() const { return active_.load(); }
    bool complete() const { return complete_.load(); }
    bool shouldStop() const { return cfg_.enabled && cfg_.stop_after_clip && complete_.load(); }
    int frameCount() const { return frame_count_.load(); }
    int clipFrameCount() const { return clip_frame_count_.load(); }
    int completedClips() const { return completed_clips_.load(); }
    int droppedFrames() const { return dropped_frames_.load(); }
    std::string clipDir() const { return clip_dir_; }

private:
    struct PairSelection {
        int left_idx = -1;
        int right_idx = -1;
        bool valid = false;
        float score = 0.0f;
        float disparity_px = -1.0f;
        float dy_px = -1.0f;
        float size_ratio = -1.0f;
    };

    struct QueuedFrame {
        std::string clip_dir;
        int clip_frame_id = 0;
        int clip_index = 0;
        int pipeline_frame_id = 0;
        double timestamp_s = 0.0;
        float fps = 0.0f;
        FrameMetadata metadata;
        cv::Mat left_gray;
        cv::Mat right_gray;
        cv::Mat left_bgr;
        cv::Mat right_bgr;
        std::vector<Detection> detections_left;
        std::vector<Detection> detections_right;
        PairSelection pair;
        int best_left_idx = -1;
        int best_right_idx = -1;
    };

    BaselineClipRecorderConfig cfg_;
    std::atomic<bool> active_{false};
    std::atomic<bool> complete_{false};
    std::atomic<bool> running_{false};
    std::atomic<int> frame_count_{0};
    std::atomic<int> clip_frame_count_{0};
    std::atomic<int> completed_clips_{0};
    std::atomic<int> dropped_frames_{0};
    std::atomic<int> copy_failures_{0};

    int target_frames_ = 0;
    int gap_frames_ = 0;
    int current_clip_index_ = 0;
    int next_start_frame_id_ = 0;
    size_t effective_max_queue_frames_ = 0;
    std::string clip_dir_;
    std::string image_ext_ = "png";
    std::string image_mode_ = "gray";

    std::thread writer_thread_;
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::deque<QueuedFrame> queue_;

    bool shouldStart(const std::vector<Detection>& left,
                     const std::vector<Detection>& right,
                     const PairSelection& pair) const;
    bool startClip();
    void writerLoop();
    void writeFrame(const QueuedFrame& frame);

    int bestDetectionIndex(const std::vector<Detection>& detections) const;
    PairSelection selectBestPair(const std::vector<Detection>& left,
                                 const std::vector<Detection>& right) const;
    bool copyGrayImage(VPIImage image, cv::Mat& out) const;
    bool copyBgrImage(VPIImage image, cv::Mat& out) const;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_BASELINE_CLIP_RECORDER_H_
