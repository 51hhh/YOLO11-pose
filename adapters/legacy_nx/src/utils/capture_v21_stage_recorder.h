/**
 * @file capture_v21_stage_recorder.h
 * @brief 把旧 NX 帧回调无损转换为主线 v2.1 staging 事件。
 *
 * 该类只属于 legacy adapter。最终契约校验、哈希和原子发布由
 * python/volleyball_vision/capture.py 完成。
 */

#ifndef STEREO_3D_PIPELINE_CAPTURE_V21_STAGE_RECORDER_H_
#define STEREO_3D_PIPELINE_CAPTURE_V21_STAGE_RECORDER_H_

#include "../pipeline/pipeline_callbacks.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <thread>
#include <vector>

namespace stereo3d {

struct CaptureV21StageRecorderConfig {
    bool valid = true;
    bool enabled = false;
    std::string output_dir;
    int frame_limit = 0;
    std::size_t queue_capacity = 256;
    std::string overflow_policy = "block";
    std::string raw_frames = "all";
    std::string image_product = "rectified_gray8";
    std::string image_format = "pgm";
    std::string left_camera_id;
    std::string right_camera_id;
    std::string pipeline_id;
    std::string loss_policy = "fail_trial";
    std::string exposure_timestamp_source;
};

class CaptureV21StageRecorder {
public:
    CaptureV21StageRecorder() = default;
    ~CaptureV21StageRecorder() { close(); }

    CaptureV21StageRecorder(const CaptureV21StageRecorder&) = delete;
    CaptureV21StageRecorder& operator=(const CaptureV21StageRecorder&) = delete;

    bool init(const CaptureV21StageRecorderConfig& config);
    void record(const FrameCallbackData& frame);
    void close();

    [[nodiscard]] bool enabled() const noexcept { return cfg_.enabled; }
    [[nodiscard]] bool shouldStop() const noexcept;
    [[nodiscard]] bool publishable() const noexcept {
        return status_publishable_.load();
    }
    [[nodiscard]] int capturedFrames() const noexcept {
        return captured_frames_.load();
    }
    [[nodiscard]] int writtenFrames() const noexcept {
        return written_frames_.load();
    }
    [[nodiscard]] std::string fatalError() const;

private:
    struct QueuedFrame {
        int frame_id = -1;
        std::uint64_t callback_timestamp_ns = 0;
        FrameMetadata metadata;
        std::vector<Detection> candidates_left;
        std::vector<Detection> candidates_right;
        std::vector<StereoRoiPairAttempt> pair_attempts;
        cv::Mat left_gray;
        cv::Mat right_gray;
        bool has_raw_images = false;
    };

    CaptureV21StageRecorderConfig cfg_;
    std::ofstream frames_file_;
    std::ofstream detections_file_;
    std::ofstream pairs_file_;
    std::thread writer_thread_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_not_empty_;
    std::condition_variable queue_not_full_;
    std::deque<QueuedFrame> queue_;
    std::atomic<bool> running_{false};
    std::atomic<bool> capture_complete_{false};
    std::atomic<bool> status_publishable_{false};
    std::atomic<int> captured_frames_{0};
    std::atomic<int> written_frames_{0};
    std::atomic<int> dropped_frames_{0};
    std::atomic<int> copy_failures_{0};
    std::atomic<int> write_failures_{0};
    std::atomic<std::uint64_t> next_pair_id_{0};
    mutable std::mutex error_mutex_;
    std::string fatal_error_;
    int last_frame_id_ = -1;
    std::uint32_t last_left_counter_ = 0;
    std::uint32_t last_right_counter_ = 0;
    std::uint64_t last_trigger_sequence_ = 0;
    bool have_previous_frame_ = false;

    bool copyGrayImage(VPIImage image, cv::Mat& out) const;
    bool writePgm(const std::string& path, const cv::Mat& image) const;
    void writerLoop() noexcept;
    bool writeFrame(const QueuedFrame& frame);
    void setFatal(std::string message);
    void writeStatus();
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_CAPTURE_V21_STAGE_RECORDER_H_

