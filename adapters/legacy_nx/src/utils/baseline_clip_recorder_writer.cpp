#include "baseline_clip_recorder.h"

#include "baseline_clip_recorder_io.h"
#include "logger.h"

#include <filesystem>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>

namespace stereo3d {

void BaselineClipRecorder::writerLoop() {
    std::deque<QueuedFrame> batch;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            queue_cv_.wait(lock, [this] {
                if (!running_.load()) return true;
                if (queue_.empty()) return false;
                return !cfg_.write_after_capture;
            });
            if (!running_.load() && queue_.empty()) break;
            if (cfg_.write_after_capture && running_.load()) continue;
            batch.swap(queue_);
        }
        for (const auto& frame : batch) {
            writeFrame(frame);
        }
        batch.clear();
    }
}

void BaselineClipRecorder::writeFrame(const QueuedFrame& frame) {
    namespace fs = std::filesystem;
    const std::string name = baselineClipFrameName(frame.clip_frame_id,
                                                   image_ext_);
    const std::string left_rel = "left/" + name;
    const std::string right_rel = "right/" + name;
    std::string left_bgr_rel;
    std::string right_bgr_rel;
    const fs::path clip_root(frame.clip_dir);
    const fs::path left_path = clip_root / left_rel;
    const fs::path right_path = clip_root / right_rel;

    std::vector<int> params;
    if (image_ext_ == "png") {
        params = {cv::IMWRITE_PNG_COMPRESSION, cfg_.png_compression};
    }
    const bool primary_bgr = image_mode_ == "bgr";
    const cv::Mat& left_primary = primary_bgr ? frame.left_bgr : frame.left_gray;
    const cv::Mat& right_primary = primary_bgr ? frame.right_bgr : frame.right_gray;
    const bool ok_left = !left_primary.empty() &&
                         cv::imwrite(left_path.string(), left_primary, params);
    const bool ok_right = !right_primary.empty() &&
                          cv::imwrite(right_path.string(), right_primary, params);
    if (!ok_left || !ok_right) {
        LOG_WARN("BaselineClipRecorder: image write failed clip_frame=%d left=%d right=%d",
                 frame.clip_frame_id, ok_left ? 1 : 0, ok_right ? 1 : 0);
    }
    if (image_mode_ == "both") {
        left_bgr_rel = "left_bgr/" + name;
        right_bgr_rel = "right_bgr/" + name;
        const bool ok_left_bgr = !frame.left_bgr.empty() &&
            cv::imwrite((clip_root / left_bgr_rel).string(), frame.left_bgr, params);
        const bool ok_right_bgr = !frame.right_bgr.empty() &&
            cv::imwrite((clip_root / right_bgr_rel).string(), frame.right_bgr, params);
        if (!ok_left_bgr || !ok_right_bgr) {
            LOG_WARN("BaselineClipRecorder: BGR image write failed clip_frame=%d left=%d right=%d",
                     frame.clip_frame_id, ok_left_bgr ? 1 : 0, ok_right_bgr ? 1 : 0);
        }
    }

    const Detection* left_det =
        (frame.best_left_idx >= 0 &&
         frame.best_left_idx < static_cast<int>(frame.detections_left.size()))
            ? &frame.detections_left[frame.best_left_idx] : nullptr;
    const Detection* right_det =
        (frame.best_right_idx >= 0 &&
         frame.best_right_idx < static_cast<int>(frame.detections_right.size()))
            ? &frame.detections_right[frame.best_right_idx] : nullptr;

    std::ofstream csv((clip_root / "frames.csv").string(),
                      std::ios::out | std::ios::app);
    if (!csv.is_open()) {
        LOG_WARN("BaselineClipRecorder: failed to append frames.csv for %s",
                 frame.clip_dir.c_str());
        return;
    }

    csv << frame.clip_frame_id << ','
         << frame.pipeline_frame_id << ','
         << std::fixed << std::setprecision(6) << frame.timestamp_s << ','
         << std::setprecision(2) << frame.fps << ','
         << left_rel << ','
         << right_rel << ','
         << left_bgr_rel << ','
         << right_bgr_rel << ','
         << frame.detections_left.size() << ','
         << frame.detections_right.size() << ','
         << frame.best_left_idx << ','
         << frame.best_right_idx << ','
         << (frame.pair.valid ? 1 : 0) << ','
         << std::setprecision(5) << frame.pair.score << ','
         << std::setprecision(3) << frame.pair.disparity_px << ','
         << frame.pair.dy_px << ','
         << frame.pair.size_ratio << ',';
    writeBaselineDetectionColumns(csv, left_det);
    csv << ',';
    writeBaselineDetectionColumns(csv, right_det);
    csv << ','
         << frame.metadata.left_timestamp_us << ','
         << frame.metadata.right_timestamp_us << ','
         << frame.metadata.left_frame_number << ','
         << frame.metadata.right_frame_number << ','
         << frame.metadata.left_frame_counter << ','
         << frame.metadata.right_frame_counter << ','
         << frame.metadata.left_trigger_index << ','
         << frame.metadata.right_trigger_index << ','
         << frame.metadata.frame_counter_delta << ','
         << frame.metadata.frame_number_delta << ','
         << frame.metadata.timestamp_delta_us << ','
         << (frame.metadata.grab_failed ? 1 : 0) << ','
         << (frame.metadata.is_detect_frame ? 1 : 0)
         << '\n';
}

}  // namespace stereo3d
