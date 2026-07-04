/**
 * @file baseline_clip_recorder.cpp
 * @brief Fixed-length stereo image sequence recorder for offline feature tests.
 */

#include "baseline_clip_recorder.h"
#include "baseline_clip_recorder_io.h"
#include "logger.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>

namespace stereo3d {
namespace {

double nowSeconds() {
    const auto now = std::chrono::system_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

}  // namespace

void BaselineClipRecorder::init(const BaselineClipRecorderConfig& config) {
    close();
    cfg_ = config;
    active_ = false;
    complete_ = false;
    running_ = false;
    frame_count_ = 0;
    clip_frame_count_ = 0;
    completed_clips_ = 0;
    dropped_frames_ = 0;
    copy_failures_ = 0;
    queue_.clear();
    clip_dir_.clear();
    image_ext_ = normalizeBaselineImageFormat(cfg_.image_format);
    image_mode_ = normalizeBaselineImageMode(cfg_.image_mode);
    if (image_mode_ != "gray" && image_ext_ == "pgm") {
        LOG_WARN("BaselineClipRecorder: image_mode=%s needs color-capable output; using png",
                 image_mode_.c_str());
        image_ext_ = "png";
    }
    cfg_.png_compression = std::clamp(cfg_.png_compression, 0, 9);
    cfg_.trigger_hz = std::max(1, cfg_.trigger_hz);
    cfg_.duration_sec = std::max(0.01, cfg_.duration_sec);
    cfg_.clip_count = std::max(1, cfg_.clip_count);
    cfg_.clip_gap_sec = std::max(0.0, cfg_.clip_gap_sec);
    cfg_.clip_gap_frames = std::max(0, cfg_.clip_gap_frames);
    target_frames_ = cfg_.frame_limit > 0
        ? cfg_.frame_limit
        : static_cast<int>(std::ceil(cfg_.duration_sec * cfg_.trigger_hz));
    target_frames_ = std::max(1, target_frames_);
    gap_frames_ = cfg_.clip_gap_frames > 0
        ? cfg_.clip_gap_frames
        : static_cast<int>(std::ceil(cfg_.clip_gap_sec * cfg_.trigger_hz));
    gap_frames_ = std::max(0, gap_frames_);
    effective_max_queue_frames_ = cfg_.max_queue_frames;
    if (cfg_.write_after_capture && effective_max_queue_frames_ > 0) {
        const size_t requested_frames =
            static_cast<size_t>(target_frames_) * static_cast<size_t>(cfg_.clip_count);
        if (effective_max_queue_frames_ < requested_frames) {
            LOG_WARN("BaselineClipRecorder: max_queue_frames=%zu is smaller than requested frames=%zu; raising memory queue limit for write_after_capture",
                     effective_max_queue_frames_, requested_frames);
            effective_max_queue_frames_ = requested_frames;
        }
    }
    current_clip_index_ = 0;
    next_start_frame_id_ = 0;

    if (cfg_.enabled) {
        running_ = true;
        writer_thread_ = std::thread(&BaselineClipRecorder::writerLoop, this);
        LOG_INFO("BaselineClipRecorder armed: output_dir=%s clips=%d duration=%.2fs target=%d frames gap=%d frames format=%s mode=%s write_after_capture=%d",
                 cfg_.output_dir.c_str(), cfg_.clip_count, cfg_.duration_sec,
                 target_frames_, gap_frames_, image_ext_.c_str(), image_mode_.c_str(),
                 cfg_.write_after_capture ? 1 : 0);
    }
}

bool BaselineClipRecorder::shouldStart(
    const std::vector<Detection>& left,
    const std::vector<Detection>& right,
    const PairSelection& pair) const {
    if (cfg_.require_left_detection && bestDetectionIndex(left) < 0) return false;
    if (cfg_.require_right_detection && bestDetectionIndex(right) < 0) return false;
    if (cfg_.require_pair_gate && !pair.valid) return false;
    return true;
}

bool BaselineClipRecorder::startClip() {
    namespace fs = std::filesystem;
    if (active_.load()) return true;

    const fs::path root(cfg_.output_dir);
    const int clip_number = completed_clips_.load() + 1;
    std::ostringstream clip_name;
    clip_name << "clip_" << baselineClipTimestampName() << "_"
              << std::setw(2) << std::setfill('0') << clip_number;
    const fs::path clip = root / clip_name.str();
    const fs::path left_dir = clip / "left";
    const fs::path right_dir = clip / "right";
    const fs::path left_bgr_dir = clip / "left_bgr";
    const fs::path right_bgr_dir = clip / "right_bgr";

    try {
        fs::create_directories(left_dir);
        fs::create_directories(right_dir);
        if (image_mode_ == "both") {
            fs::create_directories(left_bgr_dir);
            fs::create_directories(right_bgr_dir);
        }
    } catch (const std::exception& e) {
        LOG_WARN("BaselineClipRecorder: failed to create %s: %s",
                 clip.string().c_str(), e.what());
        cfg_.enabled = false;
        return false;
    }

    std::ofstream csv((clip / "frames.csv").string(), std::ios::out | std::ios::trunc);
    if (!csv.is_open()) {
        LOG_WARN("BaselineClipRecorder: failed to open frames.csv under %s",
                 clip.string().c_str());
        cfg_.enabled = false;
        return false;
    }
    csv.close();
    clip_dir_ = clip.string();
    writeBaselineClipHeader(clip_dir_);
    writeBaselineClipMetadata(clip_dir_,
                              clip_number,
                              cfg_,
                              image_ext_,
                              image_mode_,
                              target_frames_,
                              gap_frames_,
                              effective_max_queue_frames_);

    current_clip_index_ = clip_number;
    clip_frame_count_ = 0;
    active_ = true;
    LOG_INFO("BaselineClipRecorder: started clip %d/%d %s",
             clip_number, cfg_.clip_count, clip_dir_.c_str());
    return true;
}

int BaselineClipRecorder::bestDetectionIndex(
    const std::vector<Detection>& detections) const {
    int best = -1;
    float best_conf = cfg_.min_confidence;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence >= best_conf) {
            best_conf = detections[i].confidence;
            best = static_cast<int>(i);
        }
    }
    return best;
}

BaselineClipRecorder::PairSelection BaselineClipRecorder::selectBestPair(
    const std::vector<Detection>& left,
    const std::vector<Detection>& right) const {
    PairSelection best;
    best.score = -std::numeric_limits<float>::infinity();
    for (size_t li = 0; li < left.size(); ++li) {
        const auto& l = left[li];
        if (l.confidence < cfg_.min_confidence || l.width <= 0.0f || l.height <= 0.0f) {
            continue;
        }
        for (size_t ri = 0; ri < right.size(); ++ri) {
            const auto& r = right[ri];
            if (r.confidence < cfg_.min_confidence || r.width <= 0.0f || r.height <= 0.0f) {
                continue;
            }
            if (l.class_id != r.class_id) continue;
            const float disp = l.cx - r.cx;
            if (disp < cfg_.pair_min_disparity_px) continue;
            const float dy = std::abs(l.cy - r.cy);
            if (dy > cfg_.pair_y_tolerance_px) continue;
            const float wr = std::max(l.width / r.width, r.width / l.width);
            const float hr = std::max(l.height / r.height, r.height / l.height);
            const float size_ratio = std::max(wr, hr);
            if (size_ratio > cfg_.pair_max_size_ratio) continue;

            const float score = l.confidence * r.confidence
                              - 0.002f * dy
                              - 0.05f * (size_ratio - 1.0f);
            if (score > best.score) {
                best.left_idx = static_cast<int>(li);
                best.right_idx = static_cast<int>(ri);
                best.valid = true;
                best.score = score;
                best.disparity_px = disp;
                best.dy_px = dy;
                best.size_ratio = size_ratio;
            }
        }
    }
    if (!best.valid) {
        best.score = 0.0f;
    }
    return best;
}

bool BaselineClipRecorder::copyGrayImage(VPIImage image, cv::Mat& out) const {
    if (!image) return false;
    VPIImageData data;
    const VPIStatus st = vpiImageLockData(image, VPI_LOCK_READ,
                                          VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                                          &data);
    if (st != VPI_SUCCESS) {
        return false;
    }
    const int h = data.buffer.pitch.planes[0].height;
    const int w = data.buffer.pitch.planes[0].width;
    const int pitch = data.buffer.pitch.planes[0].pitchBytes;
    cv::Mat view(h, w, CV_8UC1, data.buffer.pitch.planes[0].data, pitch);
    out = view.clone();
    vpiImageUnlock(image);
    return !out.empty();
}

bool BaselineClipRecorder::copyBgrImage(VPIImage image, cv::Mat& out) const {
    if (!image) return false;
    VPIImageData data;
    const VPIStatus st = vpiImageLockData(image, VPI_LOCK_READ,
                                          VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR,
                                          &data);
    if (st != VPI_SUCCESS) {
        return false;
    }
    const int h = data.buffer.pitch.planes[0].height;
    const int w = data.buffer.pitch.planes[0].width;
    const int pitch = data.buffer.pitch.planes[0].pitchBytes;
    const void* src = data.buffer.pitch.planes[0].data;
    out.create(h, w, CV_8UC3);
    const cudaError_t err = cudaMemcpy2D(out.data, out.step[0],
                                         src, pitch,
                                         static_cast<size_t>(w) * 3u,
                                         static_cast<size_t>(h),
                                         cudaMemcpyDeviceToHost);
    vpiImageUnlock(image);
    if (err != cudaSuccess) {
        out.release();
        return false;
    }
    return !out.empty();
}

void BaselineClipRecorder::record(
    int frame_id,
    VPIImage rect_gray_left,
    VPIImage rect_gray_right,
    VPIImage rect_bgr_left,
    VPIImage rect_bgr_right,
    const std::vector<Detection>& detections_left,
    const std::vector<Detection>& detections_right,
    const FrameMetadata& metadata,
    float fps) {
    if (!cfg_.enabled || complete_.load()) return;

    const PairSelection pair = selectBestPair(detections_left, detections_right);
    if (!active_.load()) {
        if (completed_clips_.load() >= cfg_.clip_count) {
            complete_ = true;
            return;
        }
        if (frame_id < next_start_frame_id_) {
            return;
        }
        if (!shouldStart(detections_left, detections_right, pair)) {
            return;
        }
        if (!startClip()) {
            return;
        }
    }

    QueuedFrame frame;
    frame.clip_dir = clip_dir_;
    frame.clip_frame_id = clip_frame_count_.load();
    frame.clip_index = current_clip_index_;
    frame.pipeline_frame_id = frame_id;
    frame.timestamp_s = nowSeconds();
    frame.fps = fps;
    frame.metadata = metadata;
    frame.detections_left = detections_left;
    frame.detections_right = detections_right;
    frame.pair = pair;
    frame.best_left_idx = pair.valid ? pair.left_idx : bestDetectionIndex(detections_left);
    frame.best_right_idx = pair.valid ? pair.right_idx : bestDetectionIndex(detections_right);

    const bool need_gray = image_mode_ == "gray" || image_mode_ == "both";
    const bool need_bgr = image_mode_ == "bgr" || image_mode_ == "both";
    if (need_gray &&
        (!copyGrayImage(rect_gray_left, frame.left_gray) ||
         !copyGrayImage(rect_gray_right, frame.right_gray))) {
        const int failures = ++copy_failures_;
        if (failures <= 3 || failures % 30 == 0) {
            LOG_WARN("BaselineClipRecorder: VPI gray copy failed at frame=%d failures=%d",
                     frame_id, failures);
        }
        return;
    }
    if (need_bgr &&
        (!copyBgrImage(rect_bgr_left, frame.left_bgr) ||
         !copyBgrImage(rect_bgr_right, frame.right_bgr))) {
        const int failures = ++copy_failures_;
        if (failures <= 3 || failures % 30 == 0) {
            LOG_WARN("BaselineClipRecorder: VPI BGR copy failed at frame=%d failures=%d",
                     frame_id, failures);
        }
        return;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        if (effective_max_queue_frames_ > 0 &&
            queue_.size() >= effective_max_queue_frames_) {
            ++dropped_frames_;
            return;
        }
        queue_.push_back(std::move(frame));
    }
    queue_cv_.notify_one();

    const int clip_recorded = ++clip_frame_count_;
    const int total_recorded = ++frame_count_;
    if (clip_recorded >= target_frames_) {
        const int finished_clip = ++completed_clips_;
        active_ = false;
        next_start_frame_id_ = frame_id + gap_frames_;
        queue_cv_.notify_one();
        LOG_INFO("BaselineClipRecorder: captured clip %d/%d frames=%d total=%d dir=%s dropped=%d",
                 finished_clip, cfg_.clip_count, clip_recorded, total_recorded,
                 clip_dir_.c_str(), dropped_frames_.load());
        if (finished_clip >= cfg_.clip_count) {
            complete_ = true;
            running_ = false;
            queue_cv_.notify_one();
            LOG_INFO("BaselineClipRecorder: completed %d clip(s), total frames=%d",
                     finished_clip, total_recorded);
        }
    }
}

void BaselineClipRecorder::close() {
    running_ = false;
    queue_cv_.notify_one();
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
    if (active_.exchange(false)) {
        LOG_INFO("BaselineClipRecorder: closed active clip %s", clip_dir_.c_str());
    }
    if (cfg_.enabled && frame_count_.load() > 0) {
        LOG_INFO("BaselineClipRecorder: saved %d frame(s) across %d clip(s) (dropped=%d)",
                 frame_count_.load(), completed_clips_.load(), dropped_frames_.load());
    }
}

}  // namespace stereo3d
