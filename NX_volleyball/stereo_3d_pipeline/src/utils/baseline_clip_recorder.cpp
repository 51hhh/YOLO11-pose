/**
 * @file baseline_clip_recorder.cpp
 * @brief Fixed-length stereo image sequence recorder for offline feature tests.
 */

#include "baseline_clip_recorder.h"
#include "logger.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <sstream>

namespace stereo3d {
namespace {

double nowSeconds() {
    const auto now = std::chrono::system_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

std::string timestampName() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string frameName(int frame_id, const std::string& ext) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << frame_id << "." << ext;
    return oss.str();
}

std::string normalizeImageFormat(std::string fmt) {
    std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!fmt.empty() && fmt[0] == '.') {
        fmt.erase(fmt.begin());
    }
    if (fmt != "png" && fmt != "pgm") {
        LOG_WARN("BaselineClipRecorder: unsupported image_format=%s, using png", fmt.c_str());
        return "png";
    }
    return fmt;
}

std::string normalizeImageMode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode != "gray" && mode != "bgr" && mode != "both") {
        LOG_WARN("BaselineClipRecorder: unsupported image_mode=%s, using gray",
                 mode.c_str());
        return "gray";
    }
    return mode;
}

void writeDetectionColumns(std::ofstream& csv, const Detection* det) {
    if (!det) {
        csv << "-1,-1,-1,-1,0,-1";
        return;
    }
    csv << std::fixed << std::setprecision(3)
        << det->cx << ','
        << det->cy << ','
        << det->width << ','
        << det->height << ','
        << std::setprecision(5) << det->confidence << ','
        << det->class_id;
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
    image_ext_ = normalizeImageFormat(cfg_.image_format);
    image_mode_ = normalizeImageMode(cfg_.image_mode);
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
    clip_name << "clip_" << timestampName() << "_"
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
    writeHeader(clip_dir_);

    std::ofstream meta((clip / "metadata.yaml").string(), std::ios::out | std::ios::trunc);
    if (meta.is_open()) {
        meta << "format: image_sequence_csv\n"
             << "clip_index: " << clip_number << "\n"
             << "clip_count: " << cfg_.clip_count << "\n"
             << "image_format: " << image_ext_ << "\n"
             << "image_mode: " << image_mode_ << "\n"
             << "duration_sec: " << std::fixed << std::setprecision(3) << cfg_.duration_sec << "\n"
             << "target_frames: " << target_frames_ << "\n"
             << "trigger_hz: " << cfg_.trigger_hz << "\n"
             << "clip_gap_sec: " << cfg_.clip_gap_sec << "\n"
             << "clip_gap_frames: " << gap_frames_ << "\n"
             << "require_left_detection: " << (cfg_.require_left_detection ? "true" : "false") << "\n"
             << "require_right_detection: " << (cfg_.require_right_detection ? "true" : "false") << "\n"
             << "require_pair_gate: " << (cfg_.require_pair_gate ? "true" : "false") << "\n"
             << "min_confidence: " << cfg_.min_confidence << "\n"
             << "pair_y_tolerance_px: " << cfg_.pair_y_tolerance_px << "\n"
             << "pair_max_size_ratio: " << cfg_.pair_max_size_ratio << "\n"
             << "pair_min_disparity_px: " << cfg_.pair_min_disparity_px << "\n"
             << "write_after_capture: " << (cfg_.write_after_capture ? "true" : "false") << "\n"
             << "max_queue_frames: " << cfg_.max_queue_frames << "\n";
        if (effective_max_queue_frames_ != cfg_.max_queue_frames) {
            meta << "effective_max_queue_frames: " << effective_max_queue_frames_ << "\n";
        }
    }

    current_clip_index_ = clip_number;
    clip_frame_count_ = 0;
    active_ = true;
    LOG_INFO("BaselineClipRecorder: started clip %d/%d %s",
             clip_number, cfg_.clip_count, clip_dir_.c_str());
    return true;
}

void BaselineClipRecorder::writeHeader(const std::string& clip_dir) {
    namespace fs = std::filesystem;
    std::ofstream csv((fs::path(clip_dir) / "frames.csv").string(),
                      std::ios::out | std::ios::trunc);
    if (!csv.is_open()) return;
    csv << "clip_frame_id,pipeline_frame_id,timestamp_s,fps,"
        << "left_image,right_image,left_bgr_image,right_bgr_image,"
        << "left_count,right_count,"
        << "best_left_idx,best_right_idx,pair_valid,pair_score,"
        << "pair_disparity_px,pair_dy_px,pair_size_ratio,"
        << "left_cx,left_cy,left_w,left_h,left_conf,left_class_id,"
        << "right_cx,right_cy,right_w,right_h,right_conf,right_class_id,"
        << "left_timestamp_us,right_timestamp_us,"
        << "left_frame_number,right_frame_number,"
        << "left_frame_counter,right_frame_counter,"
        << "left_trigger_index,right_trigger_index,"
        << "frame_counter_delta,frame_number_delta,timestamp_delta_us,"
        << "grab_failed,is_detect_frame\n";
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
    const std::string name = frameName(frame.clip_frame_id, image_ext_);
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
    writeDetectionColumns(csv, left_det);
    csv << ',';
    writeDetectionColumns(csv, right_det);
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
