/**
 * @file baseline_clip_recorder_io.cpp
 * @brief Baseline clip recorder file naming and CSV/metadata helpers.
 */

#include "baseline_clip_recorder_io.h"

#include "baseline_clip_recorder.h"
#include "logger.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace stereo3d {

std::string baselineClipTimestampName() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string baselineClipFrameName(int frame_id, const std::string& ext) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << frame_id << "." << ext;
    return oss.str();
}

std::string normalizeBaselineImageFormat(std::string fmt) {
    std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (!fmt.empty() && fmt[0] == '.') {
        fmt.erase(fmt.begin());
    }
    if (fmt != "png" && fmt != "pgm") {
        LOG_WARN("BaselineClipRecorder: unsupported image_format=%s, using png",
                 fmt.c_str());
        return "png";
    }
    return fmt;
}

std::string normalizeBaselineImageMode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (mode != "gray" && mode != "bgr" && mode != "both") {
        LOG_WARN("BaselineClipRecorder: unsupported image_mode=%s, using gray",
                 mode.c_str());
        return "gray";
    }
    return mode;
}

void writeBaselineClipHeader(const std::string& clip_dir) {
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

void writeBaselineClipMetadata(const std::string& clip_dir,
                               int clip_number,
                               const BaselineClipRecorderConfig& cfg,
                               const std::string& image_ext,
                               const std::string& image_mode,
                               int target_frames,
                               int gap_frames,
                               std::size_t effective_max_queue_frames) {
    namespace fs = std::filesystem;
    std::ofstream meta((fs::path(clip_dir) / "metadata.yaml").string(),
                       std::ios::out | std::ios::trunc);
    if (!meta.is_open()) return;

    meta << "format: image_sequence_csv\n"
         << "clip_index: " << clip_number << "\n"
         << "clip_count: " << cfg.clip_count << "\n"
         << "image_format: " << image_ext << "\n"
         << "image_mode: " << image_mode << "\n"
         << "duration_sec: " << std::fixed << std::setprecision(3)
         << cfg.duration_sec << "\n"
         << "target_frames: " << target_frames << "\n"
         << "trigger_hz: " << cfg.trigger_hz << "\n"
         << "clip_gap_sec: " << cfg.clip_gap_sec << "\n"
         << "clip_gap_frames: " << gap_frames << "\n"
         << "require_left_detection: "
         << (cfg.require_left_detection ? "true" : "false") << "\n"
         << "require_right_detection: "
         << (cfg.require_right_detection ? "true" : "false") << "\n"
         << "require_pair_gate: "
         << (cfg.require_pair_gate ? "true" : "false") << "\n"
         << "min_confidence: " << cfg.min_confidence << "\n"
         << "pair_y_tolerance_px: " << cfg.pair_y_tolerance_px << "\n"
         << "pair_max_size_ratio: " << cfg.pair_max_size_ratio << "\n"
         << "pair_min_disparity_px: " << cfg.pair_min_disparity_px << "\n"
         << "write_after_capture: "
         << (cfg.write_after_capture ? "true" : "false") << "\n"
         << "max_queue_frames: " << cfg.max_queue_frames << "\n";
    if (effective_max_queue_frames != cfg.max_queue_frames) {
        meta << "effective_max_queue_frames: "
             << effective_max_queue_frames << "\n";
    }
}

void writeBaselineDetectionColumns(std::ostream& os, const Detection* det) {
    if (!det) {
        os << "-1,-1,-1,-1,0,-1";
        return;
    }
    os << std::fixed << std::setprecision(3)
       << det->cx << ','
       << det->cy << ','
       << det->width << ','
       << det->height << ','
       << std::setprecision(5) << det->confidence << ','
       << det->class_id;
}

}  // namespace stereo3d
