#include "main_config_loaders.h"

#include "utils/logger.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <exception>
#include <string>

stereo3d::TrajectoryPredictorConfig loadPredictorConfig(const std::string& path) {
    stereo3d::TrajectoryPredictorConfig tcfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (auto pred = root["prediction"]) {
            if (pred["gravity"])          tcfg.gravity       = pred["gravity"].as<float>();
            if (pred["air_density"])      tcfg.air_density   = pred["air_density"].as<float>();
            if (pred["ball_mass"])        tcfg.ball_mass     = pred["ball_mass"].as<float>();
            if (pred["ball_radius"])      tcfg.ball_radius   = pred["ball_radius"].as<float>();
            if (pred["drag_coeff"])       tcfg.drag_coeff    = pred["drag_coeff"].as<float>();
            if (pred["ground_y"])         tcfg.ground_y      = pred["ground_y"].as<float>();
            if (pred["max_predict_time"]) tcfg.max_predict_time = pred["max_predict_time"].as<float>();
            if (pred["rk4_dt"])           tcfg.rk4_dt        = pred["rk4_dt"].as<float>();
            if (pred["poly_min_frames"])  tcfg.poly_min_frames = pred["poly_min_frames"].as<int>();
            if (pred["history_max"])      tcfg.history_max   = pred["history_max"].as<int>();
            if (pred["min_speed"])        tcfg.min_speed_for_predict = pred["min_speed"].as<float>();
        }
    } catch (const std::exception& e) {
        LOG_WARN("prediction config: %s, using defaults", e.what());
    } catch (...) { LOG_WARN("prediction config: unknown error, using defaults"); }
    return tcfg;
}

stereo3d::TrajectoryRecorderConfig loadRecorderConfig(const std::string& path) {
    stereo3d::TrajectoryRecorderConfig rcfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (auto rec = root["recording"]) {
            if (rec["enabled"])     rcfg.enabled     = rec["enabled"].as<bool>();
            if (rec["output_path"]) rcfg.output_path = rec["output_path"].as<std::string>();
            if (rec["raw_mode"])    rcfg.raw_mode    = rec["raw_mode"].as<bool>();
            if (rec["frame_summary_enabled"])
                rcfg.frame_summary_enabled =
                    rec["frame_summary_enabled"].as<bool>();
            if (rec["frame_summary_path"])
                rcfg.frame_summary_path =
                    rec["frame_summary_path"].as<std::string>();
            if (rec["max_queue_frames"])
                rcfg.max_queue_frames = rec["max_queue_frames"].as<size_t>();
            if (rec["detail_level"]) {
                std::string level = rec["detail_level"].as<std::string>();
                std::transform(level.begin(), level.end(), level.begin(),
                               [](unsigned char c) {
                                   return static_cast<char>(std::tolower(c));
                               });
                if (level == "legacy" || level == "basic") {
                    rcfg.detail_level = stereo3d::TrajectoryRecordDetail::LEGACY;
                } else if (level == "depth_candidates" || level == "candidates" ||
                           level == "depth") {
                    rcfg.detail_level =
                        stereo3d::TrajectoryRecordDetail::DEPTH_CANDIDATES;
                } else if (level == "extended" || level == "full") {
                    rcfg.detail_level = stereo3d::TrajectoryRecordDetail::EXTENDED;
                } else {
                    LOG_WARN("recording.detail_level=%s unknown, using legacy",
                             level.c_str());
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_WARN("recording config: %s, using defaults", e.what());
    } catch (...) { LOG_WARN("recording config: unknown error, using defaults"); }
    return rcfg;
}

stereo3d::BaselineClipRecorderConfig loadBaselineClipRecorderConfig(
    const std::string& path) {
    stereo3d::BaselineClipRecorderConfig cfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        auto rec = root["baseline_recording"];
        if (!rec) rec = root["baseline_clip_recording"];
        if (rec) {
            if (rec["enabled"]) cfg.enabled = rec["enabled"].as<bool>();
            if (rec["output_dir"]) cfg.output_dir = rec["output_dir"].as<std::string>();
            if (rec["duration_sec"]) cfg.duration_sec = rec["duration_sec"].as<double>();
            if (rec["frame_limit"]) cfg.frame_limit = rec["frame_limit"].as<int>();
            if (rec["clip_count"]) cfg.clip_count = rec["clip_count"].as<int>();
            if (rec["clip_gap_sec"]) cfg.clip_gap_sec = rec["clip_gap_sec"].as<double>();
            if (rec["clip_gap_frames"]) cfg.clip_gap_frames = rec["clip_gap_frames"].as<int>();
            if (rec["require_left_detection"])
                cfg.require_left_detection = rec["require_left_detection"].as<bool>();
            if (rec["require_right_detection"])
                cfg.require_right_detection = rec["require_right_detection"].as<bool>();
            if (rec["require_pair_gate"])
                cfg.require_pair_gate = rec["require_pair_gate"].as<bool>();
            if (rec["min_confidence"]) cfg.min_confidence = rec["min_confidence"].as<float>();
            if (rec["pair_y_tolerance_px"])
                cfg.pair_y_tolerance_px = rec["pair_y_tolerance_px"].as<float>();
            if (rec["pair_max_size_ratio"])
                cfg.pair_max_size_ratio = rec["pair_max_size_ratio"].as<float>();
            if (rec["pair_min_disparity_px"])
                cfg.pair_min_disparity_px = rec["pair_min_disparity_px"].as<float>();
            if (rec["image_format"]) cfg.image_format = rec["image_format"].as<std::string>();
            if (rec["image_mode"]) cfg.image_mode = rec["image_mode"].as<std::string>();
            if (rec["png_compression"])
                cfg.png_compression = rec["png_compression"].as<int>();
            if (rec["write_after_capture"])
                cfg.write_after_capture = rec["write_after_capture"].as<bool>();
            if (rec["stop_after_clip"])
                cfg.stop_after_clip = rec["stop_after_clip"].as<bool>();
            if (rec["max_queue_frames"])
                cfg.max_queue_frames = rec["max_queue_frames"].as<size_t>();
        }
    } catch (const std::exception& e) {
        LOG_WARN("baseline_recording config: %s, using defaults", e.what());
    } catch (...) {
        LOG_WARN("baseline_recording config: unknown error, using defaults");
    }
    return cfg;
}

RealtimeDebugDumpConfig loadRealtimeDebugDumpConfig(const std::string& path) {
    RealtimeDebugDumpConfig cfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        auto node = root["debug_realtime_dump"];
        if (!node) node = root["realtime_debug_dump"];
        if (!node) return cfg;
        if (node["enabled"]) cfg.enabled = node["enabled"].as<bool>();
        if (node["output_dir"]) cfg.output_dir = node["output_dir"].as<std::string>();
        if (node["stride"]) cfg.stride = node["stride"].as<int>();
        if (node["max_frames"]) cfg.max_frames = node["max_frames"].as<int>();
        if (node["max_queue"]) cfg.max_queue = node["max_queue"].as<int>();
        if (node["dump_fallback"]) cfg.dump_fallback = node["dump_fallback"].as<bool>();
    } catch (const std::exception& e) {
        LOG_WARN("debug_realtime_dump config: %s, using defaults", e.what());
    } catch (...) {
        LOG_WARN("debug_realtime_dump config: unknown error, using defaults");
    }
    cfg.stride = std::max(0, cfg.stride);
    cfg.max_frames = std::max(0, cfg.max_frames);
    cfg.max_queue = std::max(1, cfg.max_queue);
    return cfg;
}
