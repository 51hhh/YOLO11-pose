#include "main_config_loaders.h"

#include "utils/logger.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <exception>
#include <string>
#include <filesystem>
#include <fstream>
#include <limits>
#include <cstdlib>
#include <iterator>
#include <opencv2/core.hpp>

stereo3d::TrajectoryPredictorConfig loadPredictorConfig(const std::string& path) {
    stereo3d::TrajectoryPredictorConfig tcfg;
    try {
        YAML::Node root = YAML::LoadFile(path);

        // Geometry from stereo calibration + runtime d0, so the predictor can
        // rebuild bbox_center observations independently of HybridDepth state.
        std::string calib_file;
        if (auto cal = root["calibration"]) {
            if (cal["file"]) calib_file = cal["file"].as<std::string>();
        }
        if (calib_file.empty() && root["calibration_file"]) {
            calib_file = root["calibration_file"].as<std::string>();
        }
        if (!calib_file.empty()) {
            try {
                // Resolve relative to the config file directory first, then CWD.
                namespace fs = std::filesystem;
                fs::path cfg_path(path);
                fs::path cand = fs::path(calib_file);
                if (!cand.is_absolute()) {
                    fs::path near_cfg = cfg_path.parent_path() / cand;
                    fs::path near_root = cfg_path.parent_path().parent_path() / cand;
                    if (fs::exists(near_cfg)) cand = near_cfg;
                    else if (fs::exists(near_root)) cand = near_root;
                }
                cv::FileStorage fs_cal(cand.string(), cv::FileStorage::READ);
                if (fs_cal.isOpened()) {
                    cv::Mat P1;
                    fs_cal["projection_left"] >> P1;
                    double baseline_mm = 0.0;
                    fs_cal["baseline"] >> baseline_mm;
                    if (!P1.empty()) {
                        tcfg.fx = static_cast<float>(P1.at<double>(0, 0));
                        tcfg.fy = static_cast<float>(P1.at<double>(1, 1));
                        tcfg.cx = static_cast<float>(P1.at<double>(0, 2));
                        tcfg.cy = static_cast<float>(P1.at<double>(1, 2));
                        if (baseline_mm > 0.0) {
                            tcfg.fB = tcfg.fx * static_cast<float>(baseline_mm / 1000.0);
                        } else {
                            cv::Mat P2;
                            fs_cal["projection_right"] >> P2;
                            if (!P2.empty() && std::abs(tcfg.fx) > 1e-6f) {
                                const float tx = static_cast<float>(P2.at<double>(0, 3));
                                // OpenCV stereo: P2(0,3) = -fx * baseline_mm
                                const float baseline_m = std::abs(tx / tcfg.fx) / 1000.0f;
                                tcfg.fB = tcfg.fx * baseline_m;
                            }
                        }
                        tcfg.have_geometry = (tcfg.fx > 1.0f && tcfg.fB > 1.0f);
                    }
                    fs_cal.release();
                } else {
                    LOG_WARN("prediction config: cannot open calibration file %s",
                             cand.string().c_str());
                }
            } catch (const std::exception& e) {
                LOG_WARN("prediction config: calibration load failed: %s", e.what());
            }
        }

        if (auto d0 = root["disparity_offset"]) {
            bool enabled = true;
            if (d0["enabled"]) enabled = d0["enabled"].as<bool>();
            if (enabled) {
                if (d0["d0"]) tcfg.d0 = d0["d0"].as<float>();
                if (d0["file"]) {
                    // Reuse PipelineConfig loader semantics via lightweight JSON/YAML parse.
                    // Supports the project fit JSON schema.
                    try {
                        namespace fs = std::filesystem;
                        fs::path cfg_path(path);
                        fs::path cand = fs::path(d0["file"].as<std::string>());
                        if (!cand.is_absolute()) {
                            fs::path near_cfg = cfg_path.parent_path() / cand;
                            fs::path near_root = cfg_path.parent_path().parent_path() / cand;
                            if (fs::exists(near_cfg)) cand = near_cfg;
                            else if (fs::exists(near_root)) cand = near_root;
                        }
                        // Prefer OpenCV FileStorage for YAML; for JSON use a tiny scan.
                        std::ifstream in(cand.string());
                        if (in) {
                            std::string content((std::istreambuf_iterator<char>(in)),
                                                std::istreambuf_iterator<char>());
                            auto grab = [&](const std::string& key) -> float {
                                const std::string pat = "\"" + key + "\"";
                                auto p = content.find(pat);
                                if (p == std::string::npos) return std::numeric_limits<float>::quiet_NaN();
                                p = content.find(':', p);
                                if (p == std::string::npos) return std::numeric_limits<float>::quiet_NaN();
                                return std::strtof(content.c_str() + p + 1, nullptr);
                            };
                            const float d0v = grab("d0");
                            if (std::isfinite(d0v)) tcfg.d0 = d0v;
                            // fitted fB is metadata only; runtime keeps calibration fB.
                        }
                    } catch (...) {
                        LOG_WARN("prediction config: disparity_offset file parse failed");
                    }
                }
            }
        }

        if (auto pred = root["prediction"]) {
            if (pred["enable"])           tcfg.enabled       = pred["enable"].as<bool>();
            if (pred["gravity"])          tcfg.gravity       = pred["gravity"].as<float>();
            if (pred["air_density"])      tcfg.air_density   = pred["air_density"].as<float>();
            if (pred["ball_mass"])        tcfg.ball_mass     = pred["ball_mass"].as<float>();
            if (pred["ball_radius"])      tcfg.ball_radius   = pred["ball_radius"].as<float>();
            if (pred["drag_coeff"])       tcfg.drag_coeff    = pred["drag_coeff"].as<float>();
            if (pred["ground_y"])         tcfg.ground_y      = pred["ground_y"].as<float>();
            if (pred["ground_h"])         tcfg.ground_h      = pred["ground_h"].as<float>();
            if (pred["max_predict_time"]) tcfg.max_predict_time = pred["max_predict_time"].as<float>();
            if (pred["rk4_dt"])           tcfg.rk4_dt        = pred["rk4_dt"].as<float>();
            if (pred["poly_min_frames"])  tcfg.poly_min_frames = pred["poly_min_frames"].as<int>();
            if (pred["history_max"])      tcfg.history_max   = pred["history_max"].as<int>();
            if (pred["min_speed"])        tcfg.min_speed_for_predict = pred["min_speed"].as<float>();
            if (pred["use_student_t_ekf"]) tcfg.use_student_t_ekf = pred["use_student_t_ekf"].as<bool>();
            if (pred["student_t_nu"])     tcfg.student_t_nu = pred["student_t_nu"].as<float>();
            if (pred["sigma_d_px"])       tcfg.sigma_d_px = pred["sigma_d_px"].as<float>();
            if (pred["q_pos"])            tcfg.q_pos = pred["q_pos"].as<float>();
            if (pred["q_vel"])            tcfg.q_vel = pred["q_vel"].as<float>();
            if (pred["max_dt"])           tcfg.max_dt = pred["max_dt"].as<float>();
            if (pred["prefer_bbox"])      tcfg.prefer_bbox = pred["prefer_bbox"].as<bool>();
            if (pred["enable_circle_fallback"])
                tcfg.enable_circle_fallback = pred["enable_circle_fallback"].as<bool>();
            if (pred["circle_consistency_m"])
                tcfg.circle_consistency_m = pred["circle_consistency_m"].as<float>();
            if (pred["allow_raw_fallback"])
                tcfg.allow_raw_fallback = pred["allow_raw_fallback"].as<bool>();
            if (pred["allow_filtered_fallback"])
                tcfg.allow_filtered_fallback = pred["allow_filtered_fallback"].as<bool>();
            if (pred["min_height_for_predict"])
                tcfg.min_height_for_predict = pred["min_height_for_predict"].as<float>();
            if (pred["use_g_hat"])        tcfg.use_g_hat = pred["use_g_hat"].as<bool>();
            if (pred["g_hat"] && pred["g_hat"].IsSequence() && pred["g_hat"].size() >= 3) {
                tcfg.g_hat_x = pred["g_hat"][0].as<float>();
                tcfg.g_hat_y = pred["g_hat"][1].as<float>();
                tcfg.g_hat_z = pred["g_hat"][2].as<float>();
                tcfg.use_g_hat = true;
            }
            if (pred["d0"]) tcfg.d0 = pred["d0"].as<float>();
            if (pred["fx"]) tcfg.fx = pred["fx"].as<float>();
            if (pred["fy"]) tcfg.fy = pred["fy"].as<float>();
            if (pred["cx"]) tcfg.cx = pred["cx"].as<float>();
            if (pred["cy"]) tcfg.cy = pred["cy"].as<float>();
            if (pred["fB"]) tcfg.fB = pred["fB"].as<float>();
            if (tcfg.fx > 1.0f && tcfg.fB > 1.0f) tcfg.have_geometry = true;
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
