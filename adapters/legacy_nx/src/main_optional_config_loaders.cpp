#include "main_config_loaders.h"

#include "volleyball/logging/log.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <string>

stereo3d::CaptureV21StageRecorderConfig loadCaptureV21StageRecorderConfig(
    const std::string& path) {
    stereo3d::CaptureV21StageRecorderConfig cfg;
    try {
        const YAML::Node root = YAML::LoadFile(path);
        const YAML::Node capture = root["capture_v21"];
        if (!capture) return cfg;
        if (capture["enabled"]) cfg.enabled = capture["enabled"].as<bool>();
        if (capture["output_dir"])
            cfg.output_dir = capture["output_dir"].as<std::string>();
        if (capture["frame_limit"])
            cfg.frame_limit = capture["frame_limit"].as<int>();
        if (capture["queue_capacity"])
            cfg.queue_capacity = capture["queue_capacity"].as<std::size_t>();
        if (capture["overflow_policy"])
            cfg.overflow_policy = capture["overflow_policy"].as<std::string>();
        if (capture["raw_frames"])
            cfg.raw_frames = capture["raw_frames"].as<std::string>();
        if (capture["image_product"])
            cfg.image_product = capture["image_product"].as<std::string>();
        if (capture["image_format"])
            cfg.image_format = capture["image_format"].as<std::string>();
        if (capture["left_camera_id"])
            cfg.left_camera_id = capture["left_camera_id"].as<std::string>();
        if (capture["right_camera_id"])
            cfg.right_camera_id = capture["right_camera_id"].as<std::string>();
        if (capture["pipeline_id"])
            cfg.pipeline_id = capture["pipeline_id"].as<std::string>();
        if (capture["loss_policy"])
            cfg.loss_policy = capture["loss_policy"].as<std::string>();
        if (capture["exposure_timestamp_source"])
            cfg.exposure_timestamp_source =
                capture["exposure_timestamp_source"].as<std::string>();
    } catch (const std::exception& error) {
        LOG_ERROR("capture_v21 config: %s", error.what());
        cfg.valid = false;
        cfg.enabled = false;
    } catch (...) {
        LOG_ERROR("capture_v21 config: unknown error");
        cfg.valid = false;
        cfg.enabled = false;
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
