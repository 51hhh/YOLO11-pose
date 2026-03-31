/**
 * @file main.cpp
 * @brief stereo_3d_pipeline 入口
 *
 * 加载 pipeline.yaml 配置 → 初始化 Pipeline → 运行 → 信号退出
 */

#include "pipeline/pipeline.h"
#include "utils/logger.h"

#include <yaml-cpp/yaml.h>
#include <csignal>
#include <cstdio>
#include <atomic>
#include <string>

// ==================== 全局信号 ====================

static std::atomic<bool> g_shutdown{false};

static void signalHandler(int sig) {
    (void)sig;
    g_shutdown.store(true);
}

// ==================== 配置加载 ====================

static stereo3d::PipelineConfig loadConfig(const std::string& path) {
    stereo3d::PipelineConfig cfg;

    YAML::Node root = YAML::LoadFile(path);

    // Camera
    if (auto cam = root["camera"]) {
        if (cam["serial_left"])       cfg.cam_left_serial  = cam["serial_left"].as<std::string>();
        if (cam["serial_right"])      cfg.cam_right_serial = cam["serial_right"].as<std::string>();
        if (cam["left_index"])        cfg.cam_left_index   = cam["left_index"].as<int>();
        if (cam["right_index"])       cfg.cam_right_index  = cam["right_index"].as<int>();
        if (cam["exposure_us"])       cfg.exposure_us = cam["exposure_us"].as<float>();
        if (cam["gain_db"])           cfg.gain_db = cam["gain_db"].as<float>();
        if (cam["use_trigger"])       cfg.use_trigger = cam["use_trigger"].as<bool>();
        if (cam["trigger_source"])    cfg.trigger_source = cam["trigger_source"].as<std::string>();
        if (cam["trigger_activation"]) cfg.trigger_activation = cam["trigger_activation"].as<std::string>();
        if (cam["width"])             cfg.raw_width  = cam["width"].as<int>();
        if (cam["height"])            cfg.raw_height = cam["height"].as<int>();
    }

    // Calibration
    if (auto cal = root["calibration"]) {
        if (cal["file"]) cfg.calibration_file = cal["file"].as<std::string>();
    }

    // Rectify (输出分辨率, 与相机原始分辨率分离)
    if (auto rect = root["rectify"]) {
        if (rect["output_width"])  cfg.rect_width  = rect["output_width"].as<int>();
        if (rect["output_height"]) cfg.rect_height = rect["output_height"].as<int>();
    }

    // Detector
    if (auto det = root["detector"]) {
        if (det["engine_path"])            cfg.engine_file = det["engine_path"].as<std::string>();
        if (det["use_dla"])                cfg.use_dla = det["use_dla"].as<bool>();
        if (det["dla_core"])               cfg.dla_core = det["dla_core"].as<int>();
        if (det["confidence_threshold"])   cfg.conf_threshold = det["confidence_threshold"].as<float>();
        if (det["nms_threshold"])           cfg.nms_threshold = det["nms_threshold"].as<float>();
        if (det["input_size"])             cfg.input_size = det["input_size"].as<int>();
        if (det["max_detections"])         cfg.max_detections = det["max_detections"].as<int>();
    }

    // Stereo
    if (auto st = root["stereo"]) {
        if (st["max_disparity"]) cfg.max_disparity = st["max_disparity"].as<int>();
        if (st["window_size"])   cfg.window_size   = st["window_size"].as<int>();
        if (st["quality"])       cfg.stereo_quality = st["quality"].as<int>();
        if (st["use_half_resolution"] && st["use_half_resolution"].as<bool>()) {
            cfg.disparity_strategy = stereo3d::DisparityStrategy::HALF_RESOLUTION;
        }
    }

    // Fusion
    if (auto fus = root["fusion"]) {
        if (fus["min_depth"]) cfg.min_depth = fus["min_depth"].as<float>();
        if (fus["max_depth"]) cfg.max_depth = fus["max_depth"].as<float>();
    }

    // Performance
    if (auto perf = root["performance"]) {
        if (perf["log_interval"])   cfg.stats_interval = perf["log_interval"].as<int>();
        if (perf["pwm_frequency"])  cfg.trigger_freq_hz = static_cast<int>(perf["pwm_frequency"].as<float>());
    }

    return cfg;
}

// ==================== main ====================

int main(int argc, char* argv[]) {
    // 解析命令行
    std::string config_path = "config/pipeline.yaml";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [--config <path>]\n", argv[0]);
            return 0;
        }
    }

    // 注册信号
    signal(SIGINT,  signalHandler);
    signal(SIGTERM, signalHandler);

    LOG_INFO("=== stereo_3d_pipeline ===");
    LOG_INFO("Config: %s", config_path.c_str());

    // 加载配置
    stereo3d::PipelineConfig cfg;
    try {
        cfg = loadConfig(config_path);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load config: %s", e.what());
        return 1;
    }

    // 初始化 Pipeline
    stereo3d::Pipeline pipeline;

    pipeline.setResultCallback(
        [](int frame_id, const std::vector<stereo3d::Object3D>& results) {
            for (const auto& obj : results) {
                LOG_INFO("[Frame %d] Ball: (%.2f, %.2f, %.2f) m, conf=%.2f",
                         frame_id, obj.x, obj.y, obj.z, obj.confidence);
            }
        });

    if (!pipeline.init(cfg)) {
        LOG_ERROR("Pipeline init failed");
        return 1;
    }

    LOG_INFO("Pipeline initialized, starting...");

    // 启动
    pipeline.start();

    // 主线程等待退出信号
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    LOG_INFO("Shutting down...");
    pipeline.stop();
    pipeline.printPerfReport();

    LOG_INFO("Done.");
    return 0;
}
