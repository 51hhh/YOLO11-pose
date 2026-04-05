/**
 * @file main.cpp
 * @brief stereo_3d_pipeline 入口
 *
 * 加载 pipeline.yaml 配置 → 初始化 Pipeline → 运行 → 信号退出
 */

#include "pipeline/pipeline.h"
#include "utils/logger.h"

#include <vpi/Image.h>
#include <vpi/algo/TemporalNoiseReduction.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <csignal>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

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
        if (cam["trigger_chip"])       cfg.trigger_chip = cam["trigger_chip"].as<std::string>();
        if (cam["trigger_line"])       cfg.trigger_line = cam["trigger_line"].as<int>();
        if (cam["width"])             cfg.raw_width  = cam["width"].as<int>();
        if (cam["height"])            cfg.raw_height = cam["height"].as<int>();
    }

    // Calibration
    if (auto cal = root["calibration"]) {
        if (cal["file"]) cfg.calibration_file = cal["file"].as<std::string>();
    }

    // Rectify (输出分辨率, 与相机原始分辨率分离)
    // Rectify (输出分辨率, 与相机原始分辨率分离)
    if (auto rect = root["rectify"]) {
        if (rect["output_width"])  cfg.rect_width  = rect["output_width"].as<int>();
        if (rect["output_height"]) cfg.rect_height = rect["output_height"].as<int>();
        if (rect["backend"]) cfg.rect_backend = rect["backend"].as<std::string>();
    }

    // VPI TNR (时域降噪)
    if (auto tnr = root["tnr"]) {
        if (tnr["enabled"]) cfg.tnr_enabled = tnr["enabled"].as<bool>();
        if (tnr["strength"]) cfg.tnr_strength = tnr["strength"].as<float>();
        if (tnr["preset"]) {
            std::string p = tnr["preset"].as<std::string>();
            if (p == "indoor_low")         cfg.tnr_preset = VPI_TNR_PRESET_INDOOR_LOW_LIGHT;
            else if (p == "indoor_high")   cfg.tnr_preset = VPI_TNR_PRESET_INDOOR_HIGH_LIGHT;
            else if (p == "outdoor_low")   cfg.tnr_preset = VPI_TNR_PRESET_OUTDOOR_LOW_LIGHT;
            else if (p == "outdoor_medium") cfg.tnr_preset = VPI_TNR_PRESET_OUTDOOR_MEDIUM_LIGHT;
            else if (p == "outdoor_high")  cfg.tnr_preset = VPI_TNR_PRESET_OUTDOOR_HIGH_LIGHT;
        }
        if (tnr["version"]) {
            std::string v = tnr["version"].as<std::string>();
            if (v == "v1")      cfg.tnr_version = VPI_TNR_V1;
            else if (v == "v2") cfg.tnr_version = VPI_TNR_V2;
            else if (v == "v3") cfg.tnr_version = VPI_TNR_V3;
            else                cfg.tnr_version = VPI_TNR_DEFAULT;
        }
    }

    // Detector
    if (auto det = root["detector"]) {
        if (det["engine_path"])            cfg.engine_file = det["engine_path"].as<std::string>();
        if (det["use_dla"])                cfg.use_dla = det["use_dla"].as<bool>();
        if (det["dla_core"])               cfg.dla_core = det["dla_core"].as<int>();
        if (det["dual_dla"])               cfg.dual_dla = det["dual_dla"].as<bool>();
        if (det["engine_path_dla1"])       cfg.engine_file_dla1 = det["engine_path_dla1"].as<std::string>();
        if (det["triple_backend"])         cfg.triple_backend = det["triple_backend"].as<bool>();
        if (det["engine_path_gpu"])        cfg.engine_file_gpu = det["engine_path_gpu"].as<std::string>();
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

        // 策略选择: roi_only > use_half_resolution > full_frame
        if (st["strategy"]) {
            std::string s = st["strategy"].as<std::string>();
            if (s == "roi_only" || s == "ROI_ONLY")
                cfg.disparity_strategy = stereo3d::DisparityStrategy::ROI_ONLY;
            else if (s == "half_resolution" || s == "HALF_RESOLUTION")
                cfg.disparity_strategy = stereo3d::DisparityStrategy::HALF_RESOLUTION;
            else
                cfg.disparity_strategy = stereo3d::DisparityStrategy::FULL_FRAME;
        } else if (st["use_half_resolution"] && st["use_half_resolution"].as<bool>()) {
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
    bool enable_display = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--visualize" || arg == "--display" || arg == "-v") {
            enable_display = true;
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [--config <path>] [--visualize]\n", argv[0]);
            printf("  --config, -c    Pipeline configuration YAML\n");
            printf("  --visualize, -v Show detection + distance overlay window\n");
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
            // 降低日志开销: 仅每50帧打印一次
            if (frame_id % 50 != 0) return;
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

    // 可视化: 共享显示缓冲区
    std::mutex display_mutex;
    cv::Mat display_frame;
    bool display_ready = false;

    if (enable_display) {
        pipeline.setFrameCallback(
            [&](int frame_id, VPIImage rectL,
                const std::vector<stereo3d::Detection>& detections,
                const std::vector<stereo3d::Object3D>& results,
                float fps) {
                // UI线程未消费上一帧时跳过，避免可视化拖慢主流水线
                {
                    std::lock_guard<std::mutex> lock(display_mutex);
                    if (display_ready) return;
                }

                // 从 VPI Image 拷贝到 cv::Mat
                VPIImageData imgData;
                if (vpiImageLockData(rectL, VPI_LOCK_READ,
                    VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData) != VPI_SUCCESS)
                    return;

                int h = imgData.buffer.pitch.planes[0].height;
                int w = imgData.buffer.pitch.planes[0].width;
                int pitch = imgData.buffer.pitch.planes[0].pitchBytes;
                cv::Mat gray(h, w, CV_8UC1,
                             imgData.buffer.pitch.planes[0].data, pitch);
                cv::Mat frame;
                cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
                vpiImageUnlock(rectL);

                // 绘制检测框 + 距离
                for (size_t i = 0; i < detections.size(); ++i) {
                    const auto& d = detections[i];
                    int x1 = static_cast<int>(d.cx - d.width / 2);
                    int y1 = static_cast<int>(d.cy - d.height / 2);
                    int bw = static_cast<int>(d.width);
                    int bh = static_cast<int>(d.height);
                    cv::Scalar color(0, 255, 0);
                    cv::rectangle(frame, cv::Rect(x1, y1, bw, bh), color, 2);

                    char label[128];
                    if (i < results.size()) {
                        snprintf(label, sizeof(label),
                                 "%.2fm (%.0f%%)", results[i].z, d.confidence * 100);
                    } else {
                        snprintf(label, sizeof(label),
                                 "conf=%.0f%%", d.confidence * 100);
                    }
                    cv::putText(frame, label, cv::Point(x1, y1 - 8),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

                    // 3D 坐标文本
                    if (i < results.size() && results[i].z > 0) {
                        char pos[128];
                        snprintf(pos, sizeof(pos), "(%.1f, %.1f, %.1f)",
                                 results[i].x, results[i].y, results[i].z);
                        cv::putText(frame, pos, cv::Point(x1, y1 + bh + 20),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                    cv::Scalar(255, 200, 0), 1);
                    }
                }

                // FPS + 帧号
                char hud[64];
                snprintf(hud, sizeof(hud), "FPS: %.1f  Frame: %d", fps, frame_id);
                cv::putText(frame, hud, cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(0, 200, 255), 2);

                std::lock_guard<std::mutex> lock(display_mutex);
                display_frame = std::move(frame);
                display_ready = true;
            });
    }

    // 启动
    pipeline.start();

    // 主线程等待退出信号 / 可视化显示
    if (enable_display) {
        LOG_INFO("Visualization enabled - press ESC to quit");
        try {
            cv::namedWindow("Pipeline", cv::WINDOW_AUTOSIZE);
        } catch (const cv::Exception& e) {
            LOG_WARN("Failed to create visualization window (%s), fallback to headless", e.what());
            enable_display = false;
        }
    }

    if (enable_display) {
        while (!g_shutdown.load()) {
            {
                std::lock_guard<std::mutex> lock(display_mutex);
                if (display_ready) {
                    cv::imshow("Pipeline", display_frame);
                    display_ready = false;
                }
            }
            int key = cv::waitKey(5);
            if (key == 27) {  // ESC
                g_shutdown.store(true);
            }
        }
        cv::destroyAllWindows();
    } else {
        while (!g_shutdown.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    LOG_INFO("Shutting down...");
    pipeline.stop();
    pipeline.printPerfReport();

    LOG_INFO("Done.");
    return 0;
}
