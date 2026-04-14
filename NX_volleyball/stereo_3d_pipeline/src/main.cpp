/**
 * @file main.cpp
 * @brief stereo_3d_pipeline 入口
 *
 * 加载 pipeline.yaml 配置 → 初始化 Pipeline → 运行 → 信号退出
 */

#include "pipeline/pipeline.h"
#include "fusion/trajectory_predictor.h"
#include "utils/trajectory_recorder.h"
#include "utils/logger.h"

#include <vpi/Image.h>
#include <vpi/algo/TemporalNoiseReduction.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <csignal>
#include <cstdio>
#include <cmath>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cuda_runtime.h>

// ==================== 全局信号 ====================

static std::atomic<bool> g_shutdown{false};

static void signalHandler(int sig) {
    (void)sig;
    g_shutdown.store(true);
}

// ==================== 点击测距 ====================

struct ClickMeasureState {
    std::mutex mtx;
    int click_u = -1;  ///< 点击像素 u
    int click_v = -1;  ///< 点击像素 v
    float click_x = 0, click_y = 0, click_z = 0;  ///< 3D 坐标
    bool has_click = false;
    int display_frames = 0;  ///< 剩余显示帧数
};

static ClickMeasureState g_click;

static void mouseCallback(int event, int x, int y, int /*flags*/, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        std::lock_guard<std::mutex> lock(g_click.mtx);
        g_click.click_u = x;
        g_click.click_v = y;
        g_click.has_click = true;
        g_click.display_frames = 150;  // 显示约 1.5s @ 100fps
    }
}

// ==================== 配置加载 ====================

static stereo3d::PipelineConfig loadConfig(const std::string& path) {
    stereo3d::PipelineConfig cfg;

    YAML::Node root = YAML::LoadFile(path);

    // Camera → 直接写入内嵌 CameraConfig
    if (auto cam = root["camera"]) {
        if (cam["serial_left"])       cfg.camera.serial_left  = cam["serial_left"].as<std::string>();
        if (cam["serial_right"])      cfg.camera.serial_right = cam["serial_right"].as<std::string>();
        if (cam["left_index"])        cfg.camera.camera_index_left  = cam["left_index"].as<int>();
        if (cam["right_index"])       cfg.camera.camera_index_right = cam["right_index"].as<int>();
        if (cam["exposure_us"])       cfg.camera.exposure_us = cam["exposure_us"].as<float>();
        if (cam["gain_db"])           cfg.camera.gain_db = cam["gain_db"].as<float>();
        if (cam["auto_exposure"])     cfg.camera.auto_exposure = cam["auto_exposure"].as<bool>();
        if (cam["auto_gain"])         cfg.camera.auto_gain = cam["auto_gain"].as<bool>();
        if (cam["ae_upper_us"])       cfg.camera.ae_upper_us = cam["ae_upper_us"].as<float>();
        if (cam["ae_lower_us"])       cfg.camera.ae_lower_us = cam["ae_lower_us"].as<float>();
        if (cam["ag_upper_db"])       cfg.camera.ag_upper_db = cam["ag_upper_db"].as<float>();
        if (cam["gamma_enable"])      cfg.camera.gamma_enable = cam["gamma_enable"].as<bool>();
        if (cam["gamma_value"])       cfg.camera.gamma_value = cam["gamma_value"].as<float>();
        if (cam["use_trigger"])       cfg.camera.use_trigger = cam["use_trigger"].as<bool>();
        if (cam["trigger_source"])    cfg.camera.trigger_source = cam["trigger_source"].as<std::string>();
        if (cam["trigger_activation"]) cfg.camera.trigger_activation = cam["trigger_activation"].as<std::string>();
        if (cam["trigger_chip"])       cfg.trigger_chip = cam["trigger_chip"].as<std::string>();
        if (cam["trigger_line"])       cfg.trigger_line = cam["trigger_line"].as<int>();
        if (cam["width"])             cfg.camera.width  = cam["width"].as<int>();
        if (cam["height"])            cfg.camera.height = cam["height"].as<int>();
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
        if (det["input_format"])           cfg.detector_input_format = det["input_format"].as<std::string>();
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
    // Fusion → 直接写入内嵌 HybridDepthConfig
    if (auto fus = root["fusion"]) {
        if (fus["min_depth"])       cfg.depth.min_depth       = fus["min_depth"].as<float>();
        if (fus["max_depth"])       cfg.depth.max_depth       = fus["max_depth"].as<float>();
        if (fus["object_diameter"]) cfg.depth.object_diameter = fus["object_diameter"].as<float>();
        if (fus["bbox_scale"])      cfg.depth.bbox_scale      = fus["bbox_scale"].as<float>();
        if (fus["mono_max_z"])      cfg.depth.mono_max_z      = fus["mono_max_z"].as<float>();
        if (fus["stereo_min_z"])    cfg.depth.stereo_min_z    = fus["stereo_min_z"].as<float>();
        // Kalman 观测噪声 (距离自适应基值)
        if (fus["R_mono"])          cfg.depth.R_mono          = fus["R_mono"].as<float>();
        if (fus["R_stereo"])        cfg.depth.R_stereo        = fus["R_stereo"].as<float>();
        // IVW 融合权重 (与 Kalman R 分离)
        if (fus["ivw_R_mono"])      cfg.depth.ivw_R_mono      = fus["ivw_R_mono"].as<float>();
        if (fus["ivw_R_stereo"])    cfg.depth.ivw_R_stereo    = fus["ivw_R_stereo"].as<float>();
    }

    // Performance
    if (auto perf = root["performance"]) {
        if (perf["log_interval"])   cfg.stats_interval = perf["log_interval"].as<int>();
        if (perf["pwm_frequency"])  cfg.trigger_freq_hz = static_cast<int>(perf["pwm_frequency"].as<float>());
    }

    // SOT Tracker (YOLO 帧间填充)
    if (auto trk = root["tracker"]) {
        cfg.tracker.enabled = trk["enabled"] ? trk["enabled"].as<bool>() : false;
        if (trk["type"])            cfg.tracker.type            = trk["type"].as<std::string>();
        if (trk["engine_path"])     cfg.tracker.engine_path     = trk["engine_path"].as<std::string>();
        if (trk["search_engine_path"]) cfg.tracker.search_engine_path = trk["search_engine_path"].as<std::string>();
        if (trk["head_engine_path"]) cfg.tracker.head_engine_path = trk["head_engine_path"].as<std::string>();
        if (trk["detect_interval"]) cfg.tracker.detect_interval = trk["detect_interval"].as<int>();
        if (trk["lost_threshold"])  cfg.tracker.lost_threshold  = trk["lost_threshold"].as<int>();
        if (trk["min_confidence"])  cfg.tracker.min_confidence  = trk["min_confidence"].as<float>();
    }

    return cfg;
}

static stereo3d::TrajectoryPredictorConfig loadPredictorConfig(const std::string& path) {
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

static stereo3d::TrajectoryRecorderConfig loadRecorderConfig(const std::string& path) {
    stereo3d::TrajectoryRecorderConfig rcfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (auto rec = root["recording"]) {
            if (rec["enabled"])     rcfg.enabled     = rec["enabled"].as<bool>();
            if (rec["output_path"]) rcfg.output_path = rec["output_path"].as<std::string>();
        }
    } catch (const std::exception& e) {
        LOG_WARN("recording config: %s, using defaults", e.what());
    } catch (...) { LOG_WARN("recording config: unknown error, using defaults"); }
    return rcfg;
}

// ==================== main ====================

int main(int argc, char* argv[]) {
    // 解析命令行
    std::string config_path = "config/pipeline.yaml";
    bool enable_display = false;
    std::vector<std::string> unknown_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--config" || arg == "-c") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, "Usage: %s [--config <path>] [--visualize]\n", argv[0]);
            return 1;
        } else if (arg == "--visualize" || arg == "--display" || arg == "-v") {
            enable_display = true;
        } else if (arg == "--visualizels") {
            // 兼容常见拼写误写，避免静默关闭可视化
            fprintf(stderr, "Warning: unknown option '--visualizels', treating as '--visualize'.\n");
            enable_display = true;
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [--config <path>] [--visualize]\n", argv[0]);
            printf("  --config, -c    Pipeline configuration YAML\n");
            printf("  --visualize, -v Show detection + distance overlay window\n");
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            unknown_args.push_back(arg);
        }
    }

    if (!unknown_args.empty()) {
        fprintf(stderr, "Error: unknown option(s):");
        for (const auto& opt : unknown_args) {
            fprintf(stderr, " %s", opt.c_str());
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "Hint: use --help to see supported options.\n");
        return 1;
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

    // 初始化落点预测 + 轨迹记录
    stereo3d::TrajectoryPredictor predictor;
    stereo3d::TrajectoryRecorder recorder;
    predictor.init(loadPredictorConfig(config_path));
    recorder.init(loadRecorderConfig(config_path));

    // 预测帧间隔 (dt) 需要实测, 这里用 FPS 倒数
    std::chrono::steady_clock::time_point last_result_time{};
    std::mutex pred_mutex;
    std::vector<stereo3d::LandingPrediction> latest_preds;

    pipeline.setResultCallback(
        [&](int frame_id, const std::vector<stereo3d::Object3D>& results) {
            // 计算 dt
            auto now = std::chrono::steady_clock::now();
            double dt = 0.01; // 默认 100fps
            if (last_result_time.time_since_epoch().count() > 0) {
                dt = std::chrono::duration<double>(now - last_result_time).count();
                dt = std::clamp(dt, 0.001, 0.1);
            }
            last_result_time = now;

            // 预测落点
            auto preds = predictor.update(results, dt);
            {
                std::lock_guard<std::mutex> lock(pred_mutex);
                latest_preds = preds;
            }

            // 记录轨迹
            double timestamp = std::chrono::duration<double>(
                now.time_since_epoch()).count();
            recorder.record(frame_id, timestamp, results, preds);

            // 降低日志开销: 仅每50帧打印一次
            if (frame_id % 50 != 0) return;
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& obj = results[i];
                float speed = std::sqrt(obj.vx*obj.vx + obj.vy*obj.vy + obj.vz*obj.vz);
                const char* method_str = obj.depth_method == 0 ? "M" :
                                         obj.depth_method == 1 ? "S" : "F";
                if (i < preds.size() && preds[i].valid) {
                    LOG_INFO("[Frame %d] T%d: (%.2f,%.2f,%.2f)m [%s zm=%.2f zs=%.2f] |v|=%.1fm/s → land(%.2f,%.2f) in %.2fs",
                             frame_id, obj.track_id,
                             obj.x, obj.y, obj.z, method_str,
                             obj.z_mono, obj.z_stereo, speed,
                             preds[i].x, preds[i].y, preds[i].time_to_land);
                } else {
                    LOG_INFO("[Frame %d] T%d: (%.2f,%.2f,%.2f)m [%s zm=%.2f zs=%.2f] |v|=%.1fm/s conf=%.2f",
                             frame_id, obj.track_id,
                             obj.x, obj.y, obj.z, method_str,
                             obj.z_mono, obj.z_stereo, speed,
                             obj.confidence);
                }
            }
        });

    if (!pipeline.init(cfg)) {
        LOG_ERROR("Pipeline init failed");
        return 1;
    }

    LOG_INFO("Pipeline initialized, starting...");

    // (display variables moved to DisplayJob below)

    // 构建彩色校正映射表 (用于可视化: raw Bayer → demosaic → remap → 彩色校正图)
    cv::Mat vis_map1, vis_map2;
    bool has_color_remap = false;
    if (enable_display) {
        try {
            cv::FileStorage fs(cfg.calibration_file, cv::FileStorage::READ);
            if (fs.isOpened()) {
                cv::Mat K1, D1, R1, P1;
                fs["camera_matrix_left"]           >> K1;
                fs["distortion_coefficients_left"] >> D1;
                fs["rectification_left"]           >> R1;
                fs["projection_left"]              >> P1;

                // 缩放 P1 以适配 rect_width x rect_height (标定在 raw 分辨率)
                int cal_w = 0, cal_h = 0;
                fs["image_width"]  >> cal_w;
                fs["image_height"] >> cal_h;
                if (cal_w > 0 && cal_h > 0 &&
                    (cfg.rect_width != cal_w || cfg.rect_height != cal_h)) {
                    double sx = (double)cfg.rect_width  / cal_w;
                    double sy = (double)cfg.rect_height / cal_h;
                    P1 = P1.clone();
                    P1.at<double>(0, 0) *= sx;
                    P1.at<double>(1, 1) *= sy;
                    P1.at<double>(0, 2) *= sx;
                    P1.at<double>(1, 2) *= sy;
                    P1.at<double>(0, 3) *= sx;
                }

                cv::initUndistortRectifyMap(K1, D1, R1, P1,
                    cv::Size(cfg.rect_width, cfg.rect_height),
                    CV_16SC2, vis_map1, vis_map2);
                has_color_remap = true;
                LOG_INFO("Color remap built for visualization (%dx%d -> %dx%d)",
                         cfg.camera.width, cfg.camera.height, cfg.rect_width, cfg.rect_height);
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to build color remap: %s (visualization will be grayscale)", e.what());
        }
    }

    const bool use_bgr = (cfg.detector_input_format == "bgr");

    // 显示数据: 管线线程仅做GPU→CPU拷贝, 绘制在主线程完成
    struct DisplayJob {
        cv::Mat frame;
        std::vector<stereo3d::Detection> detections;
        std::vector<stereo3d::Object3D> results;
        std::vector<stereo3d::LandingPrediction> preds;
        float fps;
        int frame_id;
        int rec_frames;
    };
    std::mutex display_job_mutex;
    DisplayJob display_job;
    bool display_job_ready = false;

    if (enable_display) {
        pipeline.setFrameCallback(
            [&, use_bgr](int frame_id, VPIImage rectL, VPIImage rawL,
                const std::vector<stereo3d::Detection>& detections,
                const std::vector<stereo3d::Object3D>& results,
                float fps) {
                // 上一帧未消费时跳过 (管线线程零阻塞)
                {
                    std::lock_guard<std::mutex> lock(display_job_mutex);
                    if (display_job_ready) return;
                }

                cv::Mat frame;

                if (use_bgr) {
                    VPIImageData imgData;
                    VPIStatus st = vpiImageLockData(rectL, VPI_LOCK_READ,
                        VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData);
                    if (st == VPI_SUCCESS) {
                        int h = imgData.buffer.pitch.planes[0].height;
                        int w = imgData.buffer.pitch.planes[0].width;
                        int gpuPitch = imgData.buffer.pitch.planes[0].pitchBytes;
                        const void* gpuPtr = imgData.buffer.pitch.planes[0].data;
                        frame.create(h, w, CV_8UC3);
                        cudaMemcpy2D(frame.data, frame.step[0],
                                     gpuPtr, gpuPitch,
                                     w * 3, h,
                                     cudaMemcpyDeviceToHost);
                        vpiImageUnlock(rectL);
                    }
                } else if (has_color_remap && rawL) {
                    VPIImageData rawData;
                    if (vpiImageLockData(rawL, VPI_LOCK_READ,
                        VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &rawData) == VPI_SUCCESS) {
                        int rh = rawData.buffer.pitch.planes[0].height;
                        int rw = rawData.buffer.pitch.planes[0].width;
                        int rp = rawData.buffer.pitch.planes[0].pitchBytes;
                        cv::Mat rawBayer(rh, rw, CV_8UC1,
                                         rawData.buffer.pitch.planes[0].data, rp);
                        cv::Mat bgrRaw;
                        cv::cvtColor(rawBayer, bgrRaw, cv::COLOR_BayerBG2BGR);
                        cv::remap(bgrRaw, frame, vis_map1, vis_map2, cv::INTER_LINEAR);
                        vpiImageUnlock(rawL);
                    }
                }

                if (frame.empty()) {
                    VPIImageData imgData;
                    if (vpiImageLockData(rectL, VPI_LOCK_READ,
                        VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData) != VPI_SUCCESS)
                        return;
                    int h = imgData.buffer.pitch.planes[0].height;
                    int w = imgData.buffer.pitch.planes[0].width;
                    int pitch = imgData.buffer.pitch.planes[0].pitchBytes;
                    cv::Mat gray(h, w, CV_8UC1,
                                 imgData.buffer.pitch.planes[0].data, pitch);
                    cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
                    vpiImageUnlock(rectL);
                }

                // 仅存数据, 绘制交给主线程
                std::lock_guard<std::mutex> lock(display_job_mutex);
                display_job.frame = std::move(frame);
                display_job.detections = detections;
                display_job.results = results;
                display_job.fps = fps;
                display_job.frame_id = frame_id;
                display_job.rec_frames = recorder.frameCount();
                {
                    std::lock_guard<std::mutex> plock(pred_mutex);
                    display_job.preds = latest_preds;
                }
                display_job_ready = true;
            });
    }

    // 启动
    pipeline.start();

    // 主线程等待退出信号 / 可视化显示
    // 计算显示尺寸: 保持原始 4:3 比例 (raw_width:raw_height)
    int disp_w = cfg.rect_width;
    int disp_h = cfg.rect_height;
    if (cfg.camera.width > 0 && cfg.camera.height > 0) {
        // 以 rect_height 为基准，按原始宽高比确定宽度
        disp_w = cfg.rect_height * cfg.camera.width / cfg.camera.height;
    }

    if (enable_display) {
        LOG_INFO("Visualization enabled - press ESC to quit, click for depth");
        LOG_INFO("Display: %dx%d (aspect %d:%d)", disp_w, disp_h,
                 cfg.camera.width, cfg.camera.height);
        try {
            cv::namedWindow("Pipeline", cv::WINDOW_NORMAL);
            cv::resizeWindow("Pipeline", disp_w, disp_h);
            cv::setMouseCallback("Pipeline", mouseCallback, nullptr);
        } catch (const cv::Exception& e) {
            LOG_WARN("Failed to create visualization window (%s), fallback to headless", e.what());
            enable_display = false;
        }
    }

    if (enable_display) {
        while (!g_shutdown.load()) {
            DisplayJob job;
            bool has_job = false;
            {
                std::lock_guard<std::mutex> lock(display_job_mutex);
                if (display_job_ready) {
                    job = std::move(display_job);
                    display_job_ready = false;
                    has_job = true;
                }
            }
            if (has_job) {
                cv::Mat& frame = job.frame;

                // 处理点击测距
                {
                    std::lock_guard<std::mutex> clik(g_click.mtx);
                    if (g_click.has_click) {
                        g_click.has_click = false;
                        g_click.click_z = 0;
                        float min_dist = 1e9f;
                        for (size_t i = 0; i < job.detections.size() && i < job.results.size(); ++i) {
                            float dx = job.detections[i].cx - g_click.click_u;
                            float dy = job.detections[i].cy - g_click.click_v;
                            float dist = dx*dx + dy*dy;
                            if (dist < min_dist && job.results[i].z > 0) {
                                min_dist = dist;
                                g_click.click_x = job.results[i].x;
                                g_click.click_y = job.results[i].y;
                                g_click.click_z = job.results[i].z;
                            }
                        }
                    }
                }

                // 绘制检测框 + 距离
                for (size_t i = 0; i < job.detections.size(); ++i) {
                    const auto& d = job.detections[i];
                    int x1 = static_cast<int>(d.cx - d.width / 2);
                    int y1 = static_cast<int>(d.cy - d.height / 2);
                    int bw = static_cast<int>(d.width);
                    int bh = static_cast<int>(d.height);
                    cv::Scalar color(0, 255, 0);
                    cv::rectangle(frame, cv::Rect(x1, y1, bw, bh), color, 2);

                    char label[128];
                    if (i < job.results.size()) {
                        snprintf(label, sizeof(label),
                                 "%.2fm (%.0f%%)", job.results[i].z, d.confidence * 100);
                    } else {
                        snprintf(label, sizeof(label),
                                 "conf=%.0f%%", d.confidence * 100);
                    }
                    cv::putText(frame, label, cv::Point(x1, y1 - 8),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

                    if (i < job.results.size() && job.results[i].z > 0) {
                        const auto& r = job.results[i];
                        float speed = std::sqrt(r.vx*r.vx + r.vy*r.vy + r.vz*r.vz);
                        char pos[160];
                        snprintf(pos, sizeof(pos), "X=%.3f Y=%.3f Z=%.3f |v|=%.1f",
                                 r.x, r.y, r.z, speed);
                        cv::putText(frame, pos, cv::Point(x1, y1 + bh + 20),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                    cv::Scalar(255, 200, 0), 1);
                        char depth_info[160];
                        const char* mstr = r.depth_method == 0 ? "M" :
                                           r.depth_method == 1 ? "S" : "B";
                        snprintf(depth_info, sizeof(depth_info),
                                 "zm=%.3f zs=%.3f [%s]",
                                 r.z_mono, r.z_stereo, mstr);
                        cv::putText(frame, depth_info, cv::Point(x1, y1 + bh + 40),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                    cv::Scalar(0, 255, 255), 1);

                        if (i < job.preds.size() && job.preds[i].valid) {
                            const auto& p = job.preds[i];
                            char pred_text[128];
                            snprintf(pred_text, sizeof(pred_text),
                                     "LAND(%.1f,%.1f) %.2fs %s",
                                     p.x, p.y, p.time_to_land,
                                     p.method == 0 ? "B" : "P");
                            cv::putText(frame, pred_text,
                                        cv::Point(x1, y1 + bh + 60),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                        cv::Scalar(0, 100, 255), 2);
                        }
                    }
                }

                // 点击测距显示
                {
                    std::lock_guard<std::mutex> clik(g_click.mtx);
                    if (g_click.display_frames > 0) {
                        cv::drawMarker(frame,
                            cv::Point(g_click.click_u, g_click.click_v),
                            cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 20, 2);
                        if (g_click.click_z > 0) {
                            char click_text[128];
                            snprintf(click_text, sizeof(click_text),
                                     "(%.2f, %.2f, %.2f)m",
                                     g_click.click_x, g_click.click_y, g_click.click_z);
                            cv::putText(frame, click_text,
                                        cv::Point(g_click.click_u + 12, g_click.click_v - 12),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                        cv::Scalar(0, 255, 255), 2);
                        } else {
                            cv::putText(frame, "No depth",
                                        cv::Point(g_click.click_u + 12, g_click.click_v - 12),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                        cv::Scalar(0, 0, 255), 2);
                        }
                        g_click.display_frames--;
                    }
                }

                // FPS + 帧号 + 记录状态
                char hud[128];
                if (job.rec_frames > 0) {
                    snprintf(hud, sizeof(hud), "FPS: %.1f  Frame: %d  REC: %d",
                             job.fps, job.frame_id, job.rec_frames);
                } else {
                    snprintf(hud, sizeof(hud), "FPS: %.1f  Frame: %d",
                             job.fps, job.frame_id);
                }
                cv::putText(frame, hud, cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(0, 200, 255), 2);

                cv::imshow("Pipeline", frame);
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
    recorder.close();
    pipeline.printPerfReport();

    LOG_INFO("Done.");
    return 0;
}
