/**
 * @file main.cpp
 * @brief stereo_3d_pipeline 入口
 *
 * 加载 pipeline.yaml 配置 → 初始化 Pipeline → 运行 → 信号退出
 */

#include "pipeline/pipeline.h"
#include "fusion/trajectory_predictor.h"
#include "utils/trajectory_recorder.h"
#include "utils/baseline_clip_recorder.h"
#include "utils/logger.h"
#include "main_realtime_debug_dump.h"
#include "main_cli_options.h"
#include "main_config_loaders.h"
#include "main_display_helpers.h"

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <csignal>
#include <cmath>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <cctype>

#ifdef HAS_ROS2
#include <rclcpp/rclcpp.hpp>
#include "ros/goal_pose_bridge.h"
#include "ros/diagnostic_publisher.h"
#endif

// ==================== 全局信号 ====================

static std::atomic<bool> g_shutdown{false};

static void signalHandler(int sig) {
    (void)sig;
    g_shutdown.store(true);
}

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

// ==================== main ====================

int main(int argc, char* argv[]) {
    // 解析命令行
    const MainCliOptions cli = parseMainCliOptions(argc, argv);
    if (cli.should_exit) return cli.exit_code;

    std::string config_path = cli.config_path;
    bool enable_display = cli.enable_display;
    bool debug_feature_matches = cli.debug_feature_matches;
    std::string debug_feature_matches_dir = cli.debug_feature_matches_dir;
    std::string recording_out_override = cli.recording_out_override;
    bool baseline_clip_cli = cli.baseline_clip_cli;
    std::string baseline_out_override = cli.baseline_out_override;
    double baseline_duration_override = cli.baseline_duration_override;
    int baseline_frames_override = cli.baseline_frames_override;
    int baseline_clips_override = cli.baseline_clips_override;
    double baseline_gap_override = cli.baseline_gap_override;
    std::string baseline_format_override = cli.baseline_format_override;
    std::string baseline_image_mode_override = cli.baseline_image_mode_override;
    bool baseline_start_immediately = cli.baseline_start_immediately;
    bool debug_realtime_dump_cli = cli.debug_realtime_dump_cli;
    std::string debug_realtime_dump_dir_override =
        cli.debug_realtime_dump_dir_override;
    int debug_realtime_dump_stride_override =
        cli.debug_realtime_dump_stride_override;
    int debug_realtime_dump_max_override =
        cli.debug_realtime_dump_max_override;

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
    stereo3d::BaselineClipRecorderConfig baseline_cfg =
        loadBaselineClipRecorderConfig(config_path);
    if (baseline_clip_cli) baseline_cfg.enabled = true;
    if (!baseline_out_override.empty()) baseline_cfg.output_dir = baseline_out_override;
    if (baseline_duration_override > 0.0) baseline_cfg.duration_sec = baseline_duration_override;
    if (baseline_frames_override > 0) baseline_cfg.frame_limit = baseline_frames_override;
    if (baseline_clips_override > 0) baseline_cfg.clip_count = baseline_clips_override;
    if (baseline_gap_override >= 0.0) baseline_cfg.clip_gap_sec = baseline_gap_override;
    if (!baseline_format_override.empty()) baseline_cfg.image_format = baseline_format_override;
    if (!baseline_image_mode_override.empty()) baseline_cfg.image_mode = baseline_image_mode_override;
    if (baseline_start_immediately) {
        baseline_cfg.require_left_detection = false;
        baseline_cfg.require_right_detection = false;
        baseline_cfg.require_pair_gate = false;
    }
    baseline_cfg.trigger_hz = cfg.trigger_freq_hz;
    if (debug_feature_matches) baseline_cfg.enabled = false;

    RealtimeDebugDumpConfig realtime_dump_cfg =
        loadRealtimeDebugDumpConfig(config_path);
    if (debug_realtime_dump_cli) realtime_dump_cfg.enabled = true;
    if (!debug_realtime_dump_dir_override.empty()) {
        realtime_dump_cfg.output_dir = debug_realtime_dump_dir_override;
    }
    if (debug_realtime_dump_stride_override >= 0) {
        realtime_dump_cfg.stride = debug_realtime_dump_stride_override;
    }
    if (debug_realtime_dump_max_override >= 0) {
        realtime_dump_cfg.max_frames = debug_realtime_dump_max_override;
    }
    realtime_dump_cfg.stride = std::max(0, realtime_dump_cfg.stride);
    realtime_dump_cfg.max_frames = std::max(0, realtime_dump_cfg.max_frames);
    realtime_dump_cfg.max_queue = std::max(1, realtime_dump_cfg.max_queue);

    if (baseline_cfg.enabled) {
        std::string baseline_mode = baseline_cfg.image_mode;
        std::transform(baseline_mode.begin(), baseline_mode.end(), baseline_mode.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if ((baseline_mode == "bgr" || baseline_mode == "both") &&
            cfg.detector_input_format != "bgr") {
            LOG_WARN("Baseline image_mode=%s requested but detector.input_format=%s; "
                     "BGR images are only valid when the color pipeline is enabled",
                     baseline_cfg.image_mode.c_str(), cfg.detector_input_format.c_str());
        }
        cfg.detection_only = true;
        cfg.disparity_strategy = stereo3d::DisparityStrategy::ROI_ONLY;
        cfg.tracker.enabled = false;
        cfg.neural_features.enabled = false;
        cfg.dual_yolo.use_for_depth = false;
        cfg.dual_yolo.fallback_to_roi_match = false;
        cfg.dual_yolo.log_matches = false;
        LOG_INFO("Baseline clip mode enabled: detection-only, stereo/depth/tracker disabled");
        if (!cfg.dual_yolo.enabled) {
            LOG_WARN("Baseline clip mode has dual_yolo.enabled=false; right detections will be empty");
        }
    }

    // 初始化 Pipeline
    stereo3d::Pipeline pipeline;

    if (debug_feature_matches) {
        LOG_INFO("Feature match debug output: %s", debug_feature_matches_dir.c_str());
        if (!pipeline.init(cfg)) {
            LOG_ERROR("Pipeline init failed");
            return 1;
        }
        const bool ok = pipeline.debugFeatureMatchesOnce(debug_feature_matches_dir);
        pipeline.printPerfReport();
        return ok ? 0 : 1;
    }

    // 初始化落点预测 + 轨迹记录 / 基准片段录制
    stereo3d::TrajectoryPredictor predictor;
    stereo3d::TrajectoryRecorder recorder;
    stereo3d::BaselineClipRecorder baseline_recorder;
    RealtimeDebugDumper realtime_debug_dumper;
    if (baseline_cfg.enabled) {
        baseline_recorder.init(baseline_cfg);
    } else {
        predictor.init(loadPredictorConfig(config_path));
        auto recorder_cfg = loadRecorderConfig(config_path);
        if (!recording_out_override.empty()) {
            recorder_cfg.output_path = recording_out_override;
            recorder_cfg.frame_summary_path.clear();
        }
        recorder.init(recorder_cfg);
        if (!recording_out_override.empty() && !recorder.isEnabled()) {
            LOG_ERROR("Trajectory recorder failed for --recording-out=%s",
                      recording_out_override.c_str());
            return 1;
        }
    }
    realtime_debug_dumper.init(realtime_dump_cfg);

    // === ROS2 Bridge 初始化 ===
#ifdef HAS_ROS2
    std::shared_ptr<rclcpp::Node> ros2_node;
    std::unique_ptr<stereo3d::GoalPoseBridge> ros2_bridge;
    std::unique_ptr<stereo3d::DiagnosticPublisher> ros2_diag;
    std::thread ros2_spin_thread;
    stereo3d::Ros2BridgeConfig ros2_cfg{};

    try {
        ros2_cfg = loadRos2Config(config_path);
    } catch (const std::exception& e) {
        LOG_WARN("ROS2 config load failed: %s, bridge disabled", e.what());
        ros2_cfg.enabled = false;
    }
    if (baseline_cfg.enabled) {
        ros2_cfg.enabled = false;
    }

    if (ros2_cfg.enabled) {
        rclcpp::init(argc, argv);
        ros2_node = std::make_shared<rclcpp::Node>("stereo_3d_pipeline");
        ros2_bridge = std::make_unique<stereo3d::GoalPoseBridge>(ros2_node, ros2_cfg);

        // 诊断发布器 (录制深度图 + 检测框)
        stereo3d::DiagnosticPublisherConfig diag_cfg;
        try {
            YAML::Node root = YAML::LoadFile(config_path);
            if (auto ros = root["ros2"]) {
                if (auto diag = ros["diagnostic"]) {
                    diag_cfg.enabled = diag["enable"].as<bool>(false);
                    if (diag["depth_full_divisor"])
                        diag_cfg.depth_full_divisor = diag["depth_full_divisor"].as<int>(6);
                }
            }
        } catch (...) {}
        ros2_diag = std::make_unique<stereo3d::DiagnosticPublisher>(ros2_node, diag_cfg);

        ros2_spin_thread = std::thread([&ros2_node]() {
            rclcpp::spin(ros2_node);
        });
        LOG_INFO("ROS2 bridge enabled: %s (diag=%d)", ros2_cfg.topic_realtime.c_str(), diag_cfg.enabled);
    }
#endif

    // === 诊断回调 (录制深度图+检测框) ===
#ifdef HAS_ROS2
    if (ros2_diag && ros2_diag->enabled()) {
        pipeline.setDiagnosticCallback(
            [&](int frame_id, const float* depth_gpu, int depth_pitch,
                int depth_w, int depth_h,
                const std::vector<stereo3d::Detection>& detections,
                const std::vector<stereo3d::Object3D>& results) {
                ros2_diag->publish(frame_id, depth_gpu, depth_pitch, depth_w, depth_h,
                                   depth_w * 2, depth_h * 2,
                                   detections, results);
            });
    }
#endif

    // 预测帧间隔 (dt) 需要实测, 这里用 FPS 倒数
    std::chrono::steady_clock::time_point last_result_time{};
    std::mutex pred_mutex;
    std::vector<stereo3d::LandingPrediction> latest_preds;

    if (!baseline_cfg.enabled) {
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

            // ROS2 发布
#ifdef HAS_ROS2
            if (ros2_bridge && ros2_bridge->enabled()) {
                auto stamp = ros2_node->get_clock()->now();
                for (size_t i = 0; i < results.size(); ++i) {
                    const auto& obj = results[i];
                    auto wpt = ros2_bridge->transformVisionToWorld(obj.x, obj.z);
                    ros2_bridge->publishRealtimeWorld(wpt.x, wpt.y, -obj.y, stamp);
                    double bx, by;
                    if (ros2_bridge->tryWorldToBase(wpt.x, wpt.y, bx, by)) {
                        ros2_bridge->publishRealtimeBase(bx, by, -obj.y, stamp);
                    }
                }
                for (size_t i = 0; i < preds.size(); ++i) {
                    if (preds[i].valid) {
                        auto wl = ros2_bridge->transformVisionToWorld(preds[i].x, preds[i].y);
                        ros2_bridge->publishLandingWorld(wl.x, wl.y, stamp);
                        double bx, by;
                        if (ros2_bridge->tryWorldToBase(wl.x, wl.y, bx, by)) {
                            ros2_bridge->publishLandingBase(bx, by, stamp);
                        }
                        break;  // 只发布第一个有效落点
                    }
                }
            }
#endif

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
    }

    if (!pipeline.init(cfg)) {
        LOG_ERROR("Pipeline init failed");
        return 1;
    }

    LOG_INFO("Pipeline initialized, starting...");

    // 构建彩色校正映射表 (用于可视化: raw Bayer → demosaic → remap → 彩色校正图)
    cv::Mat vis_map1, vis_map2;
    bool has_color_remap = false;
    if (enable_display) {
        buildVisualizationColorRemap(cfg, vis_map1, vis_map2, has_color_remap);
    }

    const bool use_bgr = (cfg.detector_input_format == "bgr");

    std::mutex display_job_mutex;
    DisplayJob display_job;
    bool display_job_ready = false;

    if (baseline_cfg.enabled || enable_display || realtime_debug_dumper.enabled()) {
        pipeline.setFrameCallback(
            [&, use_bgr](const stereo3d::FrameCallbackData& frame_data) {
                if (baseline_cfg.enabled) {
                    baseline_recorder.record(frame_data.frame_id,
                                             frame_data.rect_gray_left,
                                             frame_data.rect_gray_right,
                                             frame_data.rect_bgr_left,
                                             frame_data.rect_bgr_right,
                                             frame_data.detections_left,
                                             frame_data.detections_right,
                                             frame_data.metadata,
                                             frame_data.fps);
                    if (baseline_recorder.shouldStop()) {
                        g_shutdown.store(true);
                    }
                }

                if (realtime_debug_dumper.enabled()) {
                    realtime_debug_dumper.record(frame_data);
                }

                if (!enable_display) return;

                // 降低显示帧率: 每 display_divisor 帧才处理一次 (~30fps @ 100Hz)
                constexpr int display_divisor = 3;
                if (frame_data.frame_id % display_divisor != 0) return;

                // 上一帧未消费时跳过 (管线线程零阻塞)
                {
                    std::lock_guard<std::mutex> lock(display_job_mutex);
                    if (display_job_ready) return;
                }

                cv::Mat frame;
                if (!captureDisplayFrame(frame_data, use_bgr, frame)) return;

                // 仅存数据, 绘制交给主线程
                std::lock_guard<std::mutex> lock(display_job_mutex);
                display_job.frame = std::move(frame);
                display_job.detections = frame_data.detections_left;
                display_job.results = frame_data.results;
                display_job.fps = frame_data.fps;
                display_job.frame_id = frame_data.frame_id;
                display_job.rec_frames = baseline_cfg.enabled
                    ? baseline_recorder.frameCount()
                    : recorder.frameCount();
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
    auto [disp_w, disp_h] = computeDisplaySize(cfg);

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
                renderPipelineDisplayFrame(job, g_click);
                cv::imshow("Pipeline", job.frame);
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

#ifdef HAS_ROS2
    if (ros2_cfg.enabled) {
        rclcpp::shutdown();
        if (ros2_spin_thread.joinable()) ros2_spin_thread.join();
        LOG_INFO("ROS2 bridge shutdown");
    }
#endif

    pipeline.stop();
    realtime_debug_dumper.close();
    baseline_recorder.close();
    recorder.close();
    pipeline.printPerfReport();

    LOG_INFO("Done.");
    return 0;
}
