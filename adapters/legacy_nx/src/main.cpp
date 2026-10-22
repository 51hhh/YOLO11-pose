/**
 * @file main.cpp
 * @brief stereo_3d_pipeline 入口
 *
 * 加载由统一 CLI 生成的配置 → 初始化 Pipeline → 运行 → 信号退出
 */

#include "pipeline/pipeline.h"
#include "fusion/trajectory_predictor.h"
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

#ifdef HAS_ROS2
#include "ros/nx_ball_observation_publisher.h"
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
    RealtimeDebugDumpConfig realtime_dump_cfg =
        loadRealtimeDebugDumpConfig(config_path);

    // 初始化 Pipeline
    stereo3d::Pipeline pipeline;

    // 初始化可选落点预测、正式 v2.1 采集和诊断输出。
    stereo3d::TrajectoryPredictor predictor;
    stereo3d::CaptureV21StageRecorder capture_v21_recorder;
    RealtimeDebugDumper realtime_debug_dumper;
    bool nx_prediction_enabled = false;
    const auto capture_v21_cfg = loadCaptureV21StageRecorderConfig(config_path);
    if (!capture_v21_cfg.valid) {
        LOG_ERROR("Capture v2.1 configuration is invalid");
        return 1;
    }
    if (!capture_v21_recorder.init(capture_v21_cfg)) {
        LOG_ERROR("Capture v2.1 staging recorder initialization failed");
        return 1;
    }
    const auto predictor_cfg = loadPredictorConfig(config_path);
    nx_prediction_enabled = predictor_cfg.enabled;
    if (nx_prediction_enabled) {
        predictor.init(predictor_cfg);
    } else {
        LOG_INFO("NX landing predictor disabled; RDK owns ballistic prediction");
    }
    realtime_debug_dumper.init(realtime_dump_cfg);

    // 先完成相机、GPU 与推理资源初始化，再启动 ROS2 线程。这样初始化失败
    // 时不会遗留 joinable 的通信线程，也能让采集 staging 正常写出失败状态。
    if (!pipeline.init(cfg)) {
        LOG_ERROR("Pipeline init failed");
        return 1;
    }

    // === ROS2 Bridge 初始化 ===
#ifdef HAS_ROS2
    std::shared_ptr<rclcpp::Node> ros2_node;
    std::unique_ptr<stereo3d::GoalPoseBridge> ros2_bridge;
    std::unique_ptr<stereo3d::DiagnosticPublisher> ros2_diag;
    std::unique_ptr<stereo3d::NxBallObservationPublisher> nx_observation_pub;
    std::thread ros2_spin_thread;
    stereo3d::Ros2BridgeConfig ros2_cfg{};

    try {
        ros2_cfg = loadRos2Config(config_path);
    } catch (const std::exception& e) {
        LOG_WARN("ROS2 config load failed: %s, bridge disabled", e.what());
        ros2_cfg.enabled = false;
    }
    if (ros2_cfg.enabled) {
        rclcpp::init(argc, argv);
        ros2_node = std::make_shared<rclcpp::Node>("stereo_3d_pipeline");
        ros2_bridge = std::make_unique<stereo3d::GoalPoseBridge>(ros2_node, ros2_cfg);
        if (ros2_cfg.nx_observation_enabled) {
            nx_observation_pub = std::make_unique<stereo3d::NxBallObservationPublisher>(
                ros2_node,
                ros2_cfg.nx_observation_topic,
                ros2_cfg.nx_observation_frame_id,
                ros2_cfg.nx_observation_source_epoch_file,
                ros2_cfg.nx_observation_class_id,
                ros2_cfg.nx_observation_min_depth_m,
                ros2_cfg.nx_observation_max_depth_m,
                ros2_cfg.nx_observation_max_speed_mps,
                ros2_cfg.nx_observation_reacquire_timeout_s,
                ros2_cfg.nx_observation_reacquire_base_gate_m,
                ros2_cfg.nx_observation_allow_fallback);
        }

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

    pipeline.setResultCallback(
        [&](int frame_id,
            const std::vector<stereo3d::Object3D>& results,
            const stereo3d::FrameMetadata& metadata) {
            // 计算 dt
            auto now = std::chrono::steady_clock::now();
            double dt = 0.01; // 默认 100fps
            if (last_result_time.time_since_epoch().count() > 0) {
                dt = std::chrono::duration<double>(now - last_result_time).count();
                dt = std::clamp(dt, 0.001, 0.1);
            }
            last_result_time = now;

            // 预测落点
            auto preds = nx_prediction_enabled
                ? predictor.update(results, dt)
                : std::vector<stereo3d::LandingPrediction>{};

            // ROS2 发布并把实时控制门控审计结果写回 preds。
#ifdef HAS_ROS2
            if (nx_observation_pub) nx_observation_pub->publish(frame_id, results, metadata);
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
                // 从所有有效候选中优先选择 Student-t EKF 且置信度最高的
                // track，避免按检测数组顺序把低质量 fallback 送入控制门控。
                size_t best_pred = preds.size();
                double best_score = -1.0;
                for (size_t i = 0; i < preds.size(); ++i) {
                    if (!preds[i].valid || i >= results.size()) continue;
                    const bool method_ok = preds[i].method == 0 ||
                        (preds[i].method == 1 && ros2_cfg.control_allow_polynomial);
                    const bool obs_ok = preds[i].obs_source >= 0 &&
                        (preds[i].obs_source < 3 ||
                         ros2_cfg.control_allow_fallback_observation);
                    const bool quality_hint = method_ok && obs_ok &&
                        preds[i].confidence >= ros2_cfg.control_min_confidence &&
                        preds[i].time_to_land >= ros2_cfg.control_min_time_to_land_s &&
                        preds[i].time_to_land <= ros2_cfg.control_max_time_to_land_s &&
                        preds[i].speed_mps >= ros2_cfg.control_min_speed_mps &&
                        preds[i].student_w >= ros2_cfg.control_min_student_w &&
                        std::fabs(preds[i].x) <= ros2_cfg.control_max_abs_x_m &&
                        preds[i].y > ros2_cfg.control_min_depth_m &&
                        preds[i].y <= ros2_cfg.control_max_depth_m;
                    const double score = preds[i].confidence +
                        (quality_hint ? 4.0 : 0.0) +
                        (preds[i].method == 0 ? 2.0 : 0.0) +
                        (preds[i].obs_source >= 0 && preds[i].obs_source < 3 ? 0.5 : 0.0) +
                        0.1 * std::clamp(static_cast<double>(preds[i].student_w), 0.0, 1.0);
                    if (score > best_score) {
                        best_score = score;
                        best_pred = i;
                    }
                }
                if (best_pred < preds.size()) {
                    const size_t i = best_pred;
                    auto wl = ros2_bridge->transformVisionToWorld(preds[i].x, preds[i].y);
                    ros2_bridge->publishLandingWorld(wl.x, wl.y, stamp);
                    double bx, by;
                    if (ros2_bridge->tryWorldToBase(wl.x, wl.y, bx, by)) {
                        ros2_bridge->publishLandingBase(bx, by, stamp);
                    }
                    // 控制目标与 RViz 落点分离：仅在预测质量、范围、
                    // 坐标变换和 odom 全部有效时，才发布 debug gate topic。
                    stereo3d::ControlGoalGateResult gate_result;
                    const bool gate_passed = ros2_bridge->tryPublishControlGoalFromVision(
                        preds[i].x, preds[i].y,
                        results[i].track_id,
                        preds[i].confidence,
                        preds[i].time_to_land,
                        preds[i].speed_mps,
                        preds[i].method,
                        preds[i].student_w,
                        preds[i].obs_source,
                        stamp,
                        &gate_result);
                    preds[i].control_gate_selected = 1;
                    preds[i].control_gate_passed = gate_passed ? 1 : 0;
                    preds[i].control_gate_reason =
                        static_cast<int>(gate_result.reason);
                    preds[i].control_gate_stable_frames = gate_result.stable_frames;
                    if (gate_passed) {
                        preds[i].control_base_x = static_cast<float>(gate_result.base_x);
                        preds[i].control_base_y = static_cast<float>(gate_result.base_y);
                    }
                }
            }
#endif

            {
                std::lock_guard<std::mutex> lock(pred_mutex);
                latest_preds = preds;
            }

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

    if (capture_v21_recorder.enabled() || enable_display ||
        realtime_debug_dumper.enabled()) {
        pipeline.setFrameCallback(
            [&, use_bgr](const stereo3d::FrameCallbackData& frame_data) {
                if (capture_v21_recorder.enabled()) {
                    capture_v21_recorder.record(frame_data);
                    if (capture_v21_recorder.shouldStop()) {
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
                display_job.rec_frames =
                    capture_v21_recorder.capturedFrames();
                {
                    std::lock_guard<std::mutex> plock(pred_mutex);
                    display_job.preds = latest_preds;
                }
                display_job_ready = true;
            });
    }

    // 启动
    if (!pipeline.start()) {
        LOG_ERROR("Pipeline start failed");
#ifdef HAS_ROS2
        if (ros2_cfg.enabled) {
            rclcpp::shutdown();
            if (ros2_spin_thread.joinable()) ros2_spin_thread.join();
        }
#endif
        capture_v21_recorder.close();
        realtime_debug_dumper.close();
        return 1;
    }

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
        while (!g_shutdown.load() && pipeline.isRunning()) {
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
        while (!g_shutdown.load() && pipeline.isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    bool pipeline_failed = pipeline.hasFailed();
    if (pipeline_failed) {
        LOG_ERROR("Pipeline worker failed; beginning controlled shutdown");
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
    pipeline_failed = pipeline_failed || pipeline.hasFailed();
    capture_v21_recorder.close();
    realtime_debug_dumper.close();
    pipeline.printPerfReport();

    if (capture_v21_recorder.enabled() &&
        !capture_v21_recorder.publishable()) {
        LOG_ERROR("Capture v2.1 staging is not publishable: %s",
                  capture_v21_recorder.fatalError().c_str());
        return 2;
    }

    if (pipeline_failed) {
        return 1;
    }

    LOG_INFO("Done.");
    return 0;
}
