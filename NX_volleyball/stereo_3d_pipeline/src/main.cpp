/**
 * @file main.cpp
 * @brief stereo_3d_pipeline 入口
 *
 * 加载 pipeline.yaml 配置 → 初始化 Pipeline → 运行 → 信号退出
 */

#include "pipeline/pipeline.h"
#include "stereo/neural_feature_config.h"
#include "fusion/trajectory_predictor.h"
#include "utils/trajectory_recorder.h"
#include "utils/baseline_clip_recorder.h"
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
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cuda_runtime.h>

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

// ==================== Realtime debug dump ====================

struct RealtimeDebugDumpConfig {
    bool enabled = false;
    std::string output_dir = "debug_realtime_dumps";
    int stride = 100;
    int max_frames = 0;
    int max_queue = 4;
    bool dump_fallback = true;
};

struct RealtimeDebugDumpJob {
    int frame_id = 0;
    float fps = 0.0f;
    cv::Mat left_gray;
    cv::Mat right_gray;
    std::vector<stereo3d::Detection> left_detections;
    std::vector<stereo3d::Detection> right_detections;
    std::vector<stereo3d::Object3D> results;
    stereo3d::FrameMetadata metadata;
};

class RealtimeDebugDumper {
public:
    ~RealtimeDebugDumper() { close(); }

    void init(const RealtimeDebugDumpConfig& cfg) {
        cfg_ = cfg;
        if (!cfg_.enabled) return;
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(cfg_.output_dir, ec);
        if (ec) {
            LOG_WARN("RealtimeDebugDumper: failed to create %s: %s",
                     cfg_.output_dir.c_str(), ec.message().c_str());
            cfg_.enabled = false;
            return;
        }
        running_ = true;
        writer_ = std::thread(&RealtimeDebugDumper::writerLoop, this);
        LOG_INFO("RealtimeDebugDumper: output=%s stride=%d max_frames=%d",
                 cfg_.output_dir.c_str(), cfg_.stride, cfg_.max_frames);
    }

    bool enabled() const { return cfg_.enabled && running_; }

    void record(const stereo3d::FrameCallbackData& frame) {
        if (!enabled()) return;
        if (cfg_.max_frames > 0 && captured_count_.load() >= cfg_.max_frames) {
            return;
        }

        bool has_fallback = false;
        for (const auto& obj : frame.results) {
            if (obj.stereo_match_source == 2 || obj.stereo_match_source == 3) {
                has_fallback = true;
                break;
            }
        }
        const bool stride_hit =
            cfg_.stride > 0 && (frame.frame_id % cfg_.stride) == 0;
        if (!stride_hit && !(cfg_.dump_fallback && has_fallback)) {
            return;
        }
        if (!reserveSlot()) {
            return;
        }

        RealtimeDebugDumpJob job;
        job.frame_id = frame.frame_id;
        job.fps = frame.fps;
        job.left_detections = frame.detections_left;
        job.right_detections = frame.detections_right;
        job.results = frame.results;
        job.metadata = frame.metadata;
        if (!copyGray(frame.rect_gray_left, job.left_gray) ||
            !copyGray(frame.rect_gray_right, job.right_gray)) {
            releaseReservedSlot(false, nullptr);
            return;
        }
        releaseReservedSlot(true, &job);
        cv_.notify_one();
    }

    void close() {
        if (!cfg_.enabled && !writer_.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            running_ = false;
        }
        cv_.notify_all();
        if (writer_.joinable()) writer_.join();
        if (cfg_.enabled) {
            LOG_INFO("RealtimeDebugDumper: saved=%d dropped=%d",
                     saved_count_.load(), dropped_count_.load());
        }
        cfg_.enabled = false;
    }

private:
    bool reserveSlot() {
        std::lock_guard<std::mutex> lock(mtx_);
        const int reserved = reserved_count_;
        if (cfg_.max_frames > 0 &&
            captured_count_.load() + reserved >= cfg_.max_frames) {
            return false;
        }
        if (cfg_.max_queue > 0 &&
            queue_.size() + static_cast<size_t>(reserved) >=
                static_cast<size_t>(cfg_.max_queue)) {
            ++dropped_count_;
            return false;
        }
        ++reserved_count_;
        return true;
    }

    void releaseReservedSlot(bool enqueue, RealtimeDebugDumpJob* job) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (reserved_count_ > 0) --reserved_count_;
        if (enqueue && job) {
            queue_.push_back(std::move(*job));
            ++captured_count_;
        }
    }

    static bool copyGray(VPIImage img, cv::Mat& out) {
        VPIImageData data;
        const VPIStatus st = vpiImageLockData(img, VPI_LOCK_READ,
                                              VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                                              &data);
        if (st != VPI_SUCCESS) return false;
        try {
            const int w = data.buffer.pitch.planes[0].width;
            const int h = data.buffer.pitch.planes[0].height;
            const int pitch = data.buffer.pitch.planes[0].pitchBytes;
            cv::Mat view(h, w, CV_8UC1,
                         data.buffer.pitch.planes[0].data, pitch);
            view.copyTo(out);
        } catch (const cv::Exception&) {
            vpiImageUnlock(img);
            return false;
        }
        vpiImageUnlock(img);
        return !out.empty();
    }

    static bool validBox(float cx, float cy, float w, float h) {
        return std::isfinite(cx) && std::isfinite(cy) &&
               std::isfinite(w) && std::isfinite(h) &&
               w > 1.0f && h > 1.0f;
    }

    static cv::Rect cropAround(float cx, float cy, float w, float h,
                               const cv::Size& size) {
        const float scale = 1.8f;
        const int crop_w = std::max(32, static_cast<int>(std::round(w * scale)));
        const int crop_h = std::max(32, static_cast<int>(std::round(h * scale)));
        const int x = static_cast<int>(std::round(cx - crop_w * 0.5f));
        const int y = static_cast<int>(std::round(cy - crop_h * 0.5f));
        return cv::Rect(x, y, crop_w, crop_h) & cv::Rect(0, 0, size.width, size.height);
    }

    static void drawBox(cv::Mat& image, const cv::Rect& crop,
                        float cx, float cy, float w, float h,
                        const cv::Scalar& color) {
        if (!validBox(cx, cy, w, h)) return;
        cv::Rect box(
            static_cast<int>(std::round(cx - w * 0.5f)) - crop.x,
            static_cast<int>(std::round(cy - h * 0.5f)) - crop.y,
            static_cast<int>(std::round(w)),
            static_cast<int>(std::round(h)));
        cv::rectangle(image, box & cv::Rect(0, 0, image.cols, image.rows),
                      color, 1, cv::LINE_AA);
    }

    static void drawCircle(cv::Mat& image, const cv::Rect& crop,
                           float cx, float cy, float r,
                           const cv::Scalar& color) {
        if (!std::isfinite(cx) || !std::isfinite(cy) ||
            !std::isfinite(r) || r <= 1.0f) {
            return;
        }
        cv::circle(image,
                   cv::Point(static_cast<int>(std::round(cx)) - crop.x,
                             static_cast<int>(std::round(cy)) - crop.y),
                   static_cast<int>(std::round(r)),
                   color, 1, cv::LINE_AA);
    }

    static std::string fmt(float value, int digits = 3) {
        if (!std::isfinite(value)) return "nan";
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(digits) << value;
        return ss.str();
    }

    static void putLine(cv::Mat& image, const std::string& text, int line) {
        cv::putText(image, text, cv::Point(8, 18 + line * 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48,
                    cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
    }

    static cv::Mat makePanel(const RealtimeDebugDumpJob& job,
                             const stereo3d::Object3D& obj,
                             int index) {
        cv::Mat left_bgr;
        cv::Mat right_bgr;
        cv::cvtColor(job.left_gray, left_bgr, cv::COLOR_GRAY2BGR);
        cv::cvtColor(job.right_gray, right_bgr, cv::COLOR_GRAY2BGR);

        cv::Rect left_crop = validBox(obj.left_bbox_cx, obj.left_bbox_cy,
                                      obj.left_bbox_w, obj.left_bbox_h)
            ? cropAround(obj.left_bbox_cx, obj.left_bbox_cy,
                         obj.left_bbox_w, obj.left_bbox_h, left_bgr.size())
            : cv::Rect(0, 0, left_bgr.cols, left_bgr.rows);
        cv::Rect right_crop = validBox(obj.right_bbox_cx, obj.right_bbox_cy,
                                       obj.right_bbox_w, obj.right_bbox_h)
            ? cropAround(obj.right_bbox_cx, obj.right_bbox_cy,
                         obj.right_bbox_w, obj.right_bbox_h, right_bgr.size())
            : (left_crop & cv::Rect(0, 0, right_bgr.cols, right_bgr.rows));
        if (left_crop.empty()) left_crop = cv::Rect(0, 0, left_bgr.cols, left_bgr.rows);
        if (right_crop.empty()) right_crop = cv::Rect(0, 0, right_bgr.cols, right_bgr.rows);

        cv::Mat left = left_bgr(left_crop).clone();
        cv::Mat right = right_bgr(right_crop).clone();
        drawBox(left, left_crop, obj.left_bbox_cx, obj.left_bbox_cy,
                obj.left_bbox_w, obj.left_bbox_h, cv::Scalar(0, 255, 0));
        drawBox(right, right_crop, obj.right_bbox_cx, obj.right_bbox_cy,
                obj.right_bbox_w, obj.right_bbox_h, cv::Scalar(0, 255, 0));
        drawCircle(left, left_crop, obj.left_circle_cx, obj.left_circle_cy,
                   obj.left_circle_r, cv::Scalar(255, 255, 0));
        drawCircle(right, right_crop, obj.right_circle_cx, obj.right_circle_cy,
                   obj.right_circle_r, cv::Scalar(255, 0, 255));

        const int target_h = 220;
        const double left_scale = static_cast<double>(target_h) /
                                  static_cast<double>(std::max(1, left.rows));
        const double right_scale = static_cast<double>(target_h) /
                                   static_cast<double>(std::max(1, right.rows));
        cv::resize(left, left, cv::Size(std::max(1, static_cast<int>(left.cols * left_scale)), target_h));
        cv::resize(right, right, cv::Size(std::max(1, static_cast<int>(right.cols * right_scale)), target_h));
        if (left.rows != right.rows) {
            cv::resize(right, right, cv::Size(right.cols, left.rows));
        }

        cv::Mat panel;
        cv::hconcat(std::vector<cv::Mat>{left, right}, panel);
        putLine(panel, "frame=" + std::to_string(job.frame_id) +
                " obj=" + std::to_string(index) +
                " match=" + std::to_string(obj.stereo_match_source) +
                " depth=" + std::to_string(obj.stereo_depth_source), 0);
        putLine(panel, "bbox=" + fmt(obj.z_bbox_center) +
                " circle=" + fmt(obj.z_circle_center) +
                " edge=" + fmt(obj.z_roi_edge_centroid) +
                " radial=" + fmt(obj.z_roi_radial_center), 1);
        putLine(panel, "edgepair=" + fmt(obj.z_roi_edge_pair_center) +
                " multi=" + fmt(obj.z_roi_multi_point) +
                " patch=" + fmt(obj.z_roi_center_patch) +
                " tmpl=" + fmt(obj.z_roi_cuda_template_match), 2);
        putLine(panel, "cuda_bm=" + fmt(obj.z_roi_cuda_stereo_bm) +
                " cuda_sgm=" + fmt(obj.z_roi_cuda_stereo_sgm) +
                " fb_epi=" + fmt(obj.z_fallback_epipolar) +
                " fb=" + fmt(obj.z_fallback), 3);
        putLine(panel,
                " pair_iou=" + fmt(obj.pair_shifted_iou) +
                " Lsrc=" + std::to_string(obj.left_circle_source) +
                " Rsrc=" + std::to_string(obj.right_circle_source), 4);
        putLine(panel, "dy=" + fmt(obj.epipolar_dy, 2), 5);
        return panel;
    }

    static cv::Mat makeDetectionPanel(const RealtimeDebugDumpJob& job) {
        cv::Mat left_bgr;
        cv::Mat right_bgr;
        cv::cvtColor(job.left_gray, left_bgr, cv::COLOR_GRAY2BGR);
        cv::cvtColor(job.right_gray, right_bgr, cv::COLOR_GRAY2BGR);
        auto draw_dets = [](cv::Mat& image,
                            const std::vector<stereo3d::Detection>& detections) {
            for (size_t i = 0; i < detections.size(); ++i) {
                const auto& d = detections[i];
                cv::Rect box(
                    static_cast<int>(std::round(d.cx - d.width * 0.5f)),
                    static_cast<int>(std::round(d.cy - d.height * 0.5f)),
                    static_cast<int>(std::round(d.width)),
                    static_cast<int>(std::round(d.height)));
                cv::rectangle(image, box & cv::Rect(0, 0, image.cols, image.rows),
                              cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                cv::putText(image, "#" + std::to_string(i),
                            cv::Point(std::max(0, box.x), std::max(18, box.y - 4)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55,
                            cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
            }
        };
        draw_dets(left_bgr, job.left_detections);
        draw_dets(right_bgr, job.right_detections);
        cv::Mat panel;
        cv::hconcat(std::vector<cv::Mat>{left_bgr, right_bgr}, panel);
        putLine(panel, "frame=" + std::to_string(job.frame_id) +
                " detections L=" + std::to_string(job.left_detections.size()) +
                " R=" + std::to_string(job.right_detections.size()), 0);
        return panel;
    }

    void writeSummaryJson(const RealtimeDebugDumpJob& job,
                          const std::filesystem::path& path) {
        std::ofstream out(path.string());
        if (!out.is_open()) return;
        out << "{\n";
        out << "  \"frame_id\": " << job.frame_id << ",\n";
        out << "  \"fps\": " << job.fps << ",\n";
        out << "  \"left_count\": " << job.left_detections.size() << ",\n";
        out << "  \"right_count\": " << job.right_detections.size() << ",\n";
        out << "  \"result_count\": " << job.results.size() << ",\n";
        out << "  \"frame_counter_delta\": "
            << (static_cast<int64_t>(job.metadata.left_frame_counter) -
                static_cast<int64_t>(job.metadata.right_frame_counter)) << ",\n";
        out << "  \"frame_number_delta\": "
            << (static_cast<int64_t>(job.metadata.left_frame_number) -
                static_cast<int64_t>(job.metadata.right_frame_number)) << ",\n";
        out << "  \"results\": [\n";
        for (size_t i = 0; i < job.results.size(); ++i) {
            const auto& obj = job.results[i];
            out << "    {";
            out << "\"index\": " << i
                << ", \"match_source\": " << obj.stereo_match_source
                << ", \"depth_source\": " << obj.stereo_depth_source
                << ", \"left_circle_source\": " << obj.left_circle_source
                << ", \"right_circle_source\": " << obj.right_circle_source
                << ", \"z_bbox_center\": " << obj.z_bbox_center
                << ", \"z_circle_center\": " << obj.z_circle_center
                << ", \"z_roi_edge_centroid\": " << obj.z_roi_edge_centroid
                << ", \"z_roi_radial_center\": " << obj.z_roi_radial_center
                << ", \"z_roi_edge_pair_center\": " << obj.z_roi_edge_pair_center
                << ", \"z_roi_multi_point\": " << obj.z_roi_multi_point
                << ", \"z_roi_center_patch\": " << obj.z_roi_center_patch
                << ", \"z_roi_cuda_template_match\": " << obj.z_roi_cuda_template_match
                << ", \"z_roi_cuda_stereo_bm\": " << obj.z_roi_cuda_stereo_bm
                << ", \"z_roi_cuda_stereo_sgm\": " << obj.z_roi_cuda_stereo_sgm
                << ", \"z_fallback_epipolar\": " << obj.z_fallback_epipolar
                << ", \"z_fallback\": " << obj.z_fallback
                << ", \"pair_shifted_iou\": " << obj.pair_shifted_iou
                << "}";
            if (i + 1 < job.results.size()) out << ",";
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";
    }

    void writeJob(const RealtimeDebugDumpJob& job) {
        namespace fs = std::filesystem;
        std::ostringstream prefix;
        prefix << "frame_" << std::setw(6) << std::setfill('0') << job.frame_id;
        const fs::path root(cfg_.output_dir);

        cv::Mat panel;
        if (!job.results.empty()) {
            std::vector<cv::Mat> panels;
            for (size_t i = 0; i < job.results.size(); ++i) {
                panels.push_back(makePanel(job, job.results[i], static_cast<int>(i)));
            }
            cv::vconcat(panels, panel);
        } else {
            panel = makeDetectionPanel(job);
        }
        cv::imwrite((root / (prefix.str() + "_zoom.png")).string(), panel);
        writeSummaryJson(job, root / (prefix.str() + "_summary.json"));
        ++saved_count_;
    }

    void writerLoop() {
        while (true) {
            RealtimeDebugDumpJob job;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
                if (!running_ && queue_.empty()) break;
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            try {
                writeJob(job);
            } catch (const cv::Exception& e) {
                LOG_WARN("RealtimeDebugDumper: write failed frame=%d: %s",
                         job.frame_id, e.what());
            }
        }
    }

    RealtimeDebugDumpConfig cfg_;
    std::atomic<bool> running_{false};
    std::atomic<int> captured_count_{0};
    std::atomic<int> saved_count_{0};
    std::atomic<int> dropped_count_{0};
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<RealtimeDebugDumpJob> queue_;
    int reserved_count_ = 0;
    std::thread writer_;
};

// ==================== 配置加载 ====================

static stereo3d::PipelineConfig loadConfig(const std::string& path) {
    stereo3d::PipelineConfig cfg;
    bool camera_trigger_frequency_set = false;

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
        if (cam["trigger_frequency_hz"]) {
            cfg.camera.trigger_frequency_hz = cam["trigger_frequency_hz"].as<int>();
            camera_trigger_frequency_set = true;
        }
        if (cam["image_node_num"]) cfg.camera.image_node_num = cam["image_node_num"].as<int>();
        if (cam["embedded_info_clear_rows"])
            cfg.camera.embedded_info_clear_rows = cam["embedded_info_clear_rows"].as<int>();
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
        if (det["confidence_threshold"])   cfg.conf_threshold = det["confidence_threshold"].as<float>();
        if (det["nms_threshold"])           cfg.nms_threshold = det["nms_threshold"].as<float>();
        if (det["input_size"])             cfg.input_size = det["input_size"].as<int>();
        if (det["max_detections"])         cfg.max_detections = det["max_detections"].as<int>();
        if (det["input_format"])           cfg.detector_input_format = det["input_format"].as<std::string>();
        if (auto dual = det["dual_yolo"]) {
            if (dual["enabled"]) cfg.dual_yolo.enabled = dual["enabled"].as<bool>();
            if (dual["right_engine_path"])
                cfg.dual_yolo.right_engine_file = dual["right_engine_path"].as<std::string>();
            if (dual["right_engine_file"])
                cfg.dual_yolo.right_engine_file = dual["right_engine_file"].as<std::string>();
            if (dual["right_input_format"])
                cfg.dual_yolo.right_input_format = dual["right_input_format"].as<std::string>();
            cfg.dual_yolo.right_use_dla = cfg.use_dla;
            if (dual["right_use_dla"])
                cfg.dual_yolo.right_use_dla = dual["right_use_dla"].as<bool>();
            if (dual["right_dla_core"])
                cfg.dual_yolo.right_dla_core = dual["right_dla_core"].as<int>();
            if (dual["use_for_depth"])
                cfg.dual_yolo.use_for_depth = dual["use_for_depth"].as<bool>();
            if (dual["fallback_to_roi_match"])
                cfg.dual_yolo.fallback_to_roi_match = dual["fallback_to_roi_match"].as<bool>();
            if (dual["gpu_candidate_refine"])
                cfg.dual_yolo.gpu_candidate_refine = dual["gpu_candidate_refine"].as<bool>();
            if (dual["fallback_epipolar_search"]) {
                cfg.dual_yolo.fallback_epipolar_search = dual["fallback_epipolar_search"].as<bool>();
                cfg.dual_yolo.depth_epipolar_fallback = cfg.dual_yolo.fallback_epipolar_search;
            }
            if (dual["center_refine"])
                cfg.dual_yolo.center_refine = dual["center_refine"].as<bool>();
            if (dual["roi_denoise"])
                cfg.dual_yolo.roi_denoise = dual["roi_denoise"].as<bool>();
            if (dual["log_matches"])
                cfg.dual_yolo.log_matches = dual["log_matches"].as<bool>();
            if (dual["depth_solver"])
                cfg.dual_yolo.depth_solver = dual["depth_solver"].as<std::string>();
            if (dual["subpixel_enabled"]) {
                cfg.dual_yolo.subpixel_enabled = dual["subpixel_enabled"].as<bool>();
                cfg.dual_yolo.depth_roi_subpixel = cfg.dual_yolo.subpixel_enabled;
            }
            if (auto modes = dual["depth_modes"]) {
                if (modes["bbox_pair"])
                    cfg.dual_yolo.depth_bbox_pair = modes["bbox_pair"].as<bool>();
                if (modes["bbox_edges"])
                    cfg.dual_yolo.depth_bbox_edges = modes["bbox_edges"].as<bool>();
                if (modes["circle_center"])
                    cfg.dual_yolo.depth_circle_center = modes["circle_center"].as<bool>();
                if (modes["circle_edges"])
                    cfg.dual_yolo.depth_circle_edges = modes["circle_edges"].as<bool>();
                if (modes["roi_edge_centroid"])
                    cfg.dual_yolo.depth_roi_edge_centroid = modes["roi_edge_centroid"].as<bool>();
                if (modes["roi_radial_center"])
                    cfg.dual_yolo.depth_roi_radial_center = modes["roi_radial_center"].as<bool>();
                if (modes["roi_edge_pair_center"])
                    cfg.dual_yolo.depth_roi_edge_pair_center = modes["roi_edge_pair_center"].as<bool>();
                if (modes["roi_corner_points"])
                    cfg.dual_yolo.depth_roi_corner_points = modes["roi_corner_points"].as<bool>();
                if (modes["roi_texture_points"])
                    cfg.dual_yolo.depth_roi_texture_points = modes["roi_texture_points"].as<bool>();
                if (modes["roi_binary_points"])
                    cfg.dual_yolo.depth_roi_binary_points = modes["roi_binary_points"].as<bool>();
                if (modes["roi_orb_points"])
                    cfg.dual_yolo.depth_roi_orb_points = modes["roi_orb_points"].as<bool>();
                if (modes["roi_brisk_points"])
                    cfg.dual_yolo.depth_roi_brisk_points = modes["roi_brisk_points"].as<bool>();
                if (modes["roi_akaze_points"])
                    cfg.dual_yolo.depth_roi_akaze_points = modes["roi_akaze_points"].as<bool>();
                if (modes["roi_sift_points"])
                    cfg.dual_yolo.depth_roi_sift_points = modes["roi_sift_points"].as<bool>();
                if (modes["roi_iou_region_color_patch"])
                    cfg.dual_yolo.depth_roi_iou_region_color_patch =
                        modes["roi_iou_region_color_patch"].as<bool>();
                if (modes["roi_patch_iou_color_edge"])
                    cfg.dual_yolo.depth_roi_patch_iou_color_edge =
                        modes["roi_patch_iou_color_edge"].as<bool>();
                if (modes["roi_cuda_template_match"])
                    cfg.dual_yolo.depth_roi_cuda_template_match =
                        modes["roi_cuda_template_match"].as<bool>();
                if (modes["roi_cuda_stereo_bm"])
                    cfg.dual_yolo.depth_roi_cuda_stereo_bm =
                        modes["roi_cuda_stereo_bm"].as<bool>();
                if (modes["roi_cuda_stereo_sgm"])
                    cfg.dual_yolo.depth_roi_cuda_stereo_sgm =
                        modes["roi_cuda_stereo_sgm"].as<bool>();
                if (modes["roi_center_patch"])
                    cfg.dual_yolo.depth_roi_center_patch = modes["roi_center_patch"].as<bool>();
                if (modes["roi_subpixel"])
                    cfg.dual_yolo.depth_roi_subpixel = modes["roi_subpixel"].as<bool>();
                if (modes["epipolar_fallback"])
                    cfg.dual_yolo.depth_epipolar_fallback = modes["epipolar_fallback"].as<bool>();
                if (modes["fallback_template"])
                    cfg.dual_yolo.depth_fallback_template = modes["fallback_template"].as<bool>();
                if (modes["fallback_feature_points"])
                    cfg.dual_yolo.depth_fallback_feature_points = modes["fallback_feature_points"].as<bool>();
            }
            if (dual["subpixel_patch_radius"])
                cfg.dual_yolo.subpixel_patch_radius = dual["subpixel_patch_radius"].as<int>();
            if (dual["subpixel_search_radius_px"])
                cfg.dual_yolo.subpixel_search_radius_px = dual["subpixel_search_radius_px"].as<int>();
            if (dual["subpixel_max_points"])
                cfg.dual_yolo.subpixel_max_points = dual["subpixel_max_points"].as<int>();
            if (dual["subpixel_min_points"])
                cfg.dual_yolo.subpixel_min_points = dual["subpixel_min_points"].as<int>();
            if (dual["subpixel_min_confidence"])
                cfg.dual_yolo.subpixel_min_confidence = dual["subpixel_min_confidence"].as<float>();
            if (dual["subpixel_max_disp_delta_px"])
                cfg.dual_yolo.subpixel_max_disp_delta_px = dual["subpixel_max_disp_delta_px"].as<float>();
            if (dual["subpixel_max_disp_delta_ratio"])
                cfg.dual_yolo.subpixel_max_disp_delta_ratio = dual["subpixel_max_disp_delta_ratio"].as<float>();
            if (dual["subpixel_max_depth_delta_m"])
                cfg.dual_yolo.subpixel_max_depth_delta_m = dual["subpixel_max_depth_delta_m"].as<float>();
            if (dual["subpixel_max_stddev_px"])
                cfg.dual_yolo.subpixel_max_stddev_px = dual["subpixel_max_stddev_px"].as<float>();
            if (dual["subpixel_time_budget_ms"])
                cfg.dual_yolo.subpixel_time_budget_ms = dual["subpixel_time_budget_ms"].as<float>();
            if (dual["epipolar_y_tolerance"])
                cfg.dual_yolo.epipolar_y_tolerance = dual["epipolar_y_tolerance"].as<float>();
            if (dual["feature_y_tolerance_px"])
                cfg.dual_yolo.feature_y_tolerance_px = dual["feature_y_tolerance_px"].as<float>();
            if (dual["feature_y_slope"])
                cfg.dual_yolo.feature_y_slope = dual["feature_y_slope"].as<float>();
            if (dual["feature_y_offset_px"])
                cfg.dual_yolo.feature_y_offset_px = dual["feature_y_offset_px"].as<float>();
            if (dual["feature_reverse_check_px"])
                cfg.dual_yolo.feature_reverse_check_px = dual["feature_reverse_check_px"].as<float>();
            if (dual["feature_overlap_scale"])
                cfg.dual_yolo.feature_overlap_scale = dual["feature_overlap_scale"].as<float>();
            if (dual["feature_mad_scale"])
                cfg.dual_yolo.feature_mad_scale = dual["feature_mad_scale"].as<float>();
            if (dual["feature_ransac_gate_px"])
                cfg.dual_yolo.feature_ransac_gate_px = dual["feature_ransac_gate_px"].as<float>();
            if (dual["feature_sphere_radius_scale"])
                cfg.dual_yolo.feature_sphere_radius_scale = dual["feature_sphere_radius_scale"].as<float>();
            if (dual["feature_sphere_margin_m"])
                cfg.dual_yolo.feature_sphere_margin_m = dual["feature_sphere_margin_m"].as<float>();
            if (dual["feature_normalize_large_roi"])
                cfg.dual_yolo.feature_normalize_large_roi = dual["feature_normalize_large_roi"].as<bool>();
            if (dual["feature_normalized_diameter_px"])
                cfg.dual_yolo.feature_normalized_diameter_px = dual["feature_normalized_diameter_px"].as<int>();
            if (dual["feature_normalize_min_diameter_px"])
                cfg.dual_yolo.feature_normalize_min_diameter_px = dual["feature_normalize_min_diameter_px"].as<float>();
            if (dual["feature_normalize_margin_scale"])
                cfg.dual_yolo.feature_normalize_margin_scale = dual["feature_normalize_margin_scale"].as<float>();
            if (dual["feature_precompute_roi_maps"])
                cfg.dual_yolo.feature_precompute_roi_maps = dual["feature_precompute_roi_maps"].as<bool>();
            if (dual["max_size_ratio"])
                cfg.dual_yolo.max_size_ratio = dual["max_size_ratio"].as<float>();
            if (dual["min_shifted_iou"])
                cfg.dual_yolo.min_shifted_iou = dual["min_shifted_iou"].as<float>();
            if (dual["bbox_disparity_consistency_ratio"])
                cfg.dual_yolo.bbox_disparity_consistency_ratio =
                    dual["bbox_disparity_consistency_ratio"].as<float>();
            if (dual["bbox_disparity_consistency_min_px"])
                cfg.dual_yolo.bbox_disparity_consistency_min_px =
                    dual["bbox_disparity_consistency_min_px"].as<float>();
            if (dual["bbox_disparity_penalty_scale"])
                cfg.dual_yolo.bbox_disparity_penalty_scale =
                    dual["bbox_disparity_penalty_scale"].as<float>();
            if (dual["fallback_search_margin_px"])
                cfg.dual_yolo.fallback_search_margin_px = dual["fallback_search_margin_px"].as<int>();
            if (dual["fallback_max_width_px"])
                cfg.dual_yolo.fallback_max_width_px = dual["fallback_max_width_px"].as<int>();
            if (dual["circle_max_roi_pixels"])
                cfg.dual_yolo.circle_max_roi_pixels = dual["circle_max_roi_pixels"].as<int>();
        }
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

    // Learned ROI feature matching (TensorRT runtime, disabled by default)
    if (auto nf = root["neural_feature_matching"]) {
        if (nf["enabled"]) cfg.neural_features.enabled = nf["enabled"].as<bool>();
        if (nf["backend"]) {
            cfg.neural_features.backend_name = nf["backend"].as<std::string>();
            cfg.neural_features.backend =
                stereo3d::parseNeuralFeatureBackend(cfg.neural_features.backend_name);
        }
        if (nf["extractor_engine_path"])
            cfg.neural_features.extractor_engine_path =
                nf["extractor_engine_path"].as<std::string>();
        if (nf["matcher_engine_path"])
            cfg.neural_features.matcher_engine_path =
                nf["matcher_engine_path"].as<std::string>();
        if (nf["fused_engine_path"])
            cfg.neural_features.fused_engine_path =
                nf["fused_engine_path"].as<std::string>();
        if (nf["roi_size"])
            cfg.neural_features.roi_size = nf["roi_size"].as<int>();
        if (nf["top_k"])
            cfg.neural_features.top_k = nf["top_k"].as<int>();
        if (nf["descriptor_dim"])
            cfg.neural_features.descriptor_dim = nf["descriptor_dim"].as<int>();
        if (nf["min_matches"])
            cfg.neural_features.min_matches = nf["min_matches"].as<int>();
        if (nf["max_y_error_px"])
            cfg.neural_features.max_y_error_px = nf["max_y_error_px"].as<float>();
        if (nf["max_disp_delta_px"])
            cfg.neural_features.max_disp_delta_px = nf["max_disp_delta_px"].as<float>();
        if (nf["final_disp_gate_px"])
            cfg.neural_features.final_disp_gate_px = nf["final_disp_gate_px"].as<float>();
        if (nf["min_score"])
            cfg.neural_features.min_score = nf["min_score"].as<float>();
        if (nf["use_lightglue"])
            cfg.neural_features.use_lightglue = nf["use_lightglue"].as<bool>();
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
        if (fus["min_confidence"])  cfg.depth.min_confidence  = fus["min_confidence"].as<float>();
        // Kalman 观测噪声 (距离自适应基值)
        if (fus["R_mono"])          cfg.depth.R_mono          = fus["R_mono"].as<float>();
        if (fus["R_stereo"])        cfg.depth.R_stereo        = fus["R_stereo"].as<float>();
        // IVW 融合权重 (与 Kalman R 分离)
        if (fus["ivw_R_mono"])      cfg.depth.ivw_R_mono      = fus["ivw_R_mono"].as<float>();
        if (fus["ivw_R_stereo"])    cfg.depth.ivw_R_stereo    = fus["ivw_R_stereo"].as<float>();
        if (fus["fallback_stereo_weight_scale"])
            cfg.depth.fallback_stereo_weight_scale =
                fus["fallback_stereo_weight_scale"].as<float>();
        if (fus["fallback_obs_noise_scale"])
            cfg.depth.fallback_obs_noise_scale =
                fus["fallback_obs_noise_scale"].as<float>();
    }

    // Performance
    if (auto perf = root["performance"]) {
        if (perf["log_interval"])   cfg.stats_interval = perf["log_interval"].as<int>();
        if (perf["pwm_frequency"])  cfg.trigger_freq_hz = static_cast<int>(perf["pwm_frequency"].as<float>());
        if (perf["drop_stale_roi_frames"])
            cfg.drop_stale_roi_frames = perf["drop_stale_roi_frames"].as<bool>();
        if (perf["async_roi_stage2"])
            cfg.async_roi_stage2 = perf["async_roi_stage2"].as<bool>();
        if (perf["async_roi_buffers"])
            cfg.async_roi_buffers = perf["async_roi_buffers"].as<int>();
        if (perf["async_roi_deadline_ms"])
            cfg.async_roi_deadline_ms = perf["async_roi_deadline_ms"].as<float>();
    }
    if (!camera_trigger_frequency_set) {
        cfg.camera.trigger_frequency_hz = cfg.trigger_freq_hz;
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
        cfg.tracker.detect_interval = std::max(1, cfg.tracker.detect_interval);
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

static stereo3d::BaselineClipRecorderConfig loadBaselineClipRecorderConfig(
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

static RealtimeDebugDumpConfig loadRealtimeDebugDumpConfig(const std::string& path) {
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

#ifdef HAS_ROS2
static stereo3d::Ros2BridgeConfig loadRos2Config(const std::string& path) {
    stereo3d::Ros2BridgeConfig cfg{};
    YAML::Node root = YAML::LoadFile(path);
    auto ros = root["ros2"];
    if (!ros) return cfg;

    cfg.enabled          = ros["enable"].as<bool>(false);
    cfg.world_frame_id   = ros["world_frame_id"].as<std::string>("vision_world");
    cfg.base_frame_id    = ros["base_frame_id"].as<std::string>("base_link");
    cfg.odom_topic       = ros["odom_topic"].as<std::string>("/odom");
    cfg.odom_timeout_sec = ros["odom_timeout_sec"].as<double>(0.5);

    if (auto t = ros["topics"]) {
        cfg.topic_realtime       = t["ball_realtime"].as<std::string>("/ball/realtime");
        cfg.topic_landing        = t["ball_landing"].as<std::string>("/ball/landing");
        cfg.topic_predicted_path = t["predicted_path"].as<std::string>("/ball/predicted_path");
        cfg.topic_actual_path    = t["actual_path"].as<std::string>("/ball/actual_path");
        cfg.topic_realtime_base  = t["ball_realtime_base"].as<std::string>("/ball/realtime_base");
        cfg.topic_landing_base   = t["ball_landing_base"].as<std::string>("/ball/landing_base");
    }
    if (auto v = ros["vision_to_world"]) {
        cfg.swap_xy       = v["swap_xy"].as<bool>(false);
        cfg.invert_x      = v["invert_x"].as<bool>(false);
        cfg.invert_y      = v["invert_y"].as<bool>(false);
        cfg.rotation_deg  = v["rotation_deg"].as<double>(0.0);
        cfg.translation_x = v["translation_x"].as<double>(0.0);
        cfg.translation_y = v["translation_y"].as<double>(0.0);
    }
    return cfg;
}
#endif

// ==================== main ====================

int main(int argc, char* argv[]) {
    // 解析命令行
    std::string config_path = "config/pipeline.yaml";
    bool enable_display = false;
    bool debug_feature_matches = false;
    std::string debug_feature_matches_dir = "test_logs/feature_match_debug";
    std::string recording_out_override;
    bool baseline_clip_cli = false;
    std::string baseline_out_override;
    double baseline_duration_override = -1.0;
    int baseline_frames_override = 0;
    int baseline_clips_override = 0;
    double baseline_gap_override = -1.0;
    std::string baseline_format_override;
    std::string baseline_image_mode_override;
    bool baseline_start_immediately = false;
    bool debug_realtime_dump_cli = false;
    std::string debug_realtime_dump_dir_override;
    int debug_realtime_dump_stride_override = -1;
    int debug_realtime_dump_max_override = -1;
    std::vector<std::string> unknown_args;
    const char* usage =
        "Usage: %s [--config <path>] [--visualize] "
        "[--debug-feature-matches] [--debug-feature-matches-dir <dir>] "
        "[--debug-realtime-dump] [--debug-realtime-dump-dir <dir>] "
        "[--debug-realtime-dump-stride <n>] [--debug-realtime-dump-max <n>] "
        "[--recording-out <csv>] "
        "[--record-baseline-clip] [--baseline-out <dir>] "
        "[--baseline-duration <sec>] [--baseline-frames <n>] "
        "[--baseline-clips <n>] [--baseline-gap <sec>] "
        "[--baseline-format <png|pgm>] [--baseline-image-mode <gray|bgr|both>] "
        "[--baseline-start-immediately]\n";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") &&
            i + 1 < argc && argv[i + 1][0] != '-') {
            config_path = argv[++i];
        } else if (arg == "--config" || arg == "-c") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--visualize" || arg == "--display" || arg == "-v") {
            enable_display = true;
        } else if (arg == "--debug-feature-matches") {
            debug_feature_matches = true;
        } else if (arg == "--debug-feature-matches-dir" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            debug_feature_matches = true;
            debug_feature_matches_dir = argv[++i];
        } else if (arg == "--debug-feature-matches-dir") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--debug-realtime-dump") {
            debug_realtime_dump_cli = true;
        } else if (arg == "--debug-realtime-dump-dir" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            debug_realtime_dump_cli = true;
            debug_realtime_dump_dir_override = argv[++i];
        } else if (arg == "--debug-realtime-dump-dir") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--debug-realtime-dump-stride" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            debug_realtime_dump_cli = true;
            debug_realtime_dump_stride_override = std::stoi(argv[++i]);
        } else if (arg == "--debug-realtime-dump-stride") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--debug-realtime-dump-max" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            debug_realtime_dump_cli = true;
            debug_realtime_dump_max_override = std::stoi(argv[++i]);
        } else if (arg == "--debug-realtime-dump-max") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--recording-out" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            recording_out_override = argv[++i];
        } else if (arg == "--recording-out") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--record-baseline-clip") {
            baseline_clip_cli = true;
        } else if (arg == "--baseline-out" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            baseline_out_override = argv[++i];
        } else if (arg == "--baseline-out") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--baseline-duration" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            baseline_duration_override = std::stod(argv[++i]);
            baseline_clip_cli = true;
        } else if (arg == "--baseline-duration") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--baseline-frames" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            baseline_frames_override = std::stoi(argv[++i]);
            baseline_clip_cli = true;
        } else if (arg == "--baseline-frames") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--baseline-clips" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            baseline_clips_override = std::stoi(argv[++i]);
            baseline_clip_cli = true;
        } else if (arg == "--baseline-clips") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--baseline-gap" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            baseline_gap_override = std::stod(argv[++i]);
            baseline_clip_cli = true;
        } else if (arg == "--baseline-gap") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--baseline-format" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            baseline_format_override = argv[++i];
            baseline_clip_cli = true;
        } else if (arg == "--baseline-format") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--baseline-image-mode" &&
                   i + 1 < argc && argv[i + 1][0] != '-') {
            baseline_image_mode_override = argv[++i];
            baseline_clip_cli = true;
        } else if (arg == "--baseline-image-mode") {
            fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
            fprintf(stderr, usage, argv[0]);
            return 1;
        } else if (arg == "--baseline-start-immediately") {
            baseline_start_immediately = true;
            baseline_clip_cli = true;
        } else if (arg == "--visualizels") {
            // 兼容常见拼写误写，避免静默关闭可视化
            fprintf(stderr, "Warning: unknown option '--visualizels', treating as '--visualize'.\n");
            enable_display = true;
        } else if (arg == "--help" || arg == "-h") {
            printf(usage, argv[0]);
            printf("  --config, -c                  Pipeline configuration YAML\n");
            printf("  --visualize, -v               Show detection + distance overlay window\n");
            printf("  --debug-feature-matches       Capture one stereo pair and export ROI feature-match images\n");
            printf("  --debug-feature-matches-dir   Output directory for feature-match images\n");
            printf("  --debug-realtime-dump         Low-rate realtime zoom PNG/JSON dump (background writer)\n");
            printf("  --debug-realtime-dump-dir     Output directory for realtime debug dump\n");
            printf("  --debug-realtime-dump-stride  Dump every N frames; 0 disables periodic dumps\n");
            printf("  --debug-realtime-dump-max     Stop dumping after N frames; 0 means unlimited\n");
            printf("  --recording-out <csv>         Override trajectory recorder CSV output path\n");
            printf("  --record-baseline-clip        Record one fixed-length left/right image sequence + CSV after ball detection\n");
            printf("  --baseline-out                Output root directory for baseline clips\n");
            printf("  --baseline-duration           Clip duration in seconds, converted by trigger frequency\n");
            printf("  --baseline-frames             Exact number of frames to record\n");
            printf("  --baseline-clips              Number of clips to record\n");
            printf("  --baseline-gap                Gap between clips in seconds\n");
            printf("  --baseline-format             Lossless image format: png or pgm\n");
            printf("  --baseline-image-mode         Image mode: gray, bgr, or both\n");
            printf("  --baseline-start-immediately  Record without waiting for detections\n");
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            unknown_args.push_back(arg);
        } else {
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

                if (use_bgr) {
                    VPIImage rectL = frame_data.rect_bgr_left;
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
                }

                // 灰度快速路径 (跳过 CPU debayer+remap, 避免 10ms 阻塞)
                if (frame.empty()) {
                    VPIImage rectL = frame_data.rect_gray_left;
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
