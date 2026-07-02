/**
 * @file pipeline.h
 * @brief 四级流水线 Pipeline 类
 *
 * 流水线架构:
 *   Stage 0 [Grab + Rectify]  → CPU + PVA    ~3ms
 *   Stage 1 [Detect / DLA]    → NVDLA         ~12-15ms  ┐
 *   Stage 2 [Stereo / GPU]    → GPU CUDA      ~10-12ms  ┘ 串行执行, 帧级流水线重叠
 *   Stage 3 [Fuse + Output]   → GPU/CPU       ~1ms
 *
 * 使用三缓冲实现帧间流水线重叠:
 *   Frame N:   Stage 3 (Fuse)
 *   Frame N+1: Stage 1+2 (Detect+Stereo)
 *   Frame N+2: Stage 0 (Grab+Rect)
 *
 * 吞吐量 = 1 / max(Stage_i latency) → 60-100 FPS
 */

#ifndef STEREO_3D_PIPELINE_PIPELINE_H_
#define STEREO_3D_PIPELINE_PIPELINE_H_

#include "frame_slot.h"
#include "sync.h"
#include "../capture/hikvision_camera.h"   // CameraConfig (值类型, 必须完整定义)
#ifndef HIK_CAMERA_ENABLED
namespace stereo3d { class HikvisionCamera; }  // 仅 class 需 forward declare
#endif
#include "../calibration/pwm_trigger.h"
#include "../calibration/stereo_calibration.h"
#include "../rectify/vpi_rectifier.h"
#include "../detect/trt_detector.h"
#include "../stereo/vpi_stereo.h"
#include "../stereo/roi_stereo_matcher.h"
#include "../stereo/dual_yolo_depth_gpu.h"
#include "../stereo/neural_feature_config.h"
#include "../fusion/coordinate_3d.h"
#include "../fusion/hybrid_depth.h"
#include "../track/sot_tracker.h"
#include "../utils/profiler.h"

#include <vpi/algo/TemporalNoiseReduction.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace stereo3d {

class NeuralFeatureMatcher;

/**
 * @brief 视差策略
 */
enum class DisparityStrategy {
    FULL_FRAME,        ///< 全帧视差 (默认, 1280x720)
    HALF_RESOLUTION,   ///< 半分辨率 (640x360)
    ROI_ONLY           ///< 仅计算 ROI 区域
};

#ifdef HAS_ROS2
struct Ros2BridgeConfig {
    bool enabled;
    std::string world_frame_id;
    std::string base_frame_id;
    std::string odom_topic;
    double odom_timeout_sec;
    std::string topic_realtime;
    std::string topic_landing;
    std::string topic_predicted_path;
    std::string topic_actual_path;
    std::string topic_realtime_base;
    std::string topic_landing_base;
    bool swap_xy;
    bool invert_x;
    bool invert_y;
    double rotation_deg;
    double translation_x;
    double translation_y;
};
#endif

/**
 * @brief Pipeline 运行时参数
 */
struct PipelineConfig {
    // 相机配置 (内嵌, 避免重复定义)
    CameraConfig camera;

    // 校正后分辨率
    int rect_width  = 1280;
    int rect_height = 720;
    std::string rect_backend = "VIC"; ///< 校正后端: "VIC" (推荐,不占GPU) 或 "CUDA"

    // PWM 触发器 (Pipeline 级, 非相机级)
    std::string trigger_chip = "gpiochip2";  ///< GPIO 芯片名 (Orin NX: gpiochip2)
    int trigger_line = 7;                     ///< GPIO 线路号 (Orin NX: line 7)
    int trigger_freq_hz = 100;

    // 标定
    std::string calibration_file = "config/intrinsics.yaml";

    // 检测
    std::string engine_file = "models/yolov8n_int8.engine";
    int  input_size      = 320;    ///< 模型输入尺寸
    float conf_threshold = 0.5f;
    float nms_threshold  = 0.4f;
    int  max_detections  = 10;
    std::string detector_input_format = "gray"; ///< gray|bayer|bgr
    bool use_dla = false;          ///< 使用 NVDLA (否则 GPU)
    int  dla_core = 0;             ///< DLA 核心 ID (0 或 1)

    // 双路 YOLO + 极线语义匹配测试
    struct DualYoloConfig {
        bool enabled = false;              ///< 是否同时运行右目 YOLO
        std::string right_engine_file;     ///< 右目 engine, 为空则复用左目 engine_path
        std::string right_input_format;    ///< 右目输入格式, 为空则复用 detector_input_format
        bool right_use_dla = false;        ///< 右目是否使用 DLA
        int right_dla_core = 1;            ///< 右目 DLA core, 仅用于日志/engine 匹配
        bool use_for_depth = true;         ///< 用左右 YOLO 语义匹配结果作为 stereo 观测
        bool depth_bbox_pair = true;       ///< 计算左右 YOLO bbox 中心视差候选
        bool depth_bbox_edges = false;     ///< 计算左右 YOLO bbox 左/右边缘视差候选
        bool depth_circle_center = true;   ///< 计算 ROI 圆拟合圆心三角测距候选, 需 center_refine
        bool depth_circle_edges = false;   ///< 计算 ROI 圆左右边缘视差候选
        bool depth_roi_edge_centroid = false; ///< 计算 ROI 边缘梯度质心视差候选
        bool depth_roi_radial_center = false; ///< 计算 ROI 径向梯度中心视差候选
        bool depth_roi_edge_pair_center = false; ///< 计算 ROI 左右边缘成对中心视差候选
        bool depth_roi_corner_points = false; ///< 计算 ROI 角点特征点视差候选
        bool depth_roi_texture_points = false; ///< 计算 ROI 纹理/梯度特征点视差候选
        bool depth_roi_binary_points = false; ///< 计算 ROI 二值描述子特征点视差候选
        bool depth_roi_orb_points = false; ///< 计算 ROI ORB 特征点视差候选
        bool depth_roi_brisk_points = false; ///< 计算 ROI BRISK 特征点视差候选
        bool depth_roi_akaze_points = false; ///< 计算 ROI AKAZE 特征点视差候选
        bool depth_roi_sift_points = false; ///< 计算 ROI OpenCV CPU SIFT 特征点视差候选
        bool depth_roi_iou_region_color_patch = false; ///< 计算 ROI 彩色区域 IoU/patch 视差候选
        bool depth_roi_patch_iou_color_edge = false; ///< 计算 ROI 彩色边缘 IoU/patch 视差候选
        bool depth_roi_center_patch = false; ///< 计算 ROI 中心 patch ZNCC 视差候选
        bool depth_roi_subpixel = true;    ///< 计算 ROI 多点亚像素视差候选
        bool depth_epipolar_fallback = true; ///< 单目漏检时启用有界极线搜索候选
        bool depth_fallback_template = false; ///< 单目漏检时启用极线模板搜索候选
        bool depth_fallback_feature_points = false; ///< 单目漏检时启用极线特征点搜索候选
        bool fallback_to_roi_match = true; ///< 旧 ROI 纹理匹配回退, 双 YOLO 测试默认关闭
        bool fallback_epipolar_search = true; ///< 单目漏检时在另一目极线附近做有界搜索
        bool gpu_candidate_refine = true;  ///< 双 YOLO 直接匹配候选优先使用 GPU 批处理
        bool center_refine = true;         ///< 在 bbox/搜索 ROI 内做圆心拟合细化
        bool roi_denoise = true;           ///< 圆心拟合前做局部 3x3 读数降噪
        bool log_matches = true;           ///< 按 stats_interval 打印匹配统计
        std::string depth_solver = "circle_center"; ///< circle_center|roi_subpixel_match
        bool subpixel_enabled = true;      ///< roi_subpixel_match: 启用 ROI 多点亚像素视差细化
        int subpixel_patch_radius = 5;     ///< ZNCC 匹配块半径 (patch=2r+1)
        int subpixel_search_radius_px = 8; ///< 以圆心视差为中心的左右搜索半宽
        int subpixel_max_points = 12;      ///< ROI 内最多采样点数
        int subpixel_min_points = 4;       ///< 接受亚像素视差的最少有效点
        float subpixel_min_confidence = 0.25f; ///< 接受亚像素视差的最低置信度
        float subpixel_max_disp_delta_px = 2.0f; ///< 相对圆心视差最大允许绝对偏差
        float subpixel_max_disp_delta_ratio = 0.03f; ///< 相对圆心视差最大比例偏差
        float subpixel_max_depth_delta_m = 0.5f; ///< 亚像素视差相对圆心测距最大跳变
        float subpixel_max_stddev_px = 1.0f; ///< 多点视差最大标准差
        float subpixel_time_budget_ms = 1.5f; ///< 每帧亚像素 CPU 预算, 0 表示不限制
        float epipolar_y_tolerance = 12.0f;///< 左右中心 y 允许差值 (px)
        float feature_y_tolerance_px = 2.0f; ///< ROI 特征点严格 y 残差门限
        float feature_y_slope = 0.0f;        ///< 可选 epipolar y 线性补偿斜率
        float feature_y_offset_px = 0.0f;    ///< 可选 epipolar y 固定补偿
        float feature_reverse_check_px = 1.0f; ///< 左右反查最大误差
        float feature_overlap_scale = 0.55f; ///< 左右 bbox 投影 overlap 椭圆尺度
        float feature_mad_scale = 2.5f;      ///< MAD 离群剔除倍数
        float feature_ransac_gate_px = 0.75f; ///< 1D disparity RANSAC 最小门限
        float feature_sphere_radius_scale = 1.8f; ///< 三角点离球心最大半径倍数
        float feature_sphere_margin_m = 0.02f; ///< 物理半径门限冗余
        bool feature_normalize_large_roi = true; ///< 大 ROI 降采样到固定球直径后匹配
        int feature_normalized_diameter_px = 96; ///< 大 ROI 归一化目标球直径
        float feature_normalize_min_diameter_px = 128.0f; ///< 超过该直径才归一化
        float feature_normalize_margin_scale = 0.62f; ///< 归一化 ROI 半径相对 bbox 尺寸
        bool feature_precompute_roi_maps = true; ///< ROI 内预计算 label/edge 供匹配使用
        float max_size_ratio = 2.0f;       ///< 左右 bbox 尺寸比例上限
        int fallback_search_margin_px = 48;///< 期望视差两侧搜索半宽 (px)
        int fallback_max_width_px = 220;   ///< 极线 fallback 搜索窗口最大宽度
        int circle_max_roi_pixels = 18000; ///< CPU 圆拟合最大采样像素数, 超过后步进采样
    } dual_yolo;

    NeuralFeatureConfig neural_features;

    // SOT Tracker 补帧 (YOLO 检测间隙帧)
    struct TrackerConfig {
        bool enabled = false;              ///< 是否启用 SOT 补帧
        std::string type = "nanotrack";    ///< "nanotrack" | "mixformer"
        std::string engine_path;           ///< TRT 引擎路径 (backbone / template backbone)
        std::string search_engine_path;    ///< NanoTrack search backbone 引擎 (双backbone模式)
        std::string head_engine_path;      ///< NanoTrack head 引擎路径
        int detect_interval = 3;           ///< YOLO 检测间隔 (每N帧一次)
        int lost_threshold = 5;            ///< 连续无输出帧数 → LOST
        float min_confidence = 0.3f;       ///< tracker 最低置信度
        bool gpu_sync_mode = false;        ///< NX模式: 强制GPU同步防止YOLO/tracker并行(省电) vs 异步管线(服务器)
    } tracker;

    // 视差
    int max_disparity = 128;
    int window_size   = 5;
    int stereo_quality = 6;
    DisparityStrategy disparity_strategy = DisparityStrategy::FULL_FRAME;

    // 深度 (内嵌 HybridDepthConfig, 避免重复)
    HybridDepthConfig depth;

    // 性能
    int stats_interval = 100;      ///< 每 N 帧打印统计
    bool detection_only = false;   ///< 仅运行采集/校正/检测/回调, 跳过 stereo/depth
    bool drop_stale_roi_frames = false; ///< ROI 模式下若后一帧 YOLO 已完成则跳过旧帧后处理
    bool async_roi_stage2 = false; ///< ROI_ONLY 检测帧异步执行 IoU/ROI 特征/候选深度
    int async_roi_buffers = 3;     ///< 异步 ROI 图像快照缓冲数 (running + pending + free)
    float async_roi_deadline_ms = 10.0f; ///< 期望在下一帧 YOLO ready 前完成的软预算

    // VPI TNR (时域降噪)
    bool tnr_enabled = false;              ///< 是否启用 VPI TNR
    VPITNRPreset tnr_preset = VPI_TNR_PRESET_OUTDOOR_MEDIUM_LIGHT;
    float tnr_strength = 0.6f;             ///< 降噪强度 0.0~1.0
    VPITNRVersion tnr_version = VPI_TNR_DEFAULT;
};

/**
 * @brief 结果回调
 */
using ResultCallback = std::function<void(int frame_id, const std::vector<Object3D>& results)>;

/**
 * @brief 帧回调视图。
 *
 * 回调同步执行, 这些 VPIImage 和 vector 引用只在回调期间有效。
 */
struct FrameCallbackData {
    int frame_id;
    VPIImage rect_gray_left;
    VPIImage rect_gray_right;
    VPIImage rect_bgr_left;
    VPIImage rect_bgr_right;
    VPIImage raw_left;
    VPIImage raw_right;
    const std::vector<Detection>& detections_left;
    const std::vector<Detection>& detections_right;
    const std::vector<Object3D>& results;
    FrameMetadata metadata;
    float fps;
};

using FrameCallback = std::function<void(const FrameCallbackData& frame)>;

/**
 * @brief 诊断回调 (深度图 + 检测框 + 3D结果)
 */
using DiagnosticCallback = std::function<void(
    int frame_id, const float* depth_gpu, int depth_pitch,
    int depth_w, int depth_h,
    const std::vector<Detection>& detections,
    const std::vector<Object3D>& results)>;

/**
 * @brief 四级流水线主类
 */
class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    /**
     * @brief 初始化 Pipeline 所有组件
     * @return true 全部初始化成功
     */
    bool init(const PipelineConfig& config);

    /**
     * @brief 启动 Pipeline (在独立线程中运行)
     */
    void start();

    /**
     * @brief 停止 Pipeline 并等待线程退出
     */
    void stop();

    /**
     * @brief 设置结果回调
     */
    void setResultCallback(ResultCallback cb) { result_callback_ = std::move(cb); }

    /**
     * @brief 设置帧回调 (可视化: 图像+检测+3D)
     */
    void setFrameCallback(FrameCallback cb) { frame_callback_ = std::move(cb); }

    /**
     * @brief 设置诊断回调 (ROS2 录制用)
     */
    void setDiagnosticCallback(DiagnosticCallback cb) { diagnostic_callback_ = std::move(cb); }

    /**
     * @brief 获取当前帧率 (吞吐量)
     */
    float getCurrentFPS() const { return current_fps_.load(); }

    /**
     * @brief 打印性能报告
     */
    void printPerfReport() const;

    /**
     * @brief 抓取一对校正图并输出 ROI 特征匹配调试图
     */
    bool debugFeatureMatchesOnce(const std::string& output_dir);

private:
    // ===== Pipeline 主循环 =====
    void pipelineLoop();
    void pipelineLoopROI();   ///< ROI_ONLY 策略: 检测后多点匹配

    // ===== Stage 函数 =====
    void stage0_grab_and_rectify(FrameSlot& slot, bool grab_preloaded = false);
    void stage1_detect(FrameSlot& slot, int slot_index);
    void stage2_stereo(FrameSlot& slot);
    void stage3_fuse(FrameSlot& slot, int slot_index);

    // ROI 模式专用 Stage
    void stage2_roi_match_fuse(FrameSlot& slot, int slot_index);
    void stage2_roi_fuse_tracker(FrameSlot& slot, int slot_index); ///< tracker bbox → ROI match + depth

    // SOT Tracker 辅助
    void tracker_handle_detect_result(FrameSlot& slot);  ///< 检测帧: 用 YOLO bbox 刷新 tracker template
    void tracker_infill(FrameSlot& slot);                ///< 非检测帧: 运行 SOT 推理填充 bbox

    // Dual DLA 帧分配: 偶数帧→DLA0, 奇数帧→DLA1
    TRTDetector* getDetector(int frame_id) const;
    cudaStream_t getDLAStream(int frame_id) const;
    TRTDetector* getRightDetector() const;
    cudaStream_t getRightDLAStream(int frame_id) const;
    bool dualYoloEnabled() const;
    bool leftDetectorUsesBGR() const;
    bool rightDetectorUsesBGR() const;
    bool colorPipelineEnabled() const;
    void recordDetectDoneEvents(FrameSlot& slot) const;
    void waitDetectDone(cudaStream_t stream, const FrameSlot& slot) const;
    bool detectEventsReady(const FrameSlot& slot) const;
    void collectRightDetections(FrameSlot& slot, int slot_index);
    struct DualYoloMatchStats {
        int left_count = 0;
        int right_count = 0;
        int matched = 0;
        int left_missing = 0;
        int right_missing = 0;
        int fallback_attempted = 0;
        int fallback_matched = 0;
        int fallback_left_to_right = 0;
        int fallback_right_to_left = 0;
        int fallback_failed = 0;
        int fallback_prior_depth = 0;
        int class_mismatch = 0;
        int invalid_box = 0;
        int no_candidate = 0;
        int nonpositive_disparity = 0;
        int over_max_disparity = 0;
        int epipolar_reject = 0;
        int size_reject = 0;
        int low_iou = 0;
        int circle_fit_fail = 0;
        int subpixel_attempted = 0;
        int subpixel_refined = 0;
        int subpixel_rejected = 0;
        int subpixel_low_conf = 0;
        int subpixel_budget_skip = 0;
        int subpixel_support_sum = 0;
        int subpixel_support_max = 0;
        double subpixel_time_ms = 0.0;
        double subpixel_max_time_ms = 0.0;
        float subpixel_gate_min_px = 0.0f;
        float subpixel_gate_max_px = 0.0f;
        int depth_reject = 0;
        int image_lock_fail = 0;
    };
    struct DualYoloMatchOutput {
        std::vector<Detection> detections;
        std::vector<Object3D> results;
    };
    struct RoiStage2Input {
        int frame_id = -1;
        std::vector<Detection> left_detections;
        std::vector<Detection> right_detections;
        const uint8_t* left_cpu = nullptr;
        int left_cpu_pitch = 0;
        const uint8_t* right_cpu = nullptr;
        int right_cpu_pitch = 0;
        const uint8_t* left_gray_gpu = nullptr;
        int left_gray_pitch = 0;
        const uint8_t* right_gray_gpu = nullptr;
        int right_gray_pitch = 0;
        const uint8_t* left_bgr_gpu = nullptr;
        int left_bgr_pitch = 0;
        const uint8_t* right_bgr_gpu = nullptr;
        int right_bgr_pitch = 0;
        int width = 0;
        int height = 0;
        cudaStream_t stream = nullptr;
    };
    struct RoiStage2Output {
        std::vector<Detection> detections;
        std::vector<Object3D> roi_results;
        bool predict_only = false;
        bool detection_only = false;
    };
    DualYoloMatchOutput matchDualYoloDetections(
        const std::vector<Detection>& left_detections,
        const std::vector<Detection>& right_detections,
        const uint8_t* left_cpu, int left_pitch,
        const uint8_t* right_cpu, int right_pitch,
        const uint8_t* left_gpu, int left_gpu_pitch,
        const uint8_t* right_gpu, int right_gpu_pitch,
        const uint8_t* left_bgr_gpu, int left_bgr_pitch,
        const uint8_t* right_bgr_gpu, int right_bgr_pitch,
        int img_width, int img_height,
        cudaStream_t stream,
        DualYoloMatchStats* stats);
    void collectRoiDetections(FrameSlot& slot, int slot_index);
    bool roiStage2NeedsHostImages(const std::vector<Detection>& left_detections,
                                  const std::vector<Detection>& right_detections) const;
    RoiStage2Output runRoiStage2Core(const RoiStage2Input& input);
    void applyRoiStage2Output(FrameSlot& slot, RoiStage2Output&& output);
    void publishRoiFrameCallbacks(FrameSlot& slot);

    struct AsyncRoiBuffer {
        uint8_t* left_gray_gpu = nullptr;
        size_t left_gray_pitch = 0;
        uint8_t* right_gray_gpu = nullptr;
        size_t right_gray_pitch = 0;
        uint8_t* left_bgr_gpu = nullptr;
        size_t left_bgr_pitch = 0;
        uint8_t* right_bgr_gpu = nullptr;
        size_t right_bgr_pitch = 0;
        uint8_t* left_gray_host = nullptr;
        size_t left_gray_host_pitch = 0;
        uint8_t* right_gray_host = nullptr;
        size_t right_gray_host_pitch = 0;
        cudaEvent_t copy_done = nullptr;
        bool copy_event_recorded = false;
    };
    struct AsyncRoiTask {
        int frame_id = -1;
        int slot_index = -1;
        int buffer_index = -1;
        bool host_gray_valid = false;
        bool bgr_valid = false;
        bool copy_event_recorded = false;
        RoiStage2Input input;
    };
    struct AsyncRoiResult {
        int frame_id = -1;
        int slot_index = -1;
        double elapsed_ms = 0.0;
        RoiStage2Output output;
    };
    bool asyncRoiStage2Configured() const;
    bool initAsyncRoiStage2();
    bool startAsyncRoiStage2();
    void shutdownAsyncRoiStage2();
    void destroyAsyncRoiStage2();
    void asyncRoiWorkerLoop();
    bool submitAsyncRoiStage2(FrameSlot& slot, int slot_index);
    bool snapshotAsyncRoiImages(FrameSlot& slot,
                                AsyncRoiBuffer& buffer,
                                bool need_host_gray,
                                bool need_bgr);
    std::vector<int> drainCompletedAsyncRoiStage2();
    void expireAsyncRoiBefore(int frame_id);
    void releaseAsyncRoiBuffer(int buffer_index, const char* reason);
    void releaseAsyncRoiBufferLocked(int buffer_index);
    bool waitAsyncRoiBufferCopy(int buffer_index, const char* reason);
    void markAsyncRoiSlotCopyPendingLocked(int slot_index);
    void waitAsyncRoiSlotSnapshotDone(int slot_index, const char* reason);

    // ===== 组件 =====
    PipelineConfig config_;
    PipelineStreams streams_;
    FrameSlot slots_[RING_BUFFER_SIZE];

#ifdef HIK_CAMERA_ENABLED
    std::unique_ptr<HikvisionCamera> camera_;    ///< 双目相机 (单实例管理左右)
    std::unique_ptr<PWMTrigger> pwm_trigger_;     ///< GPIO PWM 触发器

    // ===== 异步采集线程 =====
    // 按需模式: pipeline 请求 → 采集线程写入 VPI host-mapped Image → 通知完成
    // 设计: grab 与 stage1/stage2 并行, 实现 pipeline/camera 解耦
    std::thread grab_thread_;
    void grabLoop();

    std::mutex grab_mutex_;
    std::condition_variable grab_request_cv_;  ///< pipeline→grab: 请求采集
    std::condition_variable grab_done_cv_;     ///< grab→pipeline: 采集完成
    int grab_request_slot_ = -1;               ///< 待采集 slot 索引 (-1=空闲)
    bool grab_done_ = false;                   ///< 采集完成标志
    bool grab_done_ok_ = false;                ///< 采集结果

    void requestGrab(int slot_idx);   ///< 发起异步采集请求 (非阻塞)
    bool waitGrab();                  ///< 等待异步采集完成 (阻塞至完成)
#endif
    std::unique_ptr<StereoCalibration> calibration_;
    std::unique_ptr<VPIRectifier> rectifier_;
    std::unique_ptr<TRTDetector> detector_;
    std::unique_ptr<TRTDetector> detector_right_;
    std::unique_ptr<VPIStereo> stereo_;            ///< 全帧/半分辨率视差 (FULL_FRAME/HALF_RES)
    std::unique_ptr<ROIStereoMatcher> roi_matcher_; ///< ROI 多点匹配 (ROI_ONLY)
    std::unique_ptr<DualYoloDepthGpuMatcher> dual_yolo_depth_gpu_; ///< 双 YOLO 多候选 GPU 批处理
    std::unique_ptr<NeuralFeatureMatcher> neural_feature_matcher_; ///< Learned ROI feature matching
    std::unique_ptr<Coordinate3D> fusion_;         ///< 全帧模式的 3D 融合
    std::unique_ptr<HybridDepthEstimator> hybrid_depth_; ///< 混合深度估计 (单目+双目+Kalman)

    // ===== ROI Stage2 异步后处理 =====
    bool async_roi_ready_ = false;
    bool async_roi_thread_stop_ = false;
    cudaStream_t async_roi_stream_ = nullptr;
    cudaStream_t async_roi_copy_stream_ = nullptr;
    std::thread async_roi_thread_;
    std::mutex async_roi_mutex_;
    std::condition_variable async_roi_cv_;
    std::deque<int> async_roi_free_buffers_;
    std::deque<AsyncRoiTask> async_roi_pending_;
    std::deque<AsyncRoiResult> async_roi_completed_;
    std::vector<AsyncRoiBuffer> async_roi_buffers_;
    std::array<cudaEvent_t, RING_BUFFER_SIZE> async_roi_slot_copy_done_{};
    std::array<bool, RING_BUFFER_SIZE> async_roi_slot_copy_pending_{};
    int async_roi_expire_before_frame_ = -1;
    bool async_roi_worker_busy_ = false;
    std::mutex roi_postprocess_mutex_;
    std::mutex hybrid_depth_mutex_;

    // ===== SOT Tracker =====
    std::unique_ptr<SOTTracker> tracker_;           ///< SOT 补帧跟踪器
    TrackerState tracker_state_ = TrackerState::IDLE;
    int tracker_lost_count_ = 0;                    ///< 连续丢失帧数
    int effective_detect_interval_ = 3;             ///< 运行时检测间隔 (LOST 时=1)

    // ===== VPI TNR 资源 =====
    VPIPayload tnrPayloadL_ = nullptr;     ///< 左目 TNR payload
    VPIPayload tnrPayloadR_ = nullptr;     ///< 右目 TNR payload
    VPIImage tnrNV12L_   = nullptr;        ///< 左目 NV12 输入缓冲
    VPIImage tnrNV12R_   = nullptr;        ///< 右目 NV12 输入缓冲
    VPIImage tnrOutNV12L_   = nullptr;     ///< 左目 TNR 输出
    VPIImage tnrOutNV12R_   = nullptr;     ///< 右目 TNR 输出
    bool tnrFirstFrame_ = true;            ///< 首帧标志 (prevOutput 传 NULL)

    // ===== Kalman dt 实测时间间隔 =====
    std::chrono::steady_clock::time_point last_fuse_time_{};

    // ===== 状态 =====
    std::atomic<bool> running_{false};
    std::atomic<float> current_fps_{0.0f};
    std::thread pipeline_thread_;              ///< Pipeline 工作线程

    ResultCallback result_callback_;
    FrameCallback frame_callback_;
    DiagnosticCallback diagnostic_callback_;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PIPELINE_H_
