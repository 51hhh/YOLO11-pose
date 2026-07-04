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
#include "pipeline_callbacks.h"
#include "pipeline_config.h"
#include "pipeline_feature_jobs.h"
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
#include "../stereo/roi_feature_result.h"
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
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace stereo3d {

class NeuralFeatureMatcher;

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
        int iou_color_support_max = 0;
        int iou_color_attempted_max = 0;
        int iou_edge_support_max = 0;
        int iou_edge_attempted_max = 0;
    };
    struct P2FeatureDiagnosticResultRow {
        int frame_id = -1;
        FrameMetadata metadata;
        std::string lane = "diagnostic";
        std::string mode;
        std::string status;
        bool valid = false;
        bool low_confidence = false;
        float disparity = std::numeric_limits<float>::quiet_NaN();
        float z_m = std::numeric_limits<float>::quiet_NaN();
        float confidence = std::numeric_limits<float>::quiet_NaN();
        float stddev = std::numeric_limits<float>::quiet_NaN();
        int support = 0;
        int attempted = 0;
        float initial_disparity = std::numeric_limits<float>::quiet_NaN();
        Detection left_det;
        Detection right_det;
        float anchor_cx = std::numeric_limits<float>::quiet_NaN();
        float anchor_cy = std::numeric_limits<float>::quiet_NaN();
        float right_anchor_cx = std::numeric_limits<float>::quiet_NaN();
        float right_anchor_cy = std::numeric_limits<float>::quiet_NaN();
        int debug_match_count = 0;
        std::array<SparseFeatureDebugMatch, kMaxSparseFeatureDebugMatches> debug_matches{};
        SparseFeatureDebugPatch debug_patch;
        std::string artifact_path;
        double algo_ms = 0.0;
        double queue_wait_ms = 0.0;
        double worker_elapsed_ms = 0.0;
        float deadline_ms = 0.0f;
        bool over_deadline = false;
        uint32_t depth_mode_mask = 0u;
        uint32_t triggers = 0u;
    };
    struct DualYoloMatchOutput {
        std::vector<Detection> detections;
        std::vector<Object3D> results;
        std::vector<P2FeatureDiagnosticResultRow> p2_artifact_rows;
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
        bool p2_inline_feature_jobs_enabled = true;
    };
    struct RoiStage2Output {
        std::vector<Detection> detections;
        std::vector<Object3D> roi_results;
        std::vector<P2FeatureDiagnosticResultRow> p2_artifact_rows;
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
        bool p2_inline_feature_jobs_enabled,
        int frame_id,
        DualYoloMatchStats* stats);
    void collectRoiDetections(FrameSlot& slot, int slot_index);
    bool roiStage2NeedsHostImages(const std::vector<Detection>& left_detections,
                                  const std::vector<Detection>& right_detections) const;
    bool roiStage2FallbackMayNeedHostImages(
        const std::vector<Detection>& left_detections,
        const std::vector<Detection>& right_detections) const;
    RoiStage2Output runRoiStage2Core(const RoiStage2Input& input);
    void applyRoiStage2Output(FrameSlot& slot, RoiStage2Output&& output);
    void publishRoiResultCallback(FrameSlot& slot);
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
        cudaEvent_t p2_diag_copy_done = nullptr;
        bool p2_diag_copy_event_recorded = false;
    };
    struct AsyncRoiTask {
        int frame_id = -1;
        int slot_index = -1;
        int buffer_index = -1;
        bool host_gray_valid = false;
        bool bgr_valid = false;
        bool copy_event_recorded = false;
        FrameMetadata metadata;
        RoiStage2Input input;
        P2FeatureJobDecision p2_feature_decision;
        std::vector<P2FeatureJobDescriptor> p2_feature_jobs;
    };
    struct AsyncRoiResult {
        int frame_id = -1;
        int slot_index = -1;
        double elapsed_ms = 0.0;
        FrameMetadata metadata;
        std::vector<Detection> right_detections;
        RoiStage2Output output;
    };
    struct P2FeatureDiagnosticTask {
        P2FeatureJobDescriptor job;
        FrameMetadata metadata;
        int buffer_index = -1;
        std::vector<Detection> left_detections;
        std::vector<Detection> right_detections;
        int width = 0;
        int height = 0;
        bool copy_event_recorded = false;
        std::chrono::steady_clock::time_point enqueue_time{};
    };
    struct P2FeatureDiagnosticBuffer {
        uint8_t* left_gray_gpu = nullptr;
        size_t left_gray_pitch = 0;
        uint8_t* right_gray_gpu = nullptr;
        size_t right_gray_pitch = 0;
        cudaEvent_t copy_done = nullptr;
        bool copy_event_recorded = false;
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
    bool p2FeatureDiagnosticLaneConfigured() const;
    bool startP2FeatureDiagnosticLane();
    void shutdownP2FeatureDiagnosticLane();
    bool initP2FeatureDiagnosticBuffers();
    void destroyP2FeatureDiagnosticBuffers();
    void releaseP2FeatureDiagnosticBuffer(int buffer_index);
    void p2FeatureDiagnosticWorkerLoop();
    void enqueueP2FeatureDiagnosticJobs(
        const FrameMetadata& metadata,
        const std::vector<P2FeatureJobDescriptor>& jobs);
    void enqueueP2FeatureDiagnosticJobs(
        const FrameMetadata& metadata,
        const std::vector<P2FeatureJobDescriptor>& jobs,
        const RoiStage2Input& input,
        cudaEvent_t source_copy_done,
        bool source_copy_event_recorded,
        AsyncRoiBuffer& source_buffer);
    bool openP2FeatureDiagnosticResults();
    void closeP2FeatureDiagnosticResults();
    void writeP2FeatureDiagnosticResults(
        const std::vector<P2FeatureDiagnosticResultRow>& rows);
    void writeP2FeatureDiagnosticArtifacts(
        std::vector<P2FeatureDiagnosticResultRow>& rows,
        const P2FeatureDiagnosticBuffer& buffer,
        int width,
        int height);

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

    // ===== P2 diagnostic lane =====
    bool p2_feature_diag_thread_stop_ = false;
    bool p2_feature_diag_worker_busy_ = false;
    cudaStream_t p2_feature_diag_stream_ = nullptr;
    cudaStream_t p2_feature_diag_copy_stream_ = nullptr;
    std::thread p2_feature_diag_thread_;
    std::mutex p2_feature_diag_mutex_;
    std::condition_variable p2_feature_diag_cv_;
    std::deque<P2FeatureDiagnosticTask> p2_feature_diag_pending_;
    std::deque<int> p2_feature_diag_free_buffers_;
    std::vector<P2FeatureDiagnosticBuffer> p2_feature_diag_buffers_;
    std::mutex p2_feature_diag_results_mutex_;
    std::ofstream p2_feature_diag_results_file_;
    int p2_feature_diag_artifacts_saved_ = 0;

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
