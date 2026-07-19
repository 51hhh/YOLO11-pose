/**
 * @file frame_slot.h
 * @brief FrameSlot 三缓冲定义 (Ring Buffer)
 *
 * Pipeline 各 Stage 通过 FrameSlot 交换数据。
 * 使用三缓冲 (Triple Buffering) 解除 Producer/Consumer 之间的锁竞争。
 */

#ifndef STEREO_3D_PIPELINE_FRAME_SLOT_H_
#define STEREO_3D_PIPELINE_FRAME_SLOT_H_

#include "detection_types.h"
#include "object3d_types.h"

#include <cuda_runtime.h>
#include <vpi/Image.h>
#include <cstdint>
#include <vector>

namespace stereo3d {

/**
 * @brief SOT 跟踪结果
 */
struct SOTResult {
    float cx, cy, width, height;
    float confidence;
    bool valid;
    SOTResult() : cx(0), cy(0), width(0), height(0), confidence(0), valid(false) {}
};

/**
 * @brief bbox 来源
 */
enum class BboxSource {
    NONE,       ///< 无检测
    YOLO,       ///< YOLO 检测
    TRACKER     ///< SOT 补帧
};

/**
 * @brief Tracker 状态
 */
enum class TrackerState {
    IDLE,       ///< 无目标
    TRACKING,   ///< 正常跟踪
    LOST        ///< 目标丢失，等待 YOLO 重检测
};

/**
 * @brief Per-frame stereo sync metadata exposed to callbacks and recorders.
 */
struct FrameMetadata {
    uint64_t host_capture_timestamp_ns = 0;
    uint64_t left_timestamp_us = 0;
    uint64_t right_timestamp_us = 0;
    uint32_t left_frame_number = 0;
    uint32_t right_frame_number = 0;
    uint32_t left_frame_counter = 0;
    uint32_t right_frame_counter = 0;
    uint32_t left_trigger_index = 0;
    uint32_t right_trigger_index = 0;
    int64_t frame_counter_delta = 0;
    int64_t frame_number_delta = 0;
    int64_t timestamp_delta_us = 0;
    int64_t stereo_timestamp_residual_ns = 0;
    bool grab_failed = false;
    bool is_detect_frame = true;
    bool p2_depth_modes_enabled = false;
    uint32_t p2_depth_mode_mask = 0;
    bool p2_feature_job_scaffold_enabled = false;
    bool p2_realtime_requested = false;
    bool p2_diagnostic_requested = false;
    uint32_t p2_realtime_triggers = 0;
    uint32_t p2_diagnostic_triggers = 0;
    uint32_t p2_realtime_skip_reasons = 0;
    uint32_t p2_diagnostic_skip_reasons = 0;
    int p2_feature_job_count = 0;
    int p2_left_count = 0;
    int p2_right_count = 0;
    int p2_valid_direct_pair_count = 0;
};

/**
 * @brief 一帧数据的完整生命周期容器
 *
 * Pipeline 每一帧在此结构中流转:
 *   Stage 0 写入 rawL/rawR -> rectL/rectR
 *   Stage 1 写入 detections
 *   全帧模式 Stage 2 写入 disparityMap
 *   ROI/双路 YOLO Stage 2 直接写入 results
 */
struct FrameSlot {
    // =========== 帧标识 ===========
    int frame_id = -1;                    ///< 帧序号
    bool grab_failed = false;             ///< 抓取失败标记 (帧同步跳变等)
    uint64_t left_timestamp_us = 0;
    uint64_t host_capture_timestamp_ns = 0;
    uint64_t right_timestamp_us = 0;
    uint32_t left_frame_number = 0;
    uint32_t right_frame_number = 0;
    uint32_t left_frame_counter = 0;
    uint32_t right_frame_counter = 0;
    uint32_t left_trigger_index = 0;
    uint32_t right_trigger_index = 0;
    int64_t stereo_timestamp_residual_ns = 0;

    // =========== Stage 0: 原始图像 ===========
    VPIImage rawL      = nullptr;         ///< 左原始图 (Pinned + Mapped)
    VPIImage rawR      = nullptr;         ///< 右原始图

    // =========== Color Pipeline (VPI) ===========
    VPIImage tempBGR_L      = nullptr;   ///< 左 debayer 输出 BGR (raw res)
    VPIImage tempBGR_R      = nullptr;   ///< 右 debayer 输出 BGR (raw res)
    VPIImage rectBGR_vpiL   = nullptr;   ///< 左校正 BGR (rect res, 检测用)
    VPIImage rectBGR_vpiR   = nullptr;   ///< 右校正 BGR (rect res)
    VPIImage rectGray_vpiL  = nullptr;   ///< 左校正灰度 (rect res, 立体匹配用)
    VPIImage rectGray_vpiR  = nullptr;   ///< 右校正灰度 (rect res)

    // =========== Stage 1: 检测结果 ===========
    std::vector<Detection> detections;    ///< 左目 YOLO 检测结果列表
    std::vector<Detection> detections_right; ///< 右目 YOLO 检测结果列表 (双路测试)
    bool detection_submitted = false;      ///< 本帧是否真正提交过左目 YOLO
    bool right_detection_submitted = false; ///< 本帧是否提交过右目 YOLO

    // =========== SOT Tracker 补帧 ===========
    SOTResult sot_bbox_result;            ///< SOT tracker 输出
    BboxSource bbox_source = BboxSource::NONE; ///< 最终 bbox 来源
    bool is_detect_frame = true;          ///< 是否为 YOLO 检测帧
    bool p2_depth_modes_enabled = false;
    uint32_t p2_depth_mode_mask = 0;
    bool p2_feature_job_scaffold_enabled = false;
    bool p2_realtime_requested = false;
    bool p2_diagnostic_requested = false;
    uint32_t p2_realtime_triggers = 0;
    uint32_t p2_diagnostic_triggers = 0;
    uint32_t p2_realtime_skip_reasons = 0;
    uint32_t p2_diagnostic_skip_reasons = 0;
    int p2_feature_job_count = 0;
    int p2_left_count = 0;
    int p2_right_count = 0;
    int p2_valid_direct_pair_count = 0;

    // =========== Stage 2: 视差图 ===========
    VPIImage disparityMap  = nullptr;     ///< 视差图 (S16 格式)
    VPIImage confidenceMap = nullptr;     ///< 视差置信度图

    // =========== Stage 3: 3D 定位结果 ===========
    std::vector<Object3D> results;        ///< 最终 3D 定位输出

    // =========== CUDA Event 同步 ===========
    cudaEvent_t evtRectDone   = nullptr;  ///< Stage 0 校正完成
    cudaEvent_t evtDetectDone = nullptr;  ///< Stage 1 左目检测完成
    cudaEvent_t evtDetectRightDone = nullptr; ///< Stage 1 右目检测完成
    cudaEvent_t evtStereoDone = nullptr;  ///< Stage 2 视差完成

    // =========== Cached CUDA Pointers (Tegra 统一内存优化) ===========
    // init() 时缓存, 避免每帧 VPI lock/unlock (~0.3ms/次, 8 次 = 2.4ms)
    // Tegra 统一内存: CPU/GPU 共享物理地址, CUDA 指针在 VPI Image 生命周期内固定
    struct CachedGPU {
        void* data = nullptr;
        int pitchBytes = 0;
    };
    CachedGPU rawL_gpu, rawR_gpu;             ///< 原始 Bayer 的 CUDA 指针
    CachedGPU tempBGR_L_gpu, tempBGR_R_gpu;   ///< Debayer 输出 BGR 的 CUDA 指针
    CachedGPU rectBGR_L_gpu, rectBGR_R_gpu;   ///< 校正 BGR 图 CUDA 指针
    CachedGPU rectGray_L_gpu;                 ///< 校正灰度左图 CUDA 指针
    CachedGPU rectGray_R_gpu;                 ///< 校正灰度右图 CUDA 指针

    // =========== 生命周期 ===========

    /**
     * @brief 创建 CUDA Events
     */
    bool createEvents() {
        cudaError_t err;
        err = cudaEventCreateWithFlags(&evtRectDone, cudaEventDisableTiming);
        if (err != cudaSuccess) return false;
        err = cudaEventCreateWithFlags(&evtDetectDone, cudaEventDisableTiming);
        if (err != cudaSuccess) { cudaEventDestroy(evtRectDone); evtRectDone = nullptr; return false; }
        err = cudaEventCreateWithFlags(&evtDetectRightDone, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            cudaEventDestroy(evtRectDone);
            evtRectDone = nullptr;
            cudaEventDestroy(evtDetectDone);
            evtDetectDone = nullptr;
            return false;
        }
        err = cudaEventCreateWithFlags(&evtStereoDone, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            cudaEventDestroy(evtRectDone);
            evtRectDone = nullptr;
            cudaEventDestroy(evtDetectDone);
            evtDetectDone = nullptr;
            cudaEventDestroy(evtDetectRightDone);
            evtDetectRightDone = nullptr;
            return false;
        }
        return true;
    }

    /**
     * @brief 销毁所有 VPI 和 CUDA 资源
     */
    void destroy() {
        auto destroyVPI = [](VPIImage& img) {
            if (img) { vpiImageDestroy(img); img = nullptr; }
        };
        destroyVPI(rawL);
        destroyVPI(rawR);
        destroyVPI(tempBGR_L);
        destroyVPI(tempBGR_R);
        destroyVPI(rectBGR_vpiL);
        destroyVPI(rectBGR_vpiR);
        destroyVPI(rectGray_vpiL);
        destroyVPI(rectGray_vpiR);
        destroyVPI(disparityMap);
        destroyVPI(confidenceMap);

        auto destroyEvent = [](cudaEvent_t& evt) {
            if (evt) { cudaEventDestroy(evt); evt = nullptr; }
        };
        destroyEvent(evtRectDone);
        destroyEvent(evtDetectDone);
        destroyEvent(evtDetectRightDone);
        destroyEvent(evtStereoDone);

        detections.clear();
        detections_right.clear();
        detection_submitted = false;
        right_detection_submitted = false;
        results.clear();
        sot_bbox_result = SOTResult();
        bbox_source = BboxSource::NONE;
        is_detect_frame = true;
        frame_id = -1;
        left_timestamp_us = right_timestamp_us = 0;
        host_capture_timestamp_ns = 0;
        left_frame_number = right_frame_number = 0;
        left_frame_counter = right_frame_counter = 0;
        left_trigger_index = right_trigger_index = 0;
        stereo_timestamp_residual_ns = 0;
        p2_depth_modes_enabled = false;
        p2_depth_mode_mask = 0;
        p2_feature_job_scaffold_enabled = false;
        p2_realtime_requested = false;
        p2_diagnostic_requested = false;
        p2_realtime_triggers = 0;
        p2_diagnostic_triggers = 0;
        p2_realtime_skip_reasons = 0;
        p2_diagnostic_skip_reasons = 0;
        p2_feature_job_count = 0;
        p2_left_count = 0;
        p2_right_count = 0;
        p2_valid_direct_pair_count = 0;
    }

    /**
     * @brief 重置帧数据 (不销毁资源，仅清理数据)
     */
    void reset() {
        detections.clear();
        detections_right.clear();
        detection_submitted = false;
        right_detection_submitted = false;
        results.clear();
        sot_bbox_result = SOTResult();
        bbox_source = BboxSource::NONE;
        is_detect_frame = true;
        frame_id = -1;
        grab_failed = false;
        left_timestamp_us = right_timestamp_us = 0;
        left_frame_number = right_frame_number = 0;
        left_frame_counter = right_frame_counter = 0;
        left_trigger_index = right_trigger_index = 0;
        stereo_timestamp_residual_ns = 0;
        p2_depth_modes_enabled = false;
        p2_depth_mode_mask = 0;
        p2_feature_job_scaffold_enabled = false;
        p2_realtime_requested = false;
        p2_diagnostic_requested = false;
        p2_realtime_triggers = 0;
        p2_diagnostic_triggers = 0;
        p2_realtime_skip_reasons = 0;
        p2_diagnostic_skip_reasons = 0;
        p2_feature_job_count = 0;
        p2_left_count = 0;
        p2_right_count = 0;
        p2_valid_direct_pair_count = 0;
    }
};

inline uint64_t normalizeUnixHostTimestampNs(int64_t timestamp) {
    if (timestamp <= 0) return 0;
    // Hikvision host timestamps are platform/SDK dependent. Accept values that
    // look like Unix epoch nanoseconds or microseconds; reject monotonic-like
    // counters so ROS consumers do not see stale absolute times.
    constexpr int64_t kEpochNsMin = 100000000000000000LL;  // ~1973
    constexpr int64_t kEpochUsMin = 100000000000000LL;     // ~1973
    if (timestamp >= kEpochNsMin) {
        return static_cast<uint64_t>(timestamp);
    }
    if (timestamp >= kEpochUsMin) {
        return static_cast<uint64_t>(timestamp) * 1000ULL;
    }
    return 0;
}

inline uint64_t chooseCaptureTimestampNs(
    int64_t left_host_timestamp,
    int64_t right_host_timestamp,
    uint64_t fallback_ns) {
    const uint64_t left_ns = normalizeUnixHostTimestampNs(left_host_timestamp);
    const uint64_t right_ns = normalizeUnixHostTimestampNs(right_host_timestamp);
    if (left_ns > 0 && right_ns > 0) {
        return left_ns / 2ULL + right_ns / 2ULL +
               (left_ns % 2ULL + right_ns % 2ULL) / 2ULL;
    }
    if (left_ns > 0) return left_ns;
    if (right_ns > 0) return right_ns;
    return fallback_ns;
}

inline FrameMetadata makeFrameMetadata(const FrameSlot& slot) {
    FrameMetadata meta;
    meta.host_capture_timestamp_ns = slot.host_capture_timestamp_ns;
    meta.left_timestamp_us = slot.left_timestamp_us;
    meta.right_timestamp_us = slot.right_timestamp_us;
    meta.left_frame_number = slot.left_frame_number;
    meta.right_frame_number = slot.right_frame_number;
    meta.left_frame_counter = slot.left_frame_counter;
    meta.right_frame_counter = slot.right_frame_counter;
    meta.left_trigger_index = slot.left_trigger_index;
    meta.right_trigger_index = slot.right_trigger_index;
    meta.frame_counter_delta =
        static_cast<int64_t>(slot.left_frame_counter) -
        static_cast<int64_t>(slot.right_frame_counter);
    meta.frame_number_delta =
        static_cast<int64_t>(slot.left_frame_number) -
        static_cast<int64_t>(slot.right_frame_number);
    meta.timestamp_delta_us =
        (static_cast<int64_t>(slot.left_timestamp_us) -
         static_cast<int64_t>(slot.right_timestamp_us)) / 1000;
    meta.stereo_timestamp_residual_ns = slot.stereo_timestamp_residual_ns;
    meta.grab_failed = slot.grab_failed;
    meta.is_detect_frame = slot.is_detect_frame;
    meta.p2_depth_modes_enabled = slot.p2_depth_modes_enabled;
    meta.p2_depth_mode_mask = slot.p2_depth_mode_mask;
    meta.p2_feature_job_scaffold_enabled = slot.p2_feature_job_scaffold_enabled;
    meta.p2_realtime_requested = slot.p2_realtime_requested;
    meta.p2_diagnostic_requested = slot.p2_diagnostic_requested;
    meta.p2_realtime_triggers = slot.p2_realtime_triggers;
    meta.p2_diagnostic_triggers = slot.p2_diagnostic_triggers;
    meta.p2_realtime_skip_reasons = slot.p2_realtime_skip_reasons;
    meta.p2_diagnostic_skip_reasons = slot.p2_diagnostic_skip_reasons;
    meta.p2_feature_job_count = slot.p2_feature_job_count;
    meta.p2_left_count = slot.p2_left_count;
    meta.p2_right_count = slot.p2_right_count;
    meta.p2_valid_direct_pair_count = slot.p2_valid_direct_pair_count;
    return meta;
}

/**
 * @brief 三缓冲 Ring Buffer
 *
 * slots[0..2] 轮流使用，各 Stage 的索引独立递增。
 */
static constexpr int RING_BUFFER_SIZE = 3;

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_FRAME_SLOT_H_
