/**
 * @file frame_slot.h
 * @brief FrameSlot 三缓冲定义 (Ring Buffer)
 *
 * Pipeline 各 Stage 通过 FrameSlot 交换数据。
 * 使用三缓冲 (Triple Buffering) 解除 Producer/Consumer 之间的锁竞争。
 */

#ifndef STEREO_3D_PIPELINE_FRAME_SLOT_H_
#define STEREO_3D_PIPELINE_FRAME_SLOT_H_

#include <cuda_runtime.h>
#include <vpi/Image.h>
#include <vector>
#include <cstring>

namespace stereo3d {

/**
 * @brief 单个检测结果
 */
struct Detection {
    float cx, cy;          ///< 检测框中心 (像素坐标)
    float width, height;   ///< 检测框尺寸
    float confidence;      ///< 置信度
    int class_id;          ///< 类别 ID

    Detection() : cx(0), cy(0), width(0), height(0), confidence(0), class_id(0) {}
};

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
 * @brief 3D 定位结果
 */
struct Object3D {
    float x, y, z;         ///< 3D 坐标 (米)
    float vx, vy, vz;      ///< 3D 速度 (m/s)
    float ax, ay, az;       ///< 3D 加速度 (m/s²)
    float z_mono;          ///< 单目测距 (m), 校准用
    float z_stereo;        ///< 双目测距 (m), -1=无效
    float confidence;      ///< 定位置信度
    int class_id;          ///< 类别 ID
    int track_id;          ///< 跟踪 ID (-1 = 未跟踪)
    int depth_method;      ///< 0=单目, 1=双目, 2=融合

    Object3D() : x(0), y(0), z(0), vx(0), vy(0), vz(0),
                 ax(0), ay(0), az(0), z_mono(0), z_stereo(-1),
                 confidence(0), class_id(0), track_id(-1), depth_method(0) {}
};

/**
 * @brief 一帧数据的完整生命周期容器
 *
 * Pipeline 每一帧在此结构中流转:
 *   Stage 0 写入 rawL/rawR → rectL/rectR
 *   Stage 1 写入 detections
 *   Stage 2 写入 disparityMap
 *   Stage 3 读取 detections + disparityMap → 写入 results
 */
struct FrameSlot {
    // =========== 帧标识 ===========
    int frame_id = -1;                    ///< 帧序号
    bool grab_failed = false;             ///< 抓取失败标记 (帧同步跳变等)

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
    std::vector<Detection> detections;    ///< YOLO 检测结果列表

    // =========== SOT Tracker 补帧 ===========
    SOTResult sot_bbox_result;            ///< SOT tracker 输出
    BboxSource bbox_source = BboxSource::NONE; ///< 最终 bbox 来源
    bool is_detect_frame = true;          ///< 是否为 YOLO 检测帧

    // =========== Stage 2: 视差图 ===========
    VPIImage disparityMap  = nullptr;     ///< 视差图 (S16 格式)
    VPIImage confidenceMap = nullptr;     ///< 视差置信度图

    // =========== Stage 3: 3D 定位结果 ===========
    std::vector<Object3D> results;        ///< 最终 3D 定位输出

    // =========== CUDA Event 同步 ===========
    cudaEvent_t evtRectDone   = nullptr;  ///< Stage 0 校正完成
    cudaEvent_t evtDetectDone = nullptr;  ///< Stage 1 检测完成
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
    CachedGPU rectGray_L_gpu;                  ///< 校正灰度左图 CUDA 指针 (SOT tracker 用)

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
        err = cudaEventCreateWithFlags(&evtStereoDone, cudaEventDisableTiming);
        if (err != cudaSuccess) { cudaEventDestroy(evtRectDone); evtRectDone = nullptr; cudaEventDestroy(evtDetectDone); evtDetectDone = nullptr; return false; }
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
        destroyEvent(evtStereoDone);

        detections.clear();
        results.clear();
        sot_bbox_result = SOTResult();
        bbox_source = BboxSource::NONE;
        is_detect_frame = true;
        frame_id = -1;
    }

    /**
     * @brief 重置帧数据 (不销毁资源，仅清理数据)
     */
    void reset() {
        detections.clear();
        results.clear();
        sot_bbox_result = SOTResult();
        bbox_source = BboxSource::NONE;
        is_detect_frame = true;
        frame_id = -1;
        grab_failed = false;
    }
};

/**
 * @brief 三缓冲 Ring Buffer
 *
 * slots[0..2] 轮流使用，各 Stage 的索引独立递增。
 */
static constexpr int RING_BUFFER_SIZE = 3;

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_FRAME_SLOT_H_
