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
 * @brief 3D 定位结果
 */
struct Object3D {
    float x, y, z;         ///< 3D 坐标 (米)
    float confidence;      ///< 定位置信度
    int class_id;          ///< 类别 ID

    Object3D() : x(0), y(0), z(0), confidence(0), class_id(0) {}
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

    // =========== Stage 0: 原始图像 + 校正后图像 ===========
    VPIImage rawL      = nullptr;         ///< 左原始图 (Pinned + Mapped)
    VPIImage rawR      = nullptr;         ///< 右原始图
    VPIImage rectL     = nullptr;         ///< 校正后左图
    VPIImage rectR     = nullptr;         ///< 校正后右图

    // =========== Stage 1: 检测结果 ===========
    std::vector<Detection> detections;    ///< YOLO 检测结果列表

    // =========== Stage 2: 视差图 ===========
    VPIImage disparityMap  = nullptr;     ///< 视差图 (S16 格式)
    VPIImage confidenceMap = nullptr;     ///< 视差置信度图

    // =========== Stage 3: 3D 定位结果 ===========
    std::vector<Object3D> results;        ///< 最终 3D 定位输出

    // =========== CUDA Event 同步 ===========
    cudaEvent_t evtRectDone   = nullptr;  ///< Stage 0 校正完成
    cudaEvent_t evtDetectDone = nullptr;  ///< Stage 1 检测完成
    cudaEvent_t evtStereoDone = nullptr;  ///< Stage 2 视差完成

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
        destroyVPI(rectL);
        destroyVPI(rectR);
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
        frame_id = -1;
    }

    /**
     * @brief 重置帧数据 (不销毁资源，仅清理数据)
     */
    void reset() {
        detections.clear();
        results.clear();
        frame_id = -1;
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
