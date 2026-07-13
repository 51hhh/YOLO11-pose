/**
 * @file hikvision_camera.h
 * @brief 海康工业相机采集封装
 *
 * 针对 Pipeline 架构设计:
 *   - grabFramePair(): 同步抓取左右图像
 *   - 从 SDK buffer 拷贝到调用方提供的 VPI host-mapped buffer
 *   - 支持外触发 (Line0/Line1) + 软触发
 *   - BayerRG8 原始格式输出 (后续由 CUDA/VPI 处理)
 */

#pragma once

#include <string>
#include <cstdint>
#include <memory>
#include <atomic>
#include <mutex>
#include <functional>

// Forward declare Hikvision SDK handle
// 实际编译时需要 #include "MvCameraControl.h"

namespace stereo3d {

struct CameraConfig {
    int camera_index_left = 0;
    int camera_index_right = 1;
    std::string serial_left;               // 可选: 用序列号识别
    std::string serial_right;

    float exposure_us = 9867.0f;           // 曝光时间 (微秒)
    float gain_db = 11.9906f;              // 增益 (dB)
    bool auto_exposure = false;            // 自动曝光 (Continuous)
    bool auto_gain = false;                // 自动增益 (Continuous)
    float ae_upper_us = 5000.0f;           // 自动曝光上限 (μs)
    float ae_lower_us = 100.0f;            // 自动曝光下限 (μs)
    float ag_upper_db = 10.0f;             // 自动增益上限 (dB)
    bool gamma_enable = false;             // Gamma 校正
    float gamma_value = 0.7f;             // Gamma 值 (<1 压缩高亮)

    bool use_trigger = true;               // 外触发模式
    std::string trigger_source = "Line0";  // 触发源
    std::string trigger_activation = "RisingEdge";
    int trigger_frequency_hz = 100;         // 外触发频率, 用于帧计数追帧超时
    int image_node_num = 3;                 // SDK 取流 FIFO 深度, 吸收 USB 到达抖动
    int embedded_info_clear_rows = 2;       // FrameSpecInfo 写入图像首部后清除的行数

    int width = 1440;                      // 图像宽度
    int height = 1080;                     // 图像高度
};

/**
 * @struct GrabResult
 * @brief 单帧抓取结果
 */
struct GrabResult {
    bool success = false;
    uint64_t timestamp_us = 0;    // SDK 设备时间戳原值; 当前海康 USB 实测为 ns
    int64_t host_timestamp = 0;   // 主机生成时间戳, 仅用于诊断传输/调度 jitter
    int64_t stereo_timestamp_residual_ns = 0; // 去除左右设备固定偏移后的配对残差
    uint32_t frame_number = 0;    // SDK 帧号
    uint32_t frame_counter = 0;   // 水印帧计数, 主同步键
    uint32_t trigger_index = 0;   // 水印外触发计数, 仅用于诊断
};

/**
 * @class HikvisionCamera
 * @brief 双目海康相机同步采集
 *
 * 生命周期: create → open → start → [grabFramePair ...] → stop → close
 */
class HikvisionCamera {
public:
    HikvisionCamera();
    ~HikvisionCamera();

    // 禁止拷贝
    HikvisionCamera(const HikvisionCamera&) = delete;
    HikvisionCamera& operator=(const HikvisionCamera&) = delete;

    /**
     * @brief 初始化相机
     * @param cfg 相机配置
     * @return true 成功
     */
    bool open(const CameraConfig& cfg);

    /**
     * @brief 关闭相机释放资源
     */
    void close();

    /**
     * @brief 开始采集
     */
    bool startGrabbing();

    /**
     * @brief 停止采集
     */
    void stopGrabbing();

    /**
     * @brief 同步抓取一对图像到指定缓冲
     *
     * 输出格式: BayerRG8 (H × W, uint8_t)
     * 缓冲区大小: width * height bytes
     *
     * @param dst_left  左图写入地址 (调用方分配, 可以是 VPI host-mapped 内存)
     * @param dst_right 右图写入地址
     * @param left_pitch  左图行跨度 (bytes), 0 = width
     * @param right_pitch 右图行跨度 (bytes), 0 = width
     * @param timeout_ms  超时时间
     * @param[out] result_left  左图结果
     * @param[out] result_right 右图结果
     * @return true 双目均成功
     */
    bool grabFramePair(
        uint8_t* dst_left,  uint8_t* dst_right,
        int left_pitch, int right_pitch,
        unsigned int timeout_ms,
        GrabResult& result_left, GrabResult& result_right);

    /**
     * @brief 单相机抓取 (调试用)
     */
    bool grabSingle(bool is_left,
                    uint8_t* dst, int pitch,
                    unsigned int timeout_ms,
                    GrabResult& result);

    /**
     * @brief 获取实际图像尺寸
     */
    int width()  const { return width_; }
    int height() const { return height_; }

    bool isOpened() const { return opened_; }

private:
    bool openCamera(void*& handle, int index, const std::string& serial);
    bool configureCamera(void* handle, const CameraConfig& cfg, const char* tag);
    bool grabOneFrame(void* handle, uint8_t* dst, int pitch,
                      unsigned int timeout_ms, GrabResult& result);
    void resetSyncState();

    void* handle_left_  = nullptr;
    void* handle_right_ = nullptr;

    int width_  = 0;
    int height_ = 0;
    bool opened_   = false;
    bool grabbing_ = false;

    CameraConfig config_;

    // --- 自动重连 ---
    static constexpr int MAX_CONSECUTIVE_FAILURES = 10;
    static constexpr int MAX_RECONNECT_RETRIES = 3;
    int consecutive_failures_ = 0;
    bool reconnect();

    // --- 双目配对状态 ---
    bool sync_initialized_ = false;
    int sync_baseline_samples_ = 0;
    int64_t candidate_timestamp_offset_ns_ = 0;
    int64_t expected_timestamp_offset_ns_ = 0;
    int64_t expected_frame_counter_delta_ = 0;
    int64_t expected_frame_number_delta_ = 0;
    int64_t expected_trigger_delta_ = 0;
    int consecutive_sync_mismatches_ = 0;
};

}  // namespace stereo3d
