/**
 * @file hikvision_camera.h
 * @brief 海康工业相机零拷贝采集封装
 *
 * 针对 Pipeline 架构设计:
 *   - grabFramePair(): 同步抓取左右图像
 *   - 直接写入调用方提供的 buffer (零拷贝路径)
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

    float exposure_us = 3000.0f;           // 曝光时间 (微秒)
    float gain_db = 0.0f;                  // 增益 (dB)

    bool use_trigger = true;               // 外触发模式
    std::string trigger_source = "Line0";  // 触发源
    std::string trigger_activation = "RisingEdge";

    int width = 1440;                      // 图像宽度
    int height = 1080;                     // 图像高度
};

/**
 * @struct GrabResult
 * @brief 单帧抓取结果
 */
struct GrabResult {
    bool success = false;
    uint64_t timestamp_us = 0;    // 设备时间戳
    uint32_t frame_number = 0;    // 帧号
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
     * @param dst_left  左图写入地址 (调用方分配, 可以是零拷贝内存)
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
    void configureCamera(void* handle, const CameraConfig& cfg);
    bool grabOneFrame(void* handle, uint8_t* dst, int pitch,
                      unsigned int timeout_ms, GrabResult& result);

    void* handle_left_  = nullptr;
    void* handle_right_ = nullptr;

    int width_  = 0;
    int height_ = 0;
    bool opened_   = false;
    bool grabbing_ = false;

    CameraConfig config_;
};

}  // namespace stereo3d
