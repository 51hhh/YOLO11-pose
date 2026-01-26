/**
 * @file hik_camera_wrapper.hpp
 * @brief 海康相机 C++ 封装类
 * 
 * 支持外部触发、图像采集和参数配置
 * 使用回调模式实现高性能取流
 */

#ifndef HIK_CAMERA_WRAPPER_HPP_
#define HIK_CAMERA_WRAPPER_HPP_

#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <array>
#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"

namespace volleyball {

/**
 * @struct FrameMetadata
 * @brief 帧元数据结构 (用于PWM时间戳同步)
 */
struct FrameMetadata {
    uint32_t frame_number = 0;           // 海康SDK帧号 (nFrameNum)
    uint64_t device_timestamp = 0;       // 设备时间戳 (微秒)
    uint32_t host_timestamp = 0;         // 主机时间戳 (毫秒)
    std::chrono::steady_clock::time_point receive_time;  // 接收时间点
    
    FrameMetadata() : receive_time(std::chrono::steady_clock::now()) {}
};

/**
 * @class HikCamera
 * @brief 海康工业相机封装类 (回调模式高性能版本)
 */
class HikCamera {
public:
    /**
     * @brief 构造函数
     * @param camera_index 相机索引 (0, 1, ...)
     * @param serial_number 相机序列号 (可选)
     */
    explicit HikCamera(int camera_index = 0, const std::string& serial_number = "");

    /**
     * @brief 析构函数
     */
    ~HikCamera();

    /**
     * @brief 打开相机
     * @return true 成功, false 失败
     */
    bool open();

    /**
     * @brief 关闭相机
     */
    void close();

    /**
     * @brief 设置触发模式
     * @param mode true=On, false=Off
     * @return true 成功, false 失败
     */
    bool setTriggerMode(bool mode);

    /**
     * @brief 设置触发源
     * @param source "Line0", "Line1", "Software"
     * @return true 成功, false 失败
     */
    bool setTriggerSource(const std::string& source);

    /**
     * @brief 设置触发激活方式
     * @param activation "RisingEdge", "FallingEdge"
     * @return true 成功, false 失败
     */
    bool setTriggerActivation(const std::string& activation);

    /**
     * @brief 设置曝光时间
     * @param exposure_us 曝光时间 (微秒)
     * @return true 成功, false 失败
     */
    bool setExposureTime(float exposure_us);

    /**
     * @brief 设置增益
     * @param gain_db 增益 (dB)
     * @return true 成功, false 失败
     */
    bool setGain(float gain_db);

    /**
     * @brief 开始采集 (回调模式)
     * @return true 成功, false 失败
     */
    bool startGrabbing();

    /**
     * @brief 停止采集
     */
    void stopGrabbing();

    /**
     * @brief 采集一帧图像 (轮询模式，兼容旧接口)
     * @param timeout_ms 超时时间 (毫秒)
     * @return cv::Mat 图像 (BGR 格式), 失败返回空 Mat
     */
    cv::Mat grabImage(unsigned int timeout_ms = 1000);
    
    /**
     * @brief 获取最新一帧图像 (回调模式，零等待)
     * @return cv::Mat 图像 (BGR 格式), 无新帧返回空 Mat
     */
    cv::Mat getLatestImage();
    
    /**
     * @brief 等待新帧到达 (回调模式)
     * @param timeout_ms 超时时间 (毫秒)
     * @return true 有新帧, false 超时
     */
    bool waitForNewFrame(unsigned int timeout_ms = 100);
    
    /**
     * @brief 获取最新帧的元数据 (帧号、时间戳)
     * @return FrameMetadata 帧元数据
     */
    FrameMetadata getFrameMetadata() const;

    /**
     * @brief 获取相机信息
     * @return 相机信息字符串
     */
    std::string getCameraInfo() const;

    /**
     * @brief 是否已打开
     */
    bool isOpened() const { return is_opened_; }

    /**
     * @brief 是否正在采集
     */
    bool isGrabbing() const { return is_grabbing_; }
    
    /**
     * @brief 回调函数 (SDK调用)
     */
    void onImageCallback(MV_FRAME_OUT* pFrame);

private:
    /**
     * @brief 转换像素格式到BGR8
     * @param src_data 源数据指针
     * @param width 图像宽度
     * @param height 图像高度
     * @param pixel_type 像素格式
     * @param data_len 数据长度
     * @param dst 目标Mat（必须已分配）
     * @return true 成功, false 失败
     */
    bool convertPixelToBGR(const unsigned char* src_data, int width, int height,
                          MvGvspPixelType pixel_type, unsigned int data_len, cv::Mat& dst);

private:
    int camera_index_;
    std::string serial_number_;
    void* camera_handle_;
    MV_CC_DEVICE_INFO_LIST device_list_;
    
    bool is_opened_;
    bool is_grabbing_;
    bool use_callback_mode_;  // 是否使用回调模式
    
    int width_;
    int height_;
    int pixel_format_;
    
    // 回调模式双缓冲
    cv::Mat frame_buffer_[2];
    std::array<FrameMetadata, 2> frame_metadata_;  // 帧元数据双缓冲
    std::atomic<int> write_index_{0};
    std::atomic<int> read_index_{0};
    std::atomic<bool> new_frame_ready_{false};
    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    
    // 预分配转换缓冲区
    std::vector<unsigned char> convert_buffer_;
};

}  // namespace volleyball

#endif  // HIK_CAMERA_WRAPPER_HPP_
