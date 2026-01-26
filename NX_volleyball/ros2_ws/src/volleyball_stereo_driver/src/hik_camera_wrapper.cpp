/**
 * @file hik_camera_wrapper.cpp
 * @brief 海康相机 C++ 封装实现 (高性能回调模式版本)
 */

#include "volleyball_stereo_driver/hik_camera_wrapper.hpp"
#include <iostream>
#include <sched.h>
#include <cstring>
#include <chrono>
#include <stdexcept>

namespace volleyball {

// 全局 SDK 初始化标志
static bool g_sdk_initialized = false;
static std::mutex g_sdk_mutex;

// ==================== 回调函数 (SDK调用) ====================
// MV_CC_RegisterImageCallBackEx2 的回调签名: void (*)(MV_FRAME_OUT*, void*, bool)
static void __stdcall ImageCallBackEx2(MV_FRAME_OUT* pFrame, void* pUser, bool bAutoFree) {
    (void)bAutoFree;  // 抑制未使用参数警告 (我们总是使用自动释放模式)
    
    if (pUser && pFrame && pFrame->pBufAddr) {
        HikCamera* camera = static_cast<HikCamera*>(pUser);
        camera->onImageCallback(pFrame);
    }
}

HikCamera::HikCamera(int camera_index, const std::string& serial_number)
    : camera_index_(camera_index),
      serial_number_(serial_number),
      camera_handle_(nullptr),
      is_opened_(false),
      is_grabbing_(false),
      use_callback_mode_(true),  // 默认启用回调模式
      width_(0),
      height_(0),
      pixel_format_(0) {
    memset(&device_list_, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    
    // 初始化 SDK (只需要一次)
    std::lock_guard<std::mutex> lock(g_sdk_mutex);
    if (!g_sdk_initialized) {
        int ret = MV_CC_Initialize();
        if (ret == MV_OK) {
            g_sdk_initialized = true;
            std::cout << "✅ 海康 SDK 已初始化" << std::endl;
        } else {
            std::cerr << "❌ 海康 SDK 初始化失败: " << std::hex << ret << std::endl;
        }
    }
}

HikCamera::~HikCamera() {
    close();
}

bool HikCamera::open() {
    // 枚举设备
    int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list_);
    if (ret != MV_OK) {
        std::cerr << "枚举设备失败: " << std::hex << ret << std::endl;
        return false;
    }

    if (device_list_.nDeviceNum == 0) {
        std::cerr << "未找到相机" << std::endl;
        return false;
    }

    // 选择相机
    unsigned int device_index = camera_index_;
    
    if (!serial_number_.empty()) {
        // 根据序列号查找
        bool found = false;
        for (unsigned int i = 0; i < device_list_.nDeviceNum; i++) {
            MV_CC_DEVICE_INFO* device_info = device_list_.pDeviceInfo[i];
            
            std::string serial;
            if (device_info->nTLayerType == MV_GIGE_DEVICE) {
                serial = reinterpret_cast<char*>(device_info->SpecialInfo.stGigEInfo.chSerialNumber);
            } else if (device_info->nTLayerType == MV_USB_DEVICE) {
                serial = reinterpret_cast<char*>(device_info->SpecialInfo.stUsb3VInfo.chSerialNumber);
            }
            
            if (serial == serial_number_) {
                device_index = i;
                found = true;
                break;
            }
        }
        
        if (!found) {
            std::cerr << "未找到序列号为 " << serial_number_ << " 的相机" << std::endl;
            return false;
        }
    }

    // 创建句柄
    ret = MV_CC_CreateHandle(&camera_handle_, device_list_.pDeviceInfo[device_index]);
    if (ret != MV_OK) {
        std::cerr << "创建句柄失败: " << std::hex << ret << std::endl;
        return false;
    }

    // 打开设备
    ret = MV_CC_OpenDevice(camera_handle_, MV_ACCESS_Exclusive, 0);
    if (ret != MV_OK) {
        std::cerr << "打开设备失败: " << std::hex << ret << std::endl;
        MV_CC_DestroyHandle(camera_handle_);
        camera_handle_ = nullptr;
        return false;
    }

    is_opened_ = true;

    // ==================== 性能优化配置 (USB3.0相机) ====================
    
    // 1. USB3.0相机: 检测并打印相机类型
    if (device_list_.pDeviceInfo[device_index]->nTLayerType == MV_USB_DEVICE) {
        std::cout << "✅ USB3.0 相机已识别" << std::endl;
        
        // USB3.0 特有优化: 设置传输包大小
        ret = MV_CC_SetIntValue(camera_handle_, "TransferSize", 1048576);  // 1MB 传输块
        if (ret == MV_OK) {
            std::cout << "✅ USB传输块大小: 1MB" << std::endl;
        }
        
        // 设置USB设备流带宽 (0-100%, 100%最大带宽)
        ret = MV_CC_SetIntValue(camera_handle_, "DeviceLinkThroughputLimit", 400000000);  // 400MB/s
        if (ret == MV_OK) {
            std::cout << "✅ USB带宽限制: 400MB/s" << std::endl;
        }
    } else if (device_list_.pDeviceInfo[device_index]->nTLayerType == MV_GIGE_DEVICE) {
        int nPacketSize = MV_CC_GetOptimalPacketSize(camera_handle_);
        if (nPacketSize > 0) {
            ret = MV_CC_SetIntValueEx(camera_handle_, "GevSCPSPacketSize", nPacketSize);
            if (ret == MV_OK) {
                std::cout << "✅ GigE网络包大小: " << nPacketSize << " bytes" << std::endl;
            }
        }
    }

    // 2. 查询相机支持的像素格式
    MVCC_ENUMVALUE stEnumPixelFormat;
    memset(&stEnumPixelFormat, 0, sizeof(MVCC_ENUMVALUE));
    ret = MV_CC_GetEnumValue(camera_handle_, "PixelFormat", &stEnumPixelFormat);
    if (ret == MV_OK) {
        std::cout << "📷 当前像素格式: 0x" << std::hex << stEnumPixelFormat.nCurValue << std::dec << std::endl;
    }
    
    // 尝试设置相机输出像素格式 (按优先级尝试)
    // MV-CA016-10UC 支持: BayerRG8, BayerRG10, BayerRG12, RGB8, BGR8
    bool format_set = false;
    
    // 优先: BayerRG8 (传输带宽低，100fps@9867us曝光)
    ret = MV_CC_SetEnumValue(camera_handle_, "PixelFormat", PixelType_Gvsp_BayerRG8);
    if (ret == MV_OK) {
        std::cout << "✅ 相机像素格式: BayerRG8 (100fps支持)" << std::endl;
        format_set = true;
    }
    
    // 次选: BGR8 (OpenCV原生格式，但传输带宽高，限76fps)
    if (!format_set) {
        ret = MV_CC_SetEnumValue(camera_handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed);
        if (ret == MV_OK) {
            std::cout << "⚠️  相机像素格式: BGR8 (零转换，但限76fps)" << std::endl;
            format_set = true;
        }
    }
    
    // 再次选: RGB8 (只需通道交换，但传输带宽高)
    if (!format_set) {
        ret = MV_CC_SetEnumValue(camera_handle_, "PixelFormat", PixelType_Gvsp_RGB8_Packed);
        if (ret == MV_OK) {
            std::cout << "⚠️  相机像素格式: RGB8 (通道交换，限76fps)" << std::endl;
            format_set = true;
        }
    }
    
    if (!format_set) {
        std::cout << "⚠️  像素格式保持默认" << std::endl;
    }

    // 3. 设置图像节点缓存数量 (增加缓冲减少丢帧)
    ret = MV_CC_SetImageNodeNum(camera_handle_, 5);
    if (ret == MV_OK) {
        std::cout << "✅ 图像缓冲区: 5帧" << std::endl;
    }

    // 获取图像参数 (必须在分配缓冲区之前)
    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    
    ret = MV_CC_GetIntValue(camera_handle_, "Width", &stParam);
    if (ret == MV_OK) {
        width_ = stParam.nCurValue;
    }
    
    ret = MV_CC_GetIntValue(camera_handle_, "Height", &stParam);
    if (ret == MV_OK) {
        height_ = stParam.nCurValue;
    }
    
    ret = MV_CC_GetIntValue(camera_handle_, "PixelFormat", &stParam);
    if (ret == MV_OK) {
        pixel_format_ = stParam.nCurValue;
    }

    // 4. 预分配双缓冲区 (确保使用正确的分辨率)
    if (width_ > 0 && height_ > 0) {
        convert_buffer_.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3);
        frame_buffer_[0] = cv::Mat(height_, width_, CV_8UC3);
        frame_buffer_[1] = cv::Mat(height_, width_, CV_8UC3);
    } else {
        std::cerr << "⚠️ 获取分辨率失败，使用默认缓冲区大小" << std::endl;
        // 兜底：避免未初始化导致崩溃，分配最小1x1
        width_ = std::max(width_, 1);
        height_ = std::max(height_, 1);
        convert_buffer_.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3);
        frame_buffer_[0] = cv::Mat(height_, width_, CV_8UC3);
        frame_buffer_[1] = cv::Mat(height_, width_, CV_8UC3);
    }
    write_index_ = 0;
    read_index_ = 0;
    new_frame_ready_ = false;

    std::cout << "✅ 相机已打开: " << width_ << "x" << height_ << std::endl;
    return true;
}

void HikCamera::close() {
    if (is_grabbing_) {
        stopGrabbing();
    }

    if (is_opened_ && camera_handle_) {
        MV_CC_CloseDevice(camera_handle_);
        MV_CC_DestroyHandle(camera_handle_);
        camera_handle_ = nullptr;
        is_opened_ = false;
        std::cout << "✅ 相机已关闭" << std::endl;
    }
}

bool HikCamera::setTriggerMode(bool mode) {
    if (!is_opened_) {
        std::cerr << "相机未打开" << std::endl;
        return false;
    }

    int ret = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", mode ? 1 : 0);
    if (ret != MV_OK) {
        std::cerr << "设置触发模式失败: " << std::hex << ret << std::endl;
        return false;
    }

    std::cout << "✅ 触发模式: " << (mode ? "On" : "Off") << std::endl;
    return true;
}

bool HikCamera::setTriggerSource(const std::string& source) {
    if (!is_opened_) {
        std::cerr << "相机未打开" << std::endl;
        return false;
    }

    unsigned int value = 0;
    if (source == "Line0") value = 0;
    else if (source == "Line1") value = 1;
    else if (source == "Line2") value = 2;
    else if (source == "Software") value = 7;
    else {
        std::cerr << "未知触发源: " << source << std::endl;
        return false;
    }

    int ret = MV_CC_SetEnumValue(camera_handle_, "TriggerSource", value);
    if (ret != MV_OK) {
        std::cerr << "设置触发源失败: " << std::hex << ret << std::endl;
        return false;
    }

    std::cout << "✅ 触发源: " << source << std::endl;
    return true;
}

bool HikCamera::setTriggerActivation(const std::string& activation) {
    if (!is_opened_) {
        std::cerr << "相机未打开" << std::endl;
        return false;
    }

    unsigned int value = 0;
    if (activation == "RisingEdge") value = 0;
    else if (activation == "FallingEdge") value = 1;
    else if (activation == "LevelHigh") value = 2;
    else if (activation == "LevelLow") value = 3;
    else {
        std::cerr << "未知触发激活方式: " << activation << std::endl;
        return false;
    }

    int ret = MV_CC_SetEnumValue(camera_handle_, "TriggerActivation", value);
    if (ret != MV_OK) {
        std::cerr << "设置触发激活失败: " << std::hex << ret << std::endl;
        return false;
    }

    std::cout << "✅ 触发激活: " << activation << std::endl;
    return true;
}

bool HikCamera::setExposureTime(float exposure_us) {
    if (!is_opened_) {
        std::cerr << "相机未打开" << std::endl;
        return false;
    }

    int ret = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_us);
    if (ret != MV_OK) {
        std::cerr << "设置曝光时间失败: " << std::hex << ret << std::endl;
        return false;
    }

    std::cout << "✅ 曝光时间: " << exposure_us << " us" << std::endl;
    return true;
}

bool HikCamera::setGain(float gain_db) {
    if (!is_opened_) {
        std::cerr << "相机未打开" << std::endl;
        return false;
    }

    int ret = MV_CC_SetFloatValue(camera_handle_, "Gain", gain_db);
    if (ret != MV_OK) {
        std::cerr << "设置增益失败: " << std::hex << ret << std::endl;
        return false;
    }

    std::cout << "✅ 增益: " << gain_db << " dB" << std::endl;
    return true;
}

bool HikCamera::startGrabbing() {
    if (!is_opened_) {
        std::cerr << "相机未打开" << std::endl;
        return false;
    }
    
    int ret;
    
    // 回调模式: 注册回调函数
    if (use_callback_mode_) {
        // 注册图像回调 (SDK 内部线程会调用)
        // 参数: handle, 回调函数, 用户数据, 是否复制数据(false=零拷贝)
        ret = MV_CC_RegisterImageCallBackEx2(camera_handle_, ImageCallBackEx2, this, true);
        if (ret != MV_OK) {
            std::cerr << "⚠️  注册回调失败: " << std::hex << ret << ", 降级到轮询模式" << std::dec << std::endl;
            use_callback_mode_ = false;
        } else {
            std::cout << "✅ 回调模式已启用 (零等待)" << std::endl;
        }
    }

    ret = MV_CC_StartGrabbing(camera_handle_);
    if (ret != MV_OK) {
        std::cerr << "开始采集失败: " << std::hex << ret << std::endl;
        return false;
    }

    is_grabbing_ = true;
    std::cout << "✅ 开始采集" << (use_callback_mode_ ? " [回调模式]" : " [轮询模式]") << std::endl;
    return true;
}

void HikCamera::stopGrabbing() {
    if (is_grabbing_ && camera_handle_) {
        MV_CC_StopGrabbing(camera_handle_);
        is_grabbing_ = false;
        
        // 清除回调
        if (use_callback_mode_) {
            MV_CC_RegisterImageCallBackEx2(camera_handle_, nullptr, nullptr, true);
        }
        
        std::cout << "✅ 停止采集" << std::endl;
    }
}

// ==================== 像素格式转换（私有辅助函数）====================
bool HikCamera::convertPixelToBGR(const unsigned char* src_data, int width, int height,
                                  MvGvspPixelType pixel_type, unsigned int data_len, cv::Mat& dst) {
    if (!src_data || dst.empty()) {
        return false;
    }
    
    // BGR8: 直接拷贝
    if (pixel_type == PixelType_Gvsp_BGR8_Packed) {
        memcpy(dst.data, src_data, data_len);
        return true;
    }
    
    // RGB8: 通道转换
    if (pixel_type == PixelType_Gvsp_RGB8_Packed) {
        cv::Mat src(height, width, CV_8UC3, const_cast<unsigned char*>(src_data));
        cv::cvtColor(src, dst, cv::COLOR_RGB2BGR);
        return true;
    }
    
    // BayerRG8: ⚡ 直接拷贝Bayer原始数据，交给GPU处理
    // 优势: 1) 零CPU开销 2) GPU去马赛克比CPU快10倍+ 3) 与resize融合节省带宽
    if (pixel_type == PixelType_Gvsp_BayerRG8) {
        // 直接拷贝Bayer数据（单通道，不去马赛克）
        memcpy(dst.data, src_data, data_len);
        return true;
    }
    
    // Mono8: 灰度转BGR
    if (pixel_type == PixelType_Gvsp_Mono8) {
        cv::Mat src(height, width, CV_8UC1, const_cast<unsigned char*>(src_data));
        cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
        return true;
    }
    
    // 其他格式: SDK转换
    MV_CC_PIXEL_CONVERT_PARAM convert_param;
    memset(&convert_param, 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM));
    
    convert_param.nWidth = width;
    convert_param.nHeight = height;
    convert_param.pSrcData = const_cast<unsigned char*>(src_data);
    convert_param.nSrcDataLen = data_len;
    convert_param.enSrcPixelType = pixel_type;
    convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
    convert_param.pDstBuffer = convert_buffer_.data();
    convert_param.nDstBufferSize = convert_buffer_.size();
    
    if (MV_CC_ConvertPixelType(camera_handle_, &convert_param) == MV_OK) {
        memcpy(dst.data, convert_buffer_.data(), width * height * 3);
        return true;
    }
    
    return false;
}

// ==================== 回调模式处理函数 ====================
void HikCamera::onImageCallback(MV_FRAME_OUT* pFrame) {
    if (!pFrame || !pFrame->pBufAddr) return;
    
    // 提取帧元数据（帧号 + 时间戳）
    FrameMetadata metadata;
    metadata.frame_number = pFrame->stFrameInfo.nFrameNum;
    metadata.device_timestamp = ((uint64_t)pFrame->stFrameInfo.nDevTimeStampHigh << 32) 
                              | pFrame->stFrameInfo.nDevTimeStampLow;
    metadata.host_timestamp = pFrame->stFrameInfo.nHostTimeStamp;
    metadata.receive_time = std::chrono::steady_clock::now();
    
    // 获取写入缓冲区索引（交替写入）
    int idx = write_index_.load();
    cv::Mat& dst = frame_buffer_[idx];
    
    // 像素格式转换
    convertPixelToBGR(pFrame->pBufAddr, 
                     pFrame->stFrameInfo.nWidth, 
                     pFrame->stFrameInfo.nHeight,
                     pFrame->stFrameInfo.enPixelType, 
                     pFrame->stFrameInfo.nFrameLen, 
                     dst);
    
    // 保存帧元数据
    frame_metadata_[idx] = metadata;
    
    // 🚀 调用外部回调函数（如果已设置）
    if (external_callback_) {
        external_callback_(dst, metadata);
    }
    
    // 切换写入索引
    write_index_.store(1 - idx);
    
    // 通知有新帧
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        read_index_.store(idx);  // 读取刚写入的缓冲区
        new_frame_ready_.store(true);
    }
    frame_cv_.notify_one();
}

bool HikCamera::waitForNewFrame(unsigned int timeout_ms) {
    std::unique_lock<std::mutex> lock(frame_mutex_);
    if (new_frame_ready_.load()) {
        return true;
    }
    return frame_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                               [this]{ return new_frame_ready_.load(); });
}

cv::Mat HikCamera::getLatestImage() {
    if (!use_callback_mode_) {
        return grabImage(50);  // 降级到轮询模式
    }
    
    if (!new_frame_ready_.load()) {
        return cv::Mat();  // 无新帧
    }
    
    // 获取最新帧 (读取索引指向的缓冲区)
    int idx = read_index_.load();
    new_frame_ready_.store(false);
    
    return frame_buffer_[idx].clone();
}

cv::Mat HikCamera::grabImage(unsigned int timeout_ms) {
    if (!is_grabbing_) {
        std::cerr << "未开始采集" << std::endl;
        return cv::Mat();
    }

    MV_FRAME_OUT stImageInfo;
    memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT));

    // 获取图像缓冲区 (高性能API)
    int ret = MV_CC_GetImageBuffer(camera_handle_, &stImageInfo, timeout_ms);
    
    if (ret != MV_OK) {
        return cv::Mat();
    }

    // 创建临时Mat用于转换
    cv::Mat image(stImageInfo.stFrameInfo.nHeight, stImageInfo.stFrameInfo.nWidth, CV_8UC3);
    
    // 使用公共转换函数
    if (convertPixelToBGR(stImageInfo.pBufAddr, 
                         stImageInfo.stFrameInfo.nWidth, 
                         stImageInfo.stFrameInfo.nHeight,
                         stImageInfo.stFrameInfo.enPixelType, 
                         stImageInfo.stFrameInfo.nFrameLen, 
                         image)) {
        // 转换成功，克隆一份（因为SDK缓冲区会被释放）
        cv::Mat result = image.clone();
        
        // 释放图像缓冲区（重要！）
        MV_CC_FreeImageBuffer(camera_handle_, &stImageInfo);
        
        return result;
    }
    
    // 转换失败，释放缓冲区并返回空Mat
    MV_CC_FreeImageBuffer(camera_handle_, &stImageInfo);
    return cv::Mat();
}

std::string HikCamera::getCameraInfo() const {
    if (!is_opened_ || !camera_handle_) {
        return "相机未打开";
    }

    // 获取相机信息
    MV_CC_DEVICE_INFO* device_info = device_list_.pDeviceInfo[camera_index_];
    
    std::string info = "相机信息:\n";
    
    if (device_info->nTLayerType == MV_GIGE_DEVICE) {
        info += "  类型: GigE\n";
        info += "  型号: " + std::string(reinterpret_cast<char*>(device_info->SpecialInfo.stGigEInfo.chModelName)) + "\n";
        info += "  序列号: " + std::string(reinterpret_cast<char*>(device_info->SpecialInfo.stGigEInfo.chSerialNumber)) + "\n";
    } else if (device_info->nTLayerType == MV_USB_DEVICE) {
        info += "  类型: USB\n";
        info += "  型号: " + std::string(reinterpret_cast<char*>(device_info->SpecialInfo.stUsb3VInfo.chModelName)) + "\n";
        info += "  序列号: " + std::string(reinterpret_cast<char*>(device_info->SpecialInfo.stUsb3VInfo.chSerialNumber)) + "\n";
    }
    
    info += "  分辨率: " + std::to_string(width_) + "x" + std::to_string(height_);
    
    return info;
}

// ==================== 获取帧元数据 ====================
FrameMetadata HikCamera::getFrameMetadata() const {
    int idx = read_index_.load();
    return frame_metadata_[idx];
}

void HikCamera::setFrameCallback(FrameCallback callback) {
    external_callback_ = callback;
}

}  // namespace volleyball
