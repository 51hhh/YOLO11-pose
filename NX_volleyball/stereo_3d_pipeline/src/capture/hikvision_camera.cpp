/**
 * @file hikvision_camera.cpp
 * @brief 海康工业相机零拷贝采集实现
 *
 * 核心设计:
 *   SDK GetImageBuffer → memcpy 到零拷贝对齐地址 → FreeImageBuffer
 *   不做 ISP / 格式转换, 原始 BayerRG8 输出
 *   ISP 由后续 CUDA/VPI 在 GPU 上完成
 */

#include "hikvision_camera.h"
#include "../utils/logger.h"
#include "MvCameraControl.h"

#include <cstring>
#include <chrono>

namespace stereo3d {

// ==================== 工具宏 ====================

#define MV_CHECK(fn, msg)                                               \
    do {                                                                \
        int nRet = (fn);                                                \
        if (MV_OK != nRet) {                                           \
            LOG_ERROR("[HikCam] " msg " failed, ret=0x%X", nRet);      \
            return false;                                               \
        }                                                               \
    } while(0)

// ==================== 构造 / 析构 ====================

HikvisionCamera::HikvisionCamera() = default;

HikvisionCamera::~HikvisionCamera() {
    close();
}

// ==================== 枚举 & 打开 ====================

bool HikvisionCamera::openCamera(void*& handle, int index,
                                  const std::string& serial) {
    MV_CC_DEVICE_INFO_LIST devList;
    memset(&devList, 0, sizeof(devList));

    int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &devList);
    if (MV_OK != nRet || devList.nDeviceNum == 0) {
        LOG_ERROR("[HikCam] EnumDevices failed or no device found");
        return false;
    }

    // 按序列号或索引选择
    MV_CC_DEVICE_INFO* devInfo = nullptr;
    if (!serial.empty()) {
        for (unsigned int i = 0; i < devList.nDeviceNum; ++i) {
            auto* info = devList.pDeviceInfo[i];
            // USB3 设备序列号
            if (info->nTLayerType == MV_USB_DEVICE) {
                std::string sn(reinterpret_cast<char*>(
                    info->SpecialInfo.stUsb3VInfo.chSerialNumber));
                if (sn == serial) { devInfo = info; break; }
            }
            // GigE 设备序列号
            if (info->nTLayerType == MV_GIGE_DEVICE) {
                std::string sn(reinterpret_cast<char*>(
                    info->SpecialInfo.stGigEInfo.chSerialNumber));
                if (sn == serial) { devInfo = info; break; }
            }
        }
        if (!devInfo) {
            LOG_ERROR("[HikCam] Camera SN=%s not found", serial.c_str());
            return false;
        }
    } else {
        if (index < 0 || static_cast<unsigned>(index) >= devList.nDeviceNum) {
            LOG_ERROR("[HikCam] Camera index %d out of range (%u)",
                      index, devList.nDeviceNum);
            return false;
        }
        devInfo = devList.pDeviceInfo[index];
    }

    MV_CHECK(MV_CC_CreateHandle(&handle, devInfo), "CreateHandle");
    MV_CHECK(MV_CC_OpenDevice(handle), "OpenDevice");

    return true;
}

void HikvisionCamera::configureCamera(void* handle, const CameraConfig& cfg) {
    // 像素格式: BayerRG8 (无ISP, 后续CUDA处理)
    MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_BayerRG8);

    // 曝光
    MV_CC_SetEnumValue(handle, "ExposureAuto", 0);  // Off
    MV_CC_SetFloatValue(handle, "ExposureTime", cfg.exposure_us);

    // 增益
    MV_CC_SetEnumValue(handle, "GainAuto", 0);  // Off
    MV_CC_SetFloatValue(handle, "Gain", cfg.gain_db);

    // 触发模式
    if (cfg.use_trigger) {
        MV_CC_SetEnumValue(handle, "TriggerMode", 1);  // On

        if (cfg.trigger_source == "Line0")
            MV_CC_SetEnumValue(handle, "TriggerSource", 0);
        else if (cfg.trigger_source == "Line1")
            MV_CC_SetEnumValue(handle, "TriggerSource", 1);
        else if (cfg.trigger_source == "Software")
            MV_CC_SetEnumValue(handle, "TriggerSource", 7);

        if (cfg.trigger_activation == "RisingEdge")
            MV_CC_SetEnumValue(handle, "TriggerActivation", 0);
        else if (cfg.trigger_activation == "FallingEdge")
            MV_CC_SetEnumValue(handle, "TriggerActivation", 1);
    } else {
        MV_CC_SetEnumValue(handle, "TriggerMode", 0);  // Off
    }

    LOG_INFO("[HikCam] Configured: %dx%d, exp=%.0fus, gain=%.1fdB, trigger=%s",
             cfg.width, cfg.height, cfg.exposure_us, cfg.gain_db,
             cfg.use_trigger ? cfg.trigger_source.c_str() : "FreeRun");
}

// ==================== open / close ====================

bool HikvisionCamera::open(const CameraConfig& cfg) {
    if (opened_) close();
    config_ = cfg;

    if (!openCamera(handle_left_, cfg.camera_index_left, cfg.serial_left)) {
        LOG_ERROR("[HikCam] Failed to open LEFT camera");
        return false;
    }
    configureCamera(handle_left_, cfg);

    if (!openCamera(handle_right_, cfg.camera_index_right, cfg.serial_right)) {
        LOG_ERROR("[HikCam] Failed to open RIGHT camera");
        MV_CC_CloseDevice(handle_left_);
        MV_CC_DestroyHandle(handle_left_);
        handle_left_ = nullptr;
        return false;
    }
    configureCamera(handle_right_, cfg);

    // 读取实际图像尺寸
    MVCC_INTVALUE intVal;
    if (MV_OK == MV_CC_GetIntValue(handle_left_, "Width", &intVal))
        width_ = static_cast<int>(intVal.nCurValue);
    else
        width_ = cfg.width;

    if (MV_OK == MV_CC_GetIntValue(handle_left_, "Height", &intVal))
        height_ = static_cast<int>(intVal.nCurValue);
    else
        height_ = cfg.height;

    opened_ = true;
    LOG_INFO("[HikCam] Both cameras opened, %dx%d", width_, height_);
    return true;
}

void HikvisionCamera::close() {
    stopGrabbing();

    if (handle_left_) {
        MV_CC_CloseDevice(handle_left_);
        MV_CC_DestroyHandle(handle_left_);
        handle_left_ = nullptr;
    }
    if (handle_right_) {
        MV_CC_CloseDevice(handle_right_);
        MV_CC_DestroyHandle(handle_right_);
        handle_right_ = nullptr;
    }
    opened_ = false;
}

// ==================== start / stop ====================

bool HikvisionCamera::startGrabbing() {
    if (!opened_ || grabbing_) return false;

    MV_CHECK(MV_CC_StartGrabbing(handle_left_),  "StartGrab LEFT");
    MV_CHECK(MV_CC_StartGrabbing(handle_right_), "StartGrab RIGHT");

    grabbing_ = true;
    LOG_INFO("[HikCam] Grabbing started");
    return true;
}

void HikvisionCamera::stopGrabbing() {
    if (!grabbing_) return;

    if (handle_left_)  MV_CC_StopGrabbing(handle_left_);
    if (handle_right_) MV_CC_StopGrabbing(handle_right_);

    grabbing_ = false;
}

// ==================== 帧采集 ====================

bool HikvisionCamera::grabOneFrame(void* handle, uint8_t* dst, int pitch,
                                    unsigned int timeout_ms,
                                    GrabResult& result) {
    MV_FRAME_OUT frameOut;
    memset(&frameOut, 0, sizeof(frameOut));

    int nRet = MV_CC_GetImageBuffer(handle, &frameOut, timeout_ms);
    if (MV_OK != nRet) {
        result.success = false;
        return false;
    }

    auto& info = frameOut.stFrameInfo;
    int srcWidth  = info.nWidth;
    int srcHeight = info.nHeight;
    int srcLen    = info.nFrameLen;

    // 直接 memcpy 到目标 buffer (零拷贝路径: dst 可以是 cudaHostAlloc 地址)
    int dstPitch = (pitch > 0) ? pitch : srcWidth;
    if (dstPitch == srcWidth) {
        // 连续拷贝
        memcpy(dst, frameOut.pBufAddr, srcWidth * srcHeight);
    } else {
        // 行对齐拷贝
        for (int y = 0; y < srcHeight; ++y) {
            memcpy(dst + y * dstPitch,
                   frameOut.pBufAddr + y * srcWidth,
                   srcWidth);
        }
    }

    result.success = true;
    result.timestamp_us = info.nDevTimeStampHigh;
    result.timestamp_us = (result.timestamp_us << 32) | info.nDevTimeStampLow;
    result.frame_number = info.nFrameNum;

    MV_CC_FreeImageBuffer(handle, &frameOut);
    return true;
}

bool HikvisionCamera::grabFramePair(
    uint8_t* dst_left,  uint8_t* dst_right,
    int left_pitch, int right_pitch,
    unsigned int timeout_ms,
    GrabResult& result_left, GrabResult& result_right)
{
    if (!grabbing_) return false;

    // 硬件触发模式下, 左右相机几乎同时收到触发脉冲
    // 先抓左, 再抓右 (延迟 < 1ms)
    bool okL = grabOneFrame(handle_left_,  dst_left,  left_pitch,
                            timeout_ms, result_left);
    bool okR = grabOneFrame(handle_right_, dst_right, right_pitch,
                            timeout_ms, result_right);

    if (!okL || !okR) {
        LOG_WARN("[HikCam] GrabPair partial fail: L=%d R=%d", okL, okR);
        return false;
    }

    // 检查时间戳差异 (应 < 1ms = 1000us)
    int64_t dt = static_cast<int64_t>(result_left.timestamp_us) -
                 static_cast<int64_t>(result_right.timestamp_us);
    if (std::abs(dt) > 5000) {  // > 5ms 报警
        LOG_WARN("[HikCam] Large stereo timestamp diff: %ldus", (long)dt);
    }

    return true;
}

bool HikvisionCamera::grabSingle(bool is_left,
                                  uint8_t* dst, int pitch,
                                  unsigned int timeout_ms,
                                  GrabResult& result) {
    void* handle = is_left ? handle_left_ : handle_right_;
    if (!handle || !grabbing_) return false;
    return grabOneFrame(handle, dst, pitch, timeout_ms, result);
}

}  // namespace stereo3d
