#pragma once

#include "../utils/logger.h"
#include "MvCameraControl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

namespace stereo3d {
namespace {

bool setEnumChecked(void* handle, const char* node, unsigned int value, const char* tag) {
    int ret = MV_CC_SetEnumValue(handle, node, value);
    if (ret != MV_OK) {
        LOG_ERROR("[HikCam][%s] Set %s=%u failed, ret=0x%X", tag, node, value, ret);
        return false;
    }
    return true;
}

bool setEnumOptional(void* handle, const char* node, unsigned int value, const char* tag) {
    int ret = MV_CC_SetEnumValue(handle, node, value);
    if (ret != MV_OK) {
        LOG_WARN("[HikCam][%s] Set %s=%u failed, ret=0x%X (continuing)",
                 tag, node, value, ret);
        return false;
    }
    return true;
}

bool setEnumStringOptional(void* handle, const char* node, const char* value, const char* tag) {
    int ret = MV_CC_SetEnumValueByString(handle, node, value);
    if (ret != MV_OK) {
        LOG_WARN("[HikCam][%s] Set %s=%s failed, ret=0x%X (continuing)",
                 tag, node, value, ret);
        return false;
    }
    return true;
}

bool setFloatChecked(void* handle, const char* node, float value, const char* tag) {
    int ret = MV_CC_SetFloatValue(handle, node, value);
    if (ret != MV_OK) {
        LOG_ERROR("[HikCam][%s] Set %s=%.3f failed, ret=0x%X", tag, node, value, ret);
        return false;
    }
    return true;
}

bool setFloatOptional(void* handle, const char* node, float value, const char* tag) {
    int ret = MV_CC_SetFloatValue(handle, node, value);
    if (ret != MV_OK) {
        LOG_WARN("[HikCam][%s] Set %s=%.3f failed, ret=0x%X (continuing)",
                 tag, node, value, ret);
        return false;
    }
    return true;
}

bool setBoolOptional(void* handle, const char* node, bool value, const char* tag) {
    int ret = MV_CC_SetBoolValue(handle, node, value);
    if (ret != MV_OK) {
        LOG_WARN("[HikCam][%s] Set %s=%d failed, ret=0x%X (continuing)",
                 tag, node, value ? 1 : 0, ret);
        return false;
    }
    return true;
}

bool getEnumValue(void* handle, const char* node, unsigned int& value) {
    MVCC_ENUMVALUE enumVal;
    memset(&enumVal, 0, sizeof(enumVal));
    int ret = MV_CC_GetEnumValue(handle, node, &enumVal);
    if (ret != MV_OK) return false;
    value = enumVal.nCurValue;
    return true;
}

bool getFloatValue(void* handle, const char* node, float& value) {
    MVCC_FLOATVALUE floatVal;
    memset(&floatVal, 0, sizeof(floatVal));
    int ret = MV_CC_GetFloatValue(handle, node, &floatVal);
    if (ret != MV_OK) return false;
    value = floatVal.fCurValue;
    return true;
}

bool getBoolValue(void* handle, const char* node, bool& value) {
    bool boolVal = false;
    int ret = MV_CC_GetBoolValue(handle, node, &boolVal);
    if (ret != MV_OK) return false;
    value = boolVal;
    return true;
}

const char* triggerSourceName(unsigned int value) {
    switch (value) {
    case 0: return "Line0";
    case 1: return "Line1";
    case 7: return "Software";
    default: return "Unknown";
    }
}

const char* triggerActivationName(unsigned int value) {
    switch (value) {
    case 0: return "RisingEdge";
    case 1: return "FallingEdge";
    default: return "Unknown";
    }
}

bool closeEnough(float actual, float expected, float tolerance) {
    return std::fabs(actual - expected) <= tolerance;
}

std::string deviceSerial(const MV_CC_DEVICE_INFO* info) {
    if (!info) return {};
    if (info->nTLayerType == MV_USB_DEVICE) {
        return reinterpret_cast<const char*>(
            info->SpecialInfo.stUsb3VInfo.chSerialNumber);
    }
    if (info->nTLayerType == MV_GIGE_DEVICE) {
        return reinterpret_cast<const char*>(
            info->SpecialInfo.stGigEInfo.chSerialNumber);
    }
    return {};
}

bool selectDeviceInfo(const MV_CC_DEVICE_INFO_LIST& devList,
                      int index,
                      const std::string& serial,
                      const char* tag,
                      MV_CC_DEVICE_INFO*& devInfo) {
    devInfo = nullptr;
    if (!serial.empty()) {
        for (unsigned int i = 0; i < devList.nDeviceNum; ++i) {
            auto* info = devList.pDeviceInfo[i];
            if (deviceSerial(info) == serial) {
                devInfo = info;
                return true;
            }
        }
        LOG_ERROR("[HikCam] %s camera SN=%s not found among %u devices",
                  tag, serial.c_str(), devList.nDeviceNum);
        return false;
    }

    if (index < 0 || static_cast<unsigned int>(index) >= devList.nDeviceNum) {
        LOG_ERROR("[HikCam] %s camera index %d out of range (%u)",
                  tag, index, devList.nDeviceNum);
        return false;
    }
    devInfo = devList.pDeviceInfo[index];
    return devInfo != nullptr;
}

bool createAndOpenHandle(void*& handle, MV_CC_DEVICE_INFO* devInfo, const char* tag) {
    int ret = MV_CC_CreateHandle(&handle, devInfo);
    if (ret != MV_OK) {
        LOG_ERROR("[HikCam][%s] CreateHandle failed, ret=0x%X", tag, ret);
        return false;
    }
    ret = MV_CC_OpenDevice(handle);
    if (ret != MV_OK) {
        LOG_ERROR("[HikCam][%s] OpenDevice failed, ret=0x%X", tag, ret);
        MV_CC_DestroyHandle(handle);
        handle = nullptr;
        return false;
    }
    return true;
}

bool setEnumByValueOrString(void* handle, const char* node, unsigned int value,
                            const char* value_name, const char* tag) {
    int ret = MV_CC_SetEnumValue(handle, node, value);
    if (ret == MV_OK) return true;

    int ret_str = MV_CC_SetEnumValueByString(handle, node, value_name);
    if (ret_str == MV_OK) {
        LOG_WARN("[HikCam][%s] Set %s=%u failed ret=0x%X, string '%s' succeeded",
                 tag, node, value, ret, value_name);
        return true;
    }

    LOG_ERROR("[HikCam][%s] Set %s=%u/%s failed, ret=0x%X/0x%X",
              tag, node, value, value_name, ret, ret_str);
    return false;
}

bool setFrameSpecInfo(void* handle, const char* selector,
                      bool enabled, const char* tag, bool required) {
    int ret = MV_CC_SetEnumValueByString(handle, "FrameSpecInfoSelector", selector);
    if (ret != MV_OK) {
        if (required) {
            LOG_ERROR("[HikCam][%s] Select FrameSpecInfoSelector=%s failed, ret=0x%X",
                      tag, selector, ret);
        } else {
            LOG_WARN("[HikCam][%s] Select FrameSpecInfoSelector=%s failed, ret=0x%X (continuing)",
                     tag, selector, ret);
        }
        return !required;
    }

    ret = MV_CC_SetBoolValue(handle, "FrameSpecInfo", enabled);
    if (ret != MV_OK) {
        if (required) {
            LOG_ERROR("[HikCam][%s] Set FrameSpecInfo[%s]=%d failed, ret=0x%X",
                      tag, selector, enabled ? 1 : 0, ret);
        } else {
            LOG_WARN("[HikCam][%s] Set FrameSpecInfo[%s]=%d failed, ret=0x%X (continuing)",
                     tag, selector, enabled ? 1 : 0, ret);
        }
        return !required;
    }

    bool readback = false;
    if (!getBoolValue(handle, "FrameSpecInfo", readback) || readback != enabled) {
        if (required) {
            LOG_ERROR("[HikCam][%s] FrameSpecInfo[%s] readback mismatch: expected=%d got=%d",
                      tag, selector, enabled ? 1 : 0, readback ? 1 : 0);
        } else {
            LOG_WARN("[HikCam][%s] FrameSpecInfo[%s] readback mismatch: expected=%d got=%d",
                     tag, selector, enabled ? 1 : 0, readback ? 1 : 0);
        }
        return !required;
    }

    return true;
}

bool configureMinimalFrameSpecInfo(void* handle, const char* tag, bool required) {
    // MVS GUI may persist "enable all". Override it here so the image only
    // carries the metadata needed for synchronization and diagnostics.
    const char* all_items[] = {
        "Timestamp",
        "Gain",
        "Exposure",
        "BrightnessInfo",
        "WhiteBalance",
        "Framecounter",
        "ExtTriggerCount",
        "LineInputOutput",
        "ROIPosition",
    };
    for (const char* item : all_items) {
        setFrameSpecInfo(handle, item, false, tag, false);
    }

    const char* required_items[] = {
        "Timestamp",
        "Framecounter",
        "ExtTriggerCount",
    };
    for (const char* item : required_items) {
        if (!setFrameSpecInfo(handle, item, true, tag, required)) {
            return false;
        }
    }

    LOG_INFO("[HikCam][%s] FrameSpecInfo %s: Timestamp, Framecounter, ExtTriggerCount",
             tag, required ? "enabled" : "requested");
    return true;
}

uint32_t counterStep(uint32_t older, uint32_t newer) {
    return newer - older;
}

void clearEmbeddedInfoRows(uint8_t* dst, int pitch, int width, int height,
                           int clear_rows) {
    // 海康 FrameSpecInfo 会把水印写入图像 payload 的首行附近。
    // 同步元数据已从 SDK frame info 取出，图像进入 debayer/YOLO 前清掉这些字节。
    if (!dst || clear_rows <= 0 || pitch <= 0 || width <= 0 || height <= 0) return;
    const int rows = std::min(height, clear_rows);
    for (int y = 0; y < rows; ++y) {
        std::memset(dst + y * pitch, 0, static_cast<size_t>(width));
    }
}

}  // namespace
}  // namespace stereo3d
