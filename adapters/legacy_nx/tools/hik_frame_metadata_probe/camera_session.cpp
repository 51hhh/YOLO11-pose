#include "camera_session.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace hik_metadata_probe {
namespace {

bool setEnumString(
    void* handle,
    const char* node,
    const char* value,
    const char* tag,
    bool required = false) {
    const int status = MV_CC_SetEnumValueByString(handle, node, value);
    if (status != MV_OK) {
        std::printf(
            "[%s] set %s=%s failed ret=0x%X%s\n",
            tag,
            node,
            value,
            status,
            required ? " REQUIRED" : "");
    }
    return status == MV_OK;
}

bool setEnumValue(
    void* handle,
    const char* node,
    unsigned int value,
    const char* tag,
    bool required = false) {
    const int status = MV_CC_SetEnumValue(handle, node, value);
    if (status != MV_OK) {
        std::printf(
            "[%s] set %s=%u failed ret=0x%X%s\n",
            tag,
            node,
            value,
            status,
            required ? " REQUIRED" : "");
    }
    return status == MV_OK;
}

bool setBoolValue(
    void* handle,
    const char* node,
    bool value,
    const char* tag,
    bool required = false) {
    const int status = MV_CC_SetBoolValue(handle, node, value);
    if (status != MV_OK) {
        std::printf(
            "[%s] set %s=%d failed ret=0x%X%s\n",
            tag,
            node,
            value ? 1 : 0,
            status,
            required ? " REQUIRED" : "");
    }
    return status == MV_OK;
}

bool setFloatValue(
    void* handle,
    const char* node,
    float value,
    const char* tag,
    bool required = false) {
    const int status = MV_CC_SetFloatValue(handle, node, value);
    if (status != MV_OK) {
        std::printf(
            "[%s] set %s=%.3f failed ret=0x%X%s\n",
            tag,
            node,
            value,
            status,
            required ? " REQUIRED" : "");
    }
    return status == MV_OK;
}

bool getEnumValue(void* handle, const char* node, unsigned int& value) {
    MVCC_ENUMVALUE result {};
    const int status = MV_CC_GetEnumValue(handle, node, &result);
    if (status != MV_OK) return false;
    value = result.nCurValue;
    return true;
}

bool getBoolValue(void* handle, const char* node, bool& value) {
    bool result = false;
    const int status = MV_CC_GetBoolValue(handle, node, &result);
    if (status != MV_OK) return false;
    value = result;
    return true;
}

bool getFloatValue(void* handle, const char* node, float& value) {
    MVCC_FLOATVALUE result {};
    const int status = MV_CC_GetFloatValue(handle, node, &result);
    if (status != MV_OK) return false;
    value = result.fCurValue;
    return true;
}

bool enableFrameSpecItem(
    void* handle,
    const char* item,
    const char* tag,
    bool required) {
    const int select_status = MV_CC_SetEnumValueByString(
        handle, "FrameSpecInfoSelector", item);
    if (select_status != MV_OK) {
        std::printf(
            "[%s] select FrameSpecInfoSelector=%s failed ret=0x%X%s\n",
            tag,
            item,
            select_status,
            required ? " REQUIRED" : "");
        return !required;
    }
    const int enable_status = MV_CC_SetBoolValue(
        handle, "FrameSpecInfo", true);
    if (enable_status != MV_OK) {
        std::printf(
            "[%s] enable FrameSpecInfo[%s] failed ret=0x%X%s\n",
            tag,
            item,
            enable_status,
            required ? " REQUIRED" : "");
        return !required;
    }
    return true;
}

bool enableFrameSpecInfo(void* handle, const char* tag) {
    for (const char* item : {"Timestamp", "Framecounter", "ExtTriggerCount"}) {
        if (!enableFrameSpecItem(handle, item, tag, true)) return false;
    }
    for (const char* item : {
             "Exposure", "Gain", "LineInputOutput", "ROIPosition"}) {
        (void)enableFrameSpecItem(handle, item, tag, false);
    }
    return true;
}

}  // namespace

CameraSession::~CameraSession() {
    shutdown();
}

bool CameraSession::startGrabbing(const char* tag) {
    if (handle_ == nullptr) return false;
    const int status = MV_CC_StartGrabbing(handle_);
    if (status != MV_OK) {
        std::fprintf(stderr, "[%s] StartGrabbing failed ret=0x%X\n", tag, status);
        return false;
    }
    grabbing_ = true;
    return true;
}

void CameraSession::shutdown() noexcept {
    if (handle_ != nullptr) {
        if (grabbing_) {
            (void)MV_CC_StopGrabbing(handle_);
            grabbing_ = false;
        }
        (void)MV_CC_CloseDevice(handle_);
        (void)MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
    }
    serial_.clear();
}

std::uint64_t deviceTimestamp(const MV_FRAME_OUT_INFO_EX& info) noexcept {
    return (static_cast<std::uint64_t>(info.nDevTimeStampHigh) << 32U) |
           static_cast<std::uint64_t>(info.nDevTimeStampLow);
}

std::string serialOf(const MV_CC_DEVICE_INFO* info) {
    if (info == nullptr) return {};
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

std::string modelOf(const MV_CC_DEVICE_INFO* info) {
    if (info == nullptr) return {};
    if (info->nTLayerType == MV_USB_DEVICE) {
        return reinterpret_cast<const char*>(
            info->SpecialInfo.stUsb3VInfo.chModelName);
    }
    if (info->nTLayerType == MV_GIGE_DEVICE) {
        return reinterpret_cast<const char*>(
            info->SpecialInfo.stGigEInfo.chModelName);
    }
    return {};
}

const char* layerName(const MV_CC_DEVICE_INFO* info) noexcept {
    if (info == nullptr) return "unknown";
    if (info->nTLayerType == MV_USB_DEVICE) return "USB";
    if (info->nTLayerType == MV_GIGE_DEVICE) return "GigE";
    return "other";
}

const MV_CC_DEVICE_INFO* resolveCameraInfo(
    const MV_CC_DEVICE_INFO_LIST& list,
    const std::string& wanted_serial,
    int index) {
    if (!wanted_serial.empty()) {
        for (unsigned int candidate = 0;
             candidate < list.nDeviceNum;
             ++candidate) {
            if (serialOf(list.pDeviceInfo[candidate]) == wanted_serial) {
                return list.pDeviceInfo[candidate];
            }
        }
        return nullptr;
    }
    if (index < 0 || static_cast<unsigned int>(index) >= list.nDeviceNum) {
        return nullptr;
    }
    return list.pDeviceInfo[index];
}

bool openCamera(
    const MV_CC_DEVICE_INFO_LIST& list,
    const std::string& wanted_serial,
    int index,
    const char* tag,
    CameraSession& camera) {
    camera.shutdown();
    const MV_CC_DEVICE_INFO* selected = resolveCameraInfo(
        list, wanted_serial, index);

    if (selected == nullptr) {
        std::fprintf(
            stderr,
            "[%s] camera not found, sn='%s' index=%d\n",
            tag,
            wanted_serial.c_str(),
            index);
        return false;
    }

    camera.serial_ = serialOf(selected);
    int status = MV_CC_CreateHandle(
        &camera.handle_, const_cast<MV_CC_DEVICE_INFO*>(selected));
    if (status != MV_OK) {
        std::fprintf(stderr, "[%s] CreateHandle failed ret=0x%X\n", tag, status);
        camera.handle_ = nullptr;
        camera.serial_.clear();
        return false;
    }
    status = MV_CC_OpenDevice(camera.handle_);
    if (status != MV_OK) {
        std::fprintf(stderr, "[%s] OpenDevice failed ret=0x%X\n", tag, status);
        (void)MV_CC_DestroyHandle(camera.handle_);
        camera.handle_ = nullptr;
        camera.serial_.clear();
        return false;
    }

    std::printf(
        "[%s] opened SN=%s model=%s layer=%s\n",
        tag,
        camera.serial_.c_str(),
        modelOf(selected).c_str(),
        layerName(selected));
    return true;
}

bool configureCamera(void* handle, const Args& args, const char* tag) {
    if (handle == nullptr) return false;
    if (!setEnumString(handle, "PixelFormat", "BayerRG8", tag, false) &&
        !setEnumValue(
            handle, "PixelFormat", PixelType_Gvsp_BayerRG8, tag, true)) {
        return false;
    }
    if (!setEnumString(handle, "ExposureAuto", "Off", tag, true) ||
        !setFloatValue(handle, "ExposureTime", args.exposure_us, tag, true) ||
        !setEnumString(handle, "GainAuto", "Off", tag, true) ||
        !setFloatValue(handle, "Gain", args.gain_db, tag, true) ||
        !setBoolValue(
            handle, "AcquisitionFrameRateEnable", false, tag, true)) {
        return false;
    }
    if (MV_CC_SetGrabStrategy(handle, MV_GrabStrategy_OneByOne) != MV_OK ||
        MV_CC_SetImageNodeNum(handle, 8) != MV_OK) {
        std::printf("[%s] failed to configure OneByOne/FIFO REQUIRED\n", tag);
        return false;
    }
    if (!setEnumString(handle, "TriggerMode", "On", tag, true) ||
        !setEnumString(handle, "TriggerSource", "Line0", tag, true) ||
        !setEnumString(
            handle, "TriggerActivation", "RisingEdge", tag, true)) {
        return false;
    }
    return enableFrameSpecInfo(handle, tag);
}

void printNodeReadback(void* handle, const char* tag) {
    auto print_enum = [&](const char* node) {
        unsigned int value = 0;
        if (getEnumValue(handle, node, value)) {
            std::printf("[%s] %-28s = %u\n", tag, node, value);
        } else {
            std::printf("[%s] %-28s = <read failed>\n", tag, node);
        }
    };
    auto print_bool = [&](const char* node) {
        bool value = false;
        if (getBoolValue(handle, node, value)) {
            std::printf("[%s] %-28s = %d\n", tag, node, value ? 1 : 0);
        } else {
            std::printf("[%s] %-28s = <read failed>\n", tag, node);
        }
    };
    auto print_float = [&](const char* node) {
        float value = 0.0F;
        if (getFloatValue(handle, node, value)) {
            std::printf("[%s] %-28s = %.3f\n", tag, node, value);
        } else {
            std::printf("[%s] %-28s = <read failed>\n", tag, node);
        }
    };

    print_enum("PixelFormat");
    print_enum("TriggerMode");
    print_enum("TriggerSource");
    print_enum("TriggerActivation");
    print_enum("ExposureAuto");
    print_float("ExposureTime");
    print_enum("GainAuto");
    print_float("Gain");
    print_bool("FrameSpecInfo");
    print_enum("FrameSpecInfoSelector");
    print_bool("ChunkModeActive");
    print_enum("ChunkSelector");
    print_bool("ChunkEnable");

    for (const char* item : {
             "Timestamp", "Framecounter", "ExtTriggerCount", "Exposure",
             "Gain", "LineInputOutput", "ROIPosition", "BrightnessInfo",
             "WhiteBalance"}) {
        if (MV_CC_SetEnumValueByString(
                handle, "FrameSpecInfoSelector", item) == MV_OK) {
            bool enabled = false;
            if (getBoolValue(handle, "FrameSpecInfo", enabled)) {
                std::printf(
                    "[%s] FrameSpecInfo[%s]        = %d\n",
                    tag,
                    item,
                    enabled ? 1 : 0);
            }
        }
    }
}

FrameMeta grabMeta(void* handle, int timeout_ms) {
    FrameMeta metadata;
    MV_FRAME_OUT frame {};
    const int grab_status = MV_CC_GetImageBuffer(
        handle, &frame, static_cast<unsigned int>(timeout_ms));
    metadata.ret = grab_status;
    if (grab_status != MV_OK) return metadata;

    const MV_FRAME_OUT_INFO_EX& info = frame.stFrameInfo;
    metadata.ok = true;
    metadata.frame_num = info.nFrameNum;
    metadata.dev_ts = deviceTimestamp(info);
    metadata.host_ts = static_cast<std::int64_t>(info.nHostTimeStamp);
    metadata.frame_counter = info.nFrameCounter;
    metadata.trigger_index = info.nTriggerIndex;
    metadata.second_count = info.nSecondCount;
    metadata.cycle_count = info.nCycleCount;
    metadata.cycle_offset = info.nCycleOffset;
    metadata.gain = info.fGain;
    metadata.exposure = info.fExposureTime;
    metadata.input = info.nInput;
    metadata.output = info.nOutput;
    metadata.unparsed_chunks = info.nUnparsedChunkNum;
    metadata.width = info.nExtendWidth != 0 ? info.nExtendWidth : info.nWidth;
    metadata.height = info.nExtendHeight != 0 ? info.nExtendHeight : info.nHeight;
    metadata.frame_len = info.nFrameLenEx != 0
        ? info.nFrameLenEx : info.nFrameLen;
    if (frame.pBufAddr != nullptr && metadata.frame_len > 0) {
        const auto byte_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                metadata.frame_len,
                sizeof(metadata.first_bytes)));
        std::memcpy(
            metadata.first_bytes,
            frame.pBufAddr,
            byte_count);
    }
    if (info.nUnparsedChunkNum > 0 &&
        info.UnparsedChunkList.pUnparsedChunkContent != nullptr) {
        const auto* chunk = info.UnparsedChunkList.pUnparsedChunkContent;
        std::printf(
            "    chunk0 id=0x%X len=%u data=%p\n",
            chunk[0].nChunkID,
            chunk[0].nChunkLen,
            static_cast<void*>(chunk[0].pChunkData));
    }
    const int free_status = MV_CC_FreeImageBuffer(handle, &frame);
    if (free_status != MV_OK) {
        metadata.ok = false;
        metadata.ret = free_status;
    }
    return metadata;
}

}  // namespace hik_metadata_probe
