#include "MvCameraControl.h"
#include "hik_frame_metadata_probe_args.h"
#include "hik_frame_metadata_probe_stats.h"
#include "../src/calibration/pwm_trigger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

using hik_metadata_probe::Args;
using hik_metadata_probe::FrameMeta;
using hik_metadata_probe::RunningStats;
using hik_metadata_probe::changed;
using hik_metadata_probe::counterStep;
using hik_metadata_probe::printIntervalStats;
using hik_metadata_probe::printMetaLine;

struct Cam {
    void* handle = nullptr;
    std::string tag;
    std::string serial;
};

uint64_t devTimestamp(const MV_FRAME_OUT_INFO_EX& info) {
    return (static_cast<uint64_t>(info.nDevTimeStampHigh) << 32) |
           static_cast<uint64_t>(info.nDevTimeStampLow);
}

std::string serialOf(const MV_CC_DEVICE_INFO* info) {
    if (!info) return {};
    if (info->nTLayerType == MV_USB_DEVICE) {
        return reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber);
    }
    if (info->nTLayerType == MV_GIGE_DEVICE) {
        return reinterpret_cast<const char*>(info->SpecialInfo.stGigEInfo.chSerialNumber);
    }
    return {};
}

std::string modelOf(const MV_CC_DEVICE_INFO* info) {
    if (!info) return {};
    if (info->nTLayerType == MV_USB_DEVICE) {
        return reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chModelName);
    }
    if (info->nTLayerType == MV_GIGE_DEVICE) {
        return reinterpret_cast<const char*>(info->SpecialInfo.stGigEInfo.chModelName);
    }
    return {};
}

const char* layerName(const MV_CC_DEVICE_INFO* info) {
    if (!info) return "unknown";
    if (info->nTLayerType == MV_USB_DEVICE) return "USB";
    if (info->nTLayerType == MV_GIGE_DEVICE) return "GigE";
    return "other";
}

bool setEnumString(void* h, const char* node, const char* value, const char* tag, bool required = false) {
    int ret = MV_CC_SetEnumValueByString(h, node, value);
    if (ret != MV_OK) {
        std::printf("[%s] set %s=%s failed ret=0x%X%s\n",
                    tag, node, value, ret, required ? " REQUIRED" : "");
        return !required;
    }
    return true;
}

bool setEnumValue(void* h, const char* node, unsigned int value, const char* tag, bool required = false) {
    int ret = MV_CC_SetEnumValue(h, node, value);
    if (ret != MV_OK) {
        std::printf("[%s] set %s=%u failed ret=0x%X%s\n",
                    tag, node, value, ret, required ? " REQUIRED" : "");
        return !required;
    }
    return true;
}

bool setBoolValue(void* h, const char* node, bool value, const char* tag, bool required = false) {
    int ret = MV_CC_SetBoolValue(h, node, value);
    if (ret != MV_OK) {
        std::printf("[%s] set %s=%d failed ret=0x%X%s\n",
                    tag, node, value ? 1 : 0, ret, required ? " REQUIRED" : "");
        return !required;
    }
    return true;
}

bool setFloatValue(void* h, const char* node, float value, const char* tag, bool required = false) {
    int ret = MV_CC_SetFloatValue(h, node, value);
    if (ret != MV_OK) {
        std::printf("[%s] set %s=%.3f failed ret=0x%X%s\n",
                    tag, node, value, ret, required ? " REQUIRED" : "");
        return !required;
    }
    return true;
}

bool getEnumValue(void* h, const char* node, unsigned int& value) {
    MVCC_ENUMVALUE v{};
    int ret = MV_CC_GetEnumValue(h, node, &v);
    if (ret != MV_OK) return false;
    value = v.nCurValue;
    return true;
}

bool getBoolValue(void* h, const char* node, bool& value) {
    bool v = false;
    int ret = MV_CC_GetBoolValue(h, node, &v);
    if (ret != MV_OK) return false;
    value = v;
    return true;
}

bool getFloatValue(void* h, const char* node, float& value) {
    MVCC_FLOATVALUE v{};
    int ret = MV_CC_GetFloatValue(h, node, &v);
    if (ret != MV_OK) return false;
    value = v.fCurValue;
    return true;
}

void printNodeReadback(void* h, const char* tag) {
    auto print_enum = [&](const char* node) {
        unsigned int value = 0;
        if (getEnumValue(h, node, value)) std::printf("[%s] %-28s = %u\n", tag, node, value);
        else std::printf("[%s] %-28s = <read failed>\n", tag, node);
    };
    auto print_bool = [&](const char* node) {
        bool value = false;
        if (getBoolValue(h, node, value)) std::printf("[%s] %-28s = %d\n", tag, node, value ? 1 : 0);
        else std::printf("[%s] %-28s = <read failed>\n", tag, node);
    };
    auto print_float = [&](const char* node) {
        float value = 0.0f;
        if (getFloatValue(h, node, value)) std::printf("[%s] %-28s = %.3f\n", tag, node, value);
        else std::printf("[%s] %-28s = <read failed>\n", tag, node);
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

    const char* spec_items[] = {
        "Timestamp", "Framecounter", "ExtTriggerCount", "Exposure",
        "Gain", "LineInputOutput", "ROIPosition", "BrightnessInfo", "WhiteBalance"
    };
    for (const char* item : spec_items) {
        if (MV_CC_SetEnumValueByString(h, "FrameSpecInfoSelector", item) == MV_OK) {
            bool enabled = false;
            if (getBoolValue(h, "FrameSpecInfo", enabled)) {
                std::printf("[%s] FrameSpecInfo[%s]        = %d\n", tag, item, enabled ? 1 : 0);
            }
        }
    }
}

bool enableFrameSpecInfo(void* h, const char* tag) {
    const char* spec_items[] = {
        "Timestamp", "Framecounter", "ExtTriggerCount", "Exposure",
        "Gain", "LineInputOutput", "ROIPosition"
    };
    bool any = false;
    for (const char* item : spec_items) {
        int ret = MV_CC_SetEnumValueByString(h, "FrameSpecInfoSelector", item);
        if (ret != MV_OK) {
            std::printf("[%s] select FrameSpecInfoSelector=%s failed ret=0x%X\n", tag, item, ret);
            continue;
        }
        ret = MV_CC_SetBoolValue(h, "FrameSpecInfo", true);
        if (ret != MV_OK) {
            std::printf("[%s] enable FrameSpecInfo[%s] failed ret=0x%X\n", tag, item, ret);
            continue;
        }
        any = true;
    }
    return any;
}

bool configureCamera(void* h, const Args& args, const char* tag) {
    setEnumString(h, "PixelFormat", "BayerRG8", tag);
    setEnumValue(h, "PixelFormat", PixelType_Gvsp_BayerRG8, tag);
    setEnumString(h, "ExposureAuto", "Off", tag);
    setFloatValue(h, "ExposureTime", args.exposure_us, tag);
    setEnumString(h, "GainAuto", "Off", tag);
    setFloatValue(h, "Gain", args.gain_db, tag);
    setBoolValue(h, "AcquisitionFrameRateEnable", false, tag);
    MV_CC_SetGrabStrategy(h, MV_GrabStrategy_OneByOne);
    MV_CC_SetImageNodeNum(h, 8);

    if (!setEnumString(h, "TriggerMode", "On", tag, true)) return false;
    if (!setEnumString(h, "TriggerSource", "Line0", tag, true)) return false;
    if (!setEnumString(h, "TriggerActivation", "RisingEdge", tag, true)) return false;

    if (!enableFrameSpecInfo(h, tag)) {
        std::printf("[%s] no FrameSpecInfo selector could be enabled\n", tag);
    }
    return true;
}

bool openCamera(const MV_CC_DEVICE_INFO_LIST& list,
                const std::string& wanted_sn,
                int index,
                const char* tag,
                Cam& cam) {
    MV_CC_DEVICE_INFO* selected = nullptr;
    if (!wanted_sn.empty()) {
        for (unsigned int i = 0; i < list.nDeviceNum; ++i) {
            if (serialOf(list.pDeviceInfo[i]) == wanted_sn) {
                selected = list.pDeviceInfo[i];
                break;
            }
        }
    } else if (index >= 0 && static_cast<unsigned int>(index) < list.nDeviceNum) {
        selected = list.pDeviceInfo[index];
    }

    if (!selected) {
        std::fprintf(stderr, "[%s] camera not found, sn='%s' index=%d\n",
                     tag, wanted_sn.c_str(), index);
        return false;
    }

    cam.tag = tag;
    cam.serial = serialOf(selected);
    int ret = MV_CC_CreateHandle(&cam.handle, selected);
    if (ret != MV_OK) {
        std::fprintf(stderr, "[%s] CreateHandle failed ret=0x%X\n", tag, ret);
        return false;
    }
    ret = MV_CC_OpenDevice(cam.handle);
    if (ret != MV_OK) {
        std::fprintf(stderr, "[%s] OpenDevice failed ret=0x%X\n", tag, ret);
        MV_CC_DestroyHandle(cam.handle);
        cam.handle = nullptr;
        return false;
    }

    std::printf("[%s] opened SN=%s model=%s layer=%s\n",
                tag, cam.serial.c_str(), modelOf(selected).c_str(), layerName(selected));
    return true;
}

FrameMeta grabMeta(void* handle, int timeout_ms) {
    FrameMeta meta;
    MV_FRAME_OUT frame{};
    int ret = MV_CC_GetImageBuffer(handle, &frame, timeout_ms);
    meta.ret = ret;
    if (ret != MV_OK) return meta;

    const MV_FRAME_OUT_INFO_EX& info = frame.stFrameInfo;
    meta.ok = true;
    meta.frame_num = info.nFrameNum;
    meta.dev_ts = devTimestamp(info);
    meta.host_ts = info.nHostTimeStamp;
    meta.frame_counter = info.nFrameCounter;
    meta.trigger_index = info.nTriggerIndex;
    meta.second_count = info.nSecondCount;
    meta.cycle_count = info.nCycleCount;
    meta.cycle_offset = info.nCycleOffset;
    meta.gain = info.fGain;
    meta.exposure = info.fExposureTime;
    meta.input = info.nInput;
    meta.output = info.nOutput;
    meta.unparsed_chunks = info.nUnparsedChunkNum;
    meta.width = info.nExtendWidth ? info.nExtendWidth : info.nWidth;
    meta.height = info.nExtendHeight ? info.nExtendHeight : info.nHeight;
    meta.frame_len = info.nFrameLenEx ? info.nFrameLenEx : info.nFrameLen;
    if (frame.pBufAddr) {
        std::memcpy(meta.first_bytes, frame.pBufAddr, sizeof(meta.first_bytes));
    }
    if (info.nUnparsedChunkNum > 0 && info.UnparsedChunkList.pUnparsedChunkContent) {
        const auto* chunk = info.UnparsedChunkList.pUnparsedChunkContent;
        std::printf("    chunk0 id=0x%X len=%u data=%p\n",
                    chunk[0].nChunkID, chunk[0].nChunkLen,
                    static_cast<void*>(chunk[0].pChunkData));
    }
    MV_CC_FreeImageBuffer(handle, &frame);
    return meta;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    Args args;
    if (!hik_metadata_probe::parseArgs(argc, argv, args)) return 2;

    MV_CC_DEVICE_INFO_LIST list{};
    int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &list);
    if (ret != MV_OK || list.nDeviceNum == 0) {
        std::fprintf(stderr, "EnumDevices failed ret=0x%X count=%u\n", ret, list.nDeviceNum);
        return 1;
    }
    std::printf("Enumerated %u devices\n", list.nDeviceNum);
    for (unsigned int i = 0; i < list.nDeviceNum; ++i) {
        std::printf("  [%u] SN=%s model=%s layer=%s\n",
                    i, serialOf(list.pDeviceInfo[i]).c_str(),
                    modelOf(list.pDeviceInfo[i]).c_str(),
                    layerName(list.pDeviceInfo[i]));
    }
    if (list.nDeviceNum < 2) {
        std::fprintf(stderr,
                     "ERROR: stereo metadata probe requires two enumerated cameras, got %u.\n"
                     "Check USB connection/power, close MVS or any process holding a camera, "
                     "then rerun the probe.\n",
                     list.nDeviceNum);
        return 1;
    }

    Cam left, right;
    if (!openCamera(list, args.left_sn, args.left_index, "LEFT", left)) {
        return 1;
    }
    if (!openCamera(list, args.right_sn, args.right_index, "RIGHT", right)) {
        return 1;
    }

    if (!configureCamera(left.handle, args, "LEFT") ||
        !configureCamera(right.handle, args, "RIGHT")) {
        return 1;
    }

    std::puts("\nNode readback after configuration:");
    printNodeReadback(left.handle, "LEFT");
    printNodeReadback(right.handle, "RIGHT");

    ret = MV_CC_StartGrabbing(left.handle);
    if (ret != MV_OK) {
        std::fprintf(stderr, "[LEFT] StartGrabbing failed ret=0x%X\n", ret);
        return 1;
    }
    ret = MV_CC_StartGrabbing(right.handle);
    if (ret != MV_OK) {
        std::fprintf(stderr, "[RIGHT] StartGrabbing failed ret=0x%X\n", ret);
        return 1;
    }

    std::unique_ptr<stereo3d::PWMTrigger> pwm;
    if (args.start_pwm) {
        pwm.reset(new stereo3d::PWMTrigger(args.trigger_chip, args.trigger_line, args.pwm_hz, 50.0));
        if (!pwm->start()) {
            std::fprintf(stderr, "PWM start failed, continuing; external trigger must already be present\n");
        }
    }

    int ok_pairs = 0;
    int fail_pairs = 0;
    int frame_delta_changes = 0;
    int counter_delta_changes = 0;
    int trigger_delta_changes = 0;
    int trigger_nonzero_or_changing = 0;
    int counter_nonzero_or_changing = 0;
    int l_frame_num_jumps = 0;
    int r_frame_num_jumps = 0;
    int l_frame_counter_jumps = 0;
    int r_frame_counter_jumps = 0;
    int l_trigger_index_jumps = 0;
    int r_trigger_index_jumps = 0;
    uint32_t l_max_frame_num_step = 0;
    uint32_t r_max_frame_num_step = 0;
    uint32_t l_max_frame_counter_step = 0;
    uint32_t r_max_frame_counter_step = 0;
    uint32_t l_max_trigger_index_step = 0;
    uint32_t r_max_trigger_index_step = 0;
    RunningStats l_dev_ts_step_stats;
    RunningStats r_dev_ts_step_stats;
    RunningStats l_host_ts_step_stats;
    RunningStats r_host_ts_step_stats;
    RunningStats dev_ts_delta_stats;
    int64_t first_frame_delta = 0;
    int64_t first_counter_delta = 0;
    int64_t first_trigger_delta = 0;
    bool have_first = false;
    FrameMeta prev_l, prev_r;

    std::puts("\nFrame metadata:");
    for (int i = 0; i < args.frames; ++i) {
        FrameMeta l, r;
        std::thread rt([&]() { r = grabMeta(right.handle, args.timeout_ms); });
        l = grabMeta(left.handle, args.timeout_ms);
        rt.join();

        const bool print_this = i < 30 || !l.ok || !r.ok;
        if (print_this) printMetaLine(i, l, r);

        if (!l.ok || !r.ok) {
            ++fail_pairs;
            continue;
        }
        ++ok_pairs;

        if (!have_first) {
            first_frame_delta = static_cast<int64_t>(l.frame_num) - static_cast<int64_t>(r.frame_num);
            first_counter_delta = static_cast<int64_t>(l.frame_counter) - static_cast<int64_t>(r.frame_counter);
            first_trigger_delta = static_cast<int64_t>(l.trigger_index) - static_cast<int64_t>(r.trigger_index);
            have_first = true;
        } else {
            const int64_t frame_delta = static_cast<int64_t>(l.frame_num) - static_cast<int64_t>(r.frame_num);
            const int64_t counter_delta = static_cast<int64_t>(l.frame_counter) - static_cast<int64_t>(r.frame_counter);
            const int64_t trigger_delta = static_cast<int64_t>(l.trigger_index) - static_cast<int64_t>(r.trigger_index);
            if (frame_delta != first_frame_delta) {
                ++frame_delta_changes;
                printMetaLine(i, l, r);
            }
            if (counter_delta != first_counter_delta) {
                ++counter_delta_changes;
                printMetaLine(i, l, r);
            }
            if (trigger_delta != first_trigger_delta) {
                ++trigger_delta_changes;
                printMetaLine(i, l, r);
            }
            if (changed(l.trigger_index, prev_l.trigger_index) ||
                changed(r.trigger_index, prev_r.trigger_index) ||
                l.trigger_index != 0 || r.trigger_index != 0) {
                ++trigger_nonzero_or_changing;
            }
            if (changed(l.frame_counter, prev_l.frame_counter) ||
                changed(r.frame_counter, prev_r.frame_counter) ||
                l.frame_counter != 0 || r.frame_counter != 0) {
                ++counter_nonzero_or_changing;
            }

            const uint32_t l_frame_num_step = counterStep(prev_l.frame_num, l.frame_num);
            const uint32_t r_frame_num_step = counterStep(prev_r.frame_num, r.frame_num);
            const uint32_t l_frame_counter_step = counterStep(prev_l.frame_counter, l.frame_counter);
            const uint32_t r_frame_counter_step = counterStep(prev_r.frame_counter, r.frame_counter);
            const uint32_t l_trigger_index_step = counterStep(prev_l.trigger_index, l.trigger_index);
            const uint32_t r_trigger_index_step = counterStep(prev_r.trigger_index, r.trigger_index);
            if (l.dev_ts >= prev_l.dev_ts) {
                l_dev_ts_step_stats.add(static_cast<double>(l.dev_ts - prev_l.dev_ts));
            }
            if (r.dev_ts >= prev_r.dev_ts) {
                r_dev_ts_step_stats.add(static_cast<double>(r.dev_ts - prev_r.dev_ts));
            }
            if (l.host_ts >= prev_l.host_ts) {
                l_host_ts_step_stats.add(static_cast<double>(l.host_ts - prev_l.host_ts));
            }
            if (r.host_ts >= prev_r.host_ts) {
                r_host_ts_step_stats.add(static_cast<double>(r.host_ts - prev_r.host_ts));
            }
            dev_ts_delta_stats.add(static_cast<double>(
                static_cast<int64_t>(l.dev_ts) - static_cast<int64_t>(r.dev_ts)));
            l_max_frame_num_step = std::max(l_max_frame_num_step, l_frame_num_step);
            r_max_frame_num_step = std::max(r_max_frame_num_step, r_frame_num_step);
            l_max_frame_counter_step = std::max(l_max_frame_counter_step, l_frame_counter_step);
            r_max_frame_counter_step = std::max(r_max_frame_counter_step, r_frame_counter_step);
            l_max_trigger_index_step = std::max(l_max_trigger_index_step, l_trigger_index_step);
            r_max_trigger_index_step = std::max(r_max_trigger_index_step, r_trigger_index_step);
            bool step_anomaly = false;
            if (l_frame_num_step != 1) { ++l_frame_num_jumps; step_anomaly = true; }
            if (r_frame_num_step != 1) { ++r_frame_num_jumps; step_anomaly = true; }
            if (l_frame_counter_step != 1) { ++l_frame_counter_jumps; step_anomaly = true; }
            if (r_frame_counter_step != 1) { ++r_frame_counter_jumps; step_anomaly = true; }
            if (l_trigger_index_step != 1) { ++l_trigger_index_jumps; step_anomaly = true; }
            if (r_trigger_index_step != 1) { ++r_trigger_index_jumps; step_anomaly = true; }
            if (step_anomaly) {
                std::printf("    step anomaly at %04d: L# +%u R# +%u  Lfc +%u Rfc +%u  Ltr +%u Rtr +%u\n",
                            i,
                            l_frame_num_step, r_frame_num_step,
                            l_frame_counter_step, r_frame_counter_step,
                            l_trigger_index_step, r_trigger_index_step);
            }
        }
        prev_l = l;
        prev_r = r;
    }

    if (pwm) pwm->stop();
    MV_CC_StopGrabbing(left.handle);
    MV_CC_StopGrabbing(right.handle);
    MV_CC_CloseDevice(left.handle);
    MV_CC_CloseDevice(right.handle);
    MV_CC_DestroyHandle(left.handle);
    MV_CC_DestroyHandle(right.handle);

    std::puts("\nSummary:");
    std::printf("  ok_pairs=%d fail_pairs=%d\n", ok_pairs, fail_pairs);
    std::printf("  first_frame_delta=%ld changes=%d\n",
                static_cast<long>(first_frame_delta), frame_delta_changes);
    std::printf("  first_frame_counter_delta=%ld changes=%d nonzero_or_changing_samples=%d\n",
                static_cast<long>(first_counter_delta), counter_delta_changes,
                counter_nonzero_or_changing);
    std::printf("  first_trigger_delta=%ld changes=%d nonzero_or_changing_samples=%d\n",
                static_cast<long>(first_trigger_delta), trigger_delta_changes,
                trigger_nonzero_or_changing);
    std::printf("  per-camera jumps: frame_num L/R=%d/%d max_step=%u/%u\n",
                l_frame_num_jumps, r_frame_num_jumps,
                l_max_frame_num_step, r_max_frame_num_step);
    std::printf("  per-camera jumps: frame_counter L/R=%d/%d max_step=%u/%u\n",
                l_frame_counter_jumps, r_frame_counter_jumps,
                l_max_frame_counter_step, r_max_frame_counter_step);
    std::printf("  per-camera jumps: trigger_index L/R=%d/%d max_step=%u/%u\n",
                l_trigger_index_jumps, r_trigger_index_jumps,
                l_max_trigger_index_step, r_max_trigger_index_step);
    printIntervalStats("left_dev_timestamp_step", l_dev_ts_step_stats);
    printIntervalStats("right_dev_timestamp_step", r_dev_ts_step_stats);
    printIntervalStats("left_host_timestamp_step", l_host_ts_step_stats);
    printIntervalStats("right_host_timestamp_step", r_host_ts_step_stats);
    printIntervalStats("left_minus_right_dev_timestamp", dev_ts_delta_stats);
    std::printf("  verdict: frame_counter=%s trigger_index=%s\n",
                counter_nonzero_or_changing > 0 ? "usable-looking" : "not populated",
                trigger_nonzero_or_changing > 0 ? "usable-looking" : "not populated");
    return 0;
}
