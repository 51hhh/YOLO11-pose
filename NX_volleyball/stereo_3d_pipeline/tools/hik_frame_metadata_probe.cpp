#include "MvCameraControl.h"
#include "../src/calibration/pwm_trigger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <cmath>
#include <memory>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Args {
    std::string left_sn;
    std::string right_sn;
    int left_index = 0;
    int right_index = 1;
    int frames = 300;
    int timeout_ms = 1000;
    float exposure_us = 9867.0f;
    float gain_db = 11.99f;
    bool start_pwm = true;
    std::string trigger_chip = "gpiochip2";
    unsigned int trigger_line = 7;
    double pwm_hz = 100.0;
};

struct Cam {
    void* handle = nullptr;
    std::string tag;
    std::string serial;
};

struct FrameMeta {
    bool ok = false;
    int ret = MV_OK;
    uint32_t frame_num = 0;
    uint64_t dev_ts = 0;
    int64_t host_ts = 0;
    uint32_t frame_counter = 0;
    uint32_t trigger_index = 0;
    uint32_t second_count = 0;
    uint32_t cycle_count = 0;
    uint32_t cycle_offset = 0;
    float gain = 0.0f;
    float exposure = 0.0f;
    uint32_t input = 0;
    uint32_t output = 0;
    uint32_t unparsed_chunks = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t frame_len = 0;
    uint8_t first_bytes[16] = {};
};

struct RunningStats {
    int n = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();

    void add(double v) {
        ++n;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        const double delta = v - mean;
        mean += delta / static_cast<double>(n);
        const double delta2 = v - mean;
        m2 += delta * delta2;
    }

    double stddev() const {
        return n > 1 ? std::sqrt(m2 / static_cast<double>(n - 1)) : 0.0;
    }
};

void printIntervalStats(const char* label, const RunningStats& stats) {
    if (stats.n <= 0) {
        std::printf("  %s: no samples\n", label);
        return;
    }
    const double fps = stats.mean > 0.0 ? 1.0e9 / stats.mean : 0.0;
    std::printf("  %s: n=%d mean=%.1fns min=%.1f max=%.1f std=%.1f fps=%.3f\n",
                label, stats.n, stats.mean, stats.min_v, stats.max_v,
                stats.stddev(), fps);
}

void usage(const char* argv0) {
    std::printf(
        "Usage: %s [--left-sn SN] [--right-sn SN] [--left-index N] [--right-index N]\n"
        "          [--frames N] [--timeout-ms N] [--exposure-us US] [--gain-db DB]\n"
        "          [--no-pwm] [--trigger-chip gpiochip2] [--trigger-line 7] [--pwm-hz 100]\n",
        argv0);
}

bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--left-sn") args.left_sn = need("--left-sn");
        else if (a == "--right-sn") args.right_sn = need("--right-sn");
        else if (a == "--left-index") args.left_index = std::atoi(need("--left-index"));
        else if (a == "--right-index") args.right_index = std::atoi(need("--right-index"));
        else if (a == "--frames") args.frames = std::atoi(need("--frames"));
        else if (a == "--timeout-ms") args.timeout_ms = std::atoi(need("--timeout-ms"));
        else if (a == "--exposure-us") args.exposure_us = std::atof(need("--exposure-us"));
        else if (a == "--gain-db") args.gain_db = std::atof(need("--gain-db"));
        else if (a == "--trigger-chip") args.trigger_chip = need("--trigger-chip");
        else if (a == "--trigger-line") args.trigger_line = static_cast<unsigned int>(std::atoi(need("--trigger-line")));
        else if (a == "--pwm-hz") args.pwm_hz = std::atof(need("--pwm-hz"));
        else if (a == "--no-pwm") args.start_pwm = false;
        else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            usage(argv[0]);
            return false;
        }
    }
    return true;
}

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

bool changed(uint32_t a, uint32_t b) {
    return a != b;
}

uint32_t counterStep(uint32_t prev, uint32_t cur) {
    return cur - prev;
}

void printMetaLine(int i, const FrameMeta& l, const FrameMeta& r) {
    if (!l.ok || !r.ok) {
        std::printf("[%04d] grab fail L(ok=%d ret=0x%X) R(ok=%d ret=0x%X)\n",
                    i, l.ok ? 1 : 0, l.ret, r.ok ? 1 : 0, r.ret);
        return;
    }
    const int64_t frame_delta = static_cast<int64_t>(l.frame_num) - static_cast<int64_t>(r.frame_num);
    const int64_t counter_delta = static_cast<int64_t>(l.frame_counter) - static_cast<int64_t>(r.frame_counter);
    const int64_t trigger_delta = static_cast<int64_t>(l.trigger_index) - static_cast<int64_t>(r.trigger_index);
    const int64_t ts_delta = static_cast<int64_t>(l.dev_ts) - static_cast<int64_t>(r.dev_ts);
    const int64_t host_delta = l.host_ts - r.host_ts;
    std::printf(
        "[%04d] L#%u R#%u d#=%ld  Lfc=%u Rfc=%u dfc=%ld  Ltr=%u Rtr=%u dtr=%ld  "
        "dDevTs=%ld dHostTs=%ld  Lsc/cyc/off=%u/%u/%u R=%u/%u/%u  chunk=%u/%u  "
        "first=%02X%02X%02X%02X/%02X%02X%02X%02X\n",
        i,
        l.frame_num, r.frame_num, static_cast<long>(frame_delta),
        l.frame_counter, r.frame_counter, static_cast<long>(counter_delta),
        l.trigger_index, r.trigger_index, static_cast<long>(trigger_delta),
        static_cast<long>(ts_delta), static_cast<long>(host_delta),
        l.second_count, l.cycle_count, l.cycle_offset,
        r.second_count, r.cycle_count, r.cycle_offset,
        l.unparsed_chunks, r.unparsed_chunks,
        l.first_bytes[0], l.first_bytes[1], l.first_bytes[2], l.first_bytes[3],
        r.first_bytes[0], r.first_bytes[1], r.first_bytes[2], r.first_bytes[3]);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    Args args;
    if (!parseArgs(argc, argv, args)) return 2;

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
