/**
 * @file hikvision_camera.cpp
 * @brief 海康工业相机采集实现
 *
 * 核心设计:
 *   SDK GetImageBuffer → memcpy 到有界 host/pinned buffer → FreeImageBuffer
 *   不做 ISP / 格式转换, 原始 BayerRG8 输出
 *   ISP 由后续 CUDA 在 GPU 上完成
 */

#include "volleyball/adapters/hik_mvs/hikvision_camera.hpp"

#include "mvs_helpers.hpp"
#include "volleyball/logging/log.hpp"
#include "MvCameraControl.h"

#include <algorithm>
#include <cstring>
#include <chrono>
#include <thread>
#include <cmath>
#include <limits>

namespace volleyball::adapters::hik_mvs {

namespace {

struct CameraGeometry {
    int width = 0;
    int height = 0;
    int offset_x = 0;
    int offset_y = 0;
};

bool readCameraGeometry(void* handle, const char* tag, CameraGeometry& geometry) {
    auto read = [&](const char* node, int& output) {
        MVCC_INTVALUE value;
        std::memset(&value, 0, sizeof(value));
        const int ret = MV_CC_GetIntValue(handle, node, &value);
        if (ret != MV_OK ||
            value.nCurValue > static_cast<std::uint64_t>(
                std::numeric_limits<int>::max())) {
            LOG_ERROR("[HikCam][%s] Get %s failed or overflowed, ret=0x%X",
                      tag, node, ret);
            return false;
        }
        output = static_cast<int>(value.nCurValue);
        return true;
    };
    return read("Width", geometry.width) &&
           read("Height", geometry.height) &&
           read("OffsetX", geometry.offset_x) &&
           read("OffsetY", geometry.offset_y);
}

bool geometryMatchesConfig(const CameraGeometry& geometry,
                           const CameraConfig& config) {
    return geometry.width == config.width &&
           geometry.height == config.height &&
           geometry.offset_x == 0 && geometry.offset_y == 0;
}

}  // namespace

// ==================== 构造 / 析构 ====================

HikvisionCamera::HikvisionCamera() = default;

HikvisionCamera::~HikvisionCamera() {
    close();
}

bool HikvisionCamera::configureCamera(void* handle, const CameraConfig& cfg, const char* tag) {
    // 像素格式: BayerRG8 (无ISP, 后续CUDA处理)
    if (!setEnumChecked(handle, "PixelFormat", PixelType_Gvsp_BayerRG8, tag)) return false;

    // 曝光
    if (cfg.auto_exposure) {
        if (!setEnumOptional(handle, "ExposureAuto", 2, tag))
            setEnumStringOptional(handle, "ExposureAuto", "Continuous", tag);
        setFloatOptional(handle, "AutoExposureTimeUpperLimit", cfg.ae_upper_us, tag);
        setFloatOptional(handle, "AutoExposureTimeLowerLimit", cfg.ae_lower_us, tag);
    } else {
        if (!setEnumOptional(handle, "ExposureAuto", 0, tag))
            setEnumStringOptional(handle, "ExposureAuto", "Off", tag);
        if (!setFloatChecked(handle, "ExposureTime", cfg.exposure_us, tag)) return false;
    }

    // 增益
    if (cfg.auto_gain) {
        if (!setEnumOptional(handle, "GainAuto", 2, tag))
            setEnumStringOptional(handle, "GainAuto", "Continuous", tag);
        setFloatOptional(handle, "AutoGainUpperLimit", cfg.ag_upper_db, tag);
        setFloatOptional(handle, "AutoGainLowerLimit", 0.0f, tag);
    } else {
        if (!setEnumOptional(handle, "GainAuto", 0, tag))
            setEnumStringOptional(handle, "GainAuto", "Off", tag);
        if (!setFloatChecked(handle, "Gain", cfg.gain_db, tag)) return false;
    }

    // Gamma 校正: 压缩高亮区域动态范围
    if (cfg.gamma_enable) {
        setBoolOptional(handle, "GammaEnable", true, tag);
        setEnumOptional(handle, "GammaSelector", 1, tag);  // User
        setFloatOptional(handle, "Gamma", cfg.gamma_value, tag);
    } else {
        setBoolOptional(handle, "GammaEnable", false, tag);
    }

    // 硬触发下关闭内部帧率限制。部分机型不开放该节点, 失败只报警。
    setBoolOptional(handle, "AcquisitionFrameRateEnable", false, tag);

    // 硬触发双目优先保持左右触发序列一致。LatestImagesOnly 会让两台相机
    // 独立丢弃旧帧, 容易出现 L#n/R#n+1 的错配; 小 FIFO 能吸收 USB 抖动。
    int ret = MV_CC_SetGrabStrategy(handle, MV_GrabStrategy_OneByOne);
    if (ret != MV_OK) {
        LOG_ERROR("[HikCam][%s] Set GrabStrategy=OneByOne failed, ret=0x%X",
                  tag, ret);
        return false;
    }
    const unsigned int image_node_num =
        static_cast<unsigned int>(std::clamp(cfg.image_node_num, 2, 16));
    ret = MV_CC_SetImageNodeNum(handle, image_node_num);
    if (ret != MV_OK) {
        LOG_ERROR("[HikCam][%s] Set ImageNodeNum=%u failed, ret=0x%X",
                  tag, image_node_num, ret);
        return false;
    }

    // 触发模式
    if (cfg.use_trigger) {
        if (!setEnumByValueOrString(handle, "TriggerMode", 1, "On", tag)) return false;

        if (cfg.trigger_source == "Line0")
            { if (!setEnumByValueOrString(handle, "TriggerSource", 0, "Line0", tag)) return false; }
        else if (cfg.trigger_source == "Line1")
            { if (!setEnumByValueOrString(handle, "TriggerSource", 1, "Line1", tag)) return false; }
        else if (cfg.trigger_source == "Software")
            { if (!setEnumByValueOrString(handle, "TriggerSource", 7, "Software", tag)) return false; }
        else {
            LOG_ERROR("[HikCam][%s] Unsupported trigger_source: %s",
                      tag, cfg.trigger_source.c_str());
            return false;
        }

        if (cfg.trigger_activation == "RisingEdge")
            { if (!setEnumByValueOrString(handle, "TriggerActivation", 0, "RisingEdge", tag)) return false; }
        else if (cfg.trigger_activation == "FallingEdge")
            { if (!setEnumByValueOrString(handle, "TriggerActivation", 1, "FallingEdge", tag)) return false; }
        else {
            LOG_ERROR("[HikCam][%s] Unsupported trigger_activation: %s",
                      tag, cfg.trigger_activation.c_str());
            return false;
        }
    } else {
        if (!setEnumByValueOrString(handle, "TriggerMode", 0, "Off", tag)) return false;
    }

    if (!configureMinimalFrameSpecInfo(handle, tag, cfg.use_trigger)) return false;

    unsigned int exposure_auto = 0;
    unsigned int gain_auto = 0;
    unsigned int trigger_mode = 0;
    unsigned int trigger_source = 0;
    unsigned int trigger_activation = 0;
    float exposure = 0.0f;
    float gain = 0.0f;
    bool gamma_enable = false;
    const bool have_exposure_auto = getEnumValue(handle, "ExposureAuto", exposure_auto);
    const bool have_exposure = getFloatValue(handle, "ExposureTime", exposure);
    const bool have_gain_auto = getEnumValue(handle, "GainAuto", gain_auto);
    const bool have_gain = getFloatValue(handle, "Gain", gain);
    const bool have_gamma = getBoolValue(handle, "GammaEnable", gamma_enable);
    const bool have_trigger_mode = getEnumValue(handle, "TriggerMode", trigger_mode);
    const bool have_trigger_source =
        !cfg.use_trigger || getEnumValue(handle, "TriggerSource", trigger_source);
    const bool have_trigger_activation =
        !cfg.use_trigger || getEnumValue(handle, "TriggerActivation", trigger_activation);

    if (!have_exposure || !have_gain || !have_trigger_mode || !have_trigger_source) {
        LOG_ERROR("[HikCam][%s] Critical readback failed: ExposureTime=%d, Gain=%d, "
                  "TriggerMode=%d, TriggerSource=%d",
                  tag, have_exposure, have_gain, have_trigger_mode, have_trigger_source);
        return false;
    }
    if (!have_exposure_auto || !have_gain_auto || !have_gamma || !have_trigger_activation) {
        LOG_WARN("[HikCam][%s] Optional readback missing: ExposureAuto=%d, GainAuto=%d, "
                 "GammaEnable=%d, TriggerActivation=%d",
                 tag, have_exposure_auto, have_gain_auto, have_gamma, have_trigger_activation);
    }

    const bool exposure_ok = cfg.auto_exposure || closeEnough(exposure, cfg.exposure_us, 10.0f);
    const bool gain_ok = cfg.auto_gain || closeEnough(gain, cfg.gain_db, 0.2f);
    const bool trigger_mode_ok = cfg.use_trigger ? (trigger_mode == 1) : (trigger_mode == 0);
    bool trigger_source_ok = true;
    bool trigger_activation_ok = true;
    if (cfg.use_trigger) {
        const unsigned int expected_source =
            (cfg.trigger_source == "Line0") ? 0 :
            (cfg.trigger_source == "Line1") ? 1 : 7;
        const unsigned int expected_activation =
            (cfg.trigger_activation == "RisingEdge") ? 0 : 1;
        trigger_source_ok = (trigger_source == expected_source);
        trigger_activation_ok = !have_trigger_activation ||
                                (trigger_activation == expected_activation);
    }
    if (!exposure_ok || !gain_ok || !trigger_mode_ok ||
        !trigger_source_ok || !trigger_activation_ok) {
        LOG_WARN("[HikCam][%s] Readback mismatch: exp=%.1fus(target %.1f), "
                 "gain=%.2fdB(target %.2f), triggerMode=%u, triggerSource=%u, "
                 "triggerActivation=%u",
                 tag, exposure, cfg.exposure_us, gain, cfg.gain_db,
                 trigger_mode, trigger_source, trigger_activation);
        return false;
    }

    if (cfg.use_trigger) {
        LOG_INFO("[HikCam][%s] Readback: %dx%d, ExposureAuto=%s%u, ExposureTime=%.1fus, "
                 "GainAuto=%s%u, Gain=%.2fdB, GammaEnable=%s%d, TriggerMode=%u, "
                 "TriggerSource=%u(%s), TriggerActivation=%s%u(%s)",
                 tag, cfg.width, cfg.height, have_exposure_auto ? "" : "?", exposure_auto, exposure,
                 have_gain_auto ? "" : "?", gain_auto, gain,
                 have_gamma ? "" : "?", gamma_enable ? 1 : 0,
                 trigger_mode, trigger_source, triggerSourceName(trigger_source),
                 have_trigger_activation ? "" : "?", trigger_activation,
                 triggerActivationName(trigger_activation));
    } else {
        LOG_INFO("[HikCam][%s] Readback: %dx%d, ExposureAuto=%s%u, ExposureTime=%.1fus, "
                 "GainAuto=%s%u, Gain=%.2fdB, GammaEnable=%s%d, TriggerMode=%u(FreeRun)",
                 tag, cfg.width, cfg.height, have_exposure_auto ? "" : "?", exposure_auto, exposure,
                 have_gain_auto ? "" : "?", gain_auto, gain,
                 have_gamma ? "" : "?", gamma_enable ? 1 : 0, trigger_mode);
    }

    LOG_INFO("[HikCam][%s] Configured before grabbing: %dx%d, exp=%s%.0fus, "
             "gain=%s%.1fdB, gamma=%s, trigger=%s/%s, fifo=%u, clear_rows=%d",
             tag, cfg.width, cfg.height,
             cfg.auto_exposure ? "auto/" : "", cfg.exposure_us,
             cfg.auto_gain ? "auto/" : "", cfg.gain_db,
             cfg.gamma_enable ? "on" : "off",
             cfg.use_trigger ? cfg.trigger_source.c_str() : "FreeRun",
             cfg.use_trigger ? cfg.trigger_activation.c_str() : "None",
             image_node_num,
             std::max(0, cfg.embedded_info_clear_rows));
    return true;
}

// ==================== open / close ====================

bool HikvisionCamera::open(const CameraConfig& cfg) {
    if (opened_) close();
    config_ = cfg;
    resetSyncState();

    MV_CC_DEVICE_INFO_LIST devList;
    memset(&devList, 0, sizeof(devList));
    int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &devList);
    if (MV_OK != nRet || devList.nDeviceNum == 0) {
        LOG_ERROR("[HikCam] EnumDevices failed or no device found");
        return false;
    }
    LOG_INFO("[HikCam] Enumerated %u camera device(s)", devList.nDeviceNum);
    if (cfg.serial_left.empty() || cfg.serial_right.empty()) {
        LOG_WARN("[HikCam] Opening cameras by index because serial_left/right is not fully set. "
                 "USB enumeration order can swap LEFT/RIGHT; set camera serials for stable tests.");
    }

    MV_CC_DEVICE_INFO* leftInfo = nullptr;
    MV_CC_DEVICE_INFO* rightInfo = nullptr;
    if (!selectDeviceInfo(devList, cfg.camera_index_left, cfg.serial_left,
                          "LEFT", leftInfo) ||
        !selectDeviceInfo(devList, cfg.camera_index_right, cfg.serial_right,
                          "RIGHT", rightInfo)) {
        return false;
    }
    if (leftInfo == rightInfo) {
        LOG_ERROR("[HikCam] LEFT and RIGHT resolved to the same device (SN=%s)",
                  deviceSerial(leftInfo).c_str());
        return false;
    }

    if (!createAndOpenHandle(handle_left_, leftInfo, "LEFT")) {
        LOG_ERROR("[HikCam] Failed to open LEFT camera");
        return false;
    }
    if (!configureCamera(handle_left_, cfg, "LEFT")) {
        LOG_ERROR("[HikCam] Failed to configure LEFT camera");
        MV_CC_CloseDevice(handle_left_);
        MV_CC_DestroyHandle(handle_left_);
        handle_left_ = nullptr;
        return false;
    }

    if (!createAndOpenHandle(handle_right_, rightInfo, "RIGHT")) {
        LOG_ERROR("[HikCam] Failed to open RIGHT camera");
        if (handle_right_) { MV_CC_DestroyHandle(handle_right_); handle_right_ = nullptr; }
        MV_CC_CloseDevice(handle_left_);
        MV_CC_DestroyHandle(handle_left_);
        handle_left_ = nullptr;
        return false;
    }
    if (!configureCamera(handle_right_, cfg, "RIGHT")) {
        LOG_ERROR("[HikCam] Failed to configure RIGHT camera");
        MV_CC_CloseDevice(handle_right_);
        MV_CC_DestroyHandle(handle_right_);
        handle_right_ = nullptr;
        MV_CC_CloseDevice(handle_left_);
        MV_CC_DestroyHandle(handle_left_);
        handle_left_ = nullptr;
        return false;
    }

    // 目标缓冲按配置尺寸预分配。左右相机的实际尺寸或 ROI 偏移只要不一致，
    // 后续 memcpy 就可能越界或让标定坐标失效，因此不能回退到猜测值。
    CameraGeometry left_geometry;
    CameraGeometry right_geometry;
    if (!readCameraGeometry(handle_left_, "LEFT", left_geometry) ||
        !readCameraGeometry(handle_right_, "RIGHT", right_geometry) ||
        !geometryMatchesConfig(left_geometry, cfg) ||
        !geometryMatchesConfig(right_geometry, cfg)) {
        LOG_ERROR(
            "[HikCam] Geometry mismatch: configured=%dx%d offset=(0,0), "
            "left=%dx%d offset=(%d,%d), right=%dx%d offset=(%d,%d)",
            cfg.width, cfg.height,
            left_geometry.width, left_geometry.height,
            left_geometry.offset_x, left_geometry.offset_y,
            right_geometry.width, right_geometry.height,
            right_geometry.offset_x, right_geometry.offset_y);
        close();
        return false;
    }
    width_ = cfg.width;
    height_ = cfg.height;

    opened_ = true;
    LOG_INFO("[HikCam] Both cameras opened, %dx%d", width_, height_);
    return true;
}

void HikvisionCamera::close() {
    stopGrabbing();
    resetSyncState();

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
    width_ = 0;
    height_ = 0;
}

// ==================== start / stop ====================

bool HikvisionCamera::startGrabbing() {
    if (!opened_ || grabbing_) return false;

    resetSyncState();
    int ret = MV_CC_StartGrabbing(handle_left_);
    if (ret != MV_OK) {
        LOG_ERROR("[HikCam] StartGrab LEFT failed, ret=0x%X", ret);
        return false;
    }
    ret = MV_CC_StartGrabbing(handle_right_);
    if (ret != MV_OK) {
        LOG_ERROR("[HikCam] StartGrab RIGHT failed, ret=0x%X", ret);
        // 不允许留下只有左相机正在取流的半启动状态。
        MV_CC_StopGrabbing(handle_left_);
        return false;
    }

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
    // 调用方允许复用结果对象。任何失败路径都必须先清掉上一帧身份，
    // 避免把陈旧的 FrameCounter 或到达时间误写进本次采集。
    result = GrabResult{};
    MV_FRAME_OUT frameOut;
    memset(&frameOut, 0, sizeof(frameOut));

    int nRet = MV_CC_GetImageBuffer(handle, &frameOut, timeout_ms);
    if (MV_OK != nRet) {
        result.success = false;
        return false;
    }

    // 每台相机在各自的取帧线程内独立打点。USB 到达先后可以与曝光
    // 顺序不同，因此这两组时间只用于传输延迟/乱序审计，不用于左右配对。
    result.receive_monotonic_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    result.receive_realtime_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    auto& info = frameOut.stFrameInfo;
    int srcWidth  = info.nWidth;
    int srcHeight = info.nHeight;

    // 直接 memcpy 到目标 host/pinned buffer；后续 CUDA 可复用该图像资源，
    // 但相机 SDK buffer 仍有一次拷贝。
    int dstPitch = (pitch > 0) ? pitch : srcWidth;
    if (!dst || !frameOut.pBufAddr || srcWidth != width_ ||
        srcHeight != height_ || dstPitch < srcWidth) {
        LOG_ERROR(
            "[HikCam] Reject frame geometry=%dx%d pitch=%d, expected=%dx%d",
            srcWidth, srcHeight, dstPitch, width_, height_);
        const int free_status = MV_CC_FreeImageBuffer(handle, &frameOut);
        if (free_status != MV_OK) {
            LOG_ERROR(
                "[HikCam] FreeImageBuffer failed after rejecting frame, "
                "ret=0x%X",
                free_status);
        }
        return false;
    }
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

    if (config_.use_trigger && config_.embedded_info_clear_rows > 0) {
        clearEmbeddedInfoRows(dst, dstPitch, srcWidth, srcHeight,
                              config_.embedded_info_clear_rows);
    }

    result.device_timestamp_ns = info.nDevTimeStampHigh;
    result.device_timestamp_ns =
        (result.device_timestamp_ns << 32) | info.nDevTimeStampLow;
    result.host_timestamp = info.nHostTimeStamp;
    result.frame_number = info.nFrameNum;
    result.frame_counter = info.nFrameCounter;
    result.trigger_index = info.nTriggerIndex;
    if (std::isfinite(info.fExposureTime) && info.fExposureTime > 0.0F) {
        result.exposure_us = static_cast<double>(info.fExposureTime);
    }
    if (std::isfinite(info.fGain)) {
        result.gain_db = static_cast<double>(info.fGain);
    }

    const int free_status = MV_CC_FreeImageBuffer(handle, &frameOut);
    if (free_status != MV_OK) {
        LOG_ERROR("[HikCam] FreeImageBuffer failed, ret=0x%X", free_status);
        return false;
    }
    result.success = true;
    return true;
}

bool HikvisionCamera::grabFramePair(
    uint8_t* dst_left,  uint8_t* dst_right,
    int left_pitch, int right_pitch,
    unsigned int timeout_ms,
    GrabResult& result_left, GrabResult& result_right)
{
    // 同步基线需要连续的水印样本。预热属于一次抓帧请求的内部过程，不能把
    // 前两个有效水印帧伪装成管线丢帧，否则 free-flight 的 fail_trial 会在
    // 启动时必然失败。预热次数有界；真实超时、断流和对齐失败仍立即返回。
    constexpr int kMaxSyncWarmupPairs = 16;
    for (int warmup_pairs = 0; warmup_pairs < kMaxSyncWarmupPairs;
         ++warmup_pairs) {
        result_left = GrabResult{};
        result_right = GrabResult{};
        bool sync_warmup_pending = false;
        if (grabFramePairAttempt(
                dst_left, dst_right,
                left_pitch, right_pitch,
                timeout_ms,
                result_left, result_right,
                sync_warmup_pending)) {
            return true;
        }
        if (!sync_warmup_pending || !grabbing_) return false;
        LOG_INFO("[HikCam] Sync warm-up consumed pair %d/%d",
                 warmup_pairs + 1, kMaxSyncWarmupPairs);
    }

    LOG_ERROR("[HikCam] Sync baseline did not stabilize within %d frame pairs",
              kMaxSyncWarmupPairs);
    return false;
}

bool HikvisionCamera::grabFramePairAttempt(
    uint8_t* dst_left,  uint8_t* dst_right,
    int left_pitch, int right_pitch,
    unsigned int timeout_ms,
    GrabResult& result_left, GrabResult& result_right,
    bool& sync_warmup_pending)
{
    sync_warmup_pending = false;
    if (!grabbing_) return false;

    // 并行抓取左右帧 — 确保两个 MV_CC_GetImageBuffer 同时发起,
    // 返回同一触发脉冲的帧, 避免顺序抓取的时间差导致帧不同步
    bool okL = false, okR = false;
    std::thread grabThread([&]() {
        okR = grabOneFrame(handle_right_, dst_right, right_pitch,
                           timeout_ms, result_right);
    });
    okL = grabOneFrame(handle_left_, dst_left, left_pitch,
                       timeout_ms, result_left);
    grabThread.join();

    if (!okL || !okR) {
        LOG_WARN("[HikCam] GrabPair partial fail: L=%d R=%d", okL, okR);
        consecutive_failures_++;
        if (consecutive_failures_ >= MAX_CONSECUTIVE_FAILURES) {
            LOG_WARN("[HikCam] %d consecutive failures, attempting reconnect...",
                     consecutive_failures_);
            if (reconnect()) {
                LOG_INFO("[HikCam] Reconnect successful");
                consecutive_failures_ = 0;
            } else {
                LOG_ERROR("[HikCam] Reconnect failed after %d retries", MAX_RECONNECT_RETRIES);
            }
        }
        return false;
    }

    int64_t timestamp_residual_ns =
        static_cast<int64_t>(result_left.device_timestamp_ns) -
        static_cast<int64_t>(result_right.device_timestamp_ns);

    // 检查硬触发配对: FrameCounter 是随图像 payload 到达的水印计数。
    // PWM 在两台相机 StartGrabbing 之后启动，因此同一触发脉冲应满足 Lfc==Rfc。
    // Timestamp 只用于确认相位稳定和诊断 USB/SDK 队列抖动，不替代 FrameCounter。
    if (config_.use_trigger) {
        const int64_t trigger_period_ns =
            config_.trigger_frequency_hz > 0
                ? static_cast<int64_t>(1000000000LL / config_.trigger_frequency_hz)
                : 10000000LL;
        constexpr int64_t timestamp_match_tol = 1000000LL;
        const unsigned int align_timeout_ms = std::min<unsigned int>(
            timeout_ms,
            std::max<unsigned int>(
                5U,
                static_cast<unsigned int>(2 * trigger_period_ns / 1000000LL)));
        const int kMaxAlignReads = std::clamp(config_.image_node_num + 1, 4, 16);
        constexpr int kRequiredBaselineSamples = 3;
        constexpr int64_t kExpectedFrameCounterDelta = 0;

        auto timestamp_offset = [&]() -> int64_t {
            return static_cast<int64_t>(result_left.device_timestamp_ns) -
                   static_cast<int64_t>(result_right.device_timestamp_ns);
        };
        auto frame_number_delta_value = [&]() -> int64_t {
            return static_cast<int64_t>(result_left.frame_number) -
                   static_cast<int64_t>(result_right.frame_number);
        };
        auto frame_counter_delta_value = [&]() -> int64_t {
            return static_cast<int64_t>(result_left.frame_counter) -
                   static_cast<int64_t>(result_right.frame_counter);
        };
        auto trigger_delta_value = [&]() -> int64_t {
            return static_cast<int64_t>(result_left.trigger_index) -
                   static_cast<int64_t>(result_right.trigger_index);
        };

        if (result_left.frame_counter == 0 || result_right.frame_counter == 0) {
            ++consecutive_sync_mismatches_;
            LOG_ERROR("[HikCam] FrameSpecInfo Framecounter missing: Lfc=%u Rfc=%u "
                      "(L#%llu/R#%llu Lt%u/Rt%u), dropping frame pair",
                      result_left.frame_counter, result_right.frame_counter,
                      static_cast<unsigned long long>(result_left.frame_number),
                      static_cast<unsigned long long>(result_right.frame_number),
                      result_left.trigger_index, result_right.trigger_index);
            if (consecutive_sync_mismatches_ >= 3) {
                LOG_WARN("[HikCam] Resetting sync baseline after %d consecutive metadata failures",
                         consecutive_sync_mismatches_);
                resetSyncState();
            }
            return false;
        }

        int64_t dt = timestamp_offset();
        int64_t frame_counter_delta = frame_counter_delta_value();
        int64_t frame_number_delta = frame_number_delta_value();
        int64_t trigger_delta = trigger_delta_value();

        auto update_deltas = [&]() {
            dt = timestamp_offset();
            frame_counter_delta = frame_counter_delta_value();
            frame_number_delta = frame_number_delta_value();
            trigger_delta = trigger_delta_value();
        };

        auto advance_one_side = [&](bool advance_right,
                                    const char* reason,
                                    int64_t offset_err,
                                    int& align_reads) -> bool {
            GrabResult before = advance_right ? result_right : result_left;
            GrabResult refreshed;
            bool ok = false;
            if (advance_right) {
                ok = grabOneFrame(handle_right_, dst_right, right_pitch,
                                  align_timeout_ms, refreshed);
                if (ok) result_right = refreshed;
            } else {
                ok = grabOneFrame(handle_left_, dst_left, left_pitch,
                                  align_timeout_ms, refreshed);
                if (ok) result_left = refreshed;
            }
            if (!ok) return false;

            ++align_reads;
            update_deltas();
            LOG_WARN("[HikCam] Advancing %s for %s sync: fc %u -> %u "
                     "(step=%u, frame_counter_delta=%ld, offset_err=%ldns)",
                     advance_right ? "RIGHT" : "LEFT",
                     reason,
                     before.frame_counter, refreshed.frame_counter,
                     counterStep(before.frame_counter, refreshed.frame_counter),
                     (long)frame_counter_delta,
                     (long)offset_err);
            return true;
        };

        int align_reads = 0;
        while (frame_counter_delta != kExpectedFrameCounterDelta &&
               align_reads < kMaxAlignReads) {
            const bool advance_right =
                frame_counter_delta > kExpectedFrameCounterDelta;
            if (!advance_one_side(advance_right, "frame_counter",
                                  0, align_reads)) {
                break;
            }
            if (result_left.frame_counter == 0 || result_right.frame_counter == 0) {
                break;
            }
        }

        if (result_left.frame_counter == 0 || result_right.frame_counter == 0 ||
            frame_counter_delta != kExpectedFrameCounterDelta) {
            ++consecutive_sync_mismatches_;
            LOG_WARN("[HikCam] FrameCounter mismatch after %d align read(s), "
                     "dropping pair: Lfc=%u Rfc=%u delta=%ld expected=%ld "
                     "(L#%llu/R#%llu Lt%u/Rt%u)",
                     align_reads,
                     result_left.frame_counter, result_right.frame_counter,
                     (long)frame_counter_delta,
                     (long)kExpectedFrameCounterDelta,
                     static_cast<unsigned long long>(result_left.frame_number),
                     static_cast<unsigned long long>(result_right.frame_number),
                     result_left.trigger_index, result_right.trigger_index);
            if (consecutive_sync_mismatches_ >= 3) {
                LOG_WARN("[HikCam] Resetting sync baseline after %d consecutive FrameCounter mismatches",
                         consecutive_sync_mismatches_);
                resetSyncState();
            }
            return false;
        }

        if (!sync_initialized_) {
            if (sync_baseline_samples_ == 0 ||
                std::abs(dt - candidate_timestamp_offset_ns_) > timestamp_match_tol) {
                candidate_timestamp_offset_ns_ = dt;
                sync_baseline_samples_ = 1;
                LOG_INFO("[HikCam] L/R sync baseline sample 1/%d: "
                         "frame_counter_delta=%ld timestamp_offset=%ldns",
                         kRequiredBaselineSamples,
                         (long)frame_counter_delta,
                         (long)dt);
                consecutive_failures_ = 0;
                sync_warmup_pending = true;
                return false;
            }

            ++sync_baseline_samples_;
            if (sync_baseline_samples_ < kRequiredBaselineSamples) {
                LOG_INFO("[HikCam] L/R sync baseline sample %d/%d: "
                         "frame_counter_delta=%ld timestamp_offset=%ldns",
                         sync_baseline_samples_,
                         kRequiredBaselineSamples,
                         (long)frame_counter_delta,
                         (long)dt);
                consecutive_failures_ = 0;
                sync_warmup_pending = true;
                return false;
            }

            expected_timestamp_offset_ns_ = dt;
            expected_frame_counter_delta_ = kExpectedFrameCounterDelta;
            expected_frame_number_delta_ = frame_number_delta;
            expected_trigger_delta_ = trigger_delta;
            sync_initialized_ = true;
            consecutive_sync_mismatches_ = 0;
            timestamp_residual_ns = 0;
            LOG_INFO("[HikCam] L/R sync initialized: frame_counter_delta=%ld, "
                     "frame_number_delta=%ld, trigger_delta=%ld, timestamp_offset=%ldns "
                     "(stable samples=%d)",
                     (long)expected_frame_counter_delta_,
                     (long)expected_frame_number_delta_,
                     (long)expected_trigger_delta_, (long)dt,
                     sync_baseline_samples_);
        } else {
            int64_t offset_err = dt - expected_timestamp_offset_ns_;
            while ((frame_counter_delta != expected_frame_counter_delta_ ||
                    std::abs(offset_err) > timestamp_match_tol) &&
                   align_reads < kMaxAlignReads) {
                bool advance_right = false;
                const char* reason = "timestamp";
                if (frame_counter_delta > expected_frame_counter_delta_) {
                    advance_right = true;
                    reason = "frame_counter";
                } else if (frame_counter_delta < expected_frame_counter_delta_) {
                    advance_right = false;
                    reason = "frame_counter";
                } else {
                    // FrameCounter can stay aligned if one camera misses an exposure
                    // and returns the next trigger with the same local output count.
                    advance_right = offset_err > 0;
                }

                if (!advance_one_side(advance_right, reason,
                                      offset_err, align_reads)) {
                    break;
                }
                offset_err = dt - expected_timestamp_offset_ns_;
            }

            if (frame_counter_delta != expected_frame_counter_delta_ ||
                std::abs(offset_err) > timestamp_match_tol) {
                ++consecutive_sync_mismatches_;
                LOG_WARN("[HikCam] Stereo sync mismatch: expected_offset=%ldns got=%ldns "
                         "err=%ldns tol=%ldns, frame_counter_delta expected=%ld got=%ld "
                         "after %d align read(s), dropping frame pair "
                         "(Lfc=%u/Rfc=%u L#%llu/R#%llu Lt%u/Rt%u dt=%ldns)",
                         (long)expected_timestamp_offset_ns_, (long)dt,
                         (long)offset_err, (long)timestamp_match_tol,
                         (long)expected_frame_counter_delta_,
                         (long)frame_counter_delta,
                         align_reads,
                         result_left.frame_counter, result_right.frame_counter,
                         static_cast<unsigned long long>(result_left.frame_number),
                         static_cast<unsigned long long>(result_right.frame_number),
                         result_left.trigger_index, result_right.trigger_index,
                         (long)dt);
                if (consecutive_sync_mismatches_ >= 3) {
                    LOG_WARN("[HikCam] Resetting sync baseline after %d consecutive phase mismatches",
                             consecutive_sync_mismatches_);
                    resetSyncState();
                }
                return false;
            }

            consecutive_sync_mismatches_ = 0;
            timestamp_residual_ns = offset_err;
            if (align_reads > 0) {
                LOG_WARN("[HikCam] Frame pair realigned: reads=%d "
                         "Lfc=%u/Rfc=%u frame_counter_delta=%ld "
                         "frame_delta=%ld trigger_delta=%ld offset_err=%ldns",
                         align_reads,
                         result_left.frame_counter, result_right.frame_counter,
                         (long)frame_counter_delta,
                         (long)frame_number_delta, (long)trigger_delta,
                         (long)offset_err);
            }

            if (frame_number_delta != expected_frame_number_delta_) {
                LOG_WARN("[HikCam] Frame number delta changed while Framecounter aligned: "
                         "%ld -> %ld (L#%llu/R#%llu Lfc=%u/Rfc=%u), adopting diagnostic delta",
                         (long)expected_frame_number_delta_,
                         (long)frame_number_delta,
                         static_cast<unsigned long long>(result_left.frame_number),
                         static_cast<unsigned long long>(result_right.frame_number),
                         result_left.frame_counter, result_right.frame_counter);
                expected_frame_number_delta_ = frame_number_delta;
            }

            // ExtTriggerCount was observed to have 0/+2 steps on this USB model.
            // Keep it as a diagnostic only; do not reject a Framecounter-aligned pair.
            expected_trigger_delta_ = trigger_delta;
            expected_timestamp_offset_ns_ = dt;
        }
    }

    result_left.stereo_timestamp_residual_ns = timestamp_residual_ns;
    result_right.stereo_timestamp_residual_ns = timestamp_residual_ns;

    consecutive_failures_ = 0;
    return true;
}

void HikvisionCamera::resetSyncState() {
    sync_initialized_ = false;
    sync_baseline_samples_ = 0;
    candidate_timestamp_offset_ns_ = 0;
    expected_timestamp_offset_ns_ = 0;
    expected_frame_counter_delta_ = 0;
    expected_frame_number_delta_ = 0;
    expected_trigger_delta_ = 0;
    consecutive_sync_mismatches_ = 0;
}

bool HikvisionCamera::grabSingle(bool is_left,
                                  uint8_t* dst, int pitch,
                                  unsigned int timeout_ms,
                                  GrabResult& result) {
    void* handle = is_left ? handle_left_ : handle_right_;
    if (!handle || !grabbing_) return false;
    return grabOneFrame(handle, dst, pitch, timeout_ms, result);
}

bool HikvisionCamera::reconnect() {
    for (int attempt = 0; attempt < MAX_RECONNECT_RETRIES; ++attempt) {
        LOG_INFO("[HikCam] Reconnect attempt %d/%d", attempt + 1, MAX_RECONNECT_RETRIES);
        stopGrabbing();
        close();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (open(config_) && startGrabbing()) {
            return true;
        }
    }
    return false;
}

}  // namespace volleyball::adapters::hik_mvs
