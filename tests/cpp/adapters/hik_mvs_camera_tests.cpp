#include "volleyball/adapters/hik_mvs/hikvision_camera.hpp"
#include "volleyball/adapters/hik_mvs/hik_stereo_camera_source.hpp"
#include "volleyball/core/observation.hpp"
#include "MvCameraControl.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

namespace {

namespace hik_mvs = volleyball::adapters::hik_mvs;

struct StubCamera {
    bool right = false;
    bool started = false;
    bool frame_spec_enabled = false;
    unsigned int next_counter = 1;
    unsigned int image_reads = 0;
    unsigned int freed_frames = 0;
    unsigned int width = 2;
    unsigned int height = 2;
    unsigned int offset_x = 0;
    unsigned int offset_y = 0;
    bool fail_free = false;
    std::array<unsigned char, 16> image{};
};

StubCamera left_camera;
StubCamera right_camera;
MV_CC_DEVICE_INFO left_info{};
MV_CC_DEVICE_INFO right_info{};
MV_CC_DEVICE_INFO* device_infos[] = {&left_info, &right_info};
bool unstable_timestamp_offset = false;
bool fail_next_left_grab = false;
bool fail_trigger_activation_readback = false;
float frame_exposure_us = 9867.0F;
int failures = 0;

static_assert(std::is_base_of_v<
              volleyball::ports::ManagedCameraSource,
              hik_mvs::HikStereoCameraSource>);

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

void resetStub(bool unstable) {
    left_camera = StubCamera{};
    right_camera = StubCamera{};
    right_camera.right = true;
    unstable_timestamp_offset = unstable;
    fail_next_left_grab = false;
    fail_trigger_activation_readback = false;
    frame_exposure_us = 9867.0F;
    left_info = MV_CC_DEVICE_INFO{};
    right_info = MV_CC_DEVICE_INFO{};
    left_info.nTLayerType = MV_USB_DEVICE;
    right_info.nTLayerType = MV_USB_DEVICE;
    std::strcpy(
        reinterpret_cast<char*>(left_info.SpecialInfo.stUsb3VInfo.chSerialNumber),
        "left-stub");
    std::strcpy(
        reinterpret_cast<char*>(right_info.SpecialInfo.stUsb3VInfo.chSerialNumber),
        "right-stub");
}

hik_mvs::CameraConfig config();

hik_mvs::HikStereoCameraSourceConfig sourceConfig() {
    hik_mvs::HikStereoCameraSourceConfig output;
    output.camera = config();
    output.camera.shutter_type = "global";
    output.camera.exposure_timestamp_source = "nx_gpio_rising_edge";
    output.camera.exposure_delay_status = "assumed_zero_unmeasured";
    output.camera.exposure_delay_ns = 0;
    output.camera.exposure_delay_uncertainty_ns = 0;
    output.camera.exposure_delay_uncertainty_known = false;
    output.camera.exposure_midpoint_offset_ns = 4'933'500;
    output.left_camera_id = volleyball::core::CameraId{"left-stub"};
    output.right_camera_id = volleyball::core::CameraId{"right-stub"};
    output.left_device_clock =
        volleyball::core::ClockDomainId{"left-device"};
    output.right_device_clock =
        volleyball::core::ClockDomainId{"right-device"};
    output.capture_realtime_clock =
        volleyball::core::ClockDomainId{"nx-realtime"};
    output.host_monotonic_clock =
        volleyball::core::ClockDomainId{"nx-monotonic"};
    output.timeout_ms = 50;
    output.buffer_pool_pairs = 2;
    return output;
}

hik_mvs::TriggerEdgeEvidence triggerEvidence(std::uint64_t sequence) {
    const std::uint64_t period_ns = 11'111'111;
    const std::uint64_t edge_monotonic_ns = sequence * period_ns;
    return {
        sequence,
        edge_monotonic_ns - 100,
        edge_monotonic_ns,
        1'000'000'000ULL + edge_monotonic_ns,
        200,
    };
}

hik_mvs::CameraConfig config() {
    hik_mvs::CameraConfig cfg;
    cfg.serial_left = "left-stub";
    cfg.serial_right = "right-stub";
    cfg.width = 2;
    cfg.height = 2;
    cfg.use_trigger = true;
    cfg.trigger_frequency_hz = 90;
    cfg.embedded_info_clear_rows = 1;
    return cfg;
}

void testWarmupIsInternalToFirstGrab() {
    resetStub(false);
    hik_mvs::HikvisionCamera camera;
    check(camera.open(config()), "stub cameras must open");
    check(camera.startGrabbing(), "stub cameras must start");

    std::array<std::uint8_t, 4> left{};
    std::array<std::uint8_t, 4> right{};
    hik_mvs::GrabResult left_result;
    hik_mvs::GrabResult right_result;
    check(
        camera.grabFramePair(
            left.data(), right.data(), 2, 2, 50,
            left_result, right_result),
        "first public grab must finish synchronization warm-up");
    check(left_result.frame_counter == 3 && right_result.frame_counter == 3,
          "first published pair must be the third stable watermark pair");
    check(left_camera.image_reads == 3 && right_camera.image_reads == 3,
          "warm-up pairs must be consumed inside the adapter");

    check(
        camera.grabFramePair(
            left.data(), right.data(), 2, 2, 50,
            left_result, right_result),
        "grab after warm-up must succeed");
    check(left_result.frame_counter == 4 && right_result.frame_counter == 4,
          "post-warm-up counters must remain continuous");
    check(left_camera.image_reads == 4 && right_camera.image_reads == 4,
          "post-warm-up grab must consume exactly one pair");
}

void testStereoPairRealignsAcrossFrameCounterWrap() {
    resetStub(false);
    left_camera.next_counter = std::numeric_limits<std::uint32_t>::max() - 2U;
    right_camera.next_counter = std::numeric_limits<std::uint32_t>::max() - 2U;
    hik_mvs::HikvisionCamera camera;
    check(camera.open(config()), "wrap fixture cameras must open");
    check(camera.startGrabbing(), "wrap fixture cameras must start");

    std::array<std::uint8_t, 4> left{};
    std::array<std::uint8_t, 4> right{};
    hik_mvs::GrabResult left_result;
    hik_mvs::GrabResult right_result;
    check(
        camera.grabFramePair(
            left.data(), right.data(), 2, 2, 50,
            left_result, right_result),
        "warm-up must publish UINT32_MAX as the last pre-wrap pair");
    check(
        left_result.frame_counter == std::numeric_limits<std::uint32_t>::max() &&
            right_result.frame_counter == std::numeric_limits<std::uint32_t>::max(),
        "pre-wrap pair must retain the raw UINT32_MAX watermark");

    // 模拟右相机 USB 队列仍停在回绕前一帧，而左相机已经拿到计数 0。
    // 对齐逻辑必须按 uint32 环形距离推进右侧，不能用有符号大小关系。
    right_camera.next_counter = std::numeric_limits<std::uint32_t>::max();
    check(
        camera.grabFramePair(
            left.data(), right.data(), 2, 2, 50,
            left_result, right_result),
        "stereo queues must realign across the uint32 counter wrap");
    check(left_result.frame_counter == 0 && right_result.frame_counter == 0,
          "realigned post-wrap pair must retain counter zero");
    check(left_camera.image_reads == 4 && right_camera.image_reads == 5,
          "only the lagging right queue may be advanced during wrap realignment");
}

void testUnstableBaselineFailsWithBoundedReads() {
    resetStub(true);
    hik_mvs::HikvisionCamera camera;
    check(camera.open(config()), "unstable stub cameras must open");
    check(camera.startGrabbing(), "unstable stub cameras must start");

    std::array<std::uint8_t, 4> left{};
    std::array<std::uint8_t, 4> right{};
    hik_mvs::GrabResult left_result;
    hik_mvs::GrabResult right_result;
    check(
        !camera.grabFramePair(
            left.data(), right.data(), 2, 2, 50,
            left_result, right_result),
        "unstable synchronization baseline must fail");
    check(left_camera.image_reads == 16 && right_camera.image_reads == 16,
          "unstable warm-up must stop at the bounded retry limit");
}

void testOpenRejectsMismatchedCameraGeometry() {
    resetStub(false);
    right_camera.width = 3;
    hik_mvs::HikvisionCamera camera;
    check(!camera.open(config()),
          "camera open must reject unequal left/right dimensions");

    resetStub(false);
    right_camera.offset_x = 1;
    check(!camera.open(config()),
          "camera open must reject an ROI offset not represented by calibration");
}

void testOpenRequiresTriggerActivationReadback() {
    resetStub(false);
    fail_trigger_activation_readback = true;
    hik_mvs::HikvisionCamera camera;
    check(!camera.open(config()),
          "camera open must require TriggerActivation readback");
    check(!left_camera.started && !right_camera.started,
          "readback failure must happen before acquisition starts");
}

void testGrabRejectsGeometryDriftAndShortPitch() {
    resetStub(false);
    hik_mvs::HikvisionCamera camera;
    check(camera.open(config()), "geometry drift camera must open initially");
    check(camera.startGrabbing(), "geometry drift camera must start");

    std::array<std::uint8_t, 4> image{};
    hik_mvs::GrabResult result;
    check(!camera.grabSingle(true, image.data(), 1, 50, result),
          "frame copy must reject a destination pitch smaller than its width");
    check(left_camera.freed_frames == 1,
          "rejected short-pitch frame must be returned to the MVS SDK");

    right_camera.width = 3;
    hik_mvs::GrabResult left_result;
    hik_mvs::GrabResult right_result;
    check(!camera.grabFramePair(
              image.data(), image.data(), 2, 2, 50,
              left_result, right_result),
          "frame copy must reject dimensions that drift after open");
    check(right_camera.freed_frames == 1,
          "rejected drifted frame must be returned to the MVS SDK");
}

void testGrabRejectsSdkBufferReleaseFailureAndClearsStaleResult() {
    resetStub(false);
    hik_mvs::HikvisionCamera camera;
    check(camera.open(config()), "release-failure camera must open");
    check(camera.startGrabbing(), "release-failure camera must start");

    std::array<std::uint8_t, 4> image{};
    hik_mvs::GrabResult result;
    result.success = true;
    result.frame_counter = 999;
    left_camera.fail_free = true;
    check(!camera.grabSingle(true, image.data(), 2, 50, result),
          "SDK buffer release failure must invalidate the frame");
    check(!result.success,
          "release failure must not leave a successful frame result");
    check(result.frame_counter == 1,
          "failed release may retain current diagnostics, not stale identity");

    left_camera.fail_free = false;
    left_camera.started = false;
    result.success = true;
    result.frame_counter = 999;
    check(!camera.grabSingle(true, image.data(), 2, 50, result),
          "SDK acquisition failure must reject the frame");
    check(!result.success && result.frame_counter == 0,
          "acquisition failure must clear all stale result fields");
}

void testCameraSourcePreservesAuditedFrameEvidence() {
    resetStub(false);
    auto source_config = sourceConfig();
    source_config.frame_limit = 2;
    hik_mvs::HikStereoCameraSource source(
        source_config,
        [](std::uint64_t sequence, hik_mvs::TriggerEdgeEvidence& output) {
            output = triggerEvidence(sequence);
            return true;
        });
    std::string error;
    check(source.start(error), "audited Hik source must start with valid config");
    check(error.empty(), "successful Hik source start must not retain an error");

    auto first = source.next();
    check(first.status == volleyball::ports::SourceStatus::frame &&
              first.stereo_frame.has_value(),
          "first audited source read must produce a stereo frame");
    if (first.stereo_frame) {
        const auto& left = first.stereo_frame->left;
        const auto& right = first.stereo_frame->right;
        check(left.metadata.frame_index == 3 &&
                  right.metadata.frame_index == 3,
              "source frame identity must use the synchronized FrameCounter");
        check(left.metadata.hardware_timestamp.clock_domain ==
                  volleyball::core::ClockDomainId{"left-device"} &&
                  right.metadata.hardware_timestamp.clock_domain ==
                  volleyball::core::ClockDomainId{"right-device"},
              "each hardware timestamp must retain its own device clock");
        check(left.metadata.hardware_timestamp.nanoseconds == 3 * 11'111'111 +
                  50'000 &&
                  right.metadata.hardware_timestamp.nanoseconds ==
                      3 * 11'111'111,
              "SDK device timestamps measured in nanoseconds must not be "
              "rescaled while entering the core contract");
        check(left.metadata.capture_timestamp.nanoseconds ==
                  static_cast<std::int64_t>(
                      triggerEvidence(3).edge_realtime_ns + 4'933'500ULL),
              "capture time must be GPIO edge plus half exposure");
        check(left.metadata.acquisition.trigger_sequence == 3 &&
                  right.metadata.acquisition.trigger_sequence == 3,
              "both camera records must retain the common trigger sequence");
        check(!left.metadata.acquisition.capture_timestamp_mapping_valid &&
                  !left.metadata.acquisition
                       .capture_timestamp_uncertainty_ns.has_value(),
              "unmeasured exposure delay must remain explicitly uncertain");
        check(left.metadata.exposure_us == 9867.0 &&
                  left.metadata.gain_db &&
                  std::abs(*left.metadata.gain_db - 11.9906) < 1.0e-4,
              "frame metadata must retain actual SDK exposure and gain");
        check(left.image && right.image && left.image->sizeBytes() == 4 &&
                  right.image->sizeBytes() == 4,
              "source must return owned Bayer buffers with exact geometry");
        check(static_cast<bool>(volleyball::core::validate(left.metadata)) &&
                  static_cast<bool>(volleyball::core::validate(right.metadata)),
              "source metadata must satisfy the platform-independent contract");
    }

    auto second = source.next();
    check(second.status == volleyball::ports::SourceStatus::frame,
          "source must continue until its configured frame limit");
    auto ended = source.next();
    check(ended.status == volleyball::ports::SourceStatus::end_of_stream &&
              !ended.stereo_frame,
          "finite capture source must end without a phantom frame");
    check(!source.isStarted(), "source must close cameras at end of stream");
    check(source.next().status == volleyball::ports::SourceStatus::end_of_stream,
          "end-of-stream status must be stable");
}

void testCameraSourceExtendsFrameCounterAcrossWrap() {
    resetStub(false);
    left_camera.next_counter = std::numeric_limits<std::uint32_t>::max() - 2U;
    right_camera.next_counter = std::numeric_limits<std::uint32_t>::max() - 2U;
    auto source_config = sourceConfig();
    source_config.frame_limit = 2;
    hik_mvs::HikStereoCameraSource source(
        source_config,
        [](std::uint64_t sequence, hik_mvs::TriggerEdgeEvidence& output) {
            output = {
                sequence,
                100,
                200,
                1'000'000'200,
                10,
            };
            return true;
        });
    std::string error;
    check(source.start(error), "counter-wrap source must start");

    auto before_wrap = source.next();
    check(before_wrap.status == volleyball::ports::SourceStatus::frame &&
              before_wrap.stereo_frame.has_value(),
          "counter-wrap source must publish its pre-wrap frame");
    if (before_wrap.stereo_frame) {
        const auto& metadata = before_wrap.stereo_frame->left.metadata;
        check(metadata.frame_index ==
                  std::numeric_limits<std::uint32_t>::max(),
              "pre-wrap logical frame index must equal UINT32_MAX");
        check(metadata.acquisition.sensor_frame_counter ==
                  std::numeric_limits<std::uint32_t>::max() &&
                  metadata.acquisition.trigger_sequence ==
                      std::numeric_limits<std::uint32_t>::max(),
              "pre-wrap metadata must preserve raw and logical identities");
    }

    auto after_wrap = source.next();
    check(after_wrap.status == volleyball::ports::SourceStatus::frame &&
              after_wrap.stereo_frame.has_value(),
          "counter zero must be accepted after a valid natural wrap");
    if (after_wrap.stereo_frame) {
        const auto& metadata = after_wrap.stereo_frame->left.metadata;
        const auto extended =
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) +
            1U;
        check(metadata.frame_index == extended &&
                  metadata.acquisition.trigger_sequence == extended,
              "logical frame identity must remain strictly increasing after wrap");
        check(metadata.acquisition.sensor_frame_counter == 0,
              "raw camera watermark must remain zero after wrap");
    }
}

void testCameraSourceRejectsFrameCounterGap() {
    resetStub(false);
    auto source_config = sourceConfig();
    source_config.frame_limit = 2;
    hik_mvs::HikStereoCameraSource source(
        source_config,
        [](std::uint64_t sequence, hik_mvs::TriggerEdgeEvidence& output) {
            output = triggerEvidence(sequence);
            return true;
        });
    std::string error;
    check(source.start(error), "counter-gap source must start");
    auto first = source.next();
    check(first.status == volleyball::ports::SourceStatus::frame,
          "counter-gap source must publish its first frame");

    left_camera.next_counter = 5;
    right_camera.next_counter = 5;
    auto gap = source.next();
    check(gap.status == volleyball::ports::SourceStatus::error &&
              gap.message.find("FrameCounter gap") != std::string::npos,
          "a matching stereo pair with a skipped counter must fail the trial");
    check(!source.isStarted(), "counter gap must close the camera source");
}

void testCameraSourceNeverOverwritesRetainedBuffers() {
    resetStub(false);
    auto source_config = sourceConfig();
    source_config.buffer_pool_pairs = 1;
    hik_mvs::HikStereoCameraSource source(
        source_config,
        [](std::uint64_t sequence, hik_mvs::TriggerEdgeEvidence& output) {
            output = triggerEvidence(sequence);
            return true;
        });
    std::string error;
    check(source.start(error), "single-pair pool source must start");
    const auto retained = source.next();
    check(retained.status == volleyball::ports::SourceStatus::frame,
          "single-pair pool must produce its first frame");
    const auto exhausted = source.next();
    check(exhausted.status == volleyball::ports::SourceStatus::error &&
              exhausted.message.find("buffer pool exhausted") !=
                  std::string::npos,
          "retained images must cause explicit backpressure, not overwrite");
    check(!source.isStarted(), "fatal buffer exhaustion must close cameras");
}

void testCameraSourceMarksOnlyMeasuredExposureDelayAsMapped() {
    resetStub(false);
    auto source_config = sourceConfig();
    source_config.camera.exposure_delay_status = "measured";
    source_config.camera.exposure_delay_ns = 500;
    source_config.camera.exposure_delay_uncertainty_ns = 20;
    source_config.camera.exposure_delay_uncertainty_known = true;
    source_config.frame_limit = 1;
    hik_mvs::HikStereoCameraSource source(
        source_config,
        [](std::uint64_t sequence, hik_mvs::TriggerEdgeEvidence& output) {
            output = triggerEvidence(sequence);
            return true;
        });
    std::string error;
    check(source.start(error), "measured-delay source must start");
    const auto result = source.next();
    check(result.status == volleyball::ports::SourceStatus::frame &&
              result.stereo_frame.has_value(),
          "measured-delay source must produce a frame");
    if (result.stereo_frame) {
        const auto& metadata = result.stereo_frame->left.metadata;
        const auto edge = triggerEvidence(3);
        check(metadata.exposure_start &&
                  metadata.exposure_start->nanoseconds ==
                      static_cast<std::int64_t>(edge.edge_realtime_ns + 500),
              "measured delay must shift exposure start from the GPIO edge");
        check(metadata.acquisition.capture_timestamp_mapping_valid &&
                  metadata.acquisition.capture_timestamp_uncertainty_ns == 220,
              "measured mapping uncertainty must include GPIO write and delay error");
    }
}

void testCameraSourceUsesPerFrameExposureForCaptureCenter() {
    resetStub(false);
    frame_exposure_us = 8000.0F;
    auto source_config = sourceConfig();
    source_config.frame_limit = 1;
    hik_mvs::HikStereoCameraSource source(
        source_config,
        [](std::uint64_t sequence, hik_mvs::TriggerEdgeEvidence& output) {
            output = triggerEvidence(sequence);
            return true;
        });
    std::string error;
    check(source.start(error), "per-frame exposure source must start");
    const auto result = source.next();
    check(result.status == volleyball::ports::SourceStatus::frame &&
              result.stereo_frame.has_value(),
          "per-frame exposure source must produce a frame");
    if (result.stereo_frame) {
        const auto& metadata = result.stereo_frame->left.metadata;
        const auto edge = triggerEvidence(3);
        check(metadata.exposure_us == 8000.0,
              "source must preserve exposure reported by this payload");
        check(metadata.capture_timestamp.nanoseconds ==
                  static_cast<std::int64_t>(
                      edge.edge_realtime_ns + 4'000'000ULL),
              "capture center must use half of the payload exposure");
    }
}

void testCameraSourceRejectsMissingTriggerEvidence() {
    resetStub(false);
    hik_mvs::HikStereoCameraSource source(
        sourceConfig(),
        [](std::uint64_t, hik_mvs::TriggerEdgeEvidence&) { return false; });
    std::string error;
    check(source.start(error), "missing-trigger fixture must start cameras");
    const auto result = source.next();
    check(result.status == volleyball::ports::SourceStatus::error &&
              result.message.find("GPIO rising-edge") != std::string::npos,
          "frame without matching GPIO evidence must fail the source");
}

void testCameraSourceSeparatesTimeoutFromFatalMetadataFailure() {
    resetStub(false);
    hik_mvs::HikStereoCameraSource source(
        sourceConfig(),
        [](std::uint64_t sequence, hik_mvs::TriggerEdgeEvidence& output) {
            output = triggerEvidence(sequence);
            return true;
        });
    std::string error;
    check(source.start(error), "timeout fixture source must start");
    fail_next_left_grab = true;
    const auto result = source.next();
    check(result.status == volleyball::ports::SourceStatus::timeout &&
              !result.stereo_frame,
          "SDK no-data path must remain a retryable source timeout");
    check(source.isStarted(), "one timeout must not silently close the source");
    source.stop();
}

void testCameraSourceRejectsAmbiguousIdentityBeforeOpeningHardware() {
    resetStub(false);
    auto source_config = sourceConfig();
    source_config.right_camera_id = source_config.left_camera_id;
    hik_mvs::HikStereoCameraSource source(
        source_config,
        [](std::uint64_t sequence, hik_mvs::TriggerEdgeEvidence& output) {
            output = triggerEvidence(sequence);
            return true;
        });
    std::string error;
    check(!source.start(error) && !error.empty(),
          "ambiguous camera identity must fail before opening hardware");
    check(!left_camera.started && !right_camera.started,
          "invalid source config must not touch acquisition state");
}

void testCameraSourceRejectsUnauditableTriggerTransport() {
    resetStub(false);
    auto source_config = sourceConfig();
    source_config.camera.trigger_source = "Software";
    hik_mvs::HikStereoCameraSource software_trigger_source(
        source_config,
        [](std::uint64_t sequence, hik_mvs::TriggerEdgeEvidence& output) {
            output = triggerEvidence(sequence);
            return true;
        });
    std::string error;
    check(!software_trigger_source.start(error),
          "GPIO timing contract must reject software-triggered exposure");

    resetStub(false);
    source_config = sourceConfig();
    source_config.camera.embedded_info_clear_rows = 0;
    hik_mvs::HikStereoCameraSource uncleared_watermark_source(
        source_config,
        [](std::uint64_t sequence, hik_mvs::TriggerEdgeEvidence& output) {
            output = triggerEvidence(sequence);
            return true;
        });
    error.clear();
    check(!uncleared_watermark_source.start(error),
          "formal source must reject image payloads with uncleared watermark rows");
    check(!left_camera.started && !right_camera.started,
          "invalid trigger transport must fail before opening hardware");
}

}  // namespace

extern "C" {

int MV_CC_EnumDevices(unsigned int, MV_CC_DEVICE_INFO_LIST* device_list) {
    device_list->nDeviceNum = 2;
    device_list->pDeviceInfo = device_infos;
    return MV_OK;
}

int MV_CC_CreateHandle(void** handle, MV_CC_DEVICE_INFO* device_info) {
    *handle = device_info == &left_info
        ? static_cast<void*>(&left_camera)
        : static_cast<void*>(&right_camera);
    return MV_OK;
}

int MV_CC_OpenDevice(void*) { return MV_OK; }
int MV_CC_CloseDevice(void*) { return MV_OK; }
int MV_CC_DestroyHandle(void*) { return MV_OK; }
int MV_CC_SetEnumValue(void*, const char*, unsigned int) { return MV_OK; }
int MV_CC_SetEnumValueByString(void*, const char*, const char*) { return MV_OK; }
int MV_CC_SetFloatValue(void*, const char*, float) { return MV_OK; }

int MV_CC_SetBoolValue(void* handle, const char* node, bool value) {
    if (std::string(node) == "FrameSpecInfo") {
        static_cast<StubCamera*>(handle)->frame_spec_enabled = value;
    }
    return MV_OK;
}

int MV_CC_GetEnumValue(void*, const char* node, MVCC_ENUMVALUE* value) {
    const std::string name(node);
    if (name == "TriggerActivation" &&
        fail_trigger_activation_readback) {
        return -1;
    }
    value->nCurValue = name == "TriggerMode" ? 1U : 0U;
    return MV_OK;
}

int MV_CC_GetFloatValue(void*, const char* node, MVCC_FLOATVALUE* value) {
    value->fCurValue = std::string(node) == "ExposureTime" ? 9867.0F : 11.9906F;
    return MV_OK;
}

int MV_CC_GetBoolValue(void* handle, const char* node, bool* value) {
    *value = std::string(node) == "FrameSpecInfo"
        ? static_cast<StubCamera*>(handle)->frame_spec_enabled
        : false;
    return MV_OK;
}

int MV_CC_GetIntValue(void* handle, const char* node, MVCC_INTVALUE* value) {
    const auto* camera = static_cast<StubCamera*>(handle);
    const std::string name(node);
    if (name == "Width") value->nCurValue = camera->width;
    else if (name == "Height") value->nCurValue = camera->height;
    else if (name == "OffsetX") value->nCurValue = camera->offset_x;
    else if (name == "OffsetY") value->nCurValue = camera->offset_y;
    else return -1;
    return MV_OK;
}

int MV_CC_SetGrabStrategy(void*, unsigned int) { return MV_OK; }
int MV_CC_SetImageNodeNum(void*, unsigned int) { return MV_OK; }

int MV_CC_StartGrabbing(void* handle) {
    static_cast<StubCamera*>(handle)->started = true;
    return MV_OK;
}

int MV_CC_StopGrabbing(void* handle) {
    static_cast<StubCamera*>(handle)->started = false;
    return MV_OK;
}

int MV_CC_GetImageBuffer(void* handle, MV_FRAME_OUT* frame, unsigned int) {
    auto* camera = static_cast<StubCamera*>(handle);
    if (!camera->started) return -1;
    if (!camera->right && fail_next_left_grab) {
        fail_next_left_grab = false;
        return -1;
    }
    const unsigned int counter = camera->next_counter++;
    ++camera->image_reads;
    const std::uint64_t period_ns = 11111111ULL;
    std::uint64_t timestamp_ns = counter * period_ns;
    if (camera->right && unstable_timestamp_offset && counter % 2 == 0) {
        timestamp_ns += 2000000ULL;
    } else if (!camera->right) {
        timestamp_ns += 50000ULL;
    }
    frame->pBufAddr = camera->image.data();
    frame->stFrameInfo.nWidth = camera->width;
    frame->stFrameInfo.nHeight = camera->height;
    frame->stFrameInfo.nDevTimeStampHigh =
        static_cast<unsigned int>(timestamp_ns >> 32U);
    frame->stFrameInfo.nDevTimeStampLow =
        static_cast<unsigned int>(timestamp_ns & 0xffffffffULL);
    frame->stFrameInfo.nHostTimeStamp = timestamp_ns + 1000000ULL;
    frame->stFrameInfo.nFrameNum = counter;
    frame->stFrameInfo.nFrameCounter = counter;
    frame->stFrameInfo.nTriggerIndex = counter * 2U;
    frame->stFrameInfo.fExposureTime = frame_exposure_us;
    frame->stFrameInfo.fGain = 11.9906F;
    return MV_OK;
}

int MV_CC_FreeImageBuffer(void* handle, MV_FRAME_OUT*) {
    auto* camera = static_cast<StubCamera*>(handle);
    ++camera->freed_frames;
    return camera->fail_free ? -1 : MV_OK;
}

}  // extern "C"

int main() {
    testWarmupIsInternalToFirstGrab();
    testStereoPairRealignsAcrossFrameCounterWrap();
    testUnstableBaselineFailsWithBoundedReads();
    testOpenRejectsMismatchedCameraGeometry();
    testOpenRequiresTriggerActivationReadback();
    testGrabRejectsGeometryDriftAndShortPitch();
    testGrabRejectsSdkBufferReleaseFailureAndClearsStaleResult();
    testCameraSourcePreservesAuditedFrameEvidence();
    testCameraSourceExtendsFrameCounterAcrossWrap();
    testCameraSourceRejectsFrameCounterGap();
    testCameraSourceNeverOverwritesRetainedBuffers();
    testCameraSourceMarksOnlyMeasuredExposureDelayAsMapped();
    testCameraSourceUsesPerFrameExposureForCaptureCenter();
    testCameraSourceRejectsMissingTriggerEvidence();
    testCameraSourceSeparatesTimeoutFromFatalMetadataFailure();
    testCameraSourceRejectsAmbiguousIdentityBeforeOpeningHardware();
    testCameraSourceRejectsUnauditableTriggerTransport();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "hik camera synchronization tests passed\n";
    return EXIT_SUCCESS;
}
