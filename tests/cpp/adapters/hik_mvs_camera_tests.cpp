#include "volleyball/adapters/hik_mvs/hikvision_camera.hpp"
#include "MvCameraControl.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

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
int failures = 0;

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
    testUnstableBaselineFailsWithBoundedReads();
    testOpenRejectsMismatchedCameraGeometry();
    testGrabRejectsGeometryDriftAndShortPitch();
    testGrabRejectsSdkBufferReleaseFailureAndClearsStaleResult();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "hik camera synchronization tests passed\n";
    return EXIT_SUCCESS;
}
