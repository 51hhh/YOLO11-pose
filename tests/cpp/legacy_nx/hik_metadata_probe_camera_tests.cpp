#include "tools/hik_frame_metadata_probe/camera_session.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <string>

namespace {

struct StubCamera {
    bool opened = false;
    bool started = false;
    bool stopped = false;
    bool closed = false;
    bool destroyed = false;
    bool fail_pixel_format_string = false;
    bool fail_pixel_format_enum = false;
    bool fail_free = false;
    bool fail_frame_rate_disable = false;
    float exposure_readback = 9867.0F;
    float gain_readback = 11.99F;
    std::string fail_selector;
    std::string fail_enable;
    std::string false_readback_selector;
    std::string selected_selector;
    std::set<std::string> enabled_selectors;
    std::array<unsigned char, 4> image {{0x11, 0x22, 0x33, 0x44}};
};

StubCamera left_camera;
StubCamera right_camera;
MV_CC_DEVICE_INFO left_info {};
MV_CC_DEVICE_INFO right_info {};
MV_CC_DEVICE_INFO* device_infos[] = {&left_info, &right_info};
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

void setText(unsigned char* destination, const char* value) {
    std::strcpy(reinterpret_cast<char*>(destination), value);
}

void resetStub() {
    left_camera = StubCamera {};
    right_camera = StubCamera {};
    left_info = MV_CC_DEVICE_INFO {};
    right_info = MV_CC_DEVICE_INFO {};
    left_info.nTLayerType = MV_USB_DEVICE;
    right_info.nTLayerType = MV_GIGE_DEVICE;
    setText(left_info.SpecialInfo.stUsb3VInfo.chSerialNumber, "left-001");
    setText(left_info.SpecialInfo.stUsb3VInfo.chModelName, "MV-LEFT");
    setText(right_info.SpecialInfo.stGigEInfo.chSerialNumber, "right-002");
    setText(right_info.SpecialInfo.stGigEInfo.chModelName, "MV-RIGHT");
}

MV_CC_DEVICE_INFO_LIST deviceList() {
    MV_CC_DEVICE_INFO_LIST list {};
    list.nDeviceNum = 2;
    list.pDeviceInfo = device_infos;
    return list;
}

void testIdentityAndResolution() {
    resetStub();
    const auto list = deviceList();
    check(hik_metadata_probe::serialOf(&left_info) == "left-001",
          "USB serial must be read from the USB device block");
    check(hik_metadata_probe::modelOf(&right_info) == "MV-RIGHT",
          "GigE model must be read from the GigE device block");
    check(std::string(hik_metadata_probe::layerName(&left_info)) == "USB",
          "USB transport must have a stable display name");
    check(hik_metadata_probe::resolveCameraInfo(list, "right-002", 0) ==
              &right_info,
          "serial selection must take precedence over index");
    check(hik_metadata_probe::resolveCameraInfo(list, "", 0) == &left_info,
          "index selection must resolve an enumerated camera");
    check(hik_metadata_probe::resolveCameraInfo(list, "missing", 0) == nullptr,
          "unknown serial must not silently fall back to index");
    check(hik_metadata_probe::resolveCameraInfo(list, "", 2) == nullptr,
          "out-of-range index must fail resolution");
}

void testSessionOwnsSdkLifecycle() {
    resetStub();
    hik_metadata_probe::CameraSession camera;
    check(
        hik_metadata_probe::openCamera(
            deviceList(), "left-001", 1, "TEST", camera),
        "session must open the selected camera");
    check(camera.handle() == &left_camera && camera.serial() == "left-001",
          "session must expose the resolved handle and serial");
    check(camera.startGrabbing("TEST"),
          "session must start acquisition through the SDK");
    camera.shutdown();
    check(left_camera.stopped && left_camera.closed && left_camera.destroyed,
          "shutdown must stop, close and destroy an active camera");
    check(camera.handle() == nullptr && camera.serial().empty(),
          "shutdown must clear stale session identity");
}

void testConfigurationRequiresCoreMetadata() {
    resetStub();
    hik_metadata_probe::Args args;
    check(hik_metadata_probe::configureCamera(&left_camera, args, "TEST"),
          "complete trigger and metadata configuration must succeed");
    check(left_camera.enabled_selectors.count("Timestamp") == 1 &&
              left_camera.enabled_selectors.count("Framecounter") == 1 &&
              left_camera.enabled_selectors.count("ExtTriggerCount") == 1,
          "all three core metadata selectors must be enabled");
    check(left_camera.enabled_selectors.size() == 3,
          "non-core persistent metadata selectors must be disabled");

    resetStub();
    left_camera.fail_selector = "Framecounter";
    check(!hik_metadata_probe::configureCamera(&left_camera, args, "TEST"),
          "missing Framecounter selector must be fatal");

    resetStub();
    left_camera.fail_enable = "ExtTriggerCount";
    check(!hik_metadata_probe::configureCamera(&left_camera, args, "TEST"),
          "failure to enable ExtTriggerCount must be fatal");

    resetStub();
    left_camera.false_readback_selector = "Timestamp";
    check(!hik_metadata_probe::configureCamera(&left_camera, args, "TEST"),
          "required metadata readback mismatch must be fatal");

    resetStub();
    left_camera.fail_selector = "Exposure";
    check(hik_metadata_probe::configureCamera(&left_camera, args, "TEST"),
          "optional metadata selector failure must remain diagnostic only");

    resetStub();
    left_camera.fail_pixel_format_enum = true;
    check(hik_metadata_probe::configureCamera(&left_camera, args, "TEST"),
          "string pixel-format fallback must support SDK enum differences");
    resetStub();
    left_camera.fail_pixel_format_string = true;
    left_camera.fail_pixel_format_enum = true;
    check(!hik_metadata_probe::configureCamera(&left_camera, args, "TEST"),
          "configuration must fail when both pixel-format paths fail");

    resetStub();
    left_camera.fail_frame_rate_disable = true;
    check(hik_metadata_probe::configureCamera(&left_camera, args, "TEST"),
          "unavailable internal frame-rate node must remain optional");

    resetStub();
    left_camera.exposure_readback = 9000.0F;
    check(!hik_metadata_probe::configureCamera(&left_camera, args, "TEST"),
          "exposure readback outside tolerance must be fatal");
}

void testGrabCopiesMetadataAndChecksRelease() {
    resetStub();
    const auto metadata = hik_metadata_probe::grabMeta(&left_camera, 25);
    check(metadata.ok && metadata.ret == MV_OK,
          "successful acquisition and release must produce a valid record");
    check(metadata.frame_num == 17 && metadata.frame_counter == 23 &&
              metadata.trigger_index == 29,
          "frame identity fields must be copied from the SDK payload");
    check(metadata.dev_ts == ((std::uint64_t {3} << 32U) | 5U),
          "split device timestamp must be reconstructed without truncation");
    check(metadata.width == 1920 && metadata.height == 1080 &&
              metadata.frame_len == 4,
          "extended dimensions and payload length must be preferred");
    check(metadata.first_bytes[0] == 0x11 &&
              metadata.first_bytes[3] == 0x44 &&
              metadata.first_bytes[4] == 0,
          "watermark preview must not read beyond a short SDK payload");

    resetStub();
    left_camera.fail_free = true;
    const auto release_failure = hik_metadata_probe::grabMeta(&left_camera, 25);
    check(!release_failure.ok && release_failure.ret != MV_OK,
          "SDK buffer release failure must invalidate the frame");
}

}  // namespace

extern "C" {

int MV_CC_EnumDevices(unsigned int, MV_CC_DEVICE_INFO_LIST* list) {
    *list = deviceList();
    return MV_OK;
}

int MV_CC_CreateHandle(void** handle, MV_CC_DEVICE_INFO* info) {
    *handle = info == &left_info
        ? static_cast<void*>(&left_camera)
        : static_cast<void*>(&right_camera);
    return MV_OK;
}

int MV_CC_OpenDevice(void* handle) {
    static_cast<StubCamera*>(handle)->opened = true;
    return MV_OK;
}

int MV_CC_CloseDevice(void* handle) {
    static_cast<StubCamera*>(handle)->closed = true;
    return MV_OK;
}

int MV_CC_DestroyHandle(void* handle) {
    static_cast<StubCamera*>(handle)->destroyed = true;
    return MV_OK;
}

int MV_CC_SetEnumValue(void* handle, const char* node, unsigned int) {
    auto* camera = static_cast<StubCamera*>(handle);
    if (std::string(node) == "PixelFormat" &&
        camera->fail_pixel_format_enum) {
        return -11;
    }
    return MV_OK;
}

int MV_CC_SetEnumValueByString(
    void* handle,
    const char* node,
    const char* value) {
    auto* camera = static_cast<StubCamera*>(handle);
    const std::string node_name(node);
    const std::string selected_value(value);
    if (node_name == "PixelFormat" &&
        camera->fail_pixel_format_string) {
        return -12;
    }
    if (node_name == "FrameSpecInfoSelector") {
        if (camera->fail_selector == selected_value) return -13;
        camera->selected_selector = selected_value;
    }
    return MV_OK;
}

int MV_CC_SetFloatValue(void*, const char*, float) { return MV_OK; }

int MV_CC_SetBoolValue(void* handle, const char* node, bool value) {
    auto* camera = static_cast<StubCamera*>(handle);
    const std::string node_name(node);
    if (node_name == "AcquisitionFrameRateEnable" &&
        camera->fail_frame_rate_disable) {
        return -16;
    }
    if (node_name == "FrameSpecInfo") {
        if (value && camera->fail_enable == camera->selected_selector) {
            return -14;
        }
        if (value) {
            camera->enabled_selectors.insert(camera->selected_selector);
        } else {
            camera->enabled_selectors.erase(camera->selected_selector);
        }
    }
    return MV_OK;
}

int MV_CC_GetEnumValue(void*, const char* node, MVCC_ENUMVALUE* value) {
    const std::string node_name(node);
    value->nCurValue = node_name == "TriggerMode" ? 1U : 0U;
    return MV_OK;
}

int MV_CC_GetFloatValue(
    void* handle,
    const char* node,
    MVCC_FLOATVALUE* value) {
    const auto* camera = static_cast<StubCamera*>(handle);
    value->fCurValue = std::string(node) == "ExposureTime"
        ? camera->exposure_readback
        : camera->gain_readback;
    return MV_OK;
}

int MV_CC_GetBoolValue(void* handle, const char* node, bool* value) {
    const auto* camera = static_cast<StubCamera*>(handle);
    if (std::string(node) == "FrameSpecInfo") {
        *value = camera->false_readback_selector == camera->selected_selector
            ? false
            : camera->enabled_selectors.count(camera->selected_selector) != 0;
    } else {
        *value = false;
    }
    return MV_OK;
}

int MV_CC_GetIntValue(void*, const char*, MVCC_INTVALUE* value) {
    value->nCurValue = 1;
    return MV_OK;
}

int MV_CC_SetGrabStrategy(void*, unsigned int) { return MV_OK; }
int MV_CC_SetImageNodeNum(void*, unsigned int) { return MV_OK; }

int MV_CC_StartGrabbing(void* handle) {
    static_cast<StubCamera*>(handle)->started = true;
    return MV_OK;
}

int MV_CC_StopGrabbing(void* handle) {
    auto* camera = static_cast<StubCamera*>(handle);
    camera->started = false;
    camera->stopped = true;
    return MV_OK;
}

int MV_CC_GetImageBuffer(void* handle, MV_FRAME_OUT* frame, unsigned int) {
    auto* camera = static_cast<StubCamera*>(handle);
    frame->pBufAddr = camera->image.data();
    frame->stFrameInfo.nWidth = 640;
    frame->stFrameInfo.nHeight = 480;
    frame->stFrameInfo.nExtendWidth = 1920;
    frame->stFrameInfo.nExtendHeight = 1080;
    frame->stFrameInfo.nFrameLen = 3;
    frame->stFrameInfo.nFrameLenEx = camera->image.size();
    frame->stFrameInfo.nDevTimeStampHigh = 3;
    frame->stFrameInfo.nDevTimeStampLow = 5;
    frame->stFrameInfo.nHostTimeStamp = 7;
    frame->stFrameInfo.nFrameNum = 17;
    frame->stFrameInfo.nFrameCounter = 23;
    frame->stFrameInfo.nTriggerIndex = 29;
    frame->stFrameInfo.nSecondCount = 31;
    frame->stFrameInfo.nCycleCount = 37;
    frame->stFrameInfo.nCycleOffset = 41;
    frame->stFrameInfo.fGain = 11.9F;
    frame->stFrameInfo.fExposureTime = 9867.0F;
    frame->stFrameInfo.nInput = 43;
    frame->stFrameInfo.nOutput = 47;
    return MV_OK;
}

int MV_CC_FreeImageBuffer(void* handle, MV_FRAME_OUT*) {
    return static_cast<StubCamera*>(handle)->fail_free ? -15 : MV_OK;
}

}  // extern "C"

int main() {
    testIdentityAndResolution();
    testSessionOwnsSdkLifecycle();
    testConfigurationRequiresCoreMetadata();
    testGrabCopiesMetadataAndChecksRelease();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "hik metadata camera tests passed\n";
    return 0;
}
