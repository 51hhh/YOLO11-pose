#ifndef VOLLEYBALL_HIK_METADATA_PROBE_CAMERA_SESSION_H_
#define VOLLEYBALL_HIK_METADATA_PROBE_CAMERA_SESSION_H_

#include "MvCameraControl.h"
#include "args.h"
#include "stats.h"

#include <cstdint>
#include <string>

namespace hik_metadata_probe {

class CameraSession {
public:
    CameraSession() = default;
    ~CameraSession();

    CameraSession(const CameraSession&) = delete;
    CameraSession& operator=(const CameraSession&) = delete;

    [[nodiscard]] void* handle() const noexcept { return handle_; }
    [[nodiscard]] const std::string& serial() const noexcept { return serial_; }
    [[nodiscard]] bool startGrabbing(const char* tag);
    void shutdown() noexcept;

private:
    friend bool openCamera(
        const MV_CC_DEVICE_INFO_LIST&,
        const std::string&,
        int,
        const char*,
        CameraSession&);

    void* handle_ = nullptr;
    std::string serial_;
    bool grabbing_ = false;
};

[[nodiscard]] std::uint64_t deviceTimestamp(
    const MV_FRAME_OUT_INFO_EX& info) noexcept;
[[nodiscard]] std::string serialOf(const MV_CC_DEVICE_INFO* info);
[[nodiscard]] std::string modelOf(const MV_CC_DEVICE_INFO* info);
[[nodiscard]] const char* layerName(const MV_CC_DEVICE_INFO* info) noexcept;
[[nodiscard]] const MV_CC_DEVICE_INFO* resolveCameraInfo(
    const MV_CC_DEVICE_INFO_LIST& list,
    const std::string& wanted_serial,
    int index);

[[nodiscard]] bool openCamera(
    const MV_CC_DEVICE_INFO_LIST& list,
    const std::string& wanted_serial,
    int index,
    const char* tag,
    CameraSession& camera);
[[nodiscard]] bool configureCamera(
    void* handle,
    const Args& args,
    const char* tag);
void printNodeReadback(void* handle, const char* tag);
[[nodiscard]] FrameMeta grabMeta(void* handle, int timeout_ms);

}  // namespace hik_metadata_probe

#endif  // VOLLEYBALL_HIK_METADATA_PROBE_CAMERA_SESSION_H_
