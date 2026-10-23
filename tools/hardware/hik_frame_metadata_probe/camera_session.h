#ifndef VOLLEYBALL_HIK_METADATA_PROBE_CAMERA_SESSION_H_
#define VOLLEYBALL_HIK_METADATA_PROBE_CAMERA_SESSION_H_

#include "MvCameraControl.h"
#include "args.h"

#include <cstdint>
#include <string>

namespace hik_metadata_probe {

struct FrameMeta {
    bool ok = false;
    int ret = 0;
    std::uint32_t frame_num = 0;
    std::uint64_t dev_ts = 0;
    std::int64_t host_ts = 0;
    std::uint32_t frame_counter = 0;
    std::uint32_t trigger_index = 0;
    std::uint32_t second_count = 0;
    std::uint32_t cycle_count = 0;
    std::uint32_t cycle_offset = 0;
    float gain = 0.0F;
    float exposure = 0.0F;
    std::uint32_t input = 0;
    std::uint32_t output = 0;
    std::uint32_t unparsed_chunks = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t frame_len = 0;
    std::uint8_t first_bytes[16] = {};
};

class CameraSession {
public:
    CameraSession() = default;
    ~CameraSession();

    CameraSession(const CameraSession&) = delete;
    CameraSession& operator=(const CameraSession&) = delete;

    [[nodiscard]] void* handle() const noexcept { return handle_; }
    [[nodiscard]] const std::string& serial() const noexcept { return serial_; }
    [[nodiscard]] const std::string& model() const noexcept { return model_; }
    [[nodiscard]] const std::string& firmwareVersion() const noexcept {
        return firmware_version_;
    }
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
    std::string model_;
    std::string firmware_version_;
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
