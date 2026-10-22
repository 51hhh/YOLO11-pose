#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/image_buffer.hpp"

#include <optional>
#include <string>

namespace volleyball::ports {

struct CapturedFrame {
    volleyball::core::CameraFrameRecord metadata;
    ImageBufferPtr image;
};

struct StereoFrame {
    CapturedFrame left;
    CapturedFrame right;
};

enum class SourceStatus { frame, end_of_stream, timeout, error };

struct SourceResult {
    SourceStatus status{SourceStatus::error};
    std::optional<StereoFrame> stereo_frame;
    std::string message;
};

class CameraSource {
public:
    virtual ~CameraSource() = default;
    [[nodiscard]] virtual SourceResult next() = 0;
};

}  // namespace volleyball::ports

