#pragma once

#include "volleyball/ports/captured_frame.hpp"

#include <optional>
#include <string>

namespace volleyball::ports {

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

