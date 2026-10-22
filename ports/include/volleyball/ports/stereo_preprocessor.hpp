#pragma once

#include "volleyball/ports/captured_frame.hpp"
#include "volleyball/ports/rectified_frame.hpp"

#include <optional>
#include <string>

namespace volleyball::ports {

struct PreprocessResult {
    bool success{false};
    std::optional<RectifiedStereoFrame> frame;
    std::string message;
};

// 实现负责去 Bayer、双目标定 remap 以及灰度转换。CUDA/VPI 句柄和
// stream 所有权仍由 adapter 管理，不得泄漏到服务层。
class StereoPreprocessor {
public:
    virtual ~StereoPreprocessor() = default;
    [[nodiscard]] virtual PreprocessResult prepare(
        const StereoFrame& frame) = 0;
};

}  // namespace volleyball::ports
