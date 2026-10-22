#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/image_buffer.hpp"

namespace volleyball::ports {

// 校正图像只引用原始帧身份，不复制或改写任何采集时间事实。
// detector_input 使用 BGR8；geometry_gray 使用同一校正模型的 Gray8。
struct RectifiedCameraFrame {
    volleyball::core::FrameReference source;
    ImageBufferPtr detector_input;
    ImageBufferPtr geometry_gray;
};

struct RectifiedStereoFrame {
    RectifiedCameraFrame left;
    RectifiedCameraFrame right;
};

}  // namespace volleyball::ports
