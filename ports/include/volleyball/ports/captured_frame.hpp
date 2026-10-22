#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/image_buffer.hpp"

namespace volleyball::ports {

// 帧数据与相机生命周期接口分离，使检测器和写入器无需依赖 CameraSource。
struct CapturedFrame {
    volleyball::core::CameraFrameRecord metadata;
    ImageBufferPtr image;
};

struct StereoFrame {
    CapturedFrame left;
    CapturedFrame right;
};

}  // namespace volleyball::ports
