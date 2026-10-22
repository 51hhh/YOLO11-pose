#pragma once

#include "volleyball/core/ids.hpp"

namespace volleyball::core {

struct PinholeIntrinsics {
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
};

// 已校正、水平基线的双目模型。算法层只依赖这个值对象，不依赖
// OpenCV 的矩阵或硬件句柄。
struct RectifiedStereoCalibration {
    PinholeIntrinsics left;
    PinholeIntrinsics right;
    double baseline_m{0.0};
    CoordinateFrameId rig_frame;
};

}  // namespace volleyball::core
