#pragma once

#include "volleyball/adapters/cuda_preprocess/stereo_calibration.hpp"

namespace stereo3d {

// 兼容旧 NX 源码的临时类型名；实现和 OpenCV 依赖归属 cuda_preprocess。
class StereoCalibration final
    : public volleyball::adapters::cuda_preprocess::StereoCalibration {};

}  // namespace stereo3d
