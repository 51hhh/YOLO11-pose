#ifndef STEREO_3D_PIPELINE_TRT_DETECTOR_H_
#define STEREO_3D_PIPELINE_TRT_DETECTOR_H_

#include "volleyball/adapters/tensorrt/runtime.hpp"

namespace stereo3d {

// 兼容期只保留旧名称；TensorRT 资源及实现归独立 adapter 所有。
using TRTDetector = volleyball::adapters::tensorrt::TensorRtRuntime;

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_TRT_DETECTOR_H_
