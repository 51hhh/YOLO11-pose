#pragma once

#include "volleyball/adapters/tensorrt/postprocess.hpp"
#include "volleyball/ports/detector.hpp"

#include <cstddef>
#include <vector>

namespace volleyball::adapters::tensorrt {

struct DetectorMappingConfig {
    core::PipelineId pipeline_id;
    std::size_t maximum_detections{10};
    int expected_class_id{0};
};

// 将已解码的 TensorRT 候选转换成主线事实类型。当前 Detection2D 契约
// 不携带类别，因此遇到非目标类别必须失败，不能静默丢失类别证据。
[[nodiscard]] ports::DetectionResult mapDetections(
    const ports::RectifiedCameraFrame& frame,
    std::vector<Detection> detections,
    const DetectorMappingConfig& config);

}  // namespace volleyball::adapters::tensorrt
