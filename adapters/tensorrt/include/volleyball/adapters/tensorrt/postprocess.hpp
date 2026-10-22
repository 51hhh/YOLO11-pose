#pragma once

#include "volleyball/adapters/tensorrt/engine_contract.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace volleyball::adapters::tensorrt {

struct Detection {
    float cx{0.0F};
    float cy{0.0F};
    float width{0.0F};
    float height{0.0F};
    float confidence{0.0F};
    int class_id{0};
};

struct PostprocessConfig {
    float confidence_threshold{0.5F};
    float nms_iou_threshold{0.4F};
};

struct SingleOutputView {
    const float* data{nullptr};
    std::size_t element_count{0};
    int dim1{0};
    int dim2{0};
    int class_count{0};
    SingleOutputLayout layout{SingleOutputLayout::none};
};

struct DflScaleView {
    const float* classification{nullptr};
    std::size_t classification_elements{0};
    const float* boxes{nullptr};
    std::size_t box_elements{0};
    int height{0};
    int width{0};
    int class_count{0};
    int stride{0};
};

struct PostprocessResult {
    bool success{false};
    std::vector<Detection> detections;
    std::string message;
};

[[nodiscard]] std::vector<Detection> nonMaxSuppression(
    const std::vector<Detection>& detections,
    float iou_threshold);

[[nodiscard]] PostprocessResult decodeSingleOutput(
    const SingleOutputView& output,
    float scale_x,
    float scale_y,
    const PostprocessConfig& config);

[[nodiscard]] PostprocessResult decodeDflOutputs(
    const std::vector<DflScaleView>& scales,
    int regression_bins,
    float scale_x,
    float scale_y,
    const PostprocessConfig& config);

}  // namespace volleyball::adapters::tensorrt
