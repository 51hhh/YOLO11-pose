#include "volleyball/adapters/tensorrt/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace volleyball::adapters::tensorrt {
namespace {

bool validConfig(
    const PostprocessConfig& config,
    float scale_x,
    float scale_y) noexcept {
    return std::isfinite(config.confidence_threshold) &&
           config.confidence_threshold >= 0.0F &&
           config.confidence_threshold <= 1.0F &&
           std::isfinite(config.nms_iou_threshold) &&
           config.nms_iou_threshold >= 0.0F &&
           config.nms_iou_threshold <= 1.0F &&
           std::isfinite(scale_x) && scale_x > 0.0F &&
           std::isfinite(scale_y) && scale_y > 0.0F;
}

bool checkedProduct(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t& product) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return false;
    }
    product = lhs * rhs;
    return true;
}

float intersectionOverUnion(
    const Detection& left,
    const Detection& right) noexcept {
    const float left_x1 = left.cx - left.width * 0.5F;
    const float left_y1 = left.cy - left.height * 0.5F;
    const float left_x2 = left.cx + left.width * 0.5F;
    const float left_y2 = left.cy + left.height * 0.5F;
    const float right_x1 = right.cx - right.width * 0.5F;
    const float right_y1 = right.cy - right.height * 0.5F;
    const float right_x2 = right.cx + right.width * 0.5F;
    const float right_y2 = right.cy + right.height * 0.5F;

    const float intersection_width =
        std::max(0.0F, std::min(left_x2, right_x2) -
                           std::max(left_x1, right_x1));
    const float intersection_height =
        std::max(0.0F, std::min(left_y2, right_y2) -
                           std::max(left_y1, right_y1));
    const float intersection = intersection_width * intersection_height;
    const float left_area = std::max(0.0F, left.width) *
                            std::max(0.0F, left.height);
    const float right_area = std::max(0.0F, right.width) *
                             std::max(0.0F, right.height);
    const float denominator = left_area + right_area - intersection;
    return denominator > 0.0F ? intersection / denominator : 0.0F;
}

}  // namespace

std::vector<Detection> nonMaxSuppression(
    const std::vector<Detection>& detections,
    float iou_threshold) {
    if (detections.empty()) {
        return {};
    }

    std::vector<std::size_t> order(detections.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(
        order.begin(), order.end(),
        [&detections](std::size_t left, std::size_t right) {
            return detections[left].confidence > detections[right].confidence;
        });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> selected;
    selected.reserve(detections.size());
    for (const std::size_t index : order) {
        if (suppressed[index]) {
            continue;
        }
        selected.push_back(detections[index]);
        for (const std::size_t candidate : order) {
            if (candidate == index || suppressed[candidate]) {
                continue;
            }
            if (intersectionOverUnion(
                    detections[index], detections[candidate]) >
                iou_threshold) {
                suppressed[candidate] = true;
            }
        }
    }
    return selected;
}

PostprocessResult decodeSingleOutput(
    const SingleOutputView& output,
    float scale_x,
    float scale_y,
    const PostprocessConfig& config) {
    PostprocessResult result;
    if (!validConfig(config, scale_x, scale_y)) {
        result.message = "invalid TensorRT postprocess configuration";
        return result;
    }
    if (output.data == nullptr || output.dim1 <= 0 || output.dim2 <= 0) {
        result.message = "single-output tensor view is empty";
        return result;
    }
    std::size_t required_elements = 0;
    if (!checkedProduct(
            static_cast<std::size_t>(output.dim1),
            static_cast<std::size_t>(output.dim2),
            required_elements) ||
        output.element_count < required_elements) {
        result.message = "single-output tensor buffer is shorter than its shape";
        return result;
    }

    const float* data = output.data;
    if (output.dim2 == 6 && output.dim1 <= 1000) {
        result.detections.reserve(static_cast<std::size_t>(output.dim1));
        for (int index = 0; index < output.dim1; ++index) {
            const float* row = data + index * 6;
            const float confidence = row[4];
            if (!std::isfinite(confidence) ||
                confidence < config.confidence_threshold) {
                continue;
            }
            const bool valid_box =
                std::isfinite(row[0]) && std::isfinite(row[1]) &&
                std::isfinite(row[2]) && std::isfinite(row[3]) &&
                row[2] > row[0] && row[3] > row[1];
            const bool valid_class =
                std::isfinite(row[5]) && row[5] >= 0.0F &&
                row[5] <= static_cast<float>(
                    std::numeric_limits<int>::max());
            if (!valid_box || !valid_class) {
                result.detections.clear();
                result.message =
                    "post-NMS tensor contains an invalid accepted detection";
                return result;
            }
            result.detections.push_back({
                (row[0] + row[2]) * 0.5F * scale_x,
                (row[1] + row[3]) * 0.5F * scale_y,
                (row[2] - row[0]) * scale_x,
                (row[3] - row[1]) * scale_y,
                confidence,
                static_cast<int>(row[5]),
            });
        }
        result.success = true;
        return result;
    }

    const bool transposed = output.dim1 < output.dim2;
    const int detection_count = transposed ? output.dim2 : output.dim1;
    const int channel_count = transposed ? output.dim1 : output.dim2;
    const int class_count = channel_count - 4;
    if (class_count <= 0) {
        result.message = "pre-NMS tensor has no class score";
        return result;
    }

    std::vector<Detection> raw;
    raw.reserve(static_cast<std::size_t>(detection_count));
    for (int index = 0; index < detection_count; ++index) {
        const auto value = [&](int channel) {
            return transposed
                ? data[channel * detection_count + index]
                : data[index * channel_count + channel];
        };
        float confidence = -std::numeric_limits<float>::infinity();
        int class_id = 0;
        for (int class_index = 0; class_index < class_count; ++class_index) {
            const float score = value(4 + class_index);
            if (score > confidence) {
                confidence = score;
                class_id = class_index;
            }
        }
        if (!std::isfinite(confidence) ||
            confidence < config.confidence_threshold) {
            continue;
        }
        const float cx = value(0);
        const float cy = value(1);
        const float width = value(2);
        const float height = value(3);
        if (!std::isfinite(cx) || !std::isfinite(cy) ||
            !std::isfinite(width) || !std::isfinite(height) ||
            width <= 0.0F || height <= 0.0F) {
            result.message =
                "pre-NMS tensor contains an invalid accepted detection";
            return result;
        }
        raw.push_back({
            cx * scale_x,
            cy * scale_y,
            width * scale_x,
            height * scale_y,
            confidence,
            class_id,
        });
    }
    result.detections =
        nonMaxSuppression(raw, config.nms_iou_threshold);
    result.success = true;
    return result;
}

PostprocessResult decodeDflOutputs(
    const std::vector<DflScaleView>& scales,
    int regression_bins,
    float scale_x,
    float scale_y,
    const PostprocessConfig& config) {
    PostprocessResult result;
    if (!validConfig(config, scale_x, scale_y) || regression_bins <= 0) {
        result.message = "invalid DFL postprocess configuration";
        return result;
    }
    if (scales.empty()) {
        result.message = "DFL postprocess received no output scales";
        return result;
    }

    std::vector<Detection> raw;
    for (const auto& scale : scales) {
        if (scale.classification == nullptr || scale.boxes == nullptr ||
            scale.height <= 0 || scale.width <= 0 ||
            scale.class_count <= 0 || scale.stride <= 0) {
            result.message = "DFL scale view is incomplete";
            return result;
        }
        std::size_t cells = 0;
        std::size_t required_classification = 0;
        std::size_t box_channels = 0;
        std::size_t required_boxes = 0;
        if (!checkedProduct(
                static_cast<std::size_t>(scale.height),
                static_cast<std::size_t>(scale.width), cells) ||
            !checkedProduct(
                cells,
                static_cast<std::size_t>(scale.class_count),
                required_classification) ||
            !checkedProduct(
                std::size_t{4},
                static_cast<std::size_t>(regression_bins),
                box_channels) ||
            !checkedProduct(cells, box_channels, required_boxes) ||
            scale.classification_elements < required_classification ||
            scale.box_elements < required_boxes) {
            result.message = "DFL tensor buffer is shorter than its shape";
            return result;
        }

        for (int y = 0; y < scale.height; ++y) {
            for (int x = 0; x < scale.width; ++x) {
                const std::size_t offset = static_cast<std::size_t>(
                    y * scale.width + x);
                float confidence = 0.0F;
                int class_id = 0;
                for (int class_index = 0;
                     class_index < scale.class_count;
                     ++class_index) {
                    const float logit = scale.classification[
                        offset * static_cast<std::size_t>(scale.class_count) +
                        static_cast<std::size_t>(class_index)];
                    if (!std::isfinite(logit)) {
                        result.message =
                            "DFL classification tensor contains a non-finite value";
                        return result;
                    }
                    const float score = 1.0F / (1.0F + std::exp(-logit));
                    if (score > confidence) {
                        confidence = score;
                        class_id = class_index;
                    }
                }
                if (confidence < config.confidence_threshold) {
                    continue;
                }

                const float* box = scale.boxes + offset * box_channels;
                float distances[4] = {};
                for (int edge = 0; edge < 4; ++edge) {
                    const float* logits =
                        box + edge * regression_bins;
                    if (!std::isfinite(logits[0])) {
                        result.message =
                            "DFL box tensor contains a non-finite value";
                        return result;
                    }
                    float maximum = logits[0];
                    for (int bin = 1; bin < regression_bins; ++bin) {
                        if (!std::isfinite(logits[bin])) {
                            result.message =
                                "DFL box tensor contains a non-finite value";
                            return result;
                        }
                        maximum = std::max(maximum, logits[bin]);
                    }
                    float denominator = 0.0F;
                    float numerator = 0.0F;
                    for (int bin = 0; bin < regression_bins; ++bin) {
                        const float weight = std::exp(logits[bin] - maximum);
                        denominator += weight;
                        numerator += weight * static_cast<float>(bin);
                    }
                    if (!std::isfinite(denominator) || denominator <= 0.0F ||
                        !std::isfinite(numerator)) {
                        result.message = "DFL softmax normalization failed";
                        return result;
                    }
                    distances[edge] = numerator / denominator;
                }

                const float anchor_x =
                    (static_cast<float>(x) + 0.5F) * scale.stride;
                const float anchor_y =
                    (static_cast<float>(y) + 0.5F) * scale.stride;
                const float x1 = anchor_x - distances[0] * scale.stride;
                const float y1 = anchor_y - distances[1] * scale.stride;
                const float x2 = anchor_x + distances[2] * scale.stride;
                const float y2 = anchor_y + distances[3] * scale.stride;
                raw.push_back({
                    (x1 + x2) * 0.5F * scale_x,
                    (y1 + y2) * 0.5F * scale_y,
                    (x2 - x1) * scale_x,
                    (y2 - y1) * scale_y,
                    confidence,
                    class_id,
                });
            }
        }
    }

    result.detections =
        nonMaxSuppression(raw, config.nms_iou_threshold);
    result.success = true;
    return result;
}

}  // namespace volleyball::adapters::tensorrt
