#include "volleyball/adapters/tensorrt/detector_mapping.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace volleyball::adapters::tensorrt {
namespace {

bool validDetection(const Detection& detection) noexcept {
    return std::isfinite(detection.cx) && std::isfinite(detection.cy) &&
           std::isfinite(detection.width) &&
           std::isfinite(detection.height) &&
           std::isfinite(detection.confidence) && detection.width > 0.0F &&
           detection.height > 0.0F && detection.confidence >= 0.0F &&
           detection.confidence <= 1.0F;
}

}  // namespace

ports::DetectionResult mapDetections(
    const ports::RectifiedCameraFrame& frame,
    std::vector<Detection> detections,
    const DetectorMappingConfig& config) {
    ports::DetectionResult result;
    result.batch.frame = frame.source;
    result.batch.pipeline_id = config.pipeline_id;

    if (config.pipeline_id.empty() || config.maximum_detections == 0) {
        result.message = "TensorRT detection mapping configuration is invalid";
        return result;
    }
    if (!frame.detector_input || frame.detector_input->width() == 0 ||
        frame.detector_input->height() == 0) {
        result.message = "TensorRT detection mapping received no input image";
        return result;
    }

    const double image_width = frame.detector_input->width();
    const double image_height = frame.detector_input->height();
    for (const auto& detection : detections) {
        if (!validDetection(detection)) {
            result.message = "TensorRT produced an invalid detection";
            return result;
        }
        if (detection.class_id != config.expected_class_id) {
            result.message =
                "TensorRT produced a class not representable by Detection2D";
            return result;
        }
        if (detection.cx < 0.0F || detection.cy < 0.0F ||
            detection.cx >= image_width || detection.cy >= image_height) {
            result.message = "TensorRT detection center lies outside the image";
            return result;
        }
    }

    std::stable_sort(
        detections.begin(), detections.end(),
        [](const Detection& left, const Detection& right) {
            return left.confidence > right.confidence;
        });
    if (detections.size() > config.maximum_detections) {
        detections.resize(config.maximum_detections);
    }

    result.batch.candidates.reserve(detections.size());
    for (std::size_t index = 0; index < detections.size(); ++index) {
        const auto& detection = detections[index];
        core::Detection2D candidate;
        candidate.frame = frame.source;
        candidate.candidate_id = static_cast<core::CandidateId>(index);
        candidate.center_px = {detection.cx, detection.cy};
        candidate.bounding_box = core::BoundingBox2D{
            detection.cx - detection.width * 0.5F,
            detection.cy - detection.height * 0.5F,
            detection.width,
            detection.height,
        };
        candidate.score = detection.confidence;
        candidate.selected = false;
        candidate.source_pipeline = config.pipeline_id;
        result.batch.candidates.push_back(std::move(candidate));
    }

    result.success = true;
    return result;
}

}  // namespace volleyball::adapters::tensorrt
