#pragma once

#include "volleyball/adapters/tensorrt/engine_contract.hpp"
#include "volleyball/ports/detector.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace volleyball::adapters::tensorrt {

struct TensorRtDetectorConfig {
    std::string engine_file;
    core::PipelineId pipeline_id;
    int expected_input_size{0};
    DetectorOutputContract expected_output_contract{
        DetectorOutputContract::none};
    int expected_class_count{0};
    float confidence_threshold{0.5F};
    float nms_iou_threshold{0.4F};
    std::size_t maximum_detections{10};
    int expected_class_id{0};
    bool use_dla{false};
    int dla_core{0};
};

// 同步的主线 Detector 端口桥。每个实例拥有独立 CUDA stream 和
// TensorRT runtime；调用方为左右相机分别创建实例，且不得并发调用同一实例。
class TensorRtDetector final : public ports::Detector {
public:
    explicit TensorRtDetector(TensorRtDetectorConfig config);
    ~TensorRtDetector() override;

    TensorRtDetector(const TensorRtDetector&) = delete;
    TensorRtDetector& operator=(const TensorRtDetector&) = delete;

    [[nodiscard]] bool initialize(std::string& error);
    [[nodiscard]] bool isInitialized() const noexcept;
    [[nodiscard]] ports::DetectionResult detect(
        const ports::RectifiedCameraFrame& frame) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volleyball::adapters::tensorrt
