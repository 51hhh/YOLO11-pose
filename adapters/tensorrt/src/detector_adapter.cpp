#include "volleyball/adapters/tensorrt/detector_adapter.hpp"

#include "volleyball/adapters/tensorrt/detector_mapping.hpp"
#include "volleyball/adapters/tensorrt/runtime.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace volleyball::adapters::tensorrt {
namespace {

bool validConfig(const TensorRtDetectorConfig& config) noexcept {
    return !config.engine_file.empty() && !config.pipeline_id.empty() &&
           config.expected_input_size > 0 &&
           std::isfinite(config.confidence_threshold) &&
           config.confidence_threshold >= 0.0F &&
           config.confidence_threshold <= 1.0F &&
           std::isfinite(config.nms_iou_threshold) &&
           config.nms_iou_threshold >= 0.0F &&
           config.nms_iou_threshold <= 1.0F &&
           config.maximum_detections > 0 && config.expected_class_id >= 0 &&
           config.dla_core >= 0;
}

ports::DetectionResult failure(
    const ports::RectifiedCameraFrame& frame,
    const core::PipelineId& pipeline_id,
    std::string message) {
    ports::DetectionResult result;
    result.batch.frame = frame.source;
    result.batch.pipeline_id = pipeline_id;
    result.message = std::move(message);
    return result;
}

}  // namespace

struct TensorRtDetector::Impl {
    explicit Impl(TensorRtDetectorConfig value) : config(std::move(value)) {}

    ~Impl() {
        if (stream != nullptr) {
            static_cast<void>(cudaStreamSynchronize(stream));
        }
        runtime.reset();
        if (stream != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }

    TensorRtDetectorConfig config;
    std::unique_ptr<TensorRtRuntime> runtime;
    cudaStream_t stream{nullptr};
    bool initialized{false};
};

TensorRtDetector::TensorRtDetector(TensorRtDetectorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

TensorRtDetector::~TensorRtDetector() = default;

bool TensorRtDetector::initialize(std::string& error) {
    error.clear();
    if (impl_->initialized) {
        error = "TensorRT detector is already initialized";
        return false;
    }
    if (!validConfig(impl_->config)) {
        error = "TensorRT detector configuration is invalid";
        return false;
    }

    const cudaError_t stream_status =
        cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking);
    if (stream_status != cudaSuccess) {
        error = std::string{"failed to create TensorRT CUDA stream: "} +
                cudaGetErrorString(stream_status);
        impl_->stream = nullptr;
        return false;
    }

    impl_->runtime = std::make_unique<TensorRtRuntime>();
    if (!impl_->runtime->init(
            impl_->config.engine_file,
            impl_->config.use_dla,
            impl_->config.dla_core,
            impl_->config.confidence_threshold,
            impl_->config.nms_iou_threshold,
            "bgr")) {
        error = "failed to initialize TensorRT detector runtime";
        impl_->runtime.reset();
        static_cast<void>(cudaStreamDestroy(impl_->stream));
        impl_->stream = nullptr;
        return false;
    }
    if (impl_->runtime->getInputSize() != impl_->config.expected_input_size) {
        error = "TensorRT engine input size does not match detector profile";
        impl_->runtime.reset();
        static_cast<void>(cudaStreamDestroy(impl_->stream));
        impl_->stream = nullptr;
        return false;
    }

    impl_->initialized = true;
    return true;
}

bool TensorRtDetector::isInitialized() const noexcept {
    return impl_->initialized;
}

ports::DetectionResult TensorRtDetector::detect(
    const ports::RectifiedCameraFrame& frame) {
    if (!impl_->initialized || !impl_->runtime || impl_->stream == nullptr) {
        return failure(
            frame, impl_->config.pipeline_id,
            "TensorRT detector is not initialized");
    }

    const auto& image = frame.detector_input;
    if (!image || image->memoryKind() != ports::MemoryKind::cuda_device ||
        image->pixelFormat() != core::PixelFormat::bgr8 ||
        image->nativeHandle() == 0 || image->width() == 0 ||
        image->height() == 0) {
        return failure(
            frame, impl_->config.pipeline_id,
            "TensorRT detector requires a CUDA-device BGR8 image");
    }

    const std::size_t width = image->width();
    const std::size_t height = image->height();
    const std::size_t stride = image->strideBytes();
    if (width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        stride > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        width > std::numeric_limits<std::size_t>::max() / 3 ||
        stride < width * 3 ||
        height > std::numeric_limits<std::size_t>::max() / stride ||
        image->sizeBytes() < stride * height) {
        return failure(
            frame, impl_->config.pipeline_id,
            "TensorRT detector image geometry or capacity is invalid");
    }

    auto runtime_result = impl_->runtime->detectChecked(
        reinterpret_cast<const void*>(image->nativeHandle()),
        static_cast<int>(stride),
        static_cast<int>(width),
        static_cast<int>(height),
        impl_->stream);
    if (!runtime_result.success) {
        return failure(
            frame,
            impl_->config.pipeline_id,
            runtime_result.message.empty()
                ? "TensorRT detector runtime failed"
                : runtime_result.message);
    }

    return mapDetections(
        frame,
        std::move(runtime_result.detections),
        {
            impl_->config.pipeline_id,
            impl_->config.maximum_detections,
            impl_->config.expected_class_id,
        });
}

}  // namespace volleyball::adapters::tensorrt
