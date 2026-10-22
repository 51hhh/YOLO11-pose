#include "volleyball/adapters/cuda_preprocess/stereo_preprocessor.hpp"

#include "volleyball/adapters/cuda_preprocess/kernels.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace volleyball::adapters::cuda_preprocess {
namespace {

std::string cudaFailure(const char* operation, cudaError_t status) {
    return std::string{operation} + ": " + cudaGetErrorString(status);
}

class DevicePlane {
public:
    DevicePlane() = default;
    ~DevicePlane() { reset(); }

    DevicePlane(const DevicePlane&) = delete;
    DevicePlane& operator=(const DevicePlane&) = delete;

    bool allocate(
        std::size_t row_bytes,
        std::size_t height,
        const char* label,
        std::string& error) {
        reset();
        void* allocation = nullptr;
        std::size_t pitch = 0;
        const cudaError_t status = cudaMallocPitch(
            &allocation, &pitch, row_bytes, height);
        if (status != cudaSuccess) {
            error = cudaFailure(label, status);
            return false;
        }
        data_ = static_cast<std::uint8_t*>(allocation);
        pitch_ = pitch;
        height_ = height;
        return true;
    }

    void reset() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }
        data_ = nullptr;
        pitch_ = 0;
        height_ = 0;
    }

    [[nodiscard]] std::uint8_t* data() noexcept { return data_; }
    [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t pitch() const noexcept { return pitch_; }
    [[nodiscard]] std::size_t sizeBytes() const noexcept {
        return pitch_ * height_;
    }

private:
    std::uint8_t* data_{nullptr};
    std::size_t pitch_{0};
    std::size_t height_{0};
};

class PinnedPlane {
public:
    PinnedPlane() = default;
    ~PinnedPlane() { reset(); }

    PinnedPlane(const PinnedPlane&) = delete;
    PinnedPlane& operator=(const PinnedPlane&) = delete;

    bool allocate(
        std::size_t bytes,
        const char* label,
        std::string& error) {
        reset();
        void* allocation = nullptr;
        const cudaError_t status = cudaHostAlloc(
            &allocation, bytes, cudaHostAllocDefault);
        if (status != cudaSuccess) {
            error = cudaFailure(label, status);
            return false;
        }
        data_ = static_cast<std::uint8_t*>(allocation);
        size_ = bytes;
        return true;
    }

    void reset() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFreeHost(data_));
        }
        data_ = nullptr;
        size_ = 0;
    }

    [[nodiscard]] std::uint8_t* data() noexcept { return data_; }

private:
    std::uint8_t* data_{nullptr};
    std::size_t size_{0};
};

class DeviceMapPair {
public:
    DeviceMapPair() = default;
    ~DeviceMapPair() { reset(); }

    DeviceMapPair(const DeviceMapPair&) = delete;
    DeviceMapPair& operator=(const DeviceMapPair&) = delete;

    bool allocateAndCopy(
        const std::vector<float>& map_x,
        const std::vector<float>& map_y,
        std::uint32_t width,
        std::uint32_t height,
        const char* label,
        std::string& error) {
        reset();
        const std::size_t row_bytes =
            static_cast<std::size_t>(width) * sizeof(float);
        const std::size_t rows = static_cast<std::size_t>(height);
        if (rows > std::numeric_limits<std::size_t>::max() / 2U) {
            error = std::string{label} + " map height overflows";
            return false;
        }

        void* allocation = nullptr;
        const cudaError_t allocation_status = cudaMallocPitch(
            &allocation, &pitch_, row_bytes, rows * 2U);
        if (allocation_status != cudaSuccess) {
            pitch_ = 0;
            error = cudaFailure(label, allocation_status);
            return false;
        }
        base_ = static_cast<std::uint8_t*>(allocation);
        x_ = reinterpret_cast<float*>(base_);
        y_ = reinterpret_cast<float*>(base_ + pitch_ * rows);

        cudaError_t status = cudaMemcpy2D(
            x_, pitch_, map_x.data(), row_bytes, row_bytes, rows,
            cudaMemcpyHostToDevice);
        if (status == cudaSuccess) {
            status = cudaMemcpy2D(
                y_, pitch_, map_y.data(), row_bytes, row_bytes, rows,
                cudaMemcpyHostToDevice);
        }
        if (status != cudaSuccess) {
            error = cudaFailure(label, status);
            reset();
            return false;
        }
        return true;
    }

    void reset() noexcept {
        if (base_ != nullptr) {
            static_cast<void>(cudaFree(base_));
        }
        base_ = nullptr;
        x_ = nullptr;
        y_ = nullptr;
        pitch_ = 0;
    }

    [[nodiscard]] const float* x() const noexcept { return x_; }
    [[nodiscard]] const float* y() const noexcept { return y_; }
    [[nodiscard]] std::size_t pitch() const noexcept { return pitch_; }

private:
    std::uint8_t* base_{nullptr};
    float* x_{nullptr};
    float* y_{nullptr};
    std::size_t pitch_{0};
};

struct CameraSlot {
    PinnedPlane staging;
    DevicePlane raw;
    DevicePlane temporary_bgr;
    DevicePlane rectified_bgr;
    DevicePlane rectified_gray;

    bool allocate(
        std::uint32_t width,
        std::uint32_t height,
        const char* side,
        std::string& error) {
        const std::size_t row = static_cast<std::size_t>(width);
        const std::size_t rows = static_cast<std::size_t>(height);
        if (rows > std::numeric_limits<std::size_t>::max() / row) {
            error = std::string{side} + " staging geometry overflows";
            return false;
        }
        return staging.allocate(
                   row * rows,
                   (std::string{side} + " pinned staging allocation").c_str(),
                   error) &&
               raw.allocate(
                   row, rows,
                   (std::string{side} + " raw CUDA allocation").c_str(),
                   error) &&
               temporary_bgr.allocate(
                   row * 3U, rows,
                   (std::string{side} + " temporary BGR allocation").c_str(),
                   error) &&
               rectified_bgr.allocate(
                   row * 3U, rows,
                   (std::string{side} + " rectified BGR allocation").c_str(),
                   error) &&
               rectified_gray.allocate(
                   row, rows,
                   (std::string{side} + " rectified gray allocation").c_str(),
                   error);
    }
};

struct StereoSlot {
    CameraSlot left;
    CameraSlot right;

    bool allocate(
        std::uint32_t width,
        std::uint32_t height,
        std::string& error) {
        return left.allocate(width, height, "left", error) &&
               right.allocate(width, height, "right", error);
    }
};

class CudaImageView final : public ports::ImageBuffer {
public:
    CudaImageView(
        std::shared_ptr<const StereoSlot> owner,
        const DevicePlane& plane,
        std::uint32_t width,
        std::uint32_t height,
        core::PixelFormat format)
        : owner_(std::move(owner)),
          data_(plane.data()),
          width_(width),
          height_(height),
          stride_(plane.pitch()),
          size_(plane.sizeBytes()),
          format_(format) {}

    [[nodiscard]] ports::MemoryKind memoryKind() const noexcept override {
        return ports::MemoryKind::cuda_device;
    }
    [[nodiscard]] std::uint32_t width() const noexcept override {
        return width_;
    }
    [[nodiscard]] std::uint32_t height() const noexcept override {
        return height_;
    }
    [[nodiscard]] std::size_t strideBytes() const noexcept override {
        return stride_;
    }
    [[nodiscard]] core::PixelFormat pixelFormat() const noexcept override {
        return format_;
    }
    [[nodiscard]] std::uintptr_t nativeHandle() const noexcept override {
        return reinterpret_cast<std::uintptr_t>(data_);
    }
    [[nodiscard]] std::size_t sizeBytes() const noexcept override {
        return size_;
    }

private:
    std::shared_ptr<const StereoSlot> owner_;
    const std::uint8_t* data_{nullptr};
    std::uint32_t width_{0};
    std::uint32_t height_{0};
    std::size_t stride_{0};
    std::size_t size_{0};
    core::PixelFormat format_{core::PixelFormat::gray8};
};

bool validConfig(const CudaStereoPreprocessorConfig& config) noexcept {
    const std::size_t width = config.width;
    const std::size_t height = config.height;
    return width > 0 && height > 0 &&
           width <= static_cast<std::size_t>(
               std::numeric_limits<int>::max()) &&
           height <= static_cast<std::size_t>(
               std::numeric_limits<int>::max()) &&
           width <= std::numeric_limits<std::size_t>::max() / 3U &&
           height <= std::numeric_limits<std::size_t>::max() / width &&
           config.buffer_pool_pairs > 0;
}

bool finiteMaps(const StereoRemapMaps& maps) noexcept {
    const auto all_finite = [](const std::vector<float>& values) {
        for (const float value : values) {
            if (!std::isfinite(value)) return false;
        }
        return true;
    };
    return all_finite(maps.left_x) && all_finite(maps.left_y) &&
           all_finite(maps.right_x) && all_finite(maps.right_y);
}

bool validInputImage(
    const ports::CapturedFrame& frame,
    const CudaStereoPreprocessorConfig& config,
    std::string& error) noexcept {
    const auto& image = frame.image;
    const auto& roi = frame.metadata.roi;
    if (frame.metadata.dropped || !image || image->nativeHandle() == 0 ||
        frame.metadata.pixel_format != core::PixelFormat::bayer8 ||
        image->pixelFormat() != core::PixelFormat::bayer8 ||
        (image->memoryKind() != ports::MemoryKind::host &&
         image->memoryKind() != ports::MemoryKind::host_pinned) ||
        roi.x != 0 || roi.y != 0 || roi.width != config.width ||
        roi.height != config.height || image->width() != config.width ||
        image->height() != config.height) {
        error =
            "CUDA preprocessor requires full-frame host Bayer8 input at "
            "the configured calibration geometry";
        return false;
    }
    const std::size_t stride = image->strideBytes();
    const std::size_t height = image->height();
    if (stride < image->width() ||
        height > std::numeric_limits<std::size_t>::max() / stride ||
        image->sizeBytes() < stride * height) {
        error = "CUDA preprocessor input stride or capacity is invalid";
        return false;
    }
    return true;
}

void copyToStaging(
    const ports::CapturedFrame& frame,
    CameraSlot& destination,
    std::uint32_t width,
    std::uint32_t height) noexcept {
    const auto* source = reinterpret_cast<const std::uint8_t*>(
        frame.image->nativeHandle());
    for (std::uint32_t row = 0; row < height; ++row) {
        std::memcpy(
            destination.staging.data() +
                static_cast<std::size_t>(row) * width,
            source + static_cast<std::size_t>(row) *
                frame.image->strideBytes(),
            width);
    }
}

bool submitCamera(
    CameraSlot& slot,
    const DeviceMapPair& maps,
    std::uint32_t width,
    std::uint32_t height,
    cudaStream_t stream,
    const char* side,
    std::string& error) {
    const std::size_t row = width;
    const std::size_t rows = height;
    cudaError_t status = cudaMemcpy2DAsync(
        slot.raw.data(), slot.raw.pitch(),
        slot.staging.data(), row,
        row, rows, cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
        error = cudaFailure(
            (std::string{side} + " Bayer upload").c_str(), status);
        return false;
    }

    status = launchBayerRggbToBgr8(
        slot.raw.data(), slot.raw.pitch(),
        slot.temporary_bgr.data(), slot.temporary_bgr.pitch(),
        static_cast<int>(width), static_cast<int>(height), stream);
    if (status != cudaSuccess) {
        error = cudaFailure(
            (std::string{side} + " Bayer conversion launch").c_str(),
            status);
        return false;
    }

    status = launchRemapBgrToBgrGray8(
        slot.temporary_bgr.data(), slot.temporary_bgr.pitch(),
        static_cast<int>(width), static_cast<int>(height),
        maps.x(), maps.y(), maps.pitch(),
        slot.rectified_bgr.data(), slot.rectified_bgr.pitch(),
        slot.rectified_gray.data(), slot.rectified_gray.pitch(),
        static_cast<int>(width), static_cast<int>(height), stream);
    if (status != cudaSuccess) {
        error = cudaFailure(
            (std::string{side} + " rectification launch").c_str(),
            status);
        return false;
    }
    return true;
}

core::FrameReference reference(
    const core::CameraFrameRecord& frame) {
    return {frame.camera_id, frame.role, frame.frame_index};
}

ports::PreprocessResult failure(std::string message) {
    return {false, std::nullopt, std::move(message)};
}

}  // namespace

struct CudaStereoPreprocessor::Impl {
    explicit Impl(CudaStereoPreprocessorConfig value)
        : config(std::move(value)) {}

    ~Impl() { reset(); }

    CudaStereoPreprocessorConfig config;
    mutable std::mutex mutex;
    std::atomic_bool initialized{false};
    cudaStream_t left_stream{nullptr};
    cudaStream_t right_stream{nullptr};
    DeviceMapPair left_maps;
    DeviceMapPair right_maps;
    std::vector<std::shared_ptr<StereoSlot>> slots;

    void reset() noexcept {
        initialized.store(false);
        if (left_stream != nullptr) {
            static_cast<void>(cudaStreamSynchronize(left_stream));
        }
        if (right_stream != nullptr) {
            static_cast<void>(cudaStreamSynchronize(right_stream));
        }
        slots.clear();
        left_maps.reset();
        right_maps.reset();
        if (left_stream != nullptr) {
            static_cast<void>(cudaStreamDestroy(left_stream));
            left_stream = nullptr;
        }
        if (right_stream != nullptr) {
            static_cast<void>(cudaStreamDestroy(right_stream));
            right_stream = nullptr;
        }
    }
};

CudaStereoPreprocessor::CudaStereoPreprocessor(
    CudaStereoPreprocessorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

CudaStereoPreprocessor::~CudaStereoPreprocessor() = default;

bool CudaStereoPreprocessor::initialize(
    const StereoRemapMaps& maps,
    std::string& error) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    error.clear();
    if (impl_->initialized.load()) {
        error = "CUDA stereo preprocessor is already initialized";
        return false;
    }
    if (!validConfig(impl_->config)) {
        error = "CUDA stereo preprocessor configuration is invalid";
        return false;
    }
    if (!maps.hasExpectedShape() || maps.width != impl_->config.width ||
        maps.height != impl_->config.height || !finiteMaps(maps)) {
        error =
            "stereo remap maps must be finite and match the configured geometry";
        return false;
    }

    impl_->reset();
    cudaError_t status = cudaStreamCreateWithFlags(
        &impl_->left_stream, cudaStreamNonBlocking);
    if (status == cudaSuccess) {
        status = cudaStreamCreateWithFlags(
            &impl_->right_stream, cudaStreamNonBlocking);
    }
    if (status != cudaSuccess) {
        error = cudaFailure("CUDA preprocess stream creation", status);
        impl_->reset();
        return false;
    }
    if (!impl_->left_maps.allocateAndCopy(
            maps.left_x, maps.left_y, maps.width, maps.height,
            "left remap upload", error) ||
        !impl_->right_maps.allocateAndCopy(
            maps.right_x, maps.right_y, maps.width, maps.height,
            "right remap upload", error)) {
        impl_->reset();
        return false;
    }

    try {
        impl_->slots.reserve(impl_->config.buffer_pool_pairs);
        for (std::size_t index = 0;
             index < impl_->config.buffer_pool_pairs;
             ++index) {
            auto slot = std::make_shared<StereoSlot>();
            if (!slot->allocate(
                    impl_->config.width, impl_->config.height, error)) {
                impl_->reset();
                return false;
            }
            impl_->slots.push_back(std::move(slot));
        }
    } catch (const std::exception& exception) {
        error = std::string{"cannot allocate CUDA preprocess pool: "} +
                exception.what();
        impl_->reset();
        return false;
    }
    impl_->initialized.store(true);
    return true;
}

bool CudaStereoPreprocessor::isInitialized() const noexcept {
    return impl_->initialized.load();
}

ports::PreprocessResult CudaStereoPreprocessor::prepare(
    const ports::StereoFrame& frame) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->initialized.load()) {
        return failure("CUDA stereo preprocessor is not initialized");
    }

    std::string error;
    if (frame.left.metadata.role != core::CameraRole::left ||
        frame.right.metadata.role != core::CameraRole::right ||
        frame.left.metadata.camera_id == frame.right.metadata.camera_id ||
        frame.left.metadata.frame_index != frame.right.metadata.frame_index ||
        !validInputImage(frame.left, impl_->config, error) ||
        !validInputImage(frame.right, impl_->config, error)) {
        return failure(
            error.empty() ? "invalid CUDA stereo preprocess frame identity"
                          : std::move(error));
    }

    std::shared_ptr<StereoSlot> slot;
    for (const auto& candidate : impl_->slots) {
        if (candidate.use_count() == 1) {
            slot = candidate;
            break;
        }
    }
    if (!slot) {
        return failure(
            "CUDA preprocess buffer pool is exhausted; downstream retained "
            "too many rectified frames");
    }

    copyToStaging(
        frame.left, slot->left, impl_->config.width, impl_->config.height);
    copyToStaging(
        frame.right, slot->right, impl_->config.width, impl_->config.height);

    const bool left_submitted = submitCamera(
        slot->left, impl_->left_maps,
        impl_->config.width, impl_->config.height,
        impl_->left_stream, "left", error);
    std::string right_error;
    const bool right_submitted = submitCamera(
        slot->right, impl_->right_maps,
        impl_->config.width, impl_->config.height,
        impl_->right_stream, "right", right_error);

    const cudaError_t left_sync =
        cudaStreamSynchronize(impl_->left_stream);
    const cudaError_t right_sync =
        cudaStreamSynchronize(impl_->right_stream);
    if (!left_submitted || !right_submitted ||
        left_sync != cudaSuccess || right_sync != cudaSuccess) {
        if (error.empty()) error = std::move(right_error);
        if (error.empty() && left_sync != cudaSuccess) {
            error = cudaFailure("left preprocess stream synchronization", left_sync);
        }
        if (error.empty() && right_sync != cudaSuccess) {
            error = cudaFailure(
                "right preprocess stream synchronization", right_sync);
        }
        return failure(
            error.empty() ? "CUDA stereo preprocessing failed"
                          : std::move(error));
    }

    try {
        ports::RectifiedStereoFrame output;
        output.left.source = reference(frame.left.metadata);
        output.right.source = reference(frame.right.metadata);
        output.left.detector_input = std::make_shared<CudaImageView>(
            slot, slot->left.rectified_bgr,
            impl_->config.width, impl_->config.height,
            core::PixelFormat::bgr8);
        output.left.geometry_gray = std::make_shared<CudaImageView>(
            slot, slot->left.rectified_gray,
            impl_->config.width, impl_->config.height,
            core::PixelFormat::gray8);
        output.right.detector_input = std::make_shared<CudaImageView>(
            slot, slot->right.rectified_bgr,
            impl_->config.width, impl_->config.height,
            core::PixelFormat::bgr8);
        output.right.geometry_gray = std::make_shared<CudaImageView>(
            slot, slot->right.rectified_gray,
            impl_->config.width, impl_->config.height,
            core::PixelFormat::gray8);
        return {true, std::move(output), {}};
    } catch (const std::exception& exception) {
        return failure(
            std::string{"cannot publish CUDA preprocess image views: "} +
            exception.what());
    }
}

}  // namespace volleyball::adapters::cuda_preprocess
