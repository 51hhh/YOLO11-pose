#pragma once

#include "volleyball/core/observation.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace volleyball::ports {

enum class MemoryKind { host, host_pinned, cuda_device, vpi_image, external };

// 图像所有权由缓冲对象本身管理，管线不传递裸的短寿命指针。
class ImageBuffer {
public:
    virtual ~ImageBuffer() = default;
    [[nodiscard]] virtual MemoryKind memoryKind() const noexcept = 0;
    [[nodiscard]] virtual std::uint32_t width() const noexcept = 0;
    [[nodiscard]] virtual std::uint32_t height() const noexcept = 0;
    [[nodiscard]] virtual std::size_t strideBytes() const noexcept = 0;
    [[nodiscard]] virtual volleyball::core::PixelFormat pixelFormat() const noexcept = 0;
    [[nodiscard]] virtual std::uintptr_t nativeHandle() const noexcept = 0;
    [[nodiscard]] virtual std::size_t sizeBytes() const noexcept = 0;
};

using ImageBufferPtr = std::shared_ptr<const ImageBuffer>;

}  // namespace volleyball::ports

