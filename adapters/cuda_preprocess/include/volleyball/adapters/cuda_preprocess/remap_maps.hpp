#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace volleyball::adapters::cuda_preprocess {

// 与 OpenCV 类型解耦的目标到源浮点映射。每个数组均按行紧密排列，
// 下标 y * width + x 对应一个校正输出像素的源图坐标。
struct StereoRemapMaps {
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::vector<float> left_x;
    std::vector<float> left_y;
    std::vector<float> right_x;
    std::vector<float> right_y;

    [[nodiscard]] bool hasExpectedShape() const noexcept {
        if (width == 0 || height == 0 ||
            static_cast<std::size_t>(height) >
                std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(width)) {
            return false;
        }
        const std::size_t count =
            static_cast<std::size_t>(width) * height;
        return left_x.size() == count && left_y.size() == count &&
               right_x.size() == count && right_y.size() == count;
    }
};

}  // namespace volleyball::adapters::cuda_preprocess
