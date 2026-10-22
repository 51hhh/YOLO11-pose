#pragma once

#include <array>
#include <cmath>

namespace volleyball::core {

struct Vec2 {
    double x{0.0};
    double y{0.0};
};

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct BoundingBox2D {
    double x{0.0};
    double y{0.0};
    double width{0.0};
    double height{0.0};
};

struct Covariance2D {
    // 对称 2x2 矩阵的上三角：uu, uv, vv。
    std::array<double, 3> upper{};
};

struct Covariance3D {
    // 对称 3x3 矩阵的上三角：xx, xy, xz, yy, yz, zz。
    std::array<double, 6> upper{};
};

[[nodiscard]] inline bool isFinite(const Vec2& value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] inline bool isFinite(const Vec3& value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

}  // namespace volleyball::core

