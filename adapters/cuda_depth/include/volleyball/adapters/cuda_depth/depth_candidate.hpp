#pragma once

#include "volleyball/core/ids.hpp"
#include "volleyball/core/math_types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace volleyball::adapters::cuda_depth {

struct DepthCandidate {
    core::StereoDepthMethodId method;
    core::Vec2 left_anchor_px;
    double disparity_px{0.0};
    std::optional<double> disparity_stddev_px;
    std::optional<std::uint32_t> support_points;
    std::optional<double> confidence;
};

[[nodiscard]] bool isValid(const DepthCandidate& candidate) noexcept;

// 只列出当前双侧已关联请求能够实际生成的最终直接观测方法。
[[nodiscard]] bool isSupportedSelectionMethod(
    const core::StereoDepthMethodId& method) noexcept;

struct DepthCandidateSelection {
    bool valid_input{false};
    std::optional<DepthCandidate> candidate;
    std::string message;
};

// 按冻结顺序选择第一个存在的有效候选。候选无结果是合法拒绝；未知方法、
// 重复方法或结构损坏属于配置/后端错误。
[[nodiscard]] DepthCandidateSelection selectDepthCandidate(
    const std::vector<DepthCandidate>& candidates,
    const std::vector<core::StereoDepthMethodId>& method_order);

}  // namespace volleyball::adapters::cuda_depth
