#include "volleyball/adapters/cuda_depth/depth_candidate.hpp"

#include <cmath>
#include <string_view>
#include <unordered_set>

namespace volleyball::adapters::cuda_depth {
namespace {

bool validUnitInterval(const std::optional<double>& value) noexcept {
    return !value ||
           (std::isfinite(*value) && *value >= 0.0 && *value <= 1.0);
}

}  // namespace

bool isValid(const DepthCandidate& candidate) noexcept {
    return !candidate.method.empty() &&
           core::isFinite(candidate.left_anchor_px) &&
           std::isfinite(candidate.disparity_px) &&
           candidate.disparity_px > 0.0 &&
           (!candidate.disparity_stddev_px ||
            (std::isfinite(*candidate.disparity_stddev_px) &&
             *candidate.disparity_stddev_px >= 0.0)) &&
           (!candidate.support_points || *candidate.support_points > 0U) &&
           validUnitInterval(candidate.confidence);
}

bool isSupportedSelectionMethod(
    const core::StereoDepthMethodId& method) noexcept {
    const std::string_view value = method.value();
    // 纹理视差对应球面，不是球心。保留底层 ROI 算子用于研究，但在加入
    // 有验证的球心恢复前，禁止作为正式球心选择分支。
    return value == "circle_center" || value == "radial_center" ||
           value == "edge_pair_center" || value == "edge_centroid" ||
           value == "bbox_center" || value == "bbox_edges";
}

DepthCandidateSelection selectDepthCandidate(
    const std::vector<DepthCandidate>& candidates,
    const std::vector<core::StereoDepthMethodId>& method_order) {
    if (method_order.empty()) {
        return {false, std::nullopt, "depth method order is empty"};
    }

    std::unordered_set<std::string> ordered_methods;
    for (const auto& method : method_order) {
        const auto name = method.value();
        if (!isSupportedSelectionMethod(method)) {
            return {false, std::nullopt, "unsupported depth selection method: " + name};
        }
        if (!ordered_methods.emplace(name).second) {
            return {false, std::nullopt, "duplicate depth selection method: " + name};
        }
    }

    std::unordered_set<std::string> candidate_methods;
    for (const auto& candidate : candidates) {
        if (!isValid(candidate)) {
            return {false, std::nullopt, "invalid depth candidate"};
        }
        const auto name = candidate.method.value();
        if (!isSupportedSelectionMethod(candidate.method)) {
            return {
                false,
                std::nullopt,
                "unsupported depth candidate method: " + name,
            };
        }
        if (ordered_methods.count(name) == 0U) {
            return {
                false,
                std::nullopt,
                "unconfigured depth candidate: " + name,
            };
        }
        if (!candidate_methods.emplace(name).second) {
            return {false, std::nullopt, "duplicate depth candidate: " + name};
        }
    }

    for (const auto& method : method_order) {
        for (const auto& candidate : candidates) {
            if (candidate.method == method) {
                return {true, candidate, {}};
            }
        }
    }
    return {true, std::nullopt, "no configured depth candidate produced a measurement"};
}

}  // namespace volleyball::adapters::cuda_depth
