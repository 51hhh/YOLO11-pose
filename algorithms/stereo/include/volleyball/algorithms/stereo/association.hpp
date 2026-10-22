#pragma once

#include "volleyball/core/observation.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace volleyball::algorithms::stereo {

struct AssociationConfig {
    std::int64_t max_capture_delta_ns{0};
    double epipolar_tolerance_px{2.0};
    double max_size_ratio{2.0};
    double min_shifted_iou{0.05};
    double max_disparity_px{2048.0};
    bool require_bounding_box{true};
};

struct ScoredAssociation {
    core::StereoAssociationAttempt attempt;
    double score{0.0};
};

struct AssociationBatch {
    std::vector<ScoredAssociation> attempts;
    std::optional<std::size_t> best_accepted_index;
    std::string message;

    [[nodiscard]] bool ok() const noexcept { return message.empty(); }
};

[[nodiscard]] bool isValidAssociationConfig(
    const AssociationConfig& config) noexcept;

// 保留笛卡尔积中的每一次尝试；best 只是对已接受项的确定性选择。
[[nodiscard]] AssociationBatch associateCandidates(
    const std::vector<core::Detection2D>& left,
    const std::vector<core::Detection2D>& right,
    const core::Timestamp& left_capture,
    const core::Timestamp& right_capture,
    const AssociationConfig& config,
    core::PairId first_pair_id = 0);

}  // namespace volleyball::algorithms::stereo
