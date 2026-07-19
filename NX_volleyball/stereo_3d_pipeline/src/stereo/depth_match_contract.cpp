#include "depth_match_contract.h"

#include <algorithm>
#include <cmath>

namespace stereo3d {

namespace {

struct RectF {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

RectF rectFromDetection(const Detection& det, float shift_x = 0.0f) {
    const float half_w = det.width * 0.5f;
    const float half_h = det.height * 0.5f;
    return RectF{
        det.cx - half_w + shift_x,
        det.cy - half_h,
        det.cx + half_w + shift_x,
        det.cy + half_h};
}

float rectArea(const RectF& r) {
    return std::max(0.0f, r.x2 - r.x1) * std::max(0.0f, r.y2 - r.y1);
}

float rectIoU(const RectF& a, const RectF& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float inter = rectArea(RectF{ix1, iy1, ix2, iy2});
    const float uni = rectArea(a) + rectArea(b) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

}  // namespace

const char* stereoRoiPairRejectReasonName(StereoRoiPairRejectReason reason) {
    switch (reason) {
    case StereoRoiPairRejectReason::NONE: return "none";
    case StereoRoiPairRejectReason::CLASS_MISMATCH: return "class_mismatch";
    case StereoRoiPairRejectReason::INVALID_BOX: return "invalid_box";
    case StereoRoiPairRejectReason::NONPOSITIVE_DISPARITY: return "nonpositive_disparity";
    case StereoRoiPairRejectReason::OVER_MAX_DISPARITY: return "over_max_disparity";
    case StereoRoiPairRejectReason::EPIPOLAR_REJECT: return "epipolar_reject";
    case StereoRoiPairRejectReason::SIZE_REJECT: return "size_reject";
    case StereoRoiPairRejectReason::LOW_IOU: return "low_iou";
    }
    return "unknown";
}

bool evaluateStereoRoiPair(
    const Detection& left,
    const Detection& right,
    int left_index,
    int right_index,
    const StereoRoiPairGateConfig& config,
    StereoRoiPair* pair,
    StereoRoiPairRejectReason* reason) {
    auto reject = [&](StereoRoiPairRejectReason why) {
        if (reason) *reason = why;
        return false;
    };
    if (reason) *reason = StereoRoiPairRejectReason::NONE;

    if (left.class_id != right.class_id) {
        return reject(StereoRoiPairRejectReason::CLASS_MISMATCH);
    }
    if (left.width <= 1.0f || left.height <= 1.0f ||
        right.width <= 1.0f || right.height <= 1.0f) {
        return reject(StereoRoiPairRejectReason::INVALID_BOX);
    }

    const float disparity = left.cx - right.cx;
    if (disparity <= 0.0f) {
        return reject(StereoRoiPairRejectReason::NONPOSITIVE_DISPARITY);
    }
    if (disparity > static_cast<float>(config.max_disparity)) {
        return reject(StereoRoiPairRejectReason::OVER_MAX_DISPARITY);
    }

    const float base_y_tol = std::max(1.0f, config.epipolar_y_tolerance);
    const float adaptive_y_tol =
        std::max(base_y_tol,
                 config.adaptive_y_ratio * std::max(left.height, right.height));
    const float signed_dy = left.cy - right.cy;
    const float dy = std::abs(signed_dy);
    if (dy > adaptive_y_tol) {
        return reject(StereoRoiPairRejectReason::EPIPOLAR_REJECT);
    }

    const float w_ratio = std::max(left.width / right.width,
                                   right.width / left.width);
    const float h_ratio = std::max(left.height / right.height,
                                   right.height / left.height);
    const float max_ratio = std::max(1.0f, config.max_size_ratio);
    if (w_ratio > max_ratio || h_ratio > max_ratio) {
        return reject(StereoRoiPairRejectReason::SIZE_REJECT);
    }

    const float shifted_iou =
        rectIoU(rectFromDetection(left), rectFromDetection(right, disparity));
    if (shifted_iou < std::max(0.0f, config.min_shifted_iou)) {
        return reject(StereoRoiPairRejectReason::LOW_IOU);
    }

    if (pair) {
        pair->left_index = left_index;
        pair->right_index = right_index;
        pair->left = left;
        pair->right = right;
        pair->initial_disparity = disparity;
        pair->epipolar_y_delta = signed_dy;
        pair->epipolar_dy = dy;
        pair->y_tolerance = adaptive_y_tol;
        pair->width_ratio = w_ratio;
        pair->height_ratio = h_ratio;
        pair->size_ratio = std::max(w_ratio, h_ratio);
        pair->shifted_bbox_iou = shifted_iou;
        const float size_cost = std::abs(std::log(w_ratio)) +
                                std::abs(std::log(h_ratio));
        pair->score = dy / adaptive_y_tol + size_cost - 0.25f * right.confidence;
        pair->semantic_confidence =
            std::sqrt(std::max(0.0f, left.confidence * right.confidence));
    }
    return true;
}

void accumulateStereoRoiPairReject(
    StereoRoiPairStats* stats,
    StereoRoiPairRejectReason reason) {
    if (!stats) return;
    switch (reason) {
    case StereoRoiPairRejectReason::NONE: break;
    case StereoRoiPairRejectReason::CLASS_MISMATCH: ++stats->class_mismatch; break;
    case StereoRoiPairRejectReason::INVALID_BOX: ++stats->invalid_box; break;
    case StereoRoiPairRejectReason::NONPOSITIVE_DISPARITY:
        ++stats->nonpositive_disparity;
        break;
    case StereoRoiPairRejectReason::OVER_MAX_DISPARITY:
        ++stats->over_max_disparity;
        break;
    case StereoRoiPairRejectReason::EPIPOLAR_REJECT: ++stats->epipolar_reject; break;
    case StereoRoiPairRejectReason::SIZE_REJECT: ++stats->size_reject; break;
    case StereoRoiPairRejectReason::LOW_IOU: ++stats->low_iou; break;
    }
}

std::vector<StereoRoiPair> collectStereoRoiPairCandidates(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections,
    const StereoRoiPairGateConfig& config,
    std::size_t max_pairs,
    StereoRoiPairStats* stats) {
    std::vector<StereoRoiPair> pairs;
    if (max_pairs == 0) return pairs;
    pairs.reserve(left_detections.size() * right_detections.size());
    for (std::size_t li = 0; li < left_detections.size(); ++li) {
        for (std::size_t ri = 0; ri < right_detections.size(); ++ri) {
            StereoRoiPair pair;
            StereoRoiPairRejectReason reason = StereoRoiPairRejectReason::NONE;
            if (!evaluateStereoRoiPair(left_detections[li], right_detections[ri],
                                       static_cast<int>(li),
                                       static_cast<int>(ri),
                                       config, &pair, &reason)) {
                accumulateStereoRoiPairReject(stats, reason);
                continue;
            }
            pairs.push_back(pair);
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const StereoRoiPair& a, const StereoRoiPair& b) {
                  return a.score < b.score;
              });
    if (pairs.size() > max_pairs) {
        pairs.resize(max_pairs);
    }
    return pairs;
}

bool findBestStereoRoiPair(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections,
    const StereoRoiPairGateConfig& config,
    StereoRoiPair* best_pair,
    StereoRoiPairStats* stats) {
    bool found = false;
    StereoRoiPair best;
    for (std::size_t li = 0; li < left_detections.size(); ++li) {
        for (std::size_t ri = 0; ri < right_detections.size(); ++ri) {
            StereoRoiPair pair;
            StereoRoiPairRejectReason reason = StereoRoiPairRejectReason::NONE;
            if (!evaluateStereoRoiPair(left_detections[li], right_detections[ri],
                                       static_cast<int>(li),
                                       static_cast<int>(ri),
                                       config, &pair, &reason)) {
                accumulateStereoRoiPairReject(stats, reason);
                continue;
            }
            if (!found || pair.score < best.score) {
                found = true;
                best = pair;
            }
        }
    }
    if (found && best_pair) *best_pair = best;
    return found;
}

}  // namespace stereo3d
