#ifndef STEREO_3D_PIPELINE_P0P1_SOFT_GATE_H_
#define STEREO_3D_PIPELINE_P0P1_SOFT_GATE_H_

#include <array>
#include <vector>

namespace stereo3d {

enum class P0P1SoftGateCandidate : int {
    BBOX_CENTER = 0,
    CIRCLE_CENTER = 1,
    ROI_EDGE_CENTROID = 2,
    ROI_RADIAL_CENTER = 3,
    ROI_EDGE_PAIR_CENTER = 4,
    ROI_CENTER_PATCH = 5,
    ROI_MULTI_POINT = 6,
    ROI_CUDA_TEMPLATE_MATCH = 7,
    ROI_NEURAL_XFEAT = 8,
};

constexpr int kP0P1SoftGateCandidateCount = 9;

struct P0P1SoftGateSample {
    P0P1SoftGateCandidate candidate = P0P1SoftGateCandidate::BBOX_CENTER;
    float disparity_px = -1.0f;
    float depth_m = -1.0f;
    float y_delta_px = 0.0f;
    float source_score = 1.0f;
    float geometry_score = 1.0f;
};

struct P0P1SoftGateCandidateState {
    P0P1SoftGateCandidate candidate = P0P1SoftGateCandidate::BBOX_CENTER;
    bool enabled = false;
    float depth_m = -1.0f;
};

struct P0P1SoftGateOutput {
    float dy_center_px = 0.0f;
    float dy_mad_px = 0.0f;
    int sample_count = 0;
    int untrusted_mask = 0;
    std::array<float, kP0P1SoftGateCandidateCount> trust{};

    float trustOf(P0P1SoftGateCandidate candidate) const;
};

int p0p1SoftGateBit(P0P1SoftGateCandidate candidate);

P0P1SoftGateOutput evaluateP0P1SoftGate(
    const std::vector<P0P1SoftGateSample>& samples,
    const std::vector<P0P1SoftGateCandidateState>& candidate_states);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_P0P1_SOFT_GATE_H_
