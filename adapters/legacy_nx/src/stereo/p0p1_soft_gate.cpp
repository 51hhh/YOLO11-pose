#include "p0p1_soft_gate.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace stereo3d {
namespace {

float medianCopy(std::vector<float> values) {
    if (values.empty()) {
        return 0.0f;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    float med = values[mid];
    if ((values.size() % 2) == 0 && mid > 0) {
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        med = 0.5f * (med + values[mid - 1]);
    }
    return med;
}

float robustMad(const std::vector<float>& values, float center) {
    std::vector<float> dev;
    dev.reserve(values.size());
    for (float v : values) {
        if (std::isfinite(v)) {
            dev.push_back(std::abs(v - center));
        }
    }
    return medianCopy(std::move(dev));
}

float normalizedScore(float score) {
    if (!std::isfinite(score)) {
        return 0.0f;
    }
    if (score >= 0.0f && score <= 1.0f) {
        return score;
    }
    return std::clamp((score + 1.0f) * 0.5f, 0.0f, 1.0f);
}

float gaussianConsistency(float residual, float sigma) {
    const float s = std::max(0.25f, sigma);
    const float x = residual / s;
    return std::exp(-0.5f * x * x);
}

int candidateIndex(P0P1SoftGateCandidate candidate) {
    return static_cast<int>(candidate);
}

bool sampleUsable(const P0P1SoftGateSample& sample) {
    return std::isfinite(sample.disparity_px) &&
           sample.disparity_px > 0.0f &&
           std::isfinite(sample.depth_m) &&
           sample.depth_m > 0.0f &&
           std::isfinite(sample.y_delta_px);
}

}  // namespace

float P0P1SoftGateOutput::trustOf(P0P1SoftGateCandidate candidate) const {
    const int index = candidateIndex(candidate);
    if (index < 0 || index >= kP0P1SoftGateCandidateCount) {
        return 0.0f;
    }
    return trust[static_cast<size_t>(index)];
}

int p0p1SoftGateBit(P0P1SoftGateCandidate candidate) {
    const int index = candidateIndex(candidate);
    if (index < 0 || index >= 31) {
        return 0;
    }
    return 1 << index;
}

P0P1SoftGateOutput evaluateP0P1SoftGate(
    const std::vector<P0P1SoftGateSample>& samples,
    const std::vector<P0P1SoftGateCandidateState>& candidate_states) {
    P0P1SoftGateOutput out;

    std::vector<P0P1SoftGateSample> usable_samples;
    usable_samples.reserve(samples.size());
    for (auto sample : samples) {
        if (!sampleUsable(sample)) {
            continue;
        }
        sample.source_score = normalizedScore(sample.source_score);
        sample.geometry_score =
            std::clamp(sample.geometry_score, 0.0f, 1.0f);
        usable_samples.push_back(sample);
    }

    if (!usable_samples.empty()) {
        std::vector<float> dy_values;
        std::vector<float> disparity_values;
        dy_values.reserve(usable_samples.size());
        disparity_values.reserve(usable_samples.size());
        for (const auto& sample : usable_samples) {
            dy_values.push_back(sample.y_delta_px);
            disparity_values.push_back(sample.disparity_px);
        }

        out.dy_center_px = medianCopy(dy_values);
        out.dy_mad_px = robustMad(dy_values, out.dy_center_px);

        std::vector<std::pair<float, int>> sorted_disparities;
        sorted_disparities.reserve(usable_samples.size());
        for (size_t i = 0; i < usable_samples.size(); ++i) {
            sorted_disparities.emplace_back(usable_samples[i].disparity_px,
                                            static_cast<int>(i));
        }
        std::sort(sorted_disparities.begin(), sorted_disparities.end());

        const float global_disparity_center = medianCopy(disparity_values);
        const float global_disparity_mad =
            robustMad(disparity_values, global_disparity_center);
        const float cluster_eps =
            std::max(1.5f, 2.5f * 1.4826f * global_disparity_mad);

        std::vector<int> cluster_ids(usable_samples.size(), -1);
        std::vector<int> cluster_sizes;
        int current_cluster = -1;
        float previous_disparity =
            std::numeric_limits<float>::quiet_NaN();
        for (const auto& item : sorted_disparities) {
            if (current_cluster < 0 ||
                !std::isfinite(previous_disparity) ||
                item.first - previous_disparity > cluster_eps) {
                ++current_cluster;
                cluster_sizes.push_back(0);
            }
            cluster_ids[static_cast<size_t>(item.second)] = current_cluster;
            ++cluster_sizes[static_cast<size_t>(current_cluster)];
            previous_disparity = item.first;
        }

        int main_cluster = 0;
        for (size_t i = 1; i < cluster_sizes.size(); ++i) {
            if (cluster_sizes[i] >
                cluster_sizes[static_cast<size_t>(main_cluster)]) {
                main_cluster = static_cast<int>(i);
            }
        }

        std::vector<float> main_cluster_disparities;
        main_cluster_disparities.reserve(usable_samples.size());
        for (size_t i = 0; i < usable_samples.size(); ++i) {
            if (cluster_ids[i] == main_cluster) {
                main_cluster_disparities.push_back(
                    usable_samples[i].disparity_px);
            }
        }
        const float disparity_center = main_cluster_disparities.empty()
            ? global_disparity_center
            : medianCopy(main_cluster_disparities);
        const float disparity_mad = main_cluster_disparities.empty()
            ? global_disparity_mad
            : robustMad(main_cluster_disparities, disparity_center);

        const float dy_sigma = std::max(1.0f, 1.4826f * out.dy_mad_px);
        const float disparity_sigma =
            std::max(1.0f, 1.4826f * disparity_mad);

        for (size_t i = 0; i < usable_samples.size(); ++i) {
            const auto& sample = usable_samples[i];
            const int index = candidateIndex(sample.candidate);
            if (index < 0 || index >= kP0P1SoftGateCandidateCount) {
                continue;
            }
            const float dy_score = gaussianConsistency(
                std::abs(sample.y_delta_px - out.dy_center_px), dy_sigma);
            const int cluster_id = cluster_ids[i];
            const float cluster_score =
                (cluster_id == main_cluster)
                    ? 1.0f
                    : (cluster_id >= 0 &&
                       cluster_sizes[static_cast<size_t>(cluster_id)] >= 2)
                          ? 0.45f
                          : 0.15f;
            const float disparity_score = std::max(
                cluster_score,
                0.2f * gaussianConsistency(
                    std::abs(sample.disparity_px - disparity_center),
                    disparity_sigma));
            const float trust = std::clamp(
                0.35f * dy_score +
                0.30f * disparity_score +
                0.20f * sample.source_score +
                0.15f * sample.geometry_score,
                0.0f,
                1.0f);
            out.trust[static_cast<size_t>(index)] = trust;
            if (trust < 0.35f) {
                out.untrusted_mask |= p0p1SoftGateBit(sample.candidate);
            }
        }
        out.sample_count = static_cast<int>(usable_samples.size());
    }

    for (const auto& state : candidate_states) {
        if (!state.enabled) {
            continue;
        }
        if (!std::isfinite(state.depth_m) || state.depth_m <= 0.0f) {
            out.untrusted_mask |= p0p1SoftGateBit(state.candidate);
        }
    }

    return out;
}

}  // namespace stereo3d
