#include "roi_patch_match_cpu.h"

#include <algorithm>
#include <cmath>

namespace stereo3d {

float znccPatchCPU(
    const uint8_t* left, int left_pitch,
    const uint8_t* right, int right_pitch,
    int x_left, int y_left,
    int x_right, int y_right,
    int radius,
    bool denoise)
{
    double sum_l = 0.0;
    double sum_r = 0.0;
    int n = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            sum_l += sampleGrayCPU(left, left_pitch, x_left + dx, y_left + dy, denoise);
            sum_r += sampleGrayCPU(right, right_pitch, x_right + dx, y_right + dy, denoise);
            ++n;
        }
    }
    if (n <= 1) return -2.0f;

    const double mean_l = sum_l / static_cast<double>(n);
    const double mean_r = sum_r / static_cast<double>(n);
    double cov = 0.0;
    double var_l = 0.0;
    double var_r = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const double lv = sampleGrayCPU(left, left_pitch,
                                            x_left + dx, y_left + dy, denoise) - mean_l;
            const double rv = sampleGrayCPU(right, right_pitch,
                                            x_right + dx, y_right + dy, denoise) - mean_r;
            cov += lv * rv;
            var_l += lv * lv;
            var_r += rv * rv;
        }
    }

    const double denom = std::sqrt(var_l * var_r);
    if (denom < 1e-6) return -2.0f;
    return static_cast<float>(cov / denom);
}

float censusPatchSimilarityCPU(
    const uint8_t* left, int left_pitch,
    const uint8_t* right, int right_pitch,
    int x_left, int y_left,
    int x_right, int y_right,
    int radius,
    bool denoise)
{
    radius = std::clamp(radius, 2, 8);
    const int step = std::max(1, radius / 2);
    const float center_l = sampleGrayCPU(left, left_pitch, x_left, y_left, denoise);
    const float center_r = sampleGrayCPU(right, right_pitch, x_right, y_right, denoise);
    int bits = 0;
    int same = 0;
    for (int dy = -radius; dy <= radius; dy += step) {
        for (int dx = -radius; dx <= radius; dx += step) {
            if (dx == 0 && dy == 0) continue;
            const bool bit_l = sampleGrayCPU(left, left_pitch,
                                             x_left + dx, y_left + dy,
                                             denoise) > center_l;
            const bool bit_r = sampleGrayCPU(right, right_pitch,
                                             x_right + dx, y_right + dy,
                                             denoise) > center_r;
            same += bit_l == bit_r ? 1 : 0;
            ++bits;
        }
    }
    if (bits < 8) return -2.0f;
    return static_cast<float>(same) / static_cast<float>(bits);
}

float computeSubpixelDispDeltaGateCPU(
    float initial_disp,
    float focal,
    float baseline,
    float max_disp_delta_px,
    float max_disp_delta_ratio,
    float max_depth_delta_m)
{
    const float abs_gate = std::max(0.25f, max_disp_delta_px);
    const float ratio_gate = std::max(0.25f,
        std::max(0.0f, max_disp_delta_ratio) * initial_disp);
    float gate = std::min(abs_gate, ratio_gate);

    const float fb = focal * baseline;
    if (fb > 1e-3f && initial_disp > 0.5f && max_depth_delta_m > 0.01f) {
        const float z0 = fb / initial_disp;
        const float disp_far = fb / (z0 + max_depth_delta_m);
        float depth_gate = std::max(0.0f, initial_disp - disp_far);
        if (z0 > max_depth_delta_m + 0.01f) {
            const float disp_near = fb / (z0 - max_depth_delta_m);
            depth_gate = std::min(depth_gate, std::max(0.0f, disp_near - initial_disp));
        }
        gate = std::min(gate, std::max(0.25f, depth_gate));
    }
    return std::max(0.25f, gate);
}

}  // namespace stereo3d
