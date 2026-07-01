#ifndef STEREO_3D_PIPELINE_ROI_PATCH_MATCH_CPU_H_
#define STEREO_3D_PIPELINE_ROI_PATCH_MATCH_CPU_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stereo3d {

inline float sampleGrayCPU(const uint8_t* img, int pitch, int x, int y, bool denoise)
{
    const uint8_t* row = img + y * pitch;
    if (!denoise) return static_cast<float>(row[x]);

    const uint8_t* prev = img + (y - 1) * pitch;
    const uint8_t* next = img + (y + 1) * pitch;
    const int v =
        4 * static_cast<int>(row[x]) +
        2 * (static_cast<int>(row[x - 1]) + static_cast<int>(row[x + 1]) +
             static_cast<int>(prev[x]) + static_cast<int>(next[x])) +
        static_cast<int>(prev[x - 1]) + static_cast<int>(prev[x + 1]) +
        static_cast<int>(next[x - 1]) + static_cast<int>(next[x + 1]);
    return static_cast<float>(v) * (1.0f / 16.0f);
}

inline bool patchInsideCPU(int img_w, int img_h, int x, int y, int radius, bool denoise)
{
    const int border = denoise ? 1 : 0;
    return x - radius - border >= 0 &&
           y - radius - border >= 0 &&
           x + radius + border < img_w &&
           y + radius + border < img_h;
}

inline float medianOfSortedCPU(const std::vector<float>& values)
{
    if (values.empty()) return 0.0f;
    const size_t mid = values.size() / 2;
    if ((values.size() & 1U) != 0U) return values[mid];
    return 0.5f * (values[mid - 1] + values[mid]);
}

float znccPatchCPU(
    const uint8_t* left, int left_pitch,
    const uint8_t* right, int right_pitch,
    int x_left, int y_left,
    int x_right, int y_right,
    int radius,
    bool denoise);

float censusPatchSimilarityCPU(
    const uint8_t* left, int left_pitch,
    const uint8_t* right, int right_pitch,
    int x_left, int y_left,
    int x_right, int y_right,
    int radius,
    bool denoise);

float computeSubpixelDispDeltaGateCPU(
    float initial_disp,
    float focal,
    float baseline,
    float max_disp_delta_px,
    float max_disp_delta_ratio,
    float max_depth_delta_m);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_PATCH_MATCH_CPU_H_
