#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_GPU_REDUCE_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_GPU_REDUCE_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace stereo3d {

struct CudaTemplateScorePeak {
    float value = -1.0f;
    int x = -1;
    int y = -1;
    int valid = 0;
};

cudaError_t computeCudaTemplateCcoeffNormedScoreMap(
    const uint8_t* left_gpu,
    size_t left_pitch_bytes,
    const uint8_t* right_gpu,
    size_t right_pitch_bytes,
    int templ_x,
    int templ_y,
    int search_x,
    int search_y,
    int patch_size,
    float* score_gpu,
    size_t score_pitch_bytes,
    int score_width,
    int score_height,
    cudaStream_t stream);

cudaError_t findCudaTemplateScorePeak(
    const float* score_gpu,
    size_t score_pitch_bytes,
    int width,
    int height,
    CudaTemplateScorePeak* device_result,
    CudaTemplateScorePeak* host_result,
    cudaStream_t stream);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_GPU_REDUCE_H_
