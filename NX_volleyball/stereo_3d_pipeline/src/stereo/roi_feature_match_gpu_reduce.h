#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_GPU_REDUCE_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_GPU_REDUCE_H_

#include <cuda_runtime.h>

#include <cstddef>

namespace stereo3d {

struct CudaTemplateScorePeak {
    float value = -1.0f;
    int x = -1;
    int y = -1;
    int valid = 0;
};

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
