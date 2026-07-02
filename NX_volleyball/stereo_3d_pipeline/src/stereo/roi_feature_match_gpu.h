#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_GPU_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_GPU_H_

#include "roi_feature_contract.h"
#include "roi_feature_result.h"
#include "pipeline/detection_types.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace stereo3d {

SparseFeatureDisparityResult matchOpenCVORBDisparityGPU(
    const uint8_t* left_gpu, int left_pitch,
    const uint8_t* right_gpu, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    cudaStream_t stream);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_GPU_H_
