#ifndef STEREO_3D_PIPELINE_PIPELINE_ROI_MATCH_HELPERS_H_
#define STEREO_3D_PIPELINE_PIPELINE_ROI_MATCH_HELPERS_H_

#include "pipeline.h"
#include "../stereo/dual_yolo_depth_gpu.h"
#include "../stereo/roi_feature_result.h"
#include "../stereo/roi_geometry_cpu.h"
#include "../stereo/roi_patch_match_cpu.h"

namespace stereo3d {

CircleFit2D circleFromGpuCandidate(const DualYoloGpuCircle& in,
                                   const Detection& fallback);
PointMeasure2D pointFromGpuCandidate(const DualYoloGpuPointMeasure& in);
DualYoloGpuDetection makeGpuDetection(const Detection& det);
SubpixelDisparityResult subpixelFromGpuCandidate(const DualYoloGpuDisparity& in);
SparseFeatureDisparityResult sparseFromGpuCandidate(const DualYoloGpuDisparity& in);

float estimateDisparityFromBBoxCPU(
    const Detection& det,
    float baseline,
    const HybridDepthConfig& depth_cfg,
    int max_disparity);

float bboxDisparityConsistencyPenaltyCPU(
    const Detection& left,
    const Detection& right,
    float pair_disparity,
    float baseline,
    const HybridDepthConfig& depth_cfg,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity);

void stampFrameMetadata(FrameSlot& slot);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PIPELINE_ROI_MATCH_HELPERS_H_
