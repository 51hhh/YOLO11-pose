#ifndef STEREO_3D_PIPELINE_PIPELINE_ROI_MATCH_HELPERS_H_
#define STEREO_3D_PIPELINE_PIPELINE_ROI_MATCH_HELPERS_H_

#include "pipeline.h"
#include "../stereo/depth_match_contract.h"
#include "../stereo/dual_yolo_depth_gpu.h"
#include "../stereo/roi_feature_result.h"
#include "../stereo/roi_geometry_cpu.h"
#include "../stereo/roi_patch_match_cpu.h"

#include <cstddef>
#include <vector>

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

std::vector<DualYoloGpuDetectionPair> buildGpuDetectionPairsForRefine(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections,
    const StereoRoiPairGateConfig& roi_pair_gate,
    float baseline,
    const HybridDepthConfig& depth_cfg,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity,
    std::size_t max_pairs);

CircleFit2D searchTemplateOnEpipolarCPU(
    const uint8_t* source_img,
    int source_pitch,
    const uint8_t* target_img,
    int target_pitch,
    int img_w,
    int img_h,
    const CircleFit2D& source_circle,
    float predicted_cx,
    float predicted_cy,
    float y_tolerance,
    const PipelineConfig::DualYoloConfig& dual_cfg);

SubpixelDisparityResult refineDisparityByROICenterPatchCPU(
    const uint8_t* left_img,
    int left_pitch,
    const uint8_t* right_img,
    int right_pitch,
    int img_w,
    int img_h,
    const CircleFit2D& left_circle,
    const CircleFit2D& right_circle,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity,
    float focal,
    float baseline);

SubpixelDisparityResult refineDisparityByROIMultiPointCPU(
    const uint8_t* left_img,
    int left_pitch,
    const uint8_t* right_img,
    int right_pitch,
    int img_w,
    int img_h,
    const CircleFit2D& left_circle,
    const CircleFit2D& right_circle,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity,
    float focal,
    float baseline);

void stampFrameMetadata(FrameSlot& slot);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PIPELINE_ROI_MATCH_HELPERS_H_
