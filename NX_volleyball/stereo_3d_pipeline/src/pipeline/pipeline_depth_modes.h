#ifndef STEREO_3D_PIPELINE_PIPELINE_DEPTH_MODES_H_
#define STEREO_3D_PIPELINE_PIPELINE_DEPTH_MODES_H_

#include "pipeline.h"
#include "../stereo/depth_match_contract.h"
#include "../stereo/roi_feature_contract.h"
#include "../stereo/roi_geometry_cpu.h"

#include <string>

namespace stereo3d {

bool isBGRFormat(std::string fmt);
bool isROISubpixelDepthSolver(std::string solver);

bool dualYoloBBoxDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloBBoxEdgesDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloCircleDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloCircleEdgesDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROIEdgeCentroidDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROIRadialCenterDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROIEdgePairCenterDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROICornerPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROITexturePointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROIBinaryPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROIORBPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROIBRISKPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROIAKAZEPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROISIFTPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROIIoURegionColorPatchDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROIPatchIoUColorEdgeDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROICudaTemplateMatchDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROICudaStereoBMDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROICudaStereoSGMDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloROICenterPatchDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloSubpixelDepthEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloEpipolarFallbackEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloFallbackTemplateEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloFallbackFeaturePointsEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloOpenCVCpuDescriptorDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloCpuFallbackSearchEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloNeedsCircleSeedRefine(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloAnyDepthModeEnabled(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloNeedsHostImages(const PipelineConfig::DualYoloConfig& cfg);

ROIFeatureMatchConfig makeROIFeatureMatchConfig(
    const PipelineConfig::DualYoloConfig& cfg,
    const HybridDepthConfig& depth_cfg);
ROICircleSearchConfig makeROICircleSearchConfig(
    const PipelineConfig::DualYoloConfig& cfg);
StereoRoiPairGateConfig makeStereoRoiPairGateConfig(
    const PipelineConfig& config);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PIPELINE_DEPTH_MODES_H_
