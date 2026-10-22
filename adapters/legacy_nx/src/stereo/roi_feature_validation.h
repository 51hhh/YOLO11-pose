#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_VALIDATION_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_VALIDATION_H_

#include "roi_feature_contract.h"
#include "roi_feature_result.h"
#include "pipeline/detection_types.h"

namespace stereo3d {

bool pointInsideDetectionEllipseForFeature(
    const Detection& det,
    float x,
    float y,
    float scale);

bool validateSparseFeatureGeometry(
    const SparseFeatureDisparityResult& result,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    float focal,
    float baseline);

SparseFeatureRejectReason sparseFeatureGeometryRejectReason(
    const SparseFeatureDisparityResult& result,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    float focal,
    float baseline);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_VALIDATION_H_
