#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_COMMON_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_COMMON_H_

#include "roi_feature_contract.h"
#include "roi_feature_result.h"
#include "../pipeline/detection_types.h"

#include <opencv2/core.hpp>

#include <array>
#include <vector>

namespace stereo3d {

struct RobustMatchSample {
    float left_x = 0.0f;
    float left_y = 0.0f;
    float right_x = 0.0f;
    float right_y = 0.0f;
    float disparity = 0.0f;
    float score = 0.0f;
};

struct RobustAggregate {
    bool valid = false;
    float disparity = 0.0f;
    float anchor_x = 0.0f;
    float anchor_y = 0.0f;
    float right_anchor_x = 0.0f;
    float right_anchor_y = 0.0f;
    float stddev = 0.0f;
    float mean_score = 0.0f;
    int support = 0;
    int debug_inlier_count = 0;
    std::array<RobustMatchSample, kMaxSparseFeatureDebugMatches> debug_inliers{};
};

float computeFeatureDeltaGate(
    float initial_disp,
    float focal,
    float baseline,
    const ROIFeatureMatchConfig& cfg);

cv::Rect featureROIFromDetectionCPU(
    const Detection& det,
    int img_w,
    int img_h,
    int border,
    float scale,
    int extra_margin);

float strictFeatureYTolerance(const ROIFeatureMatchConfig& cfg);

float expectedFeatureYDelta(
    float left_x,
    const Detection& left_det,
    const ROIFeatureMatchConfig& cfg);

float featureYResidual(
    const RobustMatchSample& sample,
    const Detection& left_det,
    const ROIFeatureMatchConfig& cfg);

bool pointInsideDetectionEllipse(
    const Detection& det,
    float x,
    float y,
    float scale);

bool passesFeatureOverlapGate(
    const RobustMatchSample& sample,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg);

bool passesSphereRadiusGate(
    const RobustMatchSample& sample,
    const Detection& left_det,
    float initial_disp,
    float focal,
    float baseline,
    const ROIFeatureMatchConfig& cfg);

float weightedMedianDisparity(std::vector<RobustMatchSample> samples);

RobustAggregate aggregateRobustMatches(
    const std::vector<RobustMatchSample>& samples,
    int min_points,
    int max_points,
    float initial_disp,
    float max_delta,
    float max_stddev,
    const ROIFeatureMatchConfig& cfg);

void copyDebugMatches(const RobustAggregate& robust,
                      SparseFeatureDisparityResult& result);

void setSingleDebugMatch(const RobustMatchSample& sample,
                         SparseFeatureDisparityResult& result);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_COMMON_H_
