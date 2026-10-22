#ifndef STEREO_3D_PIPELINE_ROI_GEOMETRY_CPU_H_
#define STEREO_3D_PIPELINE_ROI_GEOMETRY_CPU_H_

#include "pipeline/detection_types.h"

#include <cstdint>

namespace stereo3d {

struct CircleFit2D {
    float cx = 0.0f;
    float cy = 0.0f;
    float radius = 0.0f;
    float confidence = 0.0f;
    int source = 0;  // 0=none, 1=bbox proxy, 2=ROI fit, 3=epipolar, 4=template, 5=feature proxy
    bool valid = false;
};

constexpr int kCircleSourceBboxProxy = 1;
constexpr int kCircleSourceRoiFit = 2;
constexpr int kCircleSourceEpipolarSearch = 3;
constexpr int kCircleSourceTemplateSearch = 4;
constexpr int kCircleSourceFeatureProxy = 5;

struct PointMeasure2D {
    float cx = 0.0f;
    float cy = 0.0f;
    float confidence = 0.0f;
    bool valid = false;
};

struct CircleFitOptions {
    bool denoise = true;
    int max_roi_pixels = 18000;
    float min_radius_ratio = 0.35f;
    float max_radius_ratio = 1.65f;
    float max_center_shift = 0.0f;
};

struct ROICircleSearchConfig {
    bool denoise = true;
    int max_roi_pixels = 18000;
    int fallback_search_margin_px = 48;
    int fallback_max_width_px = 220;
};

CircleFit2D fitCircleInRegionCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    int x1, int y1, int x2, int y2,
    float expected_cx, float expected_cy, float expected_radius,
    const CircleFitOptions& options);

CircleFit2D fitCircleInBBoxCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const Detection& det, bool denoise, int max_roi_pixels);

PointMeasure2D edgeCentroidInBBoxCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const Detection& det, bool denoise, int max_roi_pixels);

PointMeasure2D radialCenterInBBoxCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const Detection& det, bool denoise, int max_roi_pixels);

PointMeasure2D edgePairCenterInBBoxCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const Detection& det, bool denoise, int max_roi_pixels);

CircleFit2D circleFromDetectionCPU(const Detection& det);
Detection detectionFromCircleCPU(const CircleFit2D& circle, const Detection& source);
Detection detectionWithCircleCenterCPU(const CircleFit2D& circle, const Detection& source);

CircleFit2D searchCircleOnEpipolarCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const CircleFit2D& source_circle,
    float predicted_cx, float predicted_cy,
    float y_tolerance,
    const ROICircleSearchConfig& config);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_GEOMETRY_CPU_H_
