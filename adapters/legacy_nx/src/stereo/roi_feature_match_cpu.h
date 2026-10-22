#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_CPU_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_CPU_H_

#include "roi_feature_contract.h"
#include "roi_feature_result.h"
#include "../pipeline/detection_types.h"

#include <opencv2/features2d.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace stereo3d {

struct DebugFeatureMatchResult {
    std::string name;
    std::vector<cv::KeyPoint> left_keypoints;
    std::vector<cv::KeyPoint> right_keypoints;
    std::vector<cv::DMatch> matches;
    int attempted_matches = 0;
    float disparity = -1.0f;
    float stddev = -1.0f;
    float confidence = 0.0f;
};

std::string sparseFeatureModeName(SparseFeatureMode mode);
const char* openCVFeatureModeName(OpenCVFeatureMode mode);

SparseFeatureDisparityResult matchSparseFeatureDisparityCPU(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    bool source_left,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    SparseFeatureMode mode);

SparseFeatureDisparityResult matchOpenCVFeatureDisparityCPU(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    bool source_left,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    OpenCVFeatureMode mode);

DebugFeatureMatchResult makeDebugSparseFeatureMatchesCPU(
    const cv::Mat& left_gray,
    const cv::Mat& right_gray,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    SparseFeatureMode mode);

DebugFeatureMatchResult makeDebugOpenCVFeatureMatchesCPU(
    const cv::Mat& left_gray,
    const cv::Mat& right_gray,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    OpenCVFeatureMode mode);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_CPU_H_
