#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_OPENCV_HELPERS_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_OPENCV_HELPERS_H_

#include "roi_feature_match_cpu.h"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace stereo3d {
namespace {

cv::Ptr<cv::Feature2D> createOpenCVFeatureExtractorCPU(
    OpenCVFeatureMode mode,
    int max_features,
    int patch_radius)
{
    max_features = std::clamp(max_features, 16, 256);
    patch_radius = std::clamp(patch_radius, 2, 12);
    const int patch_size = std::max(9, patch_radius * 2 + 1);
    const int edge_threshold = std::clamp(patch_radius + 3, 5, 16);

    switch (mode) {
    case OpenCVFeatureMode::ORB:
        return cv::ORB::create(max_features, 1.2f, 3, edge_threshold, 0, 2,
                               cv::ORB::HARRIS_SCORE, patch_size, 12);
    case OpenCVFeatureMode::BRISK:
        return cv::BRISK::create(18, 2, 1.0f);
    case OpenCVFeatureMode::AKAZE: {
        auto akaze = cv::AKAZE::create();
        akaze->setDescriptorType(cv::AKAZE::DESCRIPTOR_MLDB);
        akaze->setThreshold(0.0015);
        akaze->setNOctaves(2);
        akaze->setNOctaveLayers(2);
        return akaze;
    }
    case OpenCVFeatureMode::SIFT:
        return cv::SIFT::create(max_features, 3, 0.04, 10.0, 1.6);
    }
    return {};
}

int descriptorNormType(OpenCVFeatureMode mode)
{
    return mode == OpenCVFeatureMode::SIFT ? cv::NORM_L2 : cv::NORM_HAMMING;
}

bool descriptorDepthCompatible(OpenCVFeatureMode mode, const cv::Mat& descriptors)
{
    if (descriptors.empty()) return false;
    if (mode == OpenCVFeatureMode::SIFT) {
        return descriptors.depth() == CV_32F || descriptors.depth() == CV_8U;
    }
    return descriptors.depth() == CV_8U;
}

float descriptorMatchScore(OpenCVFeatureMode mode,
                           float distance,
                           int descriptor_cols)
{
    if (!std::isfinite(distance) || distance < 0.0f) return 0.0f;
    if (mode == OpenCVFeatureMode::SIFT) {
        const float scale = std::max(
            32.0f,
            std::sqrt(static_cast<float>(std::max(1, descriptor_cols))) * 18.0f);
        return std::clamp(std::exp(-distance / scale), 0.0f, 1.0f);
    }
    const float max_hamming = static_cast<float>(
        std::max(1, descriptor_cols * 8));
    return 1.0f - std::min(1.0f, distance / max_hamming);
}

float descriptorMinScore(OpenCVFeatureMode mode,
                         const ROIFeatureMatchConfig& cfg)
{
    if (mode == OpenCVFeatureMode::SIFT) {
        return std::max(0.12f, 0.10f + cfg.subpixel_min_confidence * 0.35f);
    }
    return std::max(0.45f, 0.35f + cfg.subpixel_min_confidence * 0.45f);
}

float descriptorRatioThreshold(OpenCVFeatureMode mode)
{
    return mode == OpenCVFeatureMode::SIFT ? 0.75f : 0.78f;
}

void detectAndDescribeOpenCVFeatureCPU(
    cv::Feature2D& extractor,
    const cv::Mat& image,
    int max_points,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors)
{
    keypoints.clear();
    descriptors.release();
    extractor.detect(image, keypoints);
    if (keypoints.empty()) return;

    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });
    if (static_cast<int>(keypoints.size()) > max_points) {
        keypoints.resize(static_cast<size_t>(max_points));
    }
    extractor.compute(image, keypoints, descriptors);
}

}  // namespace
}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_MATCH_OPENCV_HELPERS_H_
