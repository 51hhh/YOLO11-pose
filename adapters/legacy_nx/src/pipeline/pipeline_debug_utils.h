#ifndef STEREO_3D_PIPELINE_PIPELINE_DEBUG_UTILS_H_
#define STEREO_3D_PIPELINE_PIPELINE_DEBUG_UTILS_H_

#include "detection_types.h"
#include "../stereo/roi_feature_match_cpu.h"

#include <vpi/Image.h>
#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace stereo3d {

bool writeDebugImage(const std::filesystem::path& output_dir,
                     const std::string& filename,
                     const cv::Mat& image);
bool lockGrayVpiImageCopy(VPIImage img, cv::Mat& out);
bool lockBgrVpiImageCopy(VPIImage img, cv::Mat& out);
cv::Mat drawDetectionOverlay(const cv::Mat& image,
                             const std::vector<Detection>& detections,
                             const std::string& title);
void drawSelectedBbox(cv::Mat& img,
                      const Detection& detection,
                      const cv::Scalar& color);
cv::Mat drawFeatureDebugPanel(const cv::Mat& left_base,
                              const cv::Mat& right_base,
                              const DebugFeatureMatchResult& result);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PIPELINE_DEBUG_UTILS_H_
