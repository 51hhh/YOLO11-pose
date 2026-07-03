/**
 * @file stereo_calibrate_images.h
 * @brief Image pairing and chessboard detection helpers for stereo calibration.
 */

#ifndef STEREO_3D_PIPELINE_STEREO_CALIBRATE_IMAGES_H_
#define STEREO_3D_PIPELINE_STEREO_CALIBRATE_IMAGES_H_

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

struct ImagePair {
    std::string stem;
    std::string left;
    std::string right;
};

struct DetectionResult {
    bool read_ok = false;
    bool found_left = false;
    bool found_right = false;
    cv::Size left_size;
    cv::Size right_size;
    std::vector<cv::Point2f> corners_left;
    std::vector<cv::Point2f> corners_right;
};

std::vector<std::string> globImages(const std::filesystem::path& dir);

std::vector<ImagePair> pairImagesByStem(
    const std::vector<std::string>& leftFiles,
    const std::vector<std::string>& rightFiles);

DetectionResult detectPairCorners(const ImagePair& pair,
                                  cv::Size boardSize,
                                  bool use_gpu_preprocess,
                                  bool use_sb,
                                  bool use_exhaustive);

int defaultJobCount();

#endif  // STEREO_3D_PIPELINE_STEREO_CALIBRATE_IMAGES_H_
