/**
 * @file stereo_calibrate_images.cpp
 * @brief Image pairing and chessboard detection helpers for stereo calibration.
 */

#include "stereo_calibrate_images.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <map>
#include <thread>
#include <utility>

namespace fs = std::filesystem;

std::vector<std::string> globImages(const fs::path& dir) {
    std::vector<std::string> files;
    if (!fs::exists(dir)) return files;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<ImagePair> pairImagesByStem(
    const std::vector<std::string>& leftFiles,
    const std::vector<std::string>& rightFiles) {
    std::map<std::string, std::string> leftByStem;
    std::map<std::string, std::string> rightByStem;
    for (const auto& path : leftFiles) {
        const std::string stem = fs::path(path).stem().string();
        if (!leftByStem.emplace(stem, path).second) {
            printf("[WARN] Duplicate left image stem ignored: %s\n",
                   stem.c_str());
        }
    }
    for (const auto& path : rightFiles) {
        const std::string stem = fs::path(path).stem().string();
        if (!rightByStem.emplace(stem, path).second) {
            printf("[WARN] Duplicate right image stem ignored: %s\n",
                   stem.c_str());
        }
    }

    std::vector<ImagePair> pairs;
    for (const auto& [stem, left] : leftByStem) {
        auto it = rightByStem.find(stem);
        if (it == rightByStem.end()) {
            printf("[WARN] Missing right image for %s, skipping\n",
                   stem.c_str());
            continue;
        }
        pairs.push_back({stem, left, it->second});
    }
    for (const auto& [stem, right] : rightByStem) {
        if (leftByStem.find(stem) == leftByStem.end()) {
            printf("[WARN] Missing left image for %s, skipping\n", stem.c_str());
        }
    }
    return pairs;
}

namespace {

cv::Mat toGrayCPU(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) return {};
    if (img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 1 && img.type() == CV_8UC1) {
        // Hikvision BayerRG8 sensor maps to OpenCV BayerBG convention.
        try {
            cv::cvtColor(img, img, cv::COLOR_BayerBG2GRAY);
        } catch (...) {
            // Already grayscale.
        }
    }
    return img;
}

cv::Mat toGrayGPU(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) return {};
    try {
        cv::cuda::GpuMat gpu(img);
        cv::cuda::GpuMat gray_gpu;
        if (img.channels() == 3) {
            cv::cuda::cvtColor(gpu, gray_gpu, cv::COLOR_BGR2GRAY);
        } else if (img.channels() == 1 && img.type() == CV_8UC1) {
            cv::cuda::cvtColor(gpu, gray_gpu, cv::COLOR_BayerBG2GRAY);
        } else {
            return toGrayCPU(path);
        }
        cv::Mat gray;
        gray_gpu.download(gray);
        return gray;
    } catch (const cv::Exception&) {
        return toGrayCPU(path);
    }
}

cv::Mat toGray(const std::string& path, bool use_gpu_preprocess) {
    return use_gpu_preprocess ? toGrayGPU(path) : toGrayCPU(path);
}

enum class ChessboardDetectorMode {
    CLASSIC_FAST,
    CLASSIC_FILTER_QUADS,
    CLASSIC_NORMALIZED,
    CLASSIC_PLAIN,
    SB_NORMALIZED,
    SB_EXHAUSTIVE,
};

bool findCornersWithMode(const cv::Mat& gray, cv::Size boardSize,
                         ChessboardDetectorMode mode,
                         std::vector<cv::Point2f>& corners) {
    static const cv::TermCriteria subpixCrit(
        cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

    corners.clear();
    auto refine_if_found = [&](bool found) {
        if (!found) return false;
        cv::cornerSubPix(gray, corners, cv::Size(11, 11),
                         cv::Size(-1, -1), subpixCrit);
        return true;
    };

    switch (mode) {
    case ChessboardDetectorMode::CLASSIC_FAST:
        return refine_if_found(cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_FAST_CHECK));
    case ChessboardDetectorMode::CLASSIC_FILTER_QUADS:
        return refine_if_found(cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_FILTER_QUADS));
    case ChessboardDetectorMode::CLASSIC_NORMALIZED:
        return refine_if_found(cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE));
    case ChessboardDetectorMode::CLASSIC_PLAIN:
        return refine_if_found(cv::findChessboardCorners(
            gray, boardSize, corners, 0));
    case ChessboardDetectorMode::SB_NORMALIZED:
        return refine_if_found(cv::findChessboardCornersSB(
            gray, boardSize, corners, cv::CALIB_CB_NORMALIZE_IMAGE));
    case ChessboardDetectorMode::SB_EXHAUSTIVE:
        return refine_if_found(cv::findChessboardCornersSB(
            gray, boardSize, corners,
            cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE));
    }
    return false;
}

}  // namespace

DetectionResult detectPairCorners(const ImagePair& pair,
                                  cv::Size boardSize,
                                  bool use_gpu_preprocess,
                                  bool use_sb,
                                  bool use_exhaustive) {
    DetectionResult result;
    cv::Mat gL = toGray(pair.left, use_gpu_preprocess);
    cv::Mat gR = toGray(pair.right, use_gpu_preprocess);
    if (gL.empty() || gR.empty()) return result;
    result.read_ok = true;
    result.left_size = gL.size();
    result.right_size = gR.size();
    // Try each detector as a pair. Accepting left/right corners from different
    // detector families risks inconsistent ordering on rotated or oblique boards.
    std::vector<ChessboardDetectorMode> modes = {
        ChessboardDetectorMode::CLASSIC_FAST,
        ChessboardDetectorMode::CLASSIC_FILTER_QUADS,
        ChessboardDetectorMode::CLASSIC_NORMALIZED,
    };
    if (use_sb) {
        modes.push_back(ChessboardDetectorMode::SB_NORMALIZED);
    }
    if (use_exhaustive) {
        modes.push_back(ChessboardDetectorMode::SB_EXHAUSTIVE);
    }
    for (ChessboardDetectorMode mode : modes) {
        std::vector<cv::Point2f> cL;
        const bool fL = findCornersWithMode(gL, boardSize, mode, cL);
        if (!fL) continue;
        std::vector<cv::Point2f> cR;
        const bool fR = findCornersWithMode(gR, boardSize, mode, cR);
        if (fL && fR) {
            result.found_left = true;
            result.found_right = true;
            result.corners_left = std::move(cL);
            result.corners_right = std::move(cR);
            return result;
        }
    }
    return result;
}

int defaultJobCount() {
    unsigned int hw = std::thread::hardware_concurrency();
    return std::max(1, static_cast<int>(hw == 0 ? 1 : hw));
}
