#include "volleyball/adapters/cuda_preprocess/stereo_calibration.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace calibration = volleyball::adapters::cuda_preprocess;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

void testLoadsCheckedInCalibrationAndBuildsMaps() {
    const std::filesystem::path path =
        std::filesystem::path{VOLLEYBALL_TEST_SOURCE_DIR} /
        "configs/calibration/stereo_calib.yaml";
    calibration::StereoCalibration value;
    check(value.load(path.string()),
          "checked-in OpenCV calibration must load");
    check(value.imageWidth() == 1440 && value.imageHeight() == 1080,
          "calibration must retain its source image geometry");
    check(std::abs(value.getBaseline() - 0.8531544F) < 1.0e-5F,
          "millimetre baseline must be converted to metres");

    cv::Mat left_x;
    cv::Mat left_y;
    cv::Mat right_x;
    cv::Mat right_y;
    check(value.buildRemapMaps(
              left_x, left_y, right_x, right_y, 720, 540),
          "same-aspect half-resolution maps must be supported");
    check(
        left_x.rows == 540 && left_x.cols == 720 &&
            left_x.type() == CV_32FC1 &&
            left_y.size() == left_x.size() &&
            right_x.size() == left_x.size() &&
            right_y.size() == left_x.size(),
        "all four rectification maps must have one float coordinate per pixel");
    check(std::abs(value.getFocalLength() - 827.4286F) < 1.0e-3F,
          "projection focal length must scale with output resolution");

    check(!value.buildRemapMaps(
              left_x, left_y, right_x, right_y, 720, 400) &&
              left_x.empty() && left_y.empty() &&
              right_x.empty() && right_y.empty(),
          "aspect-ratio mismatch must fail without leaving partial maps");
}

void testMissingCalibrationIsRejected() {
    calibration::StereoCalibration value;
    check(!value.load("/definitely/missing/volleyball-calibration.yaml"),
          "missing calibration must be an explicit failure");
}

}  // namespace

int main() {
    testLoadsCheckedInCalibrationAndBuildsMaps();
    testMissingCalibrationIsRejected();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "stereo calibration tests passed\n";
    return EXIT_SUCCESS;
}
