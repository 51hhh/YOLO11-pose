/**
 * @file stereo_calibration.cpp
 * @brief 双目标定参数加载实现
 */

#include "stereo_calibration.h"
#include "../utils/logger.h"
#include <fstream>

namespace stereo3d {

bool StereoCalibration::load(const std::string& filepath) {
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        LOG_ERROR("Cannot open calibration file: %s", filepath.c_str());
        return false;
    }

    // 读取标定参数 (键名与 stereo_calibration.py 输出一致)
    fs["camera_matrix_left"]  >> K1_;
    fs["distortion_coefficients_left"]  >> D1_;
    fs["camera_matrix_right"] >> K2_;
    fs["distortion_coefficients_right"] >> D2_;
    fs["projection_left"]     >> P1_;
    fs["projection_right"]    >> P2_;
    fs["rectification_left"]  >> R1_;
    fs["rectification_right"] >> R2_;
    fs["rotation"]            >> R_;
    fs["translation"]         >> T_;
    fs["disparity_to_depth_map"] >> Q_;

    // 基线 (标定脚本输出单位 mm)
    if (!fs["baseline"].empty()) {
        double bl;
        fs["baseline"] >> bl;
        baseline_ = static_cast<float>(bl / 1000.0);  // mm -> m
        LOG_INFO("Baseline: %.2f mm (%.4f m)", bl, baseline_);
    } else if (!T_.empty()) {
        // 从平移向量计算
        baseline_ = static_cast<float>(cv::norm(T_) / 1000.0);
        LOG_INFO("Baseline from T: %.4f m", baseline_);
    }

    if (!fs["image_width"].empty())  fs["image_width"]  >> image_width_;
    if (!fs["image_height"].empty()) fs["image_height"] >> image_height_;

    fs.release();

    // 验证关键矩阵
    bool ok = true;
    auto check = [&](const cv::Mat& m, const char* name, int rows, int cols) {
        if (m.empty() || m.rows != rows || m.cols != cols) {
            LOG_ERROR("Invalid %s: expected %dx%d, got %s",
                      name, rows, cols,
                      m.empty() ? "empty" : (std::to_string(m.rows) + "x" + std::to_string(m.cols)).c_str());
            ok = false;
        }
    };
    check(K1_, "K1", 3, 3);
    check(K2_, "K2", 3, 3);
    check(P1_, "P1", 3, 4);
    check(P2_, "P2", 3, 4);
    // 畸变系数: 允许 4/5/8/12/14 参数 (OpenCV 支持的所有模型)
    auto checkDistortion = [&](const cv::Mat& m, const char* name) {
        if (m.empty()) {
            LOG_ERROR("Invalid %s: empty", name);
            ok = false;
        } else {
            int n = m.total();  // 元素总数
            if (n != 4 && n != 5 && n != 8 && n != 12 && n != 14) {
                LOG_ERROR("Invalid %s: expected 4/5/8/12/14 coefficients, got %d", name, n);
                ok = false;
            }
        }
    };
    checkDistortion(D1_, "D1");
    checkDistortion(D2_, "D2");

    if (ok) {
        LOG_INFO("Calibration loaded: %dx%d, RMS baseline=%.2fmm",
                 image_width_, image_height_, baseline_ * 1000.0f);
    }
    return ok;
}

float StereoCalibration::getFocalLength() const {
    const cv::Mat& P = P1_scaled_.empty() ? P1_ : P1_scaled_;
    if (P.empty()) return 0.0f;
    return static_cast<float>(P.at<double>(0, 0));
}

cv::Point2f StereoCalibration::getPrincipalPoint() const {
    const cv::Mat& P = P1_scaled_.empty() ? P1_ : P1_scaled_;
    if (P.empty()) return {0, 0};
    return {static_cast<float>(P.at<double>(0, 2)),
            static_cast<float>(P.at<double>(1, 2))};
}

void StereoCalibration::buildRemapMaps(
    cv::Mat& map1L, cv::Mat& map2L,
    cv::Mat& map1R, cv::Mat& map2R,
    int width, int height) const
{
    // 若输出分辨率与标定分辨率不同, 缩放 P1/P2 的 fx/fy/cx/cy
    cv::Mat P1_use = P1_, P2_use = P2_;
    if (image_width_ > 0 && image_height_ > 0 &&
        (width != image_width_ || height != image_height_))
    {
        double sx = (double)width / image_width_;
        double sy = (double)height / image_height_;
        P1_use = P1_.clone();
        P1_use.at<double>(0, 0) *= sx;  // fx
        P1_use.at<double>(1, 1) *= sy;  // fy
        P1_use.at<double>(0, 2) *= sx;  // cx
        P1_use.at<double>(1, 2) *= sy;  // cy
        P1_use.at<double>(0, 3) *= sx;  // Tx*fx
        P2_use = P2_.clone();
        P2_use.at<double>(0, 0) *= sx;
        P2_use.at<double>(1, 1) *= sy;
        P2_use.at<double>(0, 2) *= sx;
        P2_use.at<double>(1, 2) *= sy;
        P2_use.at<double>(0, 3) *= sx;
        LOG_INFO("Scaled P1/P2: %dx%d -> %dx%d (sx=%.3f, sy=%.3f)",
                 image_width_, image_height_, width, height, sx, sy);
    }
    P1_scaled_ = P1_use;
    P2_scaled_ = P2_use;

    // 使用 OpenCV initUndistortRectifyMap 生成映射表
    // 输出 CV_32FC1 格式供 VPI 使用
    cv::initUndistortRectifyMap(K1_, D1_, R1_, P1_use,
                                cv::Size(width, height),
                                CV_32FC1, map1L, map2L);
    cv::initUndistortRectifyMap(K2_, D2_, R2_, P2_use,
                                cv::Size(width, height),
                                CV_32FC1, map1R, map2R);

    LOG_INFO("Built remap LUTs: %dx%d", width, height);
}

}  // namespace stereo3d
