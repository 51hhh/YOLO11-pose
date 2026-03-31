/**
 * @file stereo_matcher.cpp
 * @brief 双目立体匹配实现
 */

#include "volleyball_stereo_driver/stereo_matcher.hpp"
#include <fstream>
#include <iostream>

namespace volleyball {

StereoMatcher::StereoMatcher(
    const std::string& calib_file,
    float min_disparity,
    float max_depth
) : baseline_(0.25f),
    min_disparity_(min_disparity),
    max_depth_(max_depth)
{
    if (!loadCalibration(calib_file)) {
        std::cerr << "警告: 标定文件加载失败，使用默认参数" << std::endl;
        
        // 使用默认参数 (假设已标定)
        K1_ = cv::Mat::eye(3, 3, CV_64F);
        K2_ = cv::Mat::eye(3, 3, CV_64F);
        D1_ = cv::Mat::zeros(1, 5, CV_64F);
        D2_ = cv::Mat::zeros(1, 5, CV_64F);
        P1_ = cv::Mat::eye(3, 4, CV_64F);
        P2_ = cv::Mat::eye(3, 4, CV_64F);
    }
}

bool StereoMatcher::loadCalibration(const std::string& calib_file) {
    // 注意: OpenCV 的 FileStorage 不直接支持 .npz 格式
    // 这里假设标定文件是 .yaml 或 .xml 格式
    // 如果是 .npz，需要使用 Python 转换或 cnpy 库
    
    std::string ext = calib_file.substr(calib_file.find_last_of(".") + 1);
    
    if (ext == "npz") {
        std::cerr << "错误: 暂不支持 .npz 格式，请转换为 .yaml 或 .xml" << std::endl;
        std::cerr << "提示: 使用 Python 脚本转换:" << std::endl;
        std::cerr << "  import numpy as np" << std::endl;
        std::cerr << "  import cv2" << std::endl;
        std::cerr << "  data = np.load('stereo_calib.npz')" << std::endl;
        std::cerr << "  fs = cv2.FileStorage('stereo_calib.yaml', cv2.FILE_STORAGE_WRITE)" << std::endl;
        std::cerr << "  for key in data.files:" << std::endl;
        std::cerr << "      fs.write(key, data[key])" << std::endl;
        std::cerr << "  fs.release()" << std::endl;
        return false;
    }
    
    cv::FileStorage fs(calib_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "错误: 无法打开标定文件: " << calib_file << std::endl;
        return false;
    }
    
    // 读取标定参数 (键名与 stereo_calibration.py 输出的 YAML 一致)
    fs["camera_matrix_left"] >> K1_;
    fs["distortion_coefficients_left"] >> D1_;
    fs["camera_matrix_right"] >> K2_;
    fs["distortion_coefficients_right"] >> D2_;
    fs["projection_left"] >> P1_;
    fs["projection_right"] >> P2_;
    
    // 验证读取的矩阵
    bool valid = true;
    if (K1_.empty() || K1_.rows != 3 || K1_.cols != 3) {
        std::cerr << "错误: K1 矩阵无效" << std::endl;
        valid = false;
    }
    if (K2_.empty() || K2_.rows != 3 || K2_.cols != 3) {
        std::cerr << "错误: K2 矩阵无效" << std::endl;
        valid = false;
    }
    if (D1_.empty() || D1_.rows != 1 || D1_.cols != 5) {
        std::cerr << "错误: D1 矩阵无效" << std::endl;
        valid = false;
    }
    if (D2_.empty() || D2_.rows != 1 || D2_.cols != 5) {
        std::cerr << "错误: D2 矩阵无效" << std::endl;
        valid = false;
    }
    if (P1_.empty() || P1_.rows != 3 || P1_.cols != 4) {
        std::cerr << "错误: P1 矩阵无效" << std::endl;
        valid = false;
    }
    if (P2_.empty() || P2_.rows != 3 || P2_.cols != 4) {
        std::cerr << "错误: P2 矩阵无效" << std::endl;
        valid = false;
    }
    
    if (!valid) {
        fs.release();
        return false;
    }
    
    // 读取基线 (标定脚本输出为 mm，需转换为 m)
    if (!fs["baseline"].empty()) {
        fs["baseline"] >> baseline_;
        baseline_ /= 1000.0f;  // mm → m
    } else {
        // 从投影矩阵计算基线 (P2[0][3] = -fx * baseline_mm)
        if (!P2_.empty() && P2_.cols == 4) {
            baseline_ = static_cast<float>(
                -P2_.at<double>(0, 3) / P2_.at<double>(0, 0) / 1000.0);
        }
    }
    
    fs.release();
    
    std::cout << "✅ 标定参数加载成功" << std::endl;
    std::cout << "   基线: " << baseline_ << " m (" << baseline_ * 1000.0f << " mm)" << std::endl;
    
    return true;
}

cv::Point2f StereoMatcher::undistortPoint(
    const cv::Point2f& pt,
    const cv::Mat& K,
    const cv::Mat& D,
    const cv::Mat& P
) {
    // 将点转换为向量
    std::vector<cv::Point2f> pts_in = {pt};
    std::vector<cv::Point2f> pts_out;
    
    // 去畸变
    cv::undistortPoints(pts_in, pts_out, K, D, cv::noArray(), P);
    
    return pts_out[0];
}

float StereoMatcher::computeConfidence(float disparity, float depth) {
    // 基于视差和深度计算置信度
    
    // 视差太小: 低置信度
    if (disparity < min_disparity_) {
        return 0.1f;
    }
    
    // 深度超出范围: 低置信度
    if (depth > max_depth_ || depth < 0.1f) {
        return 0.2f;
    }
    
    // 视差越大，置信度越高 (但有上限)
    float conf = std::min(1.0f, disparity / 100.0f);
    
    // 深度越远，置信度越低
    if (depth > 10.0f) {
        conf *= (1.0f - (depth - 10.0f) / (max_depth_ - 10.0f) * 0.5f);
    }
    
    return std::max(0.1f, conf);
}

StereoPoint StereoMatcher::triangulate(
    const cv::Point2f& pt_left,
    const cv::Point2f& pt_right
) {
    StereoPoint result;
    
    // 去畸变
    cv::Point2f pt_left_undist = undistortPoint(pt_left, K1_, D1_, P1_);
    cv::Point2f pt_right_undist = undistortPoint(pt_right, K2_, D2_, P2_);
    
    // 计算视差
    float disparity = pt_left_undist.x - pt_right_undist.x;
    result.disparity = disparity;
    
    // 检查视差有效性
    if (disparity < min_disparity_) {
        result.valid = false;
        return result;
    }
    
    // 三角测量
    // 将点转换为齐次坐标
    cv::Mat pts_left_mat(2, 1, CV_32F);
    pts_left_mat.at<float>(0, 0) = pt_left_undist.x;
    pts_left_mat.at<float>(1, 0) = pt_left_undist.y;
    
    cv::Mat pts_right_mat(2, 1, CV_32F);
    pts_right_mat.at<float>(0, 0) = pt_right_undist.x;
    pts_right_mat.at<float>(1, 0) = pt_right_undist.y;
    
    // 使用 OpenCV 的三角测量函数
    cv::Mat points_4d;
    cv::triangulatePoints(P1_, P2_, pts_left_mat, pts_right_mat, points_4d);
    
    // 归一化齐次坐标
    float w = points_4d.at<float>(3, 0);
    if (std::abs(w) < 1e-6) {
        result.valid = false;
        return result;
    }
    
    result.position_3d.x = points_4d.at<float>(0, 0) / w;
    result.position_3d.y = points_4d.at<float>(1, 0) / w;
    result.position_3d.z = points_4d.at<float>(2, 0) / w;
    
    // 检查深度有效性
    if (result.position_3d.z > max_depth_ || result.position_3d.z < 0.1f) {
        result.valid = false;
        return result;
    }
    
    // 计算置信度
    result.confidence = computeConfidence(disparity, result.position_3d.z);
    result.valid = true;
    
    return result;
}

std::vector<StereoPoint> StereoMatcher::triangulateBatch(
    const std::vector<cv::Point2f>& pts_left,
    const std::vector<cv::Point2f>& pts_right
) {
    std::vector<StereoPoint> results;
    
    if (pts_left.size() != pts_right.size()) {
        std::cerr << "错误: 左右点数量不匹配" << std::endl;
        return results;
    }
    
    results.reserve(pts_left.size());
    
    for (size_t i = 0; i < pts_left.size(); ++i) {
        results.push_back(triangulate(pts_left[i], pts_right[i]));
    }
    
    return results;
}

}  // namespace volleyball
