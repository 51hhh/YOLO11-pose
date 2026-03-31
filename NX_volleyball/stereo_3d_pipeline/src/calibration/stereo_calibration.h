/**
 * @file stereo_calibration.h
 * @brief 双目标定参数加载 + VPI Remap LUT 生成
 *
 * 从 OpenCV YAML 标定文件加载内参、外参、校正矩阵,
 * 生成 VPI Remap 所需的 Lookup Table (LUT)。
 */

#ifndef STEREO_3D_PIPELINE_STEREO_CALIBRATION_H_
#define STEREO_3D_PIPELINE_STEREO_CALIBRATION_H_

#include <opencv2/opencv.hpp>
#include <string>

namespace stereo3d {

class StereoCalibration {
public:
    StereoCalibration() = default;
    ~StereoCalibration() = default;

    /**
     * @brief 从 YAML 文件加载标定参数
     * @param filepath 标定文件路径 (OpenCV FileStorage 格式)
     * @return true 加载成功
     *
     * 期望键名 (与 stereo_calibration.py 输出一致):
     *   camera_matrix_left, distortion_coefficients_left
     *   camera_matrix_right, distortion_coefficients_right
     *   projection_left, projection_right
     *   rectification_left, rectification_right
     *   rotation, translation, baseline
     */
    bool load(const std::string& filepath);

    // ===== Getters =====
    const cv::Mat& cameraMatrixLeft()  const { return K1_; }
    const cv::Mat& cameraMatrixRight() const { return K2_; }
    const cv::Mat& distCoeffsLeft()    const { return D1_; }
    const cv::Mat& distCoeffsRight()   const { return D2_; }
    const cv::Mat& projectionLeft()    const { return P1_; }
    const cv::Mat& projectionRight()   const { return P2_; }
    const cv::Mat& rectificationLeft() const { return R1_; }
    const cv::Mat& rectificationRight()const { return R2_; }
    const cv::Mat& rotation()          const { return R_; }
    const cv::Mat& translation()       const { return T_; }
    const cv::Mat& Q()                 const { return Q_; }  ///< disparity-to-depth map

    float getBaseline() const { return baseline_; }
    float getFocalLength() const;  ///< 从 P1 提取 fx
    cv::Point2f getPrincipalPoint() const;  ///< 从 P1 提取 (cx, cy)

    // P1 作为 3x4 矩阵的引用 (供 fusion 使用)
    // 若已调用 buildRemapMaps 且输出分辨率与标定分辨率不同, 返回缩放后的 P1
    const cv::Mat& getProjectionLeft() const {
        return P1_scaled_.empty() ? P1_ : P1_scaled_;
    }

    /**
     * @brief 生成 OpenCV undistortRectifyMap (用于构建 VPI Remap LUT)
     * @param[out] map1L, map2L 左目映射表
     * @param[out] map1R, map2R 右目映射表
     * @param width, height 图像尺寸
     */
    void buildRemapMaps(cv::Mat& map1L, cv::Mat& map2L,
                        cv::Mat& map1R, cv::Mat& map2R,
                        int width, int height) const;

private:
    cv::Mat K1_, D1_, P1_, R1_;   // 左目
    cv::Mat K2_, D2_, P2_, R2_;   // 右目
    cv::Mat R_, T_;               // 旋转/平移
    cv::Mat Q_;                   // 视差→深度
    mutable cv::Mat P1_scaled_, P2_scaled_;  // 缩放至 rect 分辨率的投影矩阵
    float baseline_ = 0.0f;      // 基线距离 (m)
    int image_width_  = 0;
    int image_height_ = 0;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_STEREO_CALIBRATION_H_
