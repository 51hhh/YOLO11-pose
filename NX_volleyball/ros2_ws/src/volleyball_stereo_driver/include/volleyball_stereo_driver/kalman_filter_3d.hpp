/**
 * @file kalman_filter_3d.hpp
 * @brief 3D 卡尔曼滤波器
 */

#ifndef VOLLEYBALL_STEREO_DRIVER__KALMAN_FILTER_3D_HPP_
#define VOLLEYBALL_STEREO_DRIVER__KALMAN_FILTER_3D_HPP_

#include <opencv2/opencv.hpp>

namespace volleyball {

class KalmanFilter3D {
public:
    explicit KalmanFilter3D(double dt = 0.01, double process_noise = 0.01);
    ~KalmanFilter3D() = default;
    
    void init(const cv::Point3f& initial_position);
    void predict();
    void update(const cv::Point3f& measurement, float depth);
    
    cv::Point3f getPosition() const;
    cv::Point3f getVelocity() const;
    cv::Point3f getAcceleration() const;
    cv::Point3f getPredictedPosition(int steps_ahead = 1) const;
    
    void reset();
    bool isInitialized() const { return initialized_; }

private:
    cv::Mat state_;
    cv::Mat covariance_;
    cv::Mat F_;
    cv::Mat H_;
    cv::Mat Q_;
    cv::Mat R_;
    
    double dt_;
    double process_noise_;
    bool initialized_;
    
    cv::Mat R_near_;
    cv::Mat R_mid_;
    cv::Mat R_far_;
    
    void updateMeasurementNoise(float depth);
    void initializeMatrices();
};

}  // namespace volleyball

#endif  // VOLLEYBALL_STEREO_DRIVER__KALMAN_FILTER_3D_HPP_
