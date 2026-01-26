/**
 * @file kalman_filter_3d.cpp
 * @brief 3D 卡尔曼滤波器实现
 */

#include "volleyball_stereo_driver/kalman_filter_3d.hpp"

namespace volleyball {

KalmanFilter3D::KalmanFilter3D(double dt, double process_noise)
    : dt_(dt),
      process_noise_(process_noise),
      initialized_(false)
{
    initializeMatrices();
}

void KalmanFilter3D::initializeMatrices() {
    // 状态向量: [x, y, z, vx, vy, vz, ax, ay, az] (9x1)
    state_ = cv::Mat::zeros(9, 1, CV_64F);
    
    // 协方差矩阵 (9x9)
    covariance_ = cv::Mat::eye(9, 9, CV_64F) * 1000.0;
    
    // 状态转移矩阵 F (9x9)
    // x_k = F * x_{k-1}
    F_ = cv::Mat::eye(9, 9, CV_64F);
    
    // 位置 = 位置 + 速度*dt + 0.5*加速度*dt^2
    F_.at<double>(0, 3) = dt_;
    F_.at<double>(1, 4) = dt_;
    F_.at<double>(2, 5) = dt_;
    F_.at<double>(0, 6) = 0.5 * dt_ * dt_;
    F_.at<double>(1, 7) = 0.5 * dt_ * dt_;
    F_.at<double>(2, 8) = 0.5 * dt_ * dt_;
    
    // 速度 = 速度 + 加速度*dt
    F_.at<double>(3, 6) = dt_;
    F_.at<double>(4, 7) = dt_;
    F_.at<double>(5, 8) = dt_;
    
    // 观测矩阵 H (3x9) - 只观测位置
    H_ = cv::Mat::zeros(3, 9, CV_64F);
    H_.at<double>(0, 0) = 1.0;
    H_.at<double>(1, 1) = 1.0;
    H_.at<double>(2, 2) = 1.0;
    
    // 过程噪声协方差 Q (9x9)
    Q_ = cv::Mat::eye(9, 9, CV_64F);
    
    // 位置噪声
    Q_.at<double>(0, 0) = process_noise_ * 0.25 * dt_ * dt_ * dt_ * dt_;
    Q_.at<double>(1, 1) = process_noise_ * 0.25 * dt_ * dt_ * dt_ * dt_;
    Q_.at<double>(2, 2) = process_noise_ * 0.25 * dt_ * dt_ * dt_ * dt_;
    
    // 速度噪声
    Q_.at<double>(3, 3) = process_noise_ * dt_ * dt_;
    Q_.at<double>(4, 4) = process_noise_ * dt_ * dt_;
    Q_.at<double>(5, 5) = process_noise_ * dt_ * dt_;
    
    // 加速度噪声
    Q_.at<double>(6, 6) = process_noise_;
    Q_.at<double>(7, 7) = process_noise_;
    Q_.at<double>(8, 8) = process_noise_;
    
    // 观测噪声协方差 R (3x3) - 默认值，会动态调整
    R_ = cv::Mat::eye(3, 3, CV_64F) * 0.01;
    
    // 预设不同深度的观测噪声
    R_near_ = cv::Mat::eye(3, 3, CV_64F);
    R_near_.at<double>(0, 0) = 0.01;
    R_near_.at<double>(1, 1) = 0.01;
    R_near_.at<double>(2, 2) = 0.01;
    
    R_mid_ = cv::Mat::eye(3, 3, CV_64F);
    R_mid_.at<double>(0, 0) = 0.05;
    R_mid_.at<double>(1, 1) = 0.05;
    R_mid_.at<double>(2, 2) = 0.2;
    
    R_far_ = cv::Mat::eye(3, 3, CV_64F);
    R_far_.at<double>(0, 0) = 0.1;
    R_far_.at<double>(1, 1) = 0.1;
    R_far_.at<double>(2, 2) = 0.5;
}

void KalmanFilter3D::init(const cv::Point3f& initial_position) {
    // 初始化状态向量
    state_.at<double>(0, 0) = initial_position.x;
    state_.at<double>(1, 0) = initial_position.y;
    state_.at<double>(2, 0) = initial_position.z;
    
    // 速度和加速度初始化为 0
    for (int i = 3; i < 9; ++i) {
        state_.at<double>(i, 0) = 0.0;
    }
    
    initialized_ = true;
}

void KalmanFilter3D::predict() {
    if (!initialized_) {
        return;
    }
    
    // 预测状态: x_k = F * x_{k-1}
    state_ = F_ * state_;
    
    // 预测协方差: P_k = F * P_{k-1} * F^T + Q
    covariance_ = F_ * covariance_ * F_.t() + Q_;
}

void KalmanFilter3D::update(const cv::Point3f& measurement, float depth) {
    if (!initialized_) {
        init(measurement);
        return;
    }
    
    // 根据深度动态调整观测噪声
    updateMeasurementNoise(depth);
    
    // 观测向量 z (3x1)
    cv::Mat z = cv::Mat::zeros(3, 1, CV_64F);
    z.at<double>(0, 0) = measurement.x;
    z.at<double>(1, 0) = measurement.y;
    z.at<double>(2, 0) = measurement.z;
    
    // 计算卡尔曼增益: K = P * H^T * (H * P * H^T + R)^{-1}
    cv::Mat S = H_ * covariance_ * H_.t() + R_;
    cv::Mat K = covariance_ * H_.t() * S.inv();
    
    // 更新状态: x_k = x_k + K * (z - H * x_k)
    cv::Mat y = z - H_ * state_;  // 创新 (innovation)
    state_ = state_ + K * y;
    
    // 更新协方差: P_k = (I - K * H) * P_k
    cv::Mat I = cv::Mat::eye(9, 9, CV_64F);
    covariance_ = (I - K * H_) * covariance_;
}

void KalmanFilter3D::updateMeasurementNoise(float depth) {
    // 根据深度动态调整观测噪声
    if (depth < 5.0f) {
        // 近距离: 高精度
        R_ = R_near_.clone();
    } else if (depth > 12.0f) {
        // 远距离: 低精度
        R_ = R_far_.clone();
    } else {
        // 中距离: 线性插值
        float ratio = (depth - 5.0f) / (12.0f - 5.0f);
        R_ = R_near_ * (1.0 - ratio) + R_far_ * ratio;
    }
}

cv::Point3f KalmanFilter3D::getPosition() const {
    if (!initialized_) {
        return cv::Point3f(0, 0, 0);
    }
    
    return cv::Point3f(
        static_cast<float>(state_.at<double>(0, 0)),
        static_cast<float>(state_.at<double>(1, 0)),
        static_cast<float>(state_.at<double>(2, 0))
    );
}

cv::Point3f KalmanFilter3D::getVelocity() const {
    if (!initialized_) {
        return cv::Point3f(0, 0, 0);
    }
    
    return cv::Point3f(
        static_cast<float>(state_.at<double>(3, 0)),
        static_cast<float>(state_.at<double>(4, 0)),
        static_cast<float>(state_.at<double>(5, 0))
    );
}

cv::Point3f KalmanFilter3D::getAcceleration() const {
    if (!initialized_) {
        return cv::Point3f(0, 0, 0);
    }
    
    return cv::Point3f(
        static_cast<float>(state_.at<double>(6, 0)),
        static_cast<float>(state_.at<double>(7, 0)),
        static_cast<float>(state_.at<double>(8, 0))
    );
}

cv::Point3f KalmanFilter3D::getPredictedPosition(int steps_ahead) const {
    if (!initialized_ || steps_ahead <= 0) {
        return getPosition();
    }
    
    // 使用当前状态预测未来位置
    // x_future = x + v*t + 0.5*a*t^2
    double t = dt_ * steps_ahead;
    
    cv::Point3f pos = getPosition();
    cv::Point3f vel = getVelocity();
    cv::Point3f acc = getAcceleration();
    
    return cv::Point3f(
        pos.x + vel.x * t + 0.5f * acc.x * t * t,
        pos.y + vel.y * t + 0.5f * acc.y * t * t,
        pos.z + vel.z * t + 0.5f * acc.z * t * t
    );
}

void KalmanFilter3D::reset() {
    initialized_ = false;
    state_ = cv::Mat::zeros(9, 1, CV_64F);
    covariance_ = cv::Mat::eye(9, 9, CV_64F) * 1000.0;
}

}  // namespace volleyball
