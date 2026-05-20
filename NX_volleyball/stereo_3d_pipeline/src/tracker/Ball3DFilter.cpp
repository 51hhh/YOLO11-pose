#include "Ball3DFilter.h"

#include <algorithm>

namespace stereo3d {

static constexpr float G = 9.81f;

Ball3DFilter::Ball3DFilter() {
    reset();
}

void Ball3DFilter::reset() {
    inited_ = false;
    x_.setZero();
    P_.setIdentity();
    P_ *= 1.0f;

    Q_.setIdentity();
    Q_ *= 0.001f;

    R_.setIdentity();
    R_ *= 0.02f;
}

void Ball3DFilter::predict(float dt) {
    if (!inited_) return;

    Eigen::Matrix<float,6,6> F = Eigen::Matrix<float,6,6>::Identity();
    F(0,3) = dt;
    F(1,4) = dt;
    F(2,5) = dt;

    x_ = F * x_;
    x_(5) -= G * dt;

    P_ = F * P_ * F.transpose() + Q_;
}

void Ball3DFilter::predict(float dt, float ground_z, float min_clearance) {
    predict(dt);
    if (!inited_) return;

    const float z_floor = ground_z + std::max(0.0f, min_clearance);
    if (x_(2) < z_floor) {
        x_(2) = z_floor;
        if (x_(5) < 0.0f) x_(5) = 0.0f;
    }
}

void Ball3DFilter::update(const Eigen::Vector3f& z, float meas_noise) {
    R_.setIdentity();
    R_ *= meas_noise;

    if (!inited_) {
        x_.segment<3>(0) = z;
        x_.segment<3>(3).setZero();
        inited_ = true;
        return;
    }

    Eigen::Matrix<float,3,6> H;
    H.setZero();
    H(0,0) = H(1,1) = H(2,2) = 1.0f;

    Eigen::Vector3f y = z - H * x_;
    Eigen::Matrix3f S = H * P_ * H.transpose() + R_;
    Eigen::Matrix<float,6,3> K = P_ * H.transpose() * S.inverse();

    x_ += K * y;
    P_ = (Eigen::Matrix<float,6,6>::Identity() - K * H) * P_;
}

} // namespace stereo3d
