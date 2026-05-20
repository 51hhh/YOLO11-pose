#pragma once
#include <Eigen/Dense>

namespace stereo3d {

class Ball3DFilter {
public:
    Ball3DFilter();

    void reset();
    void predict(float dt);
    void predict(float dt, float ground_z, float min_clearance = 0.0f);
    void update(const Eigen::Vector3f& pos, float meas_noise = 0.05f);

    bool initialized() const { return inited_; }
    Eigen::Vector3f position() const { return x_.segment<3>(0); }
    Eigen::Vector3f velocity() const { return x_.segment<3>(3); }

private:
    bool inited_;
    Eigen::Matrix<float,6,1> x_;   // [x,y,z,vx,vy,vz]
    Eigen::Matrix<float,6,6> P_;
    Eigen::Matrix<float,6,6> Q_;
    Eigen::Matrix<float,3,3> R_;
};

} // namespace stereo3d
