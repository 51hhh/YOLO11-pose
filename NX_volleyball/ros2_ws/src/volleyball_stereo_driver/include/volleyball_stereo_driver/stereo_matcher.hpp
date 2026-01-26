/**
 * @file stereo_matcher.hpp
 * @brief 双目立体匹配
 */

#ifndef VOLLEYBALL_STEREO_DRIVER__STEREO_MATCHER_HPP_
#define VOLLEYBALL_STEREO_DRIVER__STEREO_MATCHER_HPP_

#include <string>
#include <opencv2/opencv.hpp>

namespace volleyball {

struct StereoPoint {
    cv::Point3f position_3d;
    float confidence;
    float disparity;
    bool valid;
    
    StereoPoint() : position_3d(0, 0, 0), confidence(0), disparity(0), valid(false) {}
};

class StereoMatcher {
public:
    StereoMatcher(const std::string& calib_file, float min_disparity = 10.0f, float max_depth = 15.0f);
    ~StereoMatcher() = default;
    
    StereoPoint triangulate(const cv::Point2f& pt_left, const cv::Point2f& pt_right);
    std::vector<StereoPoint> triangulateBatch(const std::vector<cv::Point2f>& pts_left, 
                                               const std::vector<cv::Point2f>& pts_right);
    
    float getBaseline() const { return baseline_; }

private:
    cv::Mat K1_, D1_, P1_;
    cv::Mat K2_, D2_, P2_;
    float baseline_;
    float min_disparity_;
    float max_depth_;
    
    bool loadCalibration(const std::string& calib_file);
    cv::Point2f undistortPoint(const cv::Point2f& pt, const cv::Mat& K, const cv::Mat& D, const cv::Mat& P);
    float computeConfidence(float disparity, float depth);
};

}  // namespace volleyball

#endif  // VOLLEYBALL_STEREO_DRIVER__STEREO_MATCHER_HPP_
