/**
 * @file roi_manager.hpp
 * @brief ROI 管理器
 */

#ifndef VOLLEYBALL_STEREO_DRIVER__ROI_MANAGER_HPP_
#define VOLLEYBALL_STEREO_DRIVER__ROI_MANAGER_HPP_

#include <opencv2/opencv.hpp>

namespace volleyball {

class ROIManager {
public:
    explicit ROIManager(int roi_size = 320);
    ~ROIManager() = default;
    
    cv::Mat cropROI(const cv::Mat& image, const cv::Point2f& predicted_center, cv::Point2f& offset);
    cv::Point2f mapToOriginal(const cv::Point2f& roi_point, const cv::Point2f& offset) const;
    
    void setROISize(int size) { roi_size_ = size; }
    int getROISize() const { return roi_size_; }
    void adjustROISize(float velocity);

private:
    int roi_size_;
    int min_roi_size_;
    int max_roi_size_;
};

}  // namespace volleyball

#endif  // VOLLEYBALL_STEREO_DRIVER__ROI_MANAGER_HPP_
