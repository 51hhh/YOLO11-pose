/**
 * @file roi_manager.cpp
 * @brief ROI 管理器实现
 */

#include "volleyball_stereo_driver/roi_manager.hpp"
#include <algorithm>

namespace volleyball {

ROIManager::ROIManager(int roi_size)
    : roi_size_(roi_size),
      min_roi_size_(160),
      max_roi_size_(640)
{
    // 确保 ROI 尺寸在合理范围内
    roi_size_ = std::max(min_roi_size_, std::min(max_roi_size_, roi_size_));
}

cv::Mat ROIManager::cropROI(
    const cv::Mat& image,
    const cv::Point2f& predicted_center,
    cv::Point2f& offset
) {
    if (image.empty()) {
        offset = cv::Point2f(0, 0);
        return cv::Mat();
    }
    
    int img_height = image.rows;
    int img_width = image.cols;
    
    // 计算 ROI 的左上角坐标
    int cx = static_cast<int>(predicted_center.x);
    int cy = static_cast<int>(predicted_center.y);
    
    int half_size = roi_size_ / 2;
    int x1 = cx - half_size;
    int y1 = cy - half_size;
    
    // 边界检查和调整
    if (x1 < 0) {
        x1 = 0;
    }
    if (y1 < 0) {
        y1 = 0;
    }
    
    int x2 = x1 + roi_size_;
    int y2 = y1 + roi_size_;
    
    // 确保不超出图像边界
    if (x2 > img_width) {
        x2 = img_width;
        x1 = std::max(0, x2 - roi_size_);
    }
    if (y2 > img_height) {
        y2 = img_height;
        y1 = std::max(0, y2 - roi_size_);
    }
    
    // 记录偏移量
    offset.x = static_cast<float>(x1);
    offset.y = static_cast<float>(y1);
    
    // 裁切 ROI
    cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
    
    // 确保ROI矩形有效
    if (roi_rect.width <= 0 || roi_rect.height <= 0) {
        offset = cv::Point2f(0, 0);
        return cv::Mat();
    }
    
    cv::Mat roi = image(roi_rect).clone();
    
    return roi;
}

cv::Point2f ROIManager::mapToOriginal(
    const cv::Point2f& roi_point,
    const cv::Point2f& offset
) const {
    return cv::Point2f(
        roi_point.x + offset.x,
        roi_point.y + offset.y
    );
}

void ROIManager::adjustROISize(float velocity) {
    // 根据速度动态调整 ROI 大小
    // 速度越快，ROI 越大，以应对快速移动
    
    // 速度阈值 (m/s)
    const float low_velocity = 5.0f;   // 慢速
    const float high_velocity = 20.0f; // 快速
    
    if (velocity < low_velocity) {
        // 慢速：使用较小的 ROI
        roi_size_ = 320;
    } else if (velocity > high_velocity) {
        // 快速：使用较大的 ROI
        roi_size_ = 480;
    } else {
        // 中速：线性插值
        float ratio = (velocity - low_velocity) / (high_velocity - low_velocity);
        roi_size_ = static_cast<int>(320 + ratio * (480 - 320));
    }
    
    // 确保在范围内
    roi_size_ = std::max(min_roi_size_, std::min(max_roi_size_, roi_size_));
}

}  // namespace volleyball
