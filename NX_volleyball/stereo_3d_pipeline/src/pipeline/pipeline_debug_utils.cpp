#include "pipeline_debug_utils.h"
#include "../utils/logger.h"

#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vpi/Image.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace stereo3d {

bool writeDebugImage(const std::filesystem::path& output_dir,
                     const std::string& filename,
                     const cv::Mat& image) {
    const std::filesystem::path image_path = output_dir / filename;
    if (image.empty()) {
        LOG_ERROR("Feature match debug: refusing to write empty image %s",
                  image_path.string().c_str());
        return false;
    }
    try {
        if (!cv::imwrite(image_path.string(), image)) {
            LOG_ERROR("Feature match debug: failed to write %s",
                      image_path.string().c_str());
            return false;
        }
    } catch (const cv::Exception& e) {
        LOG_ERROR("Feature match debug: failed to write %s: %s",
                  image_path.string().c_str(), e.what());
        return false;
    }
    return true;
}

bool lockGrayVpiImageCopy(VPIImage img, cv::Mat& out) {
    VPIImageData data;
    VPIStatus st = vpiImageLockData(img, VPI_LOCK_READ,
                                    VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                                    &data);
    if (st != VPI_SUCCESS) return false;
    try {
        const int w = data.buffer.pitch.planes[0].width;
        const int h = data.buffer.pitch.planes[0].height;
        const int pitch = data.buffer.pitch.planes[0].pitchBytes;
        cv::Mat view(h, w, CV_8UC1, data.buffer.pitch.planes[0].data, pitch);
        view.copyTo(out);
    } catch (const cv::Exception&) {
        vpiImageUnlock(img);
        return false;
    }
    vpiImageUnlock(img);
    return true;
}

bool lockBgrVpiImageCopy(VPIImage img, cv::Mat& out) {
    VPIImageData data;
    VPIStatus st = vpiImageLockData(img, VPI_LOCK_READ,
                                    VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR,
                                    &data);
    if (st != VPI_SUCCESS) return false;
    const int plane_width = data.buffer.pitch.planes[0].width;
    const int h = data.buffer.pitch.planes[0].height;
    const int pitch = data.buffer.pitch.planes[0].pitchBytes;
    const void* src = data.buffer.pitch.planes[0].data;
    if (plane_width <= 0 || h <= 0 || pitch <= 0 || !src) {
        vpiImageUnlock(img);
        return false;
    }
    int pixel_width = plane_width;
    size_t row_bytes = static_cast<size_t>(pixel_width) * 3U;
    if (row_bytes > static_cast<size_t>(pitch)) {
        if (plane_width % 3 == 0 &&
            static_cast<size_t>(plane_width) <= static_cast<size_t>(pitch)) {
            pixel_width = plane_width / 3;
            row_bytes = static_cast<size_t>(plane_width);
        } else {
            vpiImageUnlock(img);
            return false;
        }
    }
    out.create(h, pixel_width, CV_8UC3);
    const cudaError_t err = cudaMemcpy2D(out.data, out.step[0],
                                         src, static_cast<size_t>(pitch),
                                         row_bytes, static_cast<size_t>(h),
                                         cudaMemcpyDeviceToHost);
    vpiImageUnlock(img);
    if (err != cudaSuccess) {
        out.release();
        return false;
    }
    return true;
}

cv::Mat drawDetectionOverlay(const cv::Mat& image,
                             const std::vector<Detection>& detections,
                             const std::string& title) {
    cv::Mat out;
    if (image.channels() == 1) {
        cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);
    } else {
        image.copyTo(out);
    }
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        const int x = static_cast<int>(std::round(d.cx - d.width * 0.5f));
        const int y = static_cast<int>(std::round(d.cy - d.height * 0.5f));
        const int w = static_cast<int>(std::round(d.width));
        const int h = static_cast<int>(std::round(d.height));
        cv::rectangle(out, cv::Rect(x, y, w, h) &
                            cv::Rect(0, 0, out.cols, out.rows),
                      cv::Scalar(0, 255, 0), 2);
        char label[96];
        std::snprintf(label, sizeof(label), "#%zu c%d %.2f",
                      i, d.class_id, d.confidence);
        cv::putText(out, label, cv::Point(std::max(0, x), std::max(18, y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 255, 255), 2);
    }
    cv::putText(out, title, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX,
                0.8, cv::Scalar(0, 180, 255), 2);
    return out;
}

void drawSelectedBbox(cv::Mat& img,
                      const Detection& detection,
                      const cv::Scalar& color) {
    const int x = static_cast<int>(
        std::round(detection.cx - detection.width * 0.5f));
    const int y = static_cast<int>(
        std::round(detection.cy - detection.height * 0.5f));
    const int w = static_cast<int>(std::round(detection.width));
    const int h = static_cast<int>(std::round(detection.height));
    cv::rectangle(img, cv::Rect(x, y, w, h) & cv::Rect(0, 0, img.cols, img.rows),
                  color, 2);
}

cv::Mat drawFeatureDebugPanel(const cv::Mat& left_base,
                              const cv::Mat& right_base,
                              const DebugFeatureMatchResult& r) {
    cv::Mat left_panel = left_base.clone();
    cv::Mat right_panel = right_base.clone();
    std::vector<cv::Mat> side_by_side{left_panel, right_panel};
    cv::Mat canvas;
    cv::hconcat(side_by_side, canvas);
    const int x_offset = left_panel.cols;

    for (const auto& kp : r.left_keypoints) {
        cv::circle(canvas, kp.pt, 4, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }
    for (const auto& kp : r.right_keypoints) {
        cv::Point2f p(kp.pt.x + static_cast<float>(x_offset), kp.pt.y);
        cv::circle(canvas, p, 4, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
    }
    for (const auto& m : r.matches) {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= static_cast<int>(r.left_keypoints.size()) ||
            m.trainIdx >= static_cast<int>(r.right_keypoints.size())) {
            continue;
        }
        const cv::Point2f p1 = r.left_keypoints[m.queryIdx].pt;
        const cv::Point2f p2(
            r.right_keypoints[m.trainIdx].pt.x + static_cast<float>(x_offset),
            r.right_keypoints[m.trainIdx].pt.y);
        cv::line(canvas, p1, p2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::circle(canvas, p1, 5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::circle(canvas, p2, 5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    char title[224];
    std::snprintf(title, sizeof(title),
                  "%s Lkp=%zu Rkp=%zu cand=%d matches=%zu disp=%.2f std=%.2f conf=%.2f",
                  r.name.c_str(), r.left_keypoints.size(), r.right_keypoints.size(),
                  r.attempted_matches, r.matches.size(), r.disparity,
                  r.stddev, r.confidence);
    cv::putText(canvas, title, cv::Point(16, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.68,
                cv::Scalar(0, 180, 255), 2);
    cv::putText(canvas, "left keypoints", cv::Point(16, 58),
                cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 0), 1);
    cv::putText(canvas, "right candidates", cv::Point(x_offset + 16, 58),
                cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 0, 255), 1);
    return canvas;
}

}  // namespace stereo3d
