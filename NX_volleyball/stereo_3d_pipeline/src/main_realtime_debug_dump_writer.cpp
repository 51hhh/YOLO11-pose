#include "main_realtime_debug_dump.h"

#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

bool validBox(float cx, float cy, float w, float h) {
    return std::isfinite(cx) && std::isfinite(cy) &&
           std::isfinite(w) && std::isfinite(h) &&
           w > 1.0f && h > 1.0f;
}

cv::Rect cropAround(float cx, float cy, float w, float h,
                    const cv::Size& size) {
    const float scale = 1.8f;
    const int crop_w = std::max(32, static_cast<int>(std::round(w * scale)));
    const int crop_h = std::max(32, static_cast<int>(std::round(h * scale)));
    const int x = static_cast<int>(std::round(cx - crop_w * 0.5f));
    const int y = static_cast<int>(std::round(cy - crop_h * 0.5f));
    return cv::Rect(x, y, crop_w, crop_h) &
           cv::Rect(0, 0, size.width, size.height);
}

void drawBox(cv::Mat& image, const cv::Rect& crop,
             float cx, float cy, float w, float h,
             const cv::Scalar& color) {
    if (!validBox(cx, cy, w, h)) return;
    cv::Rect box(
        static_cast<int>(std::round(cx - w * 0.5f)) - crop.x,
        static_cast<int>(std::round(cy - h * 0.5f)) - crop.y,
        static_cast<int>(std::round(w)),
        static_cast<int>(std::round(h)));
    cv::rectangle(image, box & cv::Rect(0, 0, image.cols, image.rows),
                  color, 1, cv::LINE_AA);
}

void drawCircle(cv::Mat& image, const cv::Rect& crop,
                float cx, float cy, float r,
                const cv::Scalar& color) {
    if (!std::isfinite(cx) || !std::isfinite(cy) ||
        !std::isfinite(r) || r <= 1.0f) {
        return;
    }
    cv::circle(image,
               cv::Point(static_cast<int>(std::round(cx)) - crop.x,
                         static_cast<int>(std::round(cy)) - crop.y),
               static_cast<int>(std::round(r)),
               color, 1, cv::LINE_AA);
}

std::string fmt(float value, int digits = 3) {
    if (!std::isfinite(value)) return "nan";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(digits) << value;
    return ss.str();
}

void putLine(cv::Mat& image, const std::string& text, int line) {
    cv::putText(image, text, cv::Point(8, 18 + line * 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
}

cv::Mat makePanel(const RealtimeDebugDumpJob& job,
                  const stereo3d::Object3D& obj,
                  int index) {
    cv::Mat left_bgr;
    cv::Mat right_bgr;
    cv::cvtColor(job.left_gray, left_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(job.right_gray, right_bgr, cv::COLOR_GRAY2BGR);

    cv::Rect left_crop = validBox(obj.left_bbox_cx, obj.left_bbox_cy,
                                  obj.left_bbox_w, obj.left_bbox_h)
        ? cropAround(obj.left_bbox_cx, obj.left_bbox_cy,
                     obj.left_bbox_w, obj.left_bbox_h, left_bgr.size())
        : cv::Rect(0, 0, left_bgr.cols, left_bgr.rows);
    cv::Rect right_crop = validBox(obj.right_bbox_cx, obj.right_bbox_cy,
                                   obj.right_bbox_w, obj.right_bbox_h)
        ? cropAround(obj.right_bbox_cx, obj.right_bbox_cy,
                     obj.right_bbox_w, obj.right_bbox_h, right_bgr.size())
        : (left_crop & cv::Rect(0, 0, right_bgr.cols, right_bgr.rows));
    if (left_crop.empty()) left_crop = cv::Rect(0, 0, left_bgr.cols, left_bgr.rows);
    if (right_crop.empty()) right_crop = cv::Rect(0, 0, right_bgr.cols, right_bgr.rows);

    cv::Mat left = left_bgr(left_crop).clone();
    cv::Mat right = right_bgr(right_crop).clone();
    drawBox(left, left_crop, obj.left_bbox_cx, obj.left_bbox_cy,
            obj.left_bbox_w, obj.left_bbox_h, cv::Scalar(0, 255, 0));
    drawBox(right, right_crop, obj.right_bbox_cx, obj.right_bbox_cy,
            obj.right_bbox_w, obj.right_bbox_h, cv::Scalar(0, 255, 0));
    drawCircle(left, left_crop, obj.left_circle_cx, obj.left_circle_cy,
               obj.left_circle_r, cv::Scalar(255, 255, 0));
    drawCircle(right, right_crop, obj.right_circle_cx, obj.right_circle_cy,
               obj.right_circle_r, cv::Scalar(255, 0, 255));

    const int target_h = 220;
    const double left_scale = static_cast<double>(target_h) /
                              static_cast<double>(std::max(1, left.rows));
    const double right_scale = static_cast<double>(target_h) /
                               static_cast<double>(std::max(1, right.rows));
    cv::resize(left, left,
               cv::Size(std::max(1, static_cast<int>(left.cols * left_scale)),
                        target_h));
    cv::resize(right, right,
               cv::Size(std::max(1, static_cast<int>(right.cols * right_scale)),
                        target_h));
    if (left.rows != right.rows) {
        cv::resize(right, right, cv::Size(right.cols, left.rows));
    }

    cv::Mat panel;
    cv::hconcat(std::vector<cv::Mat>{left, right}, panel);
    putLine(panel, "frame=" + std::to_string(job.frame_id) +
            " obj=" + std::to_string(index) +
            " match=" + std::to_string(obj.stereo_match_source) +
            " depth=" + std::to_string(obj.stereo_depth_source), 0);
    putLine(panel, "bbox=" + fmt(obj.z_bbox_center) +
            " circle=" + fmt(obj.z_circle_center) +
            " edge=" + fmt(obj.z_roi_edge_centroid) +
            " radial=" + fmt(obj.z_roi_radial_center), 1);
    putLine(panel, "edgepair=" + fmt(obj.z_roi_edge_pair_center) +
            " multi=" + fmt(obj.z_roi_multi_point) +
            " patch=" + fmt(obj.z_roi_center_patch) +
            " tmpl=" + fmt(obj.z_roi_cuda_template_match), 2);
    putLine(panel, "cuda_bm=" + fmt(obj.z_roi_cuda_stereo_bm) +
            " cuda_sgm=" + fmt(obj.z_roi_cuda_stereo_sgm) +
            " fb_epi=" + fmt(obj.z_fallback_epipolar) +
            " fb=" + fmt(obj.z_fallback), 3);
    putLine(panel,
            " pair_iou=" + fmt(obj.pair_shifted_iou) +
            " Lsrc=" + std::to_string(obj.left_circle_source) +
            " Rsrc=" + std::to_string(obj.right_circle_source), 4);
    putLine(panel, "dy=" + fmt(obj.epipolar_dy, 2), 5);
    return panel;
}

cv::Mat makeDetectionPanel(const RealtimeDebugDumpJob& job) {
    cv::Mat left_bgr;
    cv::Mat right_bgr;
    cv::cvtColor(job.left_gray, left_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(job.right_gray, right_bgr, cv::COLOR_GRAY2BGR);
    auto draw_dets = [](cv::Mat& image,
                        const std::vector<stereo3d::Detection>& detections) {
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& d = detections[i];
            cv::Rect box(
                static_cast<int>(std::round(d.cx - d.width * 0.5f)),
                static_cast<int>(std::round(d.cy - d.height * 0.5f)),
                static_cast<int>(std::round(d.width)),
                static_cast<int>(std::round(d.height)));
            cv::rectangle(image, box & cv::Rect(0, 0, image.cols, image.rows),
                          cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::putText(image, "#" + std::to_string(i),
                        cv::Point(std::max(0, box.x), std::max(18, box.y - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }
    };
    draw_dets(left_bgr, job.left_detections);
    draw_dets(right_bgr, job.right_detections);
    cv::Mat panel;
    cv::hconcat(std::vector<cv::Mat>{left_bgr, right_bgr}, panel);
    putLine(panel, "frame=" + std::to_string(job.frame_id) +
            " detections L=" + std::to_string(job.left_detections.size()) +
            " R=" + std::to_string(job.right_detections.size()), 0);
    return panel;
}

void writeSummaryJson(const RealtimeDebugDumpJob& job,
                      const std::filesystem::path& path) {
    std::ofstream out(path.string());
    if (!out.is_open()) return;
    out << "{\n";
    out << "  \"frame_id\": " << job.frame_id << ",\n";
    out << "  \"fps\": " << job.fps << ",\n";
    out << "  \"left_count\": " << job.left_detections.size() << ",\n";
    out << "  \"right_count\": " << job.right_detections.size() << ",\n";
    out << "  \"result_count\": " << job.results.size() << ",\n";
    out << "  \"frame_counter_delta\": "
        << (static_cast<int64_t>(job.metadata.left_frame_counter) -
            static_cast<int64_t>(job.metadata.right_frame_counter)) << ",\n";
    out << "  \"frame_number_delta\": "
        << (static_cast<int64_t>(job.metadata.left_frame_number) -
            static_cast<int64_t>(job.metadata.right_frame_number)) << ",\n";
    out << "  \"results\": [\n";
    for (size_t i = 0; i < job.results.size(); ++i) {
        const auto& obj = job.results[i];
        out << "    {";
        out << "\"index\": " << i
            << ", \"match_source\": " << obj.stereo_match_source
            << ", \"depth_source\": " << obj.stereo_depth_source
            << ", \"left_circle_source\": " << obj.left_circle_source
            << ", \"right_circle_source\": " << obj.right_circle_source
            << ", \"z_bbox_center\": " << obj.z_bbox_center
            << ", \"z_circle_center\": " << obj.z_circle_center
            << ", \"z_roi_edge_centroid\": " << obj.z_roi_edge_centroid
            << ", \"z_roi_radial_center\": " << obj.z_roi_radial_center
            << ", \"z_roi_edge_pair_center\": " << obj.z_roi_edge_pair_center
            << ", \"z_roi_multi_point\": " << obj.z_roi_multi_point
            << ", \"z_roi_center_patch\": " << obj.z_roi_center_patch
            << ", \"z_roi_cuda_template_match\": " << obj.z_roi_cuda_template_match
            << ", \"z_roi_cuda_stereo_bm\": " << obj.z_roi_cuda_stereo_bm
            << ", \"z_roi_cuda_stereo_sgm\": " << obj.z_roi_cuda_stereo_sgm
            << ", \"z_fallback_epipolar\": " << obj.z_fallback_epipolar
            << ", \"z_fallback\": " << obj.z_fallback
            << ", \"pair_shifted_iou\": " << obj.pair_shifted_iou
            << "}";
        if (i + 1 < job.results.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

}  // namespace

void RealtimeDebugDumper::writeJob(const RealtimeDebugDumpJob& job) {
    namespace fs = std::filesystem;
    std::ostringstream prefix;
    prefix << "frame_" << std::setw(6) << std::setfill('0') << job.frame_id;
    const fs::path root(cfg_.output_dir);

    cv::Mat panel;
    if (!job.results.empty()) {
        std::vector<cv::Mat> panels;
        for (size_t i = 0; i < job.results.size(); ++i) {
            panels.push_back(makePanel(job, job.results[i], static_cast<int>(i)));
        }
        cv::vconcat(panels, panel);
    } else {
        panel = makeDetectionPanel(job);
    }
    cv::imwrite((root / (prefix.str() + "_zoom.png")).string(), panel);
    writeSummaryJson(job, root / (prefix.str() + "_summary.json"));
    ++saved_count_;
}

void RealtimeDebugDumper::writerLoop() {
    while (true) {
        RealtimeDebugDumpJob job;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
            if (!running_ && queue_.empty()) break;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        try {
            writeJob(job);
        } catch (const cv::Exception& e) {
            LOG_WARN("RealtimeDebugDumper: write failed frame=%d: %s",
                     job.frame_id, e.what());
        }
    }
}
