#include "pipeline.h"
#include "pipeline_debug_utils.h"
#include "pipeline_depth_modes.h"
#include "../stereo/depth_match_contract.h"
#include "../stereo/roi_feature_match_cpu.h"
#include "../utils/logger.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>

namespace stereo3d {

bool Pipeline::debugFeatureMatchesOnce(const std::string& output_dir) {
#ifndef HIK_CAMERA_ENABLED
    (void)output_dir;
    LOG_ERROR("Feature match debug capture requires Hikvision camera support");
    return false;
#else
    if (running_.load()) {
        LOG_ERROR("Feature match debug capture must run before Pipeline::start()");
        return false;
    }
    if (!camera_) {
        LOG_ERROR("Feature match debug capture requires initialized cameras");
        return false;
    }
    if (!dualYoloEnabled()) {
        LOG_ERROR("Feature match debug capture requires dual_yolo.enabled=true");
        return false;
    }
    if (!calibration_) {
        LOG_ERROR("Feature match debug capture requires stereo calibration");
        return false;
    }

    namespace fs = std::filesystem;
    const fs::path output_path(output_dir);
    if (output_path.empty()) {
        LOG_ERROR("Feature match debug output dir is empty");
        return false;
    }
    std::error_code ec;
    fs::create_directories(output_path, ec);
    if (ec) {
        LOG_ERROR("Failed to create debug output dir %s: %s",
                  output_dir.c_str(), ec.message().c_str());
        return false;
    }
    ec.clear();
    if (!fs::exists(output_path, ec) || ec) {
        LOG_ERROR("Feature match debug output dir does not exist: %s",
                  output_dir.c_str());
        return false;
    }
    ec.clear();
    if (!fs::is_directory(output_path, ec) || ec) {
        LOG_ERROR("Feature match debug output path is not a directory: %s",
                  output_dir.c_str());
        return false;
    }

    bool output_ok = true;
    auto write_debug_image = [&](const std::string& filename,
                                 const cv::Mat& image) {
        output_ok = writeDebugImage(output_path, filename, image) && output_ok;
    };

    bool grabbing_started = false;
    bool pwm_started = false;
    auto cleanup = [&]() {
        streams_.syncAll();
        if (pwm_started && pwm_trigger_) pwm_trigger_->stop();
        if (grabbing_started && camera_) camera_->stopGrabbing();
    };

    if (!camera_->startGrabbing()) {
        LOG_ERROR("Feature match debug: failed to start camera grabbing");
        return false;
    }
    grabbing_started = true;
    if (pwm_trigger_) {
        pwm_started = pwm_trigger_->start();
        if (!pwm_started) {
            LOG_WARN("Feature match debug: PWM trigger did not start; external trigger may be active");
        }
    }

    FrameSlot& slot = slots_[0];
    bool captured = false;
    constexpr int kMaxWarmupFrames = 12;
    for (int attempt = 0; attempt < kMaxWarmupFrames; ++attempt) {
        slot.reset();
        slot.frame_id = attempt;
        slot.is_detect_frame = true;

        stage0_grab_and_rectify(slot, false);
        VPIStatus vst = vpiStreamSync(streams_.vpiStreamPVA);
        if (vst != VPI_SUCCESS) {
            LOG_ERROR("Feature match debug: rectification sync failed: %d", (int)vst);
            cleanup();
            return false;
        }
        if (!slot.grab_failed) {
            captured = true;
            break;
        }
        LOG_WARN("Feature match debug: warmup/sync frame %d skipped", attempt);
    }
    if (!captured) {
        LOG_ERROR("Feature match debug: grab/rectify failed after %d attempts",
                  kMaxWarmupFrames);
        cleanup();
        return false;
    }
    cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);

    stage1_detect(slot, 0);
    recordDetectDoneEvents(slot);
    cudaStreamSynchronize(getDLAStream(slot.frame_id));
    if (slot.right_detection_submitted) {
        cudaStreamSynchronize(getRightDLAStream(slot.frame_id));
    }

    collectRoiDetections(slot, 0);

    cv::Mat left_gray;
    cv::Mat right_gray;
    if (!lockGrayVpiImageCopy(slot.rectGray_vpiL, left_gray) ||
        !lockGrayVpiImageCopy(slot.rectGray_vpiR, right_gray)) {
        LOG_ERROR("Feature match debug: failed to copy rectified gray images");
        cleanup();
        return false;
    }
    cv::Mat left_viz;
    cv::Mat right_viz;
    if (colorPipelineEnabled() &&
        lockBgrVpiImageCopy(slot.rectBGR_vpiL, left_viz) &&
        lockBgrVpiImageCopy(slot.rectBGR_vpiR, right_viz)) {
        write_debug_image("left_rect_bgr.png", left_viz);
        write_debug_image("right_rect_bgr.png", right_viz);
    } else {
        cv::cvtColor(left_gray, left_viz, cv::COLOR_GRAY2BGR);
        cv::cvtColor(right_gray, right_viz, cv::COLOR_GRAY2BGR);
    }

    write_debug_image("left_rect_gray.png", left_gray);
    write_debug_image("right_rect_gray.png", right_gray);
    write_debug_image("left_detections.png",
                      drawDetectionOverlay(left_viz, slot.detections, "left"));
    write_debug_image("right_detections.png",
                      drawDetectionOverlay(right_viz, slot.detections_right, "right"));

    StereoRoiPair debug_pair;
    if (!findBestStereoRoiPair(slot.detections,
                               slot.detections_right,
                               makeStereoRoiPairGateConfig(config_),
                               &debug_pair)) {
        LOG_ERROR("Feature match debug: no valid left/right YOLO pair (left=%zu right=%zu)",
                  slot.detections.size(), slot.detections_right.size());
        cleanup();
        return false;
    }

    const Detection& left_det = slot.detections[debug_pair.left_index];
    const Detection& right_det = slot.detections_right[debug_pair.right_index];
    const float initial_disp = debug_pair.initial_disparity;
    const auto& P1 = calibration_->getProjectionLeft();
    const float focal = static_cast<float>(P1.at<double>(0, 0));
    const float baseline = calibration_->getBaseline();
    ROIFeatureMatchConfig feature_cfg =
        makeROIFeatureMatchConfig(config_.dual_yolo, config_.depth);
    feature_cfg.disparity_zero_offset = activeDisparityOffset();

    std::vector<DebugFeatureMatchResult> results;
    results.push_back(makeDebugSparseFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        SparseFeatureMode::CORNER));
    results.push_back(makeDebugSparseFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        SparseFeatureMode::TEXTURE));
    results.push_back(makeDebugSparseFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        SparseFeatureMode::BINARY));
    results.push_back(makeDebugOpenCVFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        OpenCVFeatureMode::ORB));
    results.push_back(makeDebugOpenCVFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        OpenCVFeatureMode::BRISK));
    results.push_back(makeDebugOpenCVFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        OpenCVFeatureMode::AKAZE));
    results.push_back(makeDebugOpenCVFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        OpenCVFeatureMode::SIFT));

    cv::Mat left_color = left_viz.clone();
    cv::Mat right_color = right_viz.clone();
    drawSelectedBbox(left_color, left_det, cv::Scalar(0, 255, 0));
    drawSelectedBbox(right_color, right_det, cv::Scalar(0, 255, 0));
    cv::Mat selected_pair;
    {
        std::vector<cv::Mat> side_by_side{left_color, right_color};
        cv::hconcat(side_by_side, selected_pair);
    }
    write_debug_image("selected_pair.png", selected_pair);

    std::vector<cv::Mat> panels;
    std::ofstream summary((output_path / "summary.txt").string());
    if (!summary.is_open()) {
        LOG_ERROR("Feature match debug: failed to open %s",
                  (output_path / "summary.txt").string().c_str());
        output_ok = false;
    } else {
        summary << "left_count=" << slot.detections.size()
                << " right_count=" << slot.detections_right.size() << "\n";
        summary << "selected_left=" << debug_pair.left_index
                << " selected_right=" << debug_pair.right_index
                << " initial_disp=" << initial_disp
                << " dy=" << debug_pair.epipolar_dy
                << " shifted_iou=" << debug_pair.shifted_bbox_iou
                << " pair_score=" << debug_pair.score
                << " baseline_m=" << baseline
                << " focal_px=" << focal << "\n";
        summary << "frame_counter_delta="
                << (static_cast<int64_t>(slot.left_frame_counter) -
                    static_cast<int64_t>(slot.right_frame_counter))
                << " frame_number_delta="
                << (static_cast<int64_t>(slot.left_frame_number) -
                    static_cast<int64_t>(slot.right_frame_number))
                << " trigger_delta="
                << (static_cast<int64_t>(slot.left_trigger_index) -
                    static_cast<int64_t>(slot.right_trigger_index))
                << "\n";
    }

    for (const auto& r : results) {
        cv::Mat canvas = drawFeatureDebugPanel(left_color, right_color, r);
        write_debug_image(r.name + "_matches.png", canvas);
        panels.push_back(canvas);
        if (summary.is_open()) {
            summary << r.name
                    << " left_keypoints=" << r.left_keypoints.size()
                    << " right_keypoints=" << r.right_keypoints.size()
                    << " candidates=" << r.attempted_matches
                    << " matches=" << r.matches.size()
                    << " disparity=" << r.disparity
                    << " std=" << r.stddev
                    << " confidence=" << r.confidence << "\n";
        }
    }

    if (!panels.empty()) {
        const int target_w = panels.front().cols;
        std::vector<cv::Mat> resized;
        resized.reserve(panels.size());
        for (const auto& p : panels) {
            if (p.cols == target_w) {
                resized.push_back(p);
            } else {
                cv::Mat tmp;
                const double scale = static_cast<double>(target_w) /
                                     static_cast<double>(std::max(1, p.cols));
                cv::resize(p, tmp, cv::Size(target_w,
                    std::max(1, static_cast<int>(std::round(p.rows * scale)))));
                resized.push_back(tmp);
            }
        }
        cv::Mat contact;
        cv::vconcat(resized, contact);
        write_debug_image("feature_match_contact_sheet.png", contact);
    }

    if (summary.is_open()) {
        summary.flush();
        if (!summary.good()) {
            LOG_ERROR("Feature match debug: failed while writing %s",
                      (output_path / "summary.txt").string().c_str());
            output_ok = false;
        }
    }
    if (!output_ok) {
        cleanup();
        return false;
    }

    LOG_INFO("Feature match debug saved to %s", output_dir.c_str());
    cleanup();
    return true;
#endif
}

}  // namespace stereo3d
