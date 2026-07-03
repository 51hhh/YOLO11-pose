/**
 * @file trajectory_recorder.cpp
 * @brief CSV trajectory data recorder — async queue + background writer
 */

#include "trajectory_recorder.h"
#include "logger.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>

namespace stereo3d {

namespace {

std::string deriveFrameSummaryPath(const std::string& output_path) {
    const std::string suffix = ".csv";
    if (output_path.size() >= suffix.size() &&
        output_path.compare(output_path.size() - suffix.size(),
                            suffix.size(),
                            suffix) == 0) {
        return output_path.substr(0, output_path.size() - suffix.size()) +
               ".frames.csv";
    }
    return output_path + ".frames.csv";
}

bool ensureParentDirectory(const std::string& output_path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path parent = fs::path(output_path).parent_path();
    if (parent.empty()) {
        return true;
    }
    fs::create_directories(parent, ec);
    if (ec) {
        LOG_WARN("TrajectoryRecorder: failed to create directory %s: %s",
                 parent.string().c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

}  // namespace

void TrajectoryRecorder::init(const TrajectoryRecorderConfig& config) {
    cfg_ = config;
    if (!cfg_.enabled) return;

    ensureParentDirectory(cfg_.output_path);
    file_.open(cfg_.output_path, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        LOG_WARN("TrajectoryRecorder: failed to open %s", cfg_.output_path.c_str());
        cfg_.enabled = false;
        return;
    }

    writeHeader();
    if (cfg_.frame_summary_enabled) {
        const std::string frame_path = cfg_.frame_summary_path.empty()
            ? deriveFrameSummaryPath(cfg_.output_path)
            : cfg_.frame_summary_path;
        ensureParentDirectory(frame_path);
        frame_file_.open(frame_path, std::ios::out | std::ios::trunc);
        if (!frame_file_.is_open()) {
            LOG_WARN("TrajectoryRecorder: failed to open frame summary %s",
                     frame_path.c_str());
        } else {
            writeFrameSummaryHeader();
            LOG_INFO("TrajectoryRecorder: frame summary to %s",
                     frame_path.c_str());
        }
    }
    frame_count_ = 0;
    dropped_frame_count_ = 0;
    running_ = true;
    writer_thread_ = std::thread(&TrajectoryRecorder::writerLoop, this);
    LOG_INFO("TrajectoryRecorder: recording to %s (max_queue_frames=%zu)",
             cfg_.output_path.c_str(), cfg_.max_queue_frames);
}

void TrajectoryRecorder::writeHeader() {
    file_ << "frame_id,timestamp,track_id,"
          << "x,y,z,vx,vy,vz,ax,ay,az,"
          << "z_mono,z_stereo,depth_method,"
          << "confidence,";
    if (cfg_.recordDepthCandidates()) {
        file_ << "class_id,"
              << "z_bbox_center,z_bbox_left_edge,z_bbox_right_edge,"
              << "z_circle_center,z_circle_left_edge,z_circle_right_edge,"
              << "z_roi_edge_centroid,z_roi_radial_center,z_roi_edge_pair_center,"
              << "z_roi_corner_points,z_roi_texture_points,z_roi_binary_points,"
              << "z_roi_orb_points,z_roi_brisk_points,z_roi_akaze_points,"
              << "z_roi_sift_points,"
              << "z_roi_iou_region_color_patch,z_roi_patch_iou_color_edge,"
              << "z_roi_neural_feature,"
              << "z_roi_center_patch,z_roi_multi_point,"
              << "z_yolo_bbox_pair,z_circle,z_subpixel,z_fallback,z_fallback_template,"
              << "z_fallback_feature_points,"
              << "disparity_bbox_center,disparity_bbox_left_edge,disparity_bbox_right_edge,"
              << "disparity_circle_center,disparity_circle_left_edge,disparity_circle_right_edge,"
              << "disparity_roi_edge_centroid,disparity_roi_radial_center,"
              << "disparity_roi_edge_pair_center,disparity_roi_corner_points,"
              << "disparity_roi_texture_points,disparity_roi_binary_points,"
              << "disparity_roi_orb_points,disparity_roi_brisk_points,"
              << "disparity_roi_akaze_points,disparity_roi_sift_points,"
              << "disparity_roi_iou_region_color_patch,"
              << "disparity_roi_patch_iou_color_edge,"
              << "disparity_roi_neural_feature,"
              << "disparity_roi_center_patch,"
              << "disparity_roi_multi_point,disparity_fallback_template,"
              << "disparity_fallback_feature_points,"
              << "disparity_yolo,disparity_circle,disparity_subpixel,"
              << "epipolar_dy,size_ratio,left_circle_conf,right_circle_conf,"
              << "subpixel_valid,subpixel_attempted,subpixel_support,"
              << "subpixel_std_px,subpixel_confidence,subpixel_gate_px,"
              << "roi_corner_points_support,roi_corner_points_std_px,"
              << "roi_corner_points_confidence,roi_texture_points_support,"
              << "roi_texture_points_std_px,roi_texture_points_confidence,"
              << "roi_binary_points_support,roi_binary_points_std_px,"
              << "roi_binary_points_confidence,roi_orb_points_support,"
              << "roi_orb_points_std_px,roi_orb_points_confidence,"
              << "roi_brisk_points_support,roi_brisk_points_std_px,"
              << "roi_brisk_points_confidence,roi_akaze_points_support,"
              << "roi_akaze_points_std_px,roi_akaze_points_confidence,"
              << "roi_sift_points_support,roi_sift_points_std_px,"
              << "roi_sift_points_confidence,"
              << "roi_iou_region_color_patch_support,"
              << "roi_iou_region_color_patch_std_px,"
              << "roi_iou_region_color_patch_confidence,"
              << "roi_patch_iou_color_edge_support,"
              << "roi_patch_iou_color_edge_std_px,"
              << "roi_patch_iou_color_edge_confidence,"
              << "roi_neural_feature_support,roi_neural_feature_std_px,"
              << "roi_neural_feature_confidence,"
              << "fallback_feature_points_support,"
              << "fallback_feature_points_std_px,fallback_feature_points_confidence,"
              << "pair_initial_disparity,pair_epipolar_dy,pair_y_tolerance,"
              << "pair_size_ratio,pair_shifted_iou,pair_score,"
              << "pair_bbox_prior_penalty,pair_positive_disparity,"
              << "raw_observation_valid,predicted_z,innovation_z,"
              << "innovation_norm,kalman_sigma_z,"
              << "left_circle_source,right_circle_source,"
              << "stereo_match_source,stereo_depth_source,";
    }
    if (cfg_.recordExtendedGeometry()) {
        file_ << "left_timestamp_ns,right_timestamp_ns,"
              << "left_frame_number,right_frame_number,"
              << "left_frame_counter,right_frame_counter,"
              << "left_trigger_index,right_trigger_index,"
              << "frame_counter_delta,frame_number_delta,timestamp_delta_us,"
              << "left_bbox_cx,left_bbox_cy,left_bbox_w,left_bbox_h,left_bbox_conf,"
              << "right_bbox_cx,right_bbox_cy,right_bbox_w,right_bbox_h,right_bbox_conf,"
              << "left_circle_cx,left_circle_cy,left_circle_r,"
              << "right_circle_cx,right_circle_cy,right_circle_r,";
    }
    file_ << "landing_x,landing_y,landing_t\n";
    header_written_ = true;
}

void TrajectoryRecorder::writeFrameSummaryHeader() {
    if (!frame_file_.is_open()) return;
    frame_file_
        << "frame_id,timestamp,result_count,tracked_count,raw_observation_count,"
        << "stereo_observation_count,direct_pair_count,fallback_l2r_count,"
        << "fallback_r2l_count,pair_positive_count,pair_shifted_iou_min,"
        << "pair_shifted_iou_mean,pair_score_mean,pair_bbox_prior_penalty_mean,"
        << "pair_epipolar_dy_max,roi_iou_region_color_patch_support_max,"
        << "roi_patch_iou_color_edge_support_max,roi_neural_feature_support_max,"
        << "best_confidence\n";
}

void TrajectoryRecorder::record(
    int frame_id, double timestamp,
    const std::vector<Object3D>& results,
    const std::vector<LandingPrediction>& preds) {

    if (!cfg_.enabled || !running_) return;

    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        if (cfg_.max_queue_frames > 0 &&
            queue_.size() >= cfg_.max_queue_frames) {
            const int dropped = ++dropped_frame_count_;
            if (dropped <= 3 || dropped % 100 == 0) {
                LOG_WARN("TrajectoryRecorder: queue full, dropping frame=%d dropped=%d",
                         frame_id, dropped);
            }
            return;
        }
        queue_.push_back({frame_id, timestamp, results, preds});
    }
    queue_cv_.notify_one();
    frame_count_++;
}

void TrajectoryRecorder::writerLoop() {
    std::deque<RecordEntry> batch;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            queue_cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
            if (!running_ && queue_.empty()) break;
            batch.swap(queue_);
        }
        for (const auto& entry : batch) {
            writeFrameSummary(entry);
            writeEntry(entry);
        }
        batch.clear();
        file_.flush();
        if (frame_file_.is_open()) {
            frame_file_.flush();
        }
    }
}

void TrajectoryRecorder::writeFrameSummary(const RecordEntry& entry) {
    if (!frame_file_.is_open()) return;

    int tracked_count = 0;
    int raw_count = 0;
    int stereo_count = 0;
    int direct_count = 0;
    int fallback_l2r_count = 0;
    int fallback_r2l_count = 0;
    int pair_positive_count = 0;
    int pair_count = 0;
    int pair_score_count = 0;
    int pair_penalty_count = 0;
    float pair_iou_min = -1.0f;
    double pair_iou_sum = 0.0;
    double pair_score_sum = 0.0;
    double pair_penalty_sum = 0.0;
    float pair_epipolar_dy_max = -1.0f;
    int iou_color_support_max = 0;
    int iou_edge_support_max = 0;
    int neural_support_max = 0;
    float best_confidence = 0.0f;

    for (const auto& r : entry.results) {
        if (r.track_id >= 0) ++tracked_count;
        if (r.raw_observation_valid) ++raw_count;
        if (r.z_stereo > 0.0f) ++stereo_count;
        if (r.stereo_match_source == 1) ++direct_count;
        if (r.stereo_match_source == 2) ++fallback_l2r_count;
        if (r.stereo_match_source == 3) ++fallback_r2l_count;
        if (r.pair_positive_disparity) ++pair_positive_count;
        if (r.pair_shifted_iou >= 0.0f) {
            ++pair_count;
            pair_iou_min = pair_iou_min < 0.0f
                ? r.pair_shifted_iou
                : std::min(pair_iou_min, r.pair_shifted_iou);
            pair_iou_sum += r.pair_shifted_iou;
            if (std::isfinite(r.pair_score)) {
                pair_score_sum += r.pair_score;
                ++pair_score_count;
            }
            if (std::isfinite(r.pair_bbox_prior_penalty)) {
                pair_penalty_sum += r.pair_bbox_prior_penalty;
                ++pair_penalty_count;
            }
            if (r.pair_epipolar_dy >= 0.0f) {
                pair_epipolar_dy_max =
                    std::max(pair_epipolar_dy_max, r.pair_epipolar_dy);
            }
        }
        iou_color_support_max = std::max(
            iou_color_support_max, r.roi_iou_region_color_patch_support);
        iou_edge_support_max = std::max(
            iou_edge_support_max, r.roi_patch_iou_color_edge_support);
        neural_support_max = std::max(
            neural_support_max, r.roi_neural_feature_support);
        best_confidence = std::max(best_confidence, r.confidence);
    }

    const double pair_iou_mean =
        pair_count > 0 ? pair_iou_sum / static_cast<double>(pair_count) : -1.0;
    const double pair_score_mean =
        pair_score_count > 0
            ? pair_score_sum / static_cast<double>(pair_score_count)
            : -1.0;
    const double pair_penalty_mean =
        pair_penalty_count > 0
            ? pair_penalty_sum / static_cast<double>(pair_penalty_count)
            : -1.0;

    frame_file_ << entry.frame_id << ","
                << std::fixed << std::setprecision(6) << entry.timestamp << ","
                << entry.results.size() << ","
                << tracked_count << ","
                << raw_count << ","
                << stereo_count << ","
                << direct_count << ","
                << fallback_l2r_count << ","
                << fallback_r2l_count << ","
                << pair_positive_count << ","
                << std::setprecision(4)
                << pair_iou_min << ","
                << pair_iou_mean << ","
                << pair_score_mean << ","
                << pair_penalty_mean << ","
                << pair_epipolar_dy_max << ","
                << iou_color_support_max << ","
                << iou_edge_support_max << ","
                << neural_support_max << ","
                << best_confidence << "\n";
}

void TrajectoryRecorder::writeEntry(const RecordEntry& entry) {
    for (size_t i = 0; i < entry.results.size(); ++i) {
        const auto& r = entry.results[i];
        if (r.track_id < 0) continue;
        if (cfg_.raw_mode && !r.raw_observation_valid) continue;

        const bool use_raw = cfg_.raw_mode && r.raw_observation_valid;
        const float out_x = use_raw ? r.raw_x : r.x;
        const float out_y = use_raw ? r.raw_y : r.y;
        const float out_z = use_raw ? r.raw_z : r.z;
        const float out_vx = use_raw ? 0.0f : r.vx;
        const float out_vy = use_raw ? 0.0f : r.vy;
        const float out_vz = use_raw ? 0.0f : r.vz;
        const float out_ax = use_raw ? 0.0f : r.ax;
        const float out_ay = use_raw ? 0.0f : r.ay;
        const float out_az = use_raw ? 0.0f : r.az;

        file_ << entry.frame_id << ","
              << std::fixed << std::setprecision(6) << entry.timestamp << ","
              << r.track_id << ","
              << std::setprecision(4)
              << out_x << "," << out_y << "," << out_z << ","
              << out_vx << "," << out_vy << "," << out_vz << ","
              << out_ax << "," << out_ay << "," << out_az << ","
              << r.z_mono << "," << r.z_stereo << "," << r.depth_method << ","
              << r.confidence << ",";
        if (cfg_.recordDepthCandidates()) {
            file_ << r.class_id << ","
                  << r.z_bbox_center << "," << r.z_bbox_left_edge << ","
                  << r.z_bbox_right_edge << ","
                  << r.z_circle_center << "," << r.z_circle_left_edge << ","
                  << r.z_circle_right_edge << ","
                  << r.z_roi_edge_centroid << "," << r.z_roi_radial_center << ","
                  << r.z_roi_edge_pair_center << ","
                  << r.z_roi_corner_points << "," << r.z_roi_texture_points << ","
                  << r.z_roi_binary_points << ","
                  << r.z_roi_orb_points << "," << r.z_roi_brisk_points << ","
                  << r.z_roi_akaze_points << ","
                  << r.z_roi_sift_points << ","
                  << r.z_roi_iou_region_color_patch << ","
                  << r.z_roi_patch_iou_color_edge << ","
                  << r.z_roi_neural_feature << ","
                  << r.z_roi_center_patch << ","
                  << r.z_roi_multi_point << ","
                  << r.z_yolo_bbox_pair << "," << r.z_circle << ","
                  << r.z_subpixel << "," << r.z_fallback << ","
                  << r.z_fallback_template << ","
                  << r.z_fallback_feature_points << ","
                  << r.disparity_bbox_center << ","
                  << r.disparity_bbox_left_edge << ","
                  << r.disparity_bbox_right_edge << ","
                  << r.disparity_circle_center << ","
                  << r.disparity_circle_left_edge << ","
                  << r.disparity_circle_right_edge << ","
                  << r.disparity_roi_edge_centroid << ","
                  << r.disparity_roi_radial_center << ","
                  << r.disparity_roi_edge_pair_center << ","
                  << r.disparity_roi_corner_points << ","
                  << r.disparity_roi_texture_points << ","
                  << r.disparity_roi_binary_points << ","
                  << r.disparity_roi_orb_points << ","
                  << r.disparity_roi_brisk_points << ","
                  << r.disparity_roi_akaze_points << ","
                  << r.disparity_roi_sift_points << ","
                  << r.disparity_roi_iou_region_color_patch << ","
                  << r.disparity_roi_patch_iou_color_edge << ","
                  << r.disparity_roi_neural_feature << ","
                  << r.disparity_roi_center_patch << ","
                  << r.disparity_roi_multi_point << ","
                  << r.disparity_fallback_template << ","
                  << r.disparity_fallback_feature_points << ","
                  << r.disparity_yolo << "," << r.disparity_circle << ","
                  << r.disparity_subpixel << ","
                  << r.epipolar_dy << "," << r.size_ratio << ","
                  << r.left_circle_conf << "," << r.right_circle_conf << ","
                  << r.subpixel_valid << "," << r.subpixel_attempted << ","
                  << r.subpixel_support << ","
                  << r.subpixel_std_px << "," << r.subpixel_confidence << ","
                  << r.subpixel_gate_px << ","
                  << r.roi_corner_points_support << ","
                  << r.roi_corner_points_std_px << ","
                  << r.roi_corner_points_confidence << ","
                  << r.roi_texture_points_support << ","
                  << r.roi_texture_points_std_px << ","
                  << r.roi_texture_points_confidence << ","
                  << r.roi_binary_points_support << ","
                  << r.roi_binary_points_std_px << ","
                  << r.roi_binary_points_confidence << ","
                  << r.roi_orb_points_support << ","
                  << r.roi_orb_points_std_px << ","
                  << r.roi_orb_points_confidence << ","
                  << r.roi_brisk_points_support << ","
                  << r.roi_brisk_points_std_px << ","
                  << r.roi_brisk_points_confidence << ","
                  << r.roi_akaze_points_support << ","
                  << r.roi_akaze_points_std_px << ","
                  << r.roi_akaze_points_confidence << ","
                  << r.roi_sift_points_support << ","
                  << r.roi_sift_points_std_px << ","
                  << r.roi_sift_points_confidence << ","
                  << r.roi_iou_region_color_patch_support << ","
                  << r.roi_iou_region_color_patch_std_px << ","
                  << r.roi_iou_region_color_patch_confidence << ","
                  << r.roi_patch_iou_color_edge_support << ","
                  << r.roi_patch_iou_color_edge_std_px << ","
                  << r.roi_patch_iou_color_edge_confidence << ","
                  << r.roi_neural_feature_support << ","
                  << r.roi_neural_feature_std_px << ","
                  << r.roi_neural_feature_confidence << ","
                  << r.fallback_feature_points_support << ","
                  << r.fallback_feature_points_std_px << ","
                  << r.fallback_feature_points_confidence << ","
                  << r.pair_initial_disparity << ","
                  << r.pair_epipolar_dy << ","
                  << r.pair_y_tolerance << ","
                  << r.pair_size_ratio << ","
                  << r.pair_shifted_iou << ","
                  << r.pair_score << ","
                  << r.pair_bbox_prior_penalty << ","
                  << r.pair_positive_disparity << ","
                  << r.raw_observation_valid << ","
                  << r.predicted_z << ","
                  << r.innovation_z << ","
                  << r.innovation_norm << ","
                  << r.kalman_sigma_z << ","
                  << r.left_circle_source << "," << r.right_circle_source << ","
                  << r.stereo_match_source << "," << r.stereo_depth_source << ",";
        }
        if (cfg_.recordExtendedGeometry()) {
            file_ << r.left_timestamp_us << "," << r.right_timestamp_us << ","
                  << r.left_frame_number << "," << r.right_frame_number << ","
                  << r.left_frame_counter << "," << r.right_frame_counter << ","
                  << r.left_trigger_index << "," << r.right_trigger_index << ","
                  << r.frame_counter_delta << "," << r.frame_number_delta << ","
                  << r.timestamp_delta_us << ","
                  << r.left_bbox_cx << "," << r.left_bbox_cy << ","
                  << r.left_bbox_w << "," << r.left_bbox_h << ","
                  << r.left_bbox_conf << ","
                  << r.right_bbox_cx << "," << r.right_bbox_cy << ","
                  << r.right_bbox_w << "," << r.right_bbox_h << ","
                  << r.right_bbox_conf << ","
                  << r.left_circle_cx << "," << r.left_circle_cy << ","
                  << r.left_circle_r << ","
                  << r.right_circle_cx << "," << r.right_circle_cy << ","
                  << r.right_circle_r << ",";
        }

        if (!cfg_.raw_mode && i < entry.preds.size() && entry.preds[i].valid) {
            file_ << entry.preds[i].x << "," << entry.preds[i].y << ","
                  << entry.preds[i].time_to_land;
        } else {
            file_ << "0,0,0";
        }
        file_ << "\n";
    }
}

void TrajectoryRecorder::close() {
    if (running_) {
        running_ = false;
        queue_cv_.notify_one();
        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
    }
    if (file_.is_open()) {
        file_.flush();
        file_.close();
        if (frame_count_.load() > 0) {
            LOG_INFO("TrajectoryRecorder: saved %d frames (dropped=%d)",
                     frame_count_.load(), dropped_frame_count_.load());
        }
    }
    if (frame_file_.is_open()) {
        frame_file_.flush();
        frame_file_.close();
    }
}

}  // namespace stereo3d
