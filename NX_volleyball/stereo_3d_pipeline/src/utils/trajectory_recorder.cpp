/**
 * @file trajectory_recorder.cpp
 * @brief CSV trajectory data recorder — async queue + background writer
 */

#include "trajectory_recorder.h"
#include "logger.h"
#include <iomanip>

namespace stereo3d {

void TrajectoryRecorder::init(const TrajectoryRecorderConfig& config) {
    cfg_ = config;
    if (!cfg_.enabled) return;

    file_.open(cfg_.output_path, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        LOG_WARN("TrajectoryRecorder: failed to open %s", cfg_.output_path.c_str());
        cfg_.enabled = false;
        return;
    }

    writeHeader();
    frame_count_ = 0;
    running_ = true;
    writer_thread_ = std::thread(&TrajectoryRecorder::writerLoop, this);
    LOG_INFO("TrajectoryRecorder: recording to %s", cfg_.output_path.c_str());
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
              << "z_roi_center_patch,z_roi_multi_point,"
              << "z_yolo_bbox_pair,z_circle,z_subpixel,z_fallback,z_fallback_template,"
              << "z_fallback_feature_points,"
              << "disparity_bbox_center,disparity_bbox_left_edge,disparity_bbox_right_edge,"
              << "disparity_circle_center,disparity_circle_left_edge,disparity_circle_right_edge,"
              << "disparity_roi_edge_centroid,disparity_roi_radial_center,"
              << "disparity_roi_edge_pair_center,disparity_roi_corner_points,"
              << "disparity_roi_texture_points,disparity_roi_binary_points,"
              << "disparity_roi_orb_points,disparity_roi_brisk_points,"
              << "disparity_roi_akaze_points,"
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
              << "fallback_feature_points_support,"
              << "fallback_feature_points_std_px,fallback_feature_points_confidence,"
              << "raw_observation_valid,left_circle_source,right_circle_source,"
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

void TrajectoryRecorder::record(
    int frame_id, double timestamp,
    const std::vector<Object3D>& results,
    const std::vector<LandingPrediction>& preds) {

    if (!cfg_.enabled || !running_) return;

    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
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
            writeEntry(entry);
        }
        batch.clear();
        file_.flush();
    }
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
                  << r.fallback_feature_points_support << ","
                  << r.fallback_feature_points_std_px << ","
                  << r.fallback_feature_points_confidence << ","
                  << r.raw_observation_valid << ","
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
            LOG_INFO("TrajectoryRecorder: saved %d frames", frame_count_.load());
        }
    }
}

}  // namespace stereo3d
