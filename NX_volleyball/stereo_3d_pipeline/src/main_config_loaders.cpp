#include "main_config_loaders.h"

#include "stereo/neural_feature_config.h"
#include "utils/logger.h"

#include <vpi/algo/TemporalNoiseReduction.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <exception>
#include <string>

stereo3d::PipelineConfig loadConfig(const std::string& path) {
    stereo3d::PipelineConfig cfg;
    bool camera_trigger_frequency_set = false;

    YAML::Node root = YAML::LoadFile(path);

    // Camera → 直接写入内嵌 CameraConfig
    if (auto cam = root["camera"]) {
        if (cam["serial_left"])       cfg.camera.serial_left  = cam["serial_left"].as<std::string>();
        if (cam["serial_right"])      cfg.camera.serial_right = cam["serial_right"].as<std::string>();
        if (cam["left_index"])        cfg.camera.camera_index_left  = cam["left_index"].as<int>();
        if (cam["right_index"])       cfg.camera.camera_index_right = cam["right_index"].as<int>();
        if (cam["exposure_us"])       cfg.camera.exposure_us = cam["exposure_us"].as<float>();
        if (cam["gain_db"])           cfg.camera.gain_db = cam["gain_db"].as<float>();
        if (cam["auto_exposure"])     cfg.camera.auto_exposure = cam["auto_exposure"].as<bool>();
        if (cam["auto_gain"])         cfg.camera.auto_gain = cam["auto_gain"].as<bool>();
        if (cam["ae_upper_us"])       cfg.camera.ae_upper_us = cam["ae_upper_us"].as<float>();
        if (cam["ae_lower_us"])       cfg.camera.ae_lower_us = cam["ae_lower_us"].as<float>();
        if (cam["ag_upper_db"])       cfg.camera.ag_upper_db = cam["ag_upper_db"].as<float>();
        if (cam["gamma_enable"])      cfg.camera.gamma_enable = cam["gamma_enable"].as<bool>();
        if (cam["gamma_value"])       cfg.camera.gamma_value = cam["gamma_value"].as<float>();
        if (cam["use_trigger"])       cfg.camera.use_trigger = cam["use_trigger"].as<bool>();
        if (cam["trigger_source"])    cfg.camera.trigger_source = cam["trigger_source"].as<std::string>();
        if (cam["trigger_activation"]) cfg.camera.trigger_activation = cam["trigger_activation"].as<std::string>();
        if (cam["trigger_frequency_hz"]) {
            cfg.camera.trigger_frequency_hz = cam["trigger_frequency_hz"].as<int>();
            camera_trigger_frequency_set = true;
        }
        if (cam["image_node_num"]) cfg.camera.image_node_num = cam["image_node_num"].as<int>();
        if (cam["embedded_info_clear_rows"])
            cfg.camera.embedded_info_clear_rows = cam["embedded_info_clear_rows"].as<int>();
        if (cam["trigger_chip"])       cfg.trigger_chip = cam["trigger_chip"].as<std::string>();
        if (cam["trigger_line"])       cfg.trigger_line = cam["trigger_line"].as<int>();
        if (cam["width"])             cfg.camera.width  = cam["width"].as<int>();
        if (cam["height"])            cfg.camera.height = cam["height"].as<int>();
    }

    // Calibration
    if (auto cal = root["calibration"]) {
        if (cal["file"]) cfg.calibration_file = cal["file"].as<std::string>();
    }

    // Rectify (输出分辨率, 与相机原始分辨率分离)
    // Rectify (输出分辨率, 与相机原始分辨率分离)
    if (auto rect = root["rectify"]) {
        if (rect["output_width"])  cfg.rect_width  = rect["output_width"].as<int>();
        if (rect["output_height"]) cfg.rect_height = rect["output_height"].as<int>();
        if (rect["backend"]) cfg.rect_backend = rect["backend"].as<std::string>();
    }

    // VPI TNR (时域降噪)
    if (auto tnr = root["tnr"]) {
        if (tnr["enabled"]) cfg.tnr_enabled = tnr["enabled"].as<bool>();
        if (tnr["strength"]) cfg.tnr_strength = tnr["strength"].as<float>();
        if (tnr["preset"]) {
            std::string p = tnr["preset"].as<std::string>();
            if (p == "indoor_low")         cfg.tnr_preset = VPI_TNR_PRESET_INDOOR_LOW_LIGHT;
            else if (p == "indoor_high")   cfg.tnr_preset = VPI_TNR_PRESET_INDOOR_HIGH_LIGHT;
            else if (p == "outdoor_low")   cfg.tnr_preset = VPI_TNR_PRESET_OUTDOOR_LOW_LIGHT;
            else if (p == "outdoor_medium") cfg.tnr_preset = VPI_TNR_PRESET_OUTDOOR_MEDIUM_LIGHT;
            else if (p == "outdoor_high")  cfg.tnr_preset = VPI_TNR_PRESET_OUTDOOR_HIGH_LIGHT;
        }
        if (tnr["version"]) {
            std::string v = tnr["version"].as<std::string>();
            if (v == "v1")      cfg.tnr_version = VPI_TNR_V1;
            else if (v == "v2") cfg.tnr_version = VPI_TNR_V2;
            else if (v == "v3") cfg.tnr_version = VPI_TNR_V3;
            else                cfg.tnr_version = VPI_TNR_DEFAULT;
        }
    }

    // Detector
    if (auto det = root["detector"]) {
        if (det["engine_path"])            cfg.engine_file = det["engine_path"].as<std::string>();
        if (det["use_dla"])                cfg.use_dla = det["use_dla"].as<bool>();
        if (det["dla_core"])               cfg.dla_core = det["dla_core"].as<int>();
        if (det["confidence_threshold"])   cfg.conf_threshold = det["confidence_threshold"].as<float>();
        if (det["nms_threshold"])           cfg.nms_threshold = det["nms_threshold"].as<float>();
        if (det["input_size"])             cfg.input_size = det["input_size"].as<int>();
        if (det["max_detections"])         cfg.max_detections = det["max_detections"].as<int>();
        if (det["input_format"])           cfg.detector_input_format = det["input_format"].as<std::string>();
        if (auto dual = det["dual_yolo"]) {
            if (dual["enabled"]) cfg.dual_yolo.enabled = dual["enabled"].as<bool>();
            if (dual["right_engine_path"])
                cfg.dual_yolo.right_engine_file = dual["right_engine_path"].as<std::string>();
            if (dual["right_engine_file"])
                cfg.dual_yolo.right_engine_file = dual["right_engine_file"].as<std::string>();
            if (dual["right_input_format"])
                cfg.dual_yolo.right_input_format = dual["right_input_format"].as<std::string>();
            cfg.dual_yolo.right_use_dla = cfg.use_dla;
            if (dual["right_use_dla"])
                cfg.dual_yolo.right_use_dla = dual["right_use_dla"].as<bool>();
            if (dual["right_dla_core"])
                cfg.dual_yolo.right_dla_core = dual["right_dla_core"].as<int>();
            if (dual["use_for_depth"])
                cfg.dual_yolo.use_for_depth = dual["use_for_depth"].as<bool>();
            if (dual["fallback_to_roi_match"])
                cfg.dual_yolo.fallback_to_roi_match = dual["fallback_to_roi_match"].as<bool>();
            if (dual["gpu_candidate_refine"])
                cfg.dual_yolo.gpu_candidate_refine = dual["gpu_candidate_refine"].as<bool>();
            if (dual["fallback_epipolar_search"]) {
                cfg.dual_yolo.fallback_epipolar_search = dual["fallback_epipolar_search"].as<bool>();
                cfg.dual_yolo.depth_epipolar_fallback = cfg.dual_yolo.fallback_epipolar_search;
            }
            if (dual["center_refine"])
                cfg.dual_yolo.center_refine = dual["center_refine"].as<bool>();
            if (dual["roi_denoise"])
                cfg.dual_yolo.roi_denoise = dual["roi_denoise"].as<bool>();
            if (dual["log_matches"])
                cfg.dual_yolo.log_matches = dual["log_matches"].as<bool>();
            if (dual["depth_solver"])
                cfg.dual_yolo.depth_solver = dual["depth_solver"].as<std::string>();
            if (dual["subpixel_enabled"]) {
                cfg.dual_yolo.subpixel_enabled = dual["subpixel_enabled"].as<bool>();
                cfg.dual_yolo.depth_roi_subpixel = cfg.dual_yolo.subpixel_enabled;
            }
            if (auto modes = dual["depth_modes"]) {
                if (modes["bbox_pair"])
                    cfg.dual_yolo.depth_bbox_pair = modes["bbox_pair"].as<bool>();
                if (modes["bbox_edges"])
                    cfg.dual_yolo.depth_bbox_edges = modes["bbox_edges"].as<bool>();
                if (modes["circle_center"])
                    cfg.dual_yolo.depth_circle_center = modes["circle_center"].as<bool>();
                if (modes["circle_edges"])
                    cfg.dual_yolo.depth_circle_edges = modes["circle_edges"].as<bool>();
                if (modes["roi_edge_centroid"])
                    cfg.dual_yolo.depth_roi_edge_centroid = modes["roi_edge_centroid"].as<bool>();
                if (modes["roi_radial_center"])
                    cfg.dual_yolo.depth_roi_radial_center = modes["roi_radial_center"].as<bool>();
                if (modes["roi_edge_pair_center"])
                    cfg.dual_yolo.depth_roi_edge_pair_center = modes["roi_edge_pair_center"].as<bool>();
                if (modes["roi_corner_points"])
                    cfg.dual_yolo.depth_roi_corner_points = modes["roi_corner_points"].as<bool>();
                if (modes["roi_texture_points"])
                    cfg.dual_yolo.depth_roi_texture_points = modes["roi_texture_points"].as<bool>();
                if (modes["roi_binary_points"])
                    cfg.dual_yolo.depth_roi_binary_points = modes["roi_binary_points"].as<bool>();
                if (modes["roi_orb_points"])
                    cfg.dual_yolo.depth_roi_orb_points = modes["roi_orb_points"].as<bool>();
                if (modes["roi_brisk_points"])
                    cfg.dual_yolo.depth_roi_brisk_points = modes["roi_brisk_points"].as<bool>();
                if (modes["roi_akaze_points"])
                    cfg.dual_yolo.depth_roi_akaze_points = modes["roi_akaze_points"].as<bool>();
                if (modes["roi_sift_points"])
                    cfg.dual_yolo.depth_roi_sift_points = modes["roi_sift_points"].as<bool>();
                if (modes["roi_iou_region_color_patch"])
                    cfg.dual_yolo.depth_roi_iou_region_color_patch =
                        modes["roi_iou_region_color_patch"].as<bool>();
                if (modes["roi_patch_iou_color_edge"])
                    cfg.dual_yolo.depth_roi_patch_iou_color_edge =
                        modes["roi_patch_iou_color_edge"].as<bool>();
                if (modes["roi_cuda_template_match"])
                    cfg.dual_yolo.depth_roi_cuda_template_match =
                        modes["roi_cuda_template_match"].as<bool>();
                if (modes["roi_cuda_stereo_bm"])
                    cfg.dual_yolo.depth_roi_cuda_stereo_bm =
                        modes["roi_cuda_stereo_bm"].as<bool>();
                if (modes["roi_cuda_stereo_sgm"])
                    cfg.dual_yolo.depth_roi_cuda_stereo_sgm =
                        modes["roi_cuda_stereo_sgm"].as<bool>();
                if (modes["roi_center_patch"])
                    cfg.dual_yolo.depth_roi_center_patch = modes["roi_center_patch"].as<bool>();
                if (modes["roi_subpixel"])
                    cfg.dual_yolo.depth_roi_subpixel = modes["roi_subpixel"].as<bool>();
                if (modes["epipolar_fallback"])
                    cfg.dual_yolo.depth_epipolar_fallback = modes["epipolar_fallback"].as<bool>();
                if (modes["fallback_template"])
                    cfg.dual_yolo.depth_fallback_template = modes["fallback_template"].as<bool>();
                if (modes["fallback_feature_points"])
                    cfg.dual_yolo.depth_fallback_feature_points = modes["fallback_feature_points"].as<bool>();
            }
            if (dual["subpixel_patch_radius"])
                cfg.dual_yolo.subpixel_patch_radius = dual["subpixel_patch_radius"].as<int>();
            if (dual["subpixel_search_radius_px"])
                cfg.dual_yolo.subpixel_search_radius_px = dual["subpixel_search_radius_px"].as<int>();
            if (dual["subpixel_max_points"])
                cfg.dual_yolo.subpixel_max_points = dual["subpixel_max_points"].as<int>();
            if (dual["subpixel_min_points"])
                cfg.dual_yolo.subpixel_min_points = dual["subpixel_min_points"].as<int>();
            if (dual["subpixel_min_confidence"])
                cfg.dual_yolo.subpixel_min_confidence = dual["subpixel_min_confidence"].as<float>();
            if (dual["subpixel_max_disp_delta_px"])
                cfg.dual_yolo.subpixel_max_disp_delta_px = dual["subpixel_max_disp_delta_px"].as<float>();
            if (dual["subpixel_max_disp_delta_ratio"])
                cfg.dual_yolo.subpixel_max_disp_delta_ratio = dual["subpixel_max_disp_delta_ratio"].as<float>();
            if (dual["subpixel_max_depth_delta_m"])
                cfg.dual_yolo.subpixel_max_depth_delta_m = dual["subpixel_max_depth_delta_m"].as<float>();
            if (dual["subpixel_max_stddev_px"])
                cfg.dual_yolo.subpixel_max_stddev_px = dual["subpixel_max_stddev_px"].as<float>();
            if (dual["subpixel_time_budget_ms"])
                cfg.dual_yolo.subpixel_time_budget_ms = dual["subpixel_time_budget_ms"].as<float>();
            if (dual["epipolar_y_tolerance"])
                cfg.dual_yolo.epipolar_y_tolerance = dual["epipolar_y_tolerance"].as<float>();
            if (dual["feature_y_tolerance_px"])
                cfg.dual_yolo.feature_y_tolerance_px = dual["feature_y_tolerance_px"].as<float>();
            if (dual["feature_y_slope"])
                cfg.dual_yolo.feature_y_slope = dual["feature_y_slope"].as<float>();
            if (dual["feature_y_offset_px"])
                cfg.dual_yolo.feature_y_offset_px = dual["feature_y_offset_px"].as<float>();
            if (dual["feature_reverse_check_px"])
                cfg.dual_yolo.feature_reverse_check_px = dual["feature_reverse_check_px"].as<float>();
            if (dual["feature_overlap_scale"])
                cfg.dual_yolo.feature_overlap_scale = dual["feature_overlap_scale"].as<float>();
            if (dual["feature_mad_scale"])
                cfg.dual_yolo.feature_mad_scale = dual["feature_mad_scale"].as<float>();
            if (dual["feature_ransac_gate_px"])
                cfg.dual_yolo.feature_ransac_gate_px = dual["feature_ransac_gate_px"].as<float>();
            if (dual["feature_sphere_radius_scale"])
                cfg.dual_yolo.feature_sphere_radius_scale = dual["feature_sphere_radius_scale"].as<float>();
            if (dual["feature_sphere_margin_m"])
                cfg.dual_yolo.feature_sphere_margin_m = dual["feature_sphere_margin_m"].as<float>();
            if (dual["feature_normalize_large_roi"])
                cfg.dual_yolo.feature_normalize_large_roi = dual["feature_normalize_large_roi"].as<bool>();
            if (dual["feature_normalized_diameter_px"])
                cfg.dual_yolo.feature_normalized_diameter_px = dual["feature_normalized_diameter_px"].as<int>();
            if (dual["feature_normalize_min_diameter_px"])
                cfg.dual_yolo.feature_normalize_min_diameter_px = dual["feature_normalize_min_diameter_px"].as<float>();
            if (dual["feature_normalize_margin_scale"])
                cfg.dual_yolo.feature_normalize_margin_scale = dual["feature_normalize_margin_scale"].as<float>();
            if (dual["feature_precompute_roi_maps"])
                cfg.dual_yolo.feature_precompute_roi_maps = dual["feature_precompute_roi_maps"].as<bool>();
            if (dual["max_size_ratio"])
                cfg.dual_yolo.max_size_ratio = dual["max_size_ratio"].as<float>();
            if (dual["min_shifted_iou"])
                cfg.dual_yolo.min_shifted_iou = dual["min_shifted_iou"].as<float>();
            if (dual["bbox_disparity_consistency_ratio"])
                cfg.dual_yolo.bbox_disparity_consistency_ratio =
                    dual["bbox_disparity_consistency_ratio"].as<float>();
            if (dual["bbox_disparity_consistency_min_px"])
                cfg.dual_yolo.bbox_disparity_consistency_min_px =
                    dual["bbox_disparity_consistency_min_px"].as<float>();
            if (dual["bbox_disparity_penalty_scale"])
                cfg.dual_yolo.bbox_disparity_penalty_scale =
                    dual["bbox_disparity_penalty_scale"].as<float>();
            if (dual["fallback_search_margin_px"])
                cfg.dual_yolo.fallback_search_margin_px = dual["fallback_search_margin_px"].as<int>();
            if (dual["fallback_max_width_px"])
                cfg.dual_yolo.fallback_max_width_px = dual["fallback_max_width_px"].as<int>();
            if (dual["circle_max_roi_pixels"])
                cfg.dual_yolo.circle_max_roi_pixels = dual["circle_max_roi_pixels"].as<int>();
        }
    }

    // Stereo
    if (auto st = root["stereo"]) {
        if (st["max_disparity"]) cfg.max_disparity = st["max_disparity"].as<int>();
        if (st["window_size"])   cfg.window_size   = st["window_size"].as<int>();
        if (st["quality"])       cfg.stereo_quality = st["quality"].as<int>();

        // 策略选择: roi_only > use_half_resolution > full_frame
        if (st["strategy"]) {
            std::string s = st["strategy"].as<std::string>();
            if (s == "roi_only" || s == "ROI_ONLY")
                cfg.disparity_strategy = stereo3d::DisparityStrategy::ROI_ONLY;
            else if (s == "half_resolution" || s == "HALF_RESOLUTION")
                cfg.disparity_strategy = stereo3d::DisparityStrategy::HALF_RESOLUTION;
            else
                cfg.disparity_strategy = stereo3d::DisparityStrategy::FULL_FRAME;
        } else if (st["use_half_resolution"] && st["use_half_resolution"].as<bool>()) {
            cfg.disparity_strategy = stereo3d::DisparityStrategy::HALF_RESOLUTION;
        }
    }

    // Learned ROI feature matching (TensorRT runtime, disabled by default)
    if (auto nf = root["neural_feature_matching"]) {
        if (nf["enabled"]) cfg.neural_features.enabled = nf["enabled"].as<bool>();
        if (nf["backend"]) {
            cfg.neural_features.backend_name = nf["backend"].as<std::string>();
            cfg.neural_features.backend =
                stereo3d::parseNeuralFeatureBackend(cfg.neural_features.backend_name);
        }
        if (nf["extractor_engine_path"])
            cfg.neural_features.extractor_engine_path =
                nf["extractor_engine_path"].as<std::string>();
        if (nf["matcher_engine_path"])
            cfg.neural_features.matcher_engine_path =
                nf["matcher_engine_path"].as<std::string>();
        if (nf["fused_engine_path"])
            cfg.neural_features.fused_engine_path =
                nf["fused_engine_path"].as<std::string>();
        if (nf["roi_size"])
            cfg.neural_features.roi_size = nf["roi_size"].as<int>();
        if (nf["top_k"])
            cfg.neural_features.top_k = nf["top_k"].as<int>();
        if (nf["descriptor_dim"])
            cfg.neural_features.descriptor_dim = nf["descriptor_dim"].as<int>();
        if (nf["min_matches"])
            cfg.neural_features.min_matches = nf["min_matches"].as<int>();
        if (nf["max_y_error_px"])
            cfg.neural_features.max_y_error_px = nf["max_y_error_px"].as<float>();
        if (nf["max_disp_delta_px"])
            cfg.neural_features.max_disp_delta_px = nf["max_disp_delta_px"].as<float>();
        if (nf["final_disp_gate_px"])
            cfg.neural_features.final_disp_gate_px = nf["final_disp_gate_px"].as<float>();
        if (nf["min_score"])
            cfg.neural_features.min_score = nf["min_score"].as<float>();
        if (nf["use_lightglue"])
            cfg.neural_features.use_lightglue = nf["use_lightglue"].as<bool>();
    }

    // Fusion
    // Fusion → 直接写入内嵌 HybridDepthConfig
    if (auto fus = root["fusion"]) {
        if (fus["min_depth"])       cfg.depth.min_depth       = fus["min_depth"].as<float>();
        if (fus["max_depth"])       cfg.depth.max_depth       = fus["max_depth"].as<float>();
        if (fus["object_diameter"]) cfg.depth.object_diameter = fus["object_diameter"].as<float>();
        if (fus["bbox_scale"])      cfg.depth.bbox_scale      = fus["bbox_scale"].as<float>();
        if (fus["mono_max_z"])      cfg.depth.mono_max_z      = fus["mono_max_z"].as<float>();
        if (fus["stereo_min_z"])    cfg.depth.stereo_min_z    = fus["stereo_min_z"].as<float>();
        if (fus["min_confidence"])  cfg.depth.min_confidence  = fus["min_confidence"].as<float>();
        // Kalman 观测噪声 (距离自适应基值)
        if (fus["R_mono"])          cfg.depth.R_mono          = fus["R_mono"].as<float>();
        if (fus["R_stereo"])        cfg.depth.R_stereo        = fus["R_stereo"].as<float>();
        // IVW 融合权重 (与 Kalman R 分离)
        if (fus["ivw_R_mono"])      cfg.depth.ivw_R_mono      = fus["ivw_R_mono"].as<float>();
        if (fus["ivw_R_stereo"])    cfg.depth.ivw_R_stereo    = fus["ivw_R_stereo"].as<float>();
        if (fus["fallback_stereo_weight_scale"])
            cfg.depth.fallback_stereo_weight_scale =
                fus["fallback_stereo_weight_scale"].as<float>();
        if (fus["fallback_obs_noise_scale"])
            cfg.depth.fallback_obs_noise_scale =
                fus["fallback_obs_noise_scale"].as<float>();
    }

    // Performance
    if (auto perf = root["performance"]) {
        if (perf["log_interval"])   cfg.stats_interval = perf["log_interval"].as<int>();
        if (perf["pwm_frequency"])  cfg.trigger_freq_hz = static_cast<int>(perf["pwm_frequency"].as<float>());
        if (perf["drop_stale_roi_frames"])
            cfg.drop_stale_roi_frames = perf["drop_stale_roi_frames"].as<bool>();
        if (perf["async_roi_stage2"])
            cfg.async_roi_stage2 = perf["async_roi_stage2"].as<bool>();
        if (perf["async_roi_buffers"])
            cfg.async_roi_buffers = perf["async_roi_buffers"].as<int>();
        if (perf["async_roi_deadline_ms"])
            cfg.async_roi_deadline_ms = perf["async_roi_deadline_ms"].as<float>();
    }
    if (!camera_trigger_frequency_set) {
        cfg.camera.trigger_frequency_hz = cfg.trigger_freq_hz;
    }

    // SOT Tracker (YOLO 帧间填充)
    if (auto trk = root["tracker"]) {
        cfg.tracker.enabled = trk["enabled"] ? trk["enabled"].as<bool>() : false;
        if (trk["type"])            cfg.tracker.type            = trk["type"].as<std::string>();
        if (trk["engine_path"])     cfg.tracker.engine_path     = trk["engine_path"].as<std::string>();
        if (trk["search_engine_path"]) cfg.tracker.search_engine_path = trk["search_engine_path"].as<std::string>();
        if (trk["head_engine_path"]) cfg.tracker.head_engine_path = trk["head_engine_path"].as<std::string>();
        if (trk["detect_interval"]) cfg.tracker.detect_interval = trk["detect_interval"].as<int>();
        if (trk["lost_threshold"])  cfg.tracker.lost_threshold  = trk["lost_threshold"].as<int>();
        if (trk["min_confidence"])  cfg.tracker.min_confidence  = trk["min_confidence"].as<float>();
        cfg.tracker.detect_interval = std::max(1, cfg.tracker.detect_interval);
    }

    return cfg;
}

stereo3d::TrajectoryPredictorConfig loadPredictorConfig(const std::string& path) {
    stereo3d::TrajectoryPredictorConfig tcfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (auto pred = root["prediction"]) {
            if (pred["gravity"])          tcfg.gravity       = pred["gravity"].as<float>();
            if (pred["air_density"])      tcfg.air_density   = pred["air_density"].as<float>();
            if (pred["ball_mass"])        tcfg.ball_mass     = pred["ball_mass"].as<float>();
            if (pred["ball_radius"])      tcfg.ball_radius   = pred["ball_radius"].as<float>();
            if (pred["drag_coeff"])       tcfg.drag_coeff    = pred["drag_coeff"].as<float>();
            if (pred["ground_y"])         tcfg.ground_y      = pred["ground_y"].as<float>();
            if (pred["max_predict_time"]) tcfg.max_predict_time = pred["max_predict_time"].as<float>();
            if (pred["rk4_dt"])           tcfg.rk4_dt        = pred["rk4_dt"].as<float>();
            if (pred["poly_min_frames"])  tcfg.poly_min_frames = pred["poly_min_frames"].as<int>();
            if (pred["history_max"])      tcfg.history_max   = pred["history_max"].as<int>();
            if (pred["min_speed"])        tcfg.min_speed_for_predict = pred["min_speed"].as<float>();
        }
    } catch (const std::exception& e) {
        LOG_WARN("prediction config: %s, using defaults", e.what());
    } catch (...) { LOG_WARN("prediction config: unknown error, using defaults"); }
    return tcfg;
}

stereo3d::TrajectoryRecorderConfig loadRecorderConfig(const std::string& path) {
    stereo3d::TrajectoryRecorderConfig rcfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (auto rec = root["recording"]) {
            if (rec["enabled"])     rcfg.enabled     = rec["enabled"].as<bool>();
            if (rec["output_path"]) rcfg.output_path = rec["output_path"].as<std::string>();
            if (rec["raw_mode"])    rcfg.raw_mode    = rec["raw_mode"].as<bool>();
            if (rec["frame_summary_enabled"])
                rcfg.frame_summary_enabled =
                    rec["frame_summary_enabled"].as<bool>();
            if (rec["frame_summary_path"])
                rcfg.frame_summary_path =
                    rec["frame_summary_path"].as<std::string>();
            if (rec["max_queue_frames"])
                rcfg.max_queue_frames = rec["max_queue_frames"].as<size_t>();
            if (rec["detail_level"]) {
                std::string level = rec["detail_level"].as<std::string>();
                std::transform(level.begin(), level.end(), level.begin(),
                               [](unsigned char c) {
                                   return static_cast<char>(std::tolower(c));
                               });
                if (level == "legacy" || level == "basic") {
                    rcfg.detail_level = stereo3d::TrajectoryRecordDetail::LEGACY;
                } else if (level == "depth_candidates" || level == "candidates" ||
                           level == "depth") {
                    rcfg.detail_level =
                        stereo3d::TrajectoryRecordDetail::DEPTH_CANDIDATES;
                } else if (level == "extended" || level == "full") {
                    rcfg.detail_level = stereo3d::TrajectoryRecordDetail::EXTENDED;
                } else {
                    LOG_WARN("recording.detail_level=%s unknown, using legacy",
                             level.c_str());
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_WARN("recording config: %s, using defaults", e.what());
    } catch (...) { LOG_WARN("recording config: unknown error, using defaults"); }
    return rcfg;
}

stereo3d::BaselineClipRecorderConfig loadBaselineClipRecorderConfig(
    const std::string& path) {
    stereo3d::BaselineClipRecorderConfig cfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        auto rec = root["baseline_recording"];
        if (!rec) rec = root["baseline_clip_recording"];
        if (rec) {
            if (rec["enabled"]) cfg.enabled = rec["enabled"].as<bool>();
            if (rec["output_dir"]) cfg.output_dir = rec["output_dir"].as<std::string>();
            if (rec["duration_sec"]) cfg.duration_sec = rec["duration_sec"].as<double>();
            if (rec["frame_limit"]) cfg.frame_limit = rec["frame_limit"].as<int>();
            if (rec["clip_count"]) cfg.clip_count = rec["clip_count"].as<int>();
            if (rec["clip_gap_sec"]) cfg.clip_gap_sec = rec["clip_gap_sec"].as<double>();
            if (rec["clip_gap_frames"]) cfg.clip_gap_frames = rec["clip_gap_frames"].as<int>();
            if (rec["require_left_detection"])
                cfg.require_left_detection = rec["require_left_detection"].as<bool>();
            if (rec["require_right_detection"])
                cfg.require_right_detection = rec["require_right_detection"].as<bool>();
            if (rec["require_pair_gate"])
                cfg.require_pair_gate = rec["require_pair_gate"].as<bool>();
            if (rec["min_confidence"]) cfg.min_confidence = rec["min_confidence"].as<float>();
            if (rec["pair_y_tolerance_px"])
                cfg.pair_y_tolerance_px = rec["pair_y_tolerance_px"].as<float>();
            if (rec["pair_max_size_ratio"])
                cfg.pair_max_size_ratio = rec["pair_max_size_ratio"].as<float>();
            if (rec["pair_min_disparity_px"])
                cfg.pair_min_disparity_px = rec["pair_min_disparity_px"].as<float>();
            if (rec["image_format"]) cfg.image_format = rec["image_format"].as<std::string>();
            if (rec["image_mode"]) cfg.image_mode = rec["image_mode"].as<std::string>();
            if (rec["png_compression"])
                cfg.png_compression = rec["png_compression"].as<int>();
            if (rec["write_after_capture"])
                cfg.write_after_capture = rec["write_after_capture"].as<bool>();
            if (rec["stop_after_clip"])
                cfg.stop_after_clip = rec["stop_after_clip"].as<bool>();
            if (rec["max_queue_frames"])
                cfg.max_queue_frames = rec["max_queue_frames"].as<size_t>();
        }
    } catch (const std::exception& e) {
        LOG_WARN("baseline_recording config: %s, using defaults", e.what());
    } catch (...) {
        LOG_WARN("baseline_recording config: unknown error, using defaults");
    }
    return cfg;
}

RealtimeDebugDumpConfig loadRealtimeDebugDumpConfig(const std::string& path) {
    RealtimeDebugDumpConfig cfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        auto node = root["debug_realtime_dump"];
        if (!node) node = root["realtime_debug_dump"];
        if (!node) return cfg;
        if (node["enabled"]) cfg.enabled = node["enabled"].as<bool>();
        if (node["output_dir"]) cfg.output_dir = node["output_dir"].as<std::string>();
        if (node["stride"]) cfg.stride = node["stride"].as<int>();
        if (node["max_frames"]) cfg.max_frames = node["max_frames"].as<int>();
        if (node["max_queue"]) cfg.max_queue = node["max_queue"].as<int>();
        if (node["dump_fallback"]) cfg.dump_fallback = node["dump_fallback"].as<bool>();
    } catch (const std::exception& e) {
        LOG_WARN("debug_realtime_dump config: %s, using defaults", e.what());
    } catch (...) {
        LOG_WARN("debug_realtime_dump config: unknown error, using defaults");
    }
    cfg.stride = std::max(0, cfg.stride);
    cfg.max_frames = std::max(0, cfg.max_frames);
    cfg.max_queue = std::max(1, cfg.max_queue);
    return cfg;
}

#ifdef HAS_ROS2
stereo3d::Ros2BridgeConfig loadRos2Config(const std::string& path) {
    stereo3d::Ros2BridgeConfig cfg{};
    YAML::Node root = YAML::LoadFile(path);
    auto ros = root["ros2"];
    if (!ros) return cfg;

    cfg.enabled          = ros["enable"].as<bool>(false);
    cfg.world_frame_id   = ros["world_frame_id"].as<std::string>("vision_world");
    cfg.base_frame_id    = ros["base_frame_id"].as<std::string>("base_link");
    cfg.odom_topic       = ros["odom_topic"].as<std::string>("/odom");
    cfg.odom_timeout_sec = ros["odom_timeout_sec"].as<double>(0.5);

    if (auto t = ros["topics"]) {
        cfg.topic_realtime       = t["ball_realtime"].as<std::string>("/ball/realtime");
        cfg.topic_landing        = t["ball_landing"].as<std::string>("/ball/landing");
        cfg.topic_predicted_path = t["predicted_path"].as<std::string>("/ball/predicted_path");
        cfg.topic_actual_path    = t["actual_path"].as<std::string>("/ball/actual_path");
        cfg.topic_realtime_base  = t["ball_realtime_base"].as<std::string>("/ball/realtime_base");
        cfg.topic_landing_base   = t["ball_landing_base"].as<std::string>("/ball/landing_base");
    }
    if (auto v = ros["vision_to_world"]) {
        cfg.swap_xy       = v["swap_xy"].as<bool>(false);
        cfg.invert_x      = v["invert_x"].as<bool>(false);
        cfg.invert_y      = v["invert_y"].as<bool>(false);
        cfg.rotation_deg  = v["rotation_deg"].as<double>(0.0);
        cfg.translation_x = v["translation_x"].as<double>(0.0);
        cfg.translation_y = v["translation_y"].as<double>(0.0);
    }
    return cfg;
}
#endif
