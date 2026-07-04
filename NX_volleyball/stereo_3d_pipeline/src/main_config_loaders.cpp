#include "main_config_loaders.h"

#include "main_dual_yolo_config.h"
#include "stereo/neural_feature_config.h"

#include <vpi/algo/TemporalNoiseReduction.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
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
            loadDualYoloConfig(dual, cfg);
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
        if (perf["p2_feature_job_scaffold_enabled"])
            cfg.p2_feature_job_scaffold_enabled =
                perf["p2_feature_job_scaffold_enabled"].as<bool>();
        if (perf["p2_realtime_lane_decision_enabled"])
            cfg.p2_realtime_lane_decision_enabled =
                perf["p2_realtime_lane_decision_enabled"].as<bool>();
        if (perf["p2_diagnostic_lane_decision_enabled"])
            cfg.p2_diagnostic_lane_decision_enabled =
                perf["p2_diagnostic_lane_decision_enabled"].as<bool>();
        if (perf["p2_selective_trigger"])
            cfg.p2_selective_trigger = perf["p2_selective_trigger"].as<bool>();
        if (perf["p2_trigger_on_fallback"])
            cfg.p2_trigger_on_fallback = perf["p2_trigger_on_fallback"].as<bool>();
        if (perf["p2_trigger_on_direct_pair"])
            cfg.p2_trigger_on_direct_pair = perf["p2_trigger_on_direct_pair"].as<bool>();
        if (perf["p2_trigger_on_host_gray"])
            cfg.p2_trigger_on_host_gray = perf["p2_trigger_on_host_gray"].as<bool>();
        if (perf["p2_trigger_on_bgr"])
            cfg.p2_trigger_on_bgr = perf["p2_trigger_on_bgr"].as<bool>();
        if (perf["p2_trigger_on_pair_quality"])
            cfg.p2_trigger_on_pair_quality =
                perf["p2_trigger_on_pair_quality"].as<bool>();
        if (perf["p2_trigger_on_no_valid_direct_pair"])
            cfg.p2_trigger_on_no_valid_direct_pair =
                perf["p2_trigger_on_no_valid_direct_pair"].as<bool>();
        if (perf["p2_pair_quality_min_shifted_iou"])
            cfg.p2_pair_quality_min_shifted_iou =
                perf["p2_pair_quality_min_shifted_iou"].as<float>();
        if (perf["p2_pair_quality_max_epipolar_dy"])
            cfg.p2_pair_quality_max_epipolar_dy =
                perf["p2_pair_quality_max_epipolar_dy"].as<float>();
        if (perf["p2_pair_quality_min_confidence"])
            cfg.p2_pair_quality_min_confidence =
                perf["p2_pair_quality_min_confidence"].as<float>();
        if (perf["p2_diagnostic_stride"])
            cfg.p2_diagnostic_stride = perf["p2_diagnostic_stride"].as<int>();
        if (perf["p2_diagnostic_max_in_flight"])
            cfg.p2_diagnostic_max_in_flight =
                perf["p2_diagnostic_max_in_flight"].as<int>();
        if (perf["p2_realtime_deadline_ms"])
            cfg.p2_realtime_deadline_ms = perf["p2_realtime_deadline_ms"].as<float>();
        if (perf["p2_diagnostic_deadline_ms"])
            cfg.p2_diagnostic_deadline_ms =
                perf["p2_diagnostic_deadline_ms"].as<float>();
        if (perf["p2_diagnostic_results_enabled"])
            cfg.p2_diagnostic_results_enabled =
                perf["p2_diagnostic_results_enabled"].as<bool>();
        if (perf["p2_diagnostic_results_path"])
            cfg.p2_diagnostic_results_path =
                perf["p2_diagnostic_results_path"].as<std::string>();
        if (perf["p2_diagnostic_artifacts_enabled"])
            cfg.p2_diagnostic_artifacts_enabled =
                perf["p2_diagnostic_artifacts_enabled"].as<bool>();
        if (perf["p2_diagnostic_artifacts_dir"])
            cfg.p2_diagnostic_artifacts_dir =
                perf["p2_diagnostic_artifacts_dir"].as<std::string>();
        if (perf["p2_diagnostic_artifacts_max"])
            cfg.p2_diagnostic_artifacts_max =
                perf["p2_diagnostic_artifacts_max"].as<int>();
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
