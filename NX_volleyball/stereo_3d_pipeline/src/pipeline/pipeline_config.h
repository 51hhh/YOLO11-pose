/**
 * @file pipeline_config.h
 * @brief Pipeline public runtime configuration.
 */

#ifndef STEREO_3D_PIPELINE_PIPELINE_CONFIG_H_
#define STEREO_3D_PIPELINE_PIPELINE_CONFIG_H_

#include "../capture/hikvision_camera.h"
#include "../fusion/hybrid_depth.h"
#include "../stereo/neural_feature_config.h"

#include <vpi/algo/TemporalNoiseReduction.h>

#include <string>

namespace stereo3d {

/**
 * @brief 视差策略
 */
enum class DisparityStrategy {
    FULL_FRAME,        ///< 全帧视差 (默认, 1280x720)
    HALF_RESOLUTION,   ///< 半分辨率 (640x360)
    ROI_ONLY           ///< 仅计算 ROI 区域
};

#ifdef HAS_ROS2
struct Ros2BridgeConfig {
    bool enabled;
    std::string world_frame_id;
    std::string base_frame_id;
    std::string odom_topic;
    double odom_timeout_sec;
    std::string topic_realtime;
    std::string topic_landing;
    std::string topic_predicted_path;
    std::string topic_actual_path;
    std::string topic_realtime_base;
    std::string topic_landing_base;
    bool swap_xy;
    bool invert_x;
    bool invert_y;
    double rotation_deg;
    double translation_x;
    double translation_y;
};
#endif

/**
 * @brief Pipeline 运行时参数
 */
struct PipelineConfig {
    // 相机配置 (内嵌, 避免重复定义)
    CameraConfig camera;

    // 校正后分辨率
    int rect_width  = 1280;
    int rect_height = 720;
    std::string rect_backend = "VIC"; ///< 校正后端: "VIC" (推荐,不占GPU) 或 "CUDA"

    // PWM 触发器 (Pipeline 级, 非相机级)
    std::string trigger_chip = "gpiochip2";  ///< GPIO 芯片名 (Orin NX: gpiochip2)
    int trigger_line = 7;                     ///< GPIO 线路号 (Orin NX: line 7)
    int trigger_freq_hz = 100;

    // 标定
    std::string calibration_file = "config/intrinsics.yaml";

    // 检测
    std::string engine_file = "models/yolov8n_int8.engine";
    int  input_size      = 320;    ///< 模型输入尺寸
    float conf_threshold = 0.5f;
    float nms_threshold  = 0.4f;
    int  max_detections  = 10;
    std::string detector_input_format = "gray"; ///< gray|bayer|bgr
    bool use_dla = false;          ///< 使用 NVDLA (否则 GPU)
    int  dla_core = 0;             ///< DLA 核心 ID (0 或 1)

    // 双路 YOLO + 极线语义匹配测试
    struct DualYoloConfig {
        bool enabled = false;              ///< 是否同时运行右目 YOLO
        std::string right_engine_file;     ///< 右目 engine, 为空则复用左目 engine_path
        std::string right_input_format;    ///< 右目输入格式, 为空则复用 detector_input_format
        bool right_use_dla = false;        ///< 右目是否使用 DLA
        int right_dla_core = 1;            ///< 右目 DLA core, 仅用于日志/engine 匹配
        bool use_for_depth = true;         ///< 用左右 YOLO 语义匹配结果作为 stereo 观测
        bool depth_bbox_pair = true;       ///< 计算左右 YOLO bbox 中心视差候选
        bool depth_bbox_edges = false;     ///< 计算左右 YOLO bbox 左/右边缘视差候选
        bool depth_circle_center = true;   ///< 计算 ROI 圆拟合圆心三角测距候选, 需 center_refine
        bool depth_circle_edges = false;   ///< 计算 ROI 圆左右边缘视差候选
        bool depth_roi_edge_centroid = false; ///< 计算 ROI 边缘梯度质心视差候选
        bool depth_roi_radial_center = false; ///< 计算 ROI 径向梯度中心视差候选
        bool depth_roi_edge_pair_center = false; ///< 计算 ROI 左右边缘成对中心视差候选
        bool depth_roi_corner_points = false; ///< 计算 ROI 角点特征点视差候选
        bool depth_roi_texture_points = false; ///< 计算 ROI 纹理/梯度特征点视差候选
        bool depth_roi_binary_points = false; ///< 计算 ROI 二值描述子特征点视差候选
        bool depth_roi_orb_points = false; ///< 计算 ROI ORB 特征点视差候选
        bool depth_roi_brisk_points = false; ///< 计算 ROI BRISK 特征点视差候选
        bool depth_roi_akaze_points = false; ///< 计算 ROI AKAZE 特征点视差候选
        bool depth_roi_sift_points = false; ///< 计算 ROI OpenCV CPU SIFT 特征点视差候选
        bool depth_roi_iou_region_color_patch = false; ///< 计算 ROI 彩色区域 IoU/patch 视差候选
        bool depth_roi_patch_iou_color_edge = false; ///< 计算 ROI 彩色边缘 IoU/patch 视差候选
        bool depth_roi_cuda_template_match = false; ///< 计算 OpenCV CUDA TemplateMatching 小 ROI 极线视差候选
        bool depth_roi_cuda_stereo_bm = false; ///< 计算 OpenCV CUDA StereoBM 小 ROI dense 视差候选
        bool depth_roi_cuda_stereo_sgm = false; ///< 计算 OpenCV CUDA StereoSGM 小 ROI dense 视差候选
        bool depth_roi_ring_edge_profile = false; ///< 计算 CUDA ROI ring/edge profile 小范围极线视差候选
        bool depth_roi_vpi_template_match = false; ///< 计算 VPI CUDA TemplateMatching 小 ROI 极线视差候选
        bool depth_roi_vpi_stereo_disparity = false; ///< 计算 VPI CUDA StereoDisparity 小 ROI 视差候选
        bool depth_roi_vpi_harris_lk = false; ///< 计算 VPI Harris + Pyramidal LK 小 ROI 视差候选
        bool depth_roi_vpi_orb = false; ///< 计算 VPI ORB 小 ROI 视差候选
        bool depth_roi_cuda_gftt_lk = false; ///< 计算 OpenCV CUDA GFTT/Harris + SparsePyrLK 视差候选
        bool depth_roi_cuda_sift = false; ///< 计算第三方 CUDA-SIFT ROI 视差候选
        bool depth_roi_libsgm = false; ///< 计算 Fixstars libSGM ROI 视差候选
        bool depth_roi_cuda_hough_circle = false; ///< 计算 CUDA Canny/HoughCircles 圆心 refinement 视差候选
        bool depth_roi_center_patch = false; ///< 计算 ROI 中心 patch ZNCC 视差候选
        bool depth_roi_subpixel = true;    ///< 计算 ROI 多点亚像素视差候选
        bool depth_epipolar_fallback = true; ///< 单目漏检时启用有界极线搜索候选
        bool depth_fallback_template = false; ///< 单目漏检时启用极线模板搜索候选
        bool depth_fallback_feature_points = false; ///< 单目漏检时启用极线特征点搜索候选
        bool fallback_to_roi_match = true; ///< 旧 ROI 纹理匹配回退, 双 YOLO 测试默认关闭
        bool fallback_epipolar_search = true; ///< 单目漏检时在另一目极线附近做有界搜索
        bool gpu_candidate_refine = true;  ///< 双 YOLO 直接匹配候选优先使用 GPU 批处理
        bool center_refine = true;         ///< 在 bbox/搜索 ROI 内做圆心拟合细化
        bool roi_denoise = true;           ///< 圆心拟合前做局部 3x3 读数降噪
        bool log_matches = true;           ///< 按 stats_interval 打印匹配统计
        std::string depth_solver = "circle_center"; ///< circle_center|roi_subpixel_match
        bool subpixel_enabled = true;      ///< roi_subpixel_match: 启用 ROI 多点亚像素视差细化
        int subpixel_patch_radius = 5;     ///< ZNCC 匹配块半径 (patch=2r+1)
        int subpixel_search_radius_px = 8; ///< 以圆心视差为中心的左右搜索半宽
        int subpixel_max_points = 12;      ///< ROI 内最多采样点数
        int subpixel_min_points = 4;       ///< 接受亚像素视差的最少有效点
        float subpixel_min_confidence = 0.25f; ///< 接受亚像素视差的最低置信度
        float subpixel_max_disp_delta_px = 2.0f; ///< 相对圆心视差最大允许绝对偏差
        float subpixel_max_disp_delta_ratio = 0.03f; ///< 相对圆心视差最大比例偏差
        float subpixel_max_depth_delta_m = 0.5f; ///< 亚像素视差相对圆心测距最大跳变
        float subpixel_max_stddev_px = 1.0f; ///< 多点视差最大标准差
        float subpixel_time_budget_ms = 1.5f; ///< 每帧亚像素 CPU 预算, 0 表示不限制
        float epipolar_y_tolerance = 12.0f;///< 左右中心 y 允许差值 (px)
        float feature_y_tolerance_px = 2.0f; ///< ROI 特征点严格 y 残差门限
        float feature_y_slope = 0.0f;        ///< 可选 epipolar y 线性补偿斜率
        float feature_y_offset_px = 0.0f;    ///< 可选 epipolar y 固定补偿
        float feature_reverse_check_px = 1.0f; ///< 左右反查最大误差
        float feature_overlap_scale = 0.55f; ///< 左右 bbox 投影 overlap 椭圆尺度
        float feature_mad_scale = 2.5f;      ///< MAD 离群剔除倍数
        float feature_ransac_gate_px = 0.75f; ///< 1D disparity RANSAC 最小门限
        float feature_sphere_radius_scale = 1.8f; ///< 三角点离球心最大半径倍数
        float feature_sphere_margin_m = 0.02f; ///< 物理半径门限冗余
        bool feature_normalize_large_roi = true; ///< 大 ROI 降采样到固定球直径后匹配
        int feature_normalized_diameter_px = 96; ///< 大 ROI 归一化目标球直径
        float feature_normalize_min_diameter_px = 128.0f; ///< 超过该直径才归一化
        float feature_normalize_margin_scale = 0.62f; ///< 归一化 ROI 半径相对 bbox 尺寸
        bool feature_precompute_roi_maps = true; ///< ROI 内预计算 label/edge 供匹配使用
        float max_size_ratio = 2.0f;       ///< 左右 bbox 尺寸比例上限
        float min_shifted_iou = 0.05f;     ///< 平移到同一 x 后左右 bbox 最小重叠
        float bbox_disparity_consistency_ratio = 0.30f; ///< bbox 物理视差一致性相对门限
        float bbox_disparity_consistency_min_px = 45.0f; ///< bbox 物理视差一致性最小门限
        float bbox_disparity_penalty_scale = 0.75f; ///< 多候选排序的物理视差不一致惩罚
        int fallback_search_margin_px = 48;///< 期望视差两侧搜索半宽 (px)
        int fallback_max_width_px = 220;   ///< 极线 fallback 搜索窗口最大宽度
        int circle_max_roi_pixels = 18000; ///< CPU 圆拟合最大采样像素数, 超过后步进采样
    } dual_yolo;

    NeuralFeatureConfig neural_features;

    // SOT Tracker 补帧 (YOLO 检测间隙帧)
    struct TrackerConfig {
        bool enabled = false;              ///< 是否启用 SOT 补帧
        std::string type = "nanotrack";    ///< "nanotrack" | "mixformer"
        std::string engine_path;           ///< TRT 引擎路径 (backbone / template backbone)
        std::string search_engine_path;    ///< NanoTrack search backbone 引擎 (双backbone模式)
        std::string head_engine_path;      ///< NanoTrack head 引擎路径
        int detect_interval = 3;           ///< YOLO 检测间隔 (每N帧一次)
        int lost_threshold = 5;            ///< 连续无输出帧数 -> LOST
        float min_confidence = 0.3f;       ///< tracker 最低置信度
        bool gpu_sync_mode = false;        ///< NX模式: 强制GPU同步防止YOLO/tracker并行(省电) vs 异步管线(服务器)
    } tracker;

    // 视差
    int max_disparity = 128;
    int window_size   = 5;
    int stereo_quality = 6;
    DisparityStrategy disparity_strategy = DisparityStrategy::FULL_FRAME;

    // 深度 (内嵌 HybridDepthConfig, 避免重复)
    HybridDepthConfig depth;

    // 性能
    int stats_interval = 100;      ///< 每 N 帧打印统计
    bool detection_only = false;   ///< 仅运行采集/校正/检测/回调, 跳过 stereo/depth
    bool drop_stale_roi_frames = false; ///< ROI 模式下若后一帧 YOLO 已完成则跳过旧帧后处理
    bool async_roi_stage2 = false; ///< ROI_ONLY 检测帧异步执行 IoU/ROI 特征/候选深度
    int async_roi_buffers = 3;     ///< 异步 ROI 图像快照缓冲数 (running + pending + free)
    float async_roi_deadline_ms = 10.0f; ///< 期望在下一帧 YOLO ready 前完成的软预算
    bool p2_feature_job_scaffold_enabled = false; ///< 预留: P2 独立 FeatureJob 骨架开关
    bool p2_realtime_lane_decision_enabled = true; ///< P2 realtime lane 决策观测开关
    bool p2_diagnostic_lane_decision_enabled = false; ///< P2 diagnostic lane 决策观测开关
    bool p2_selective_trigger = false; ///< 仅在 fallback/质量异常等条件触发 P2
    bool p2_trigger_on_fallback = true; ///< selective 模式下单侧漏检触发 P2
    bool p2_trigger_on_direct_pair = false; ///< selective 模式下直接 pair 也触发 P2
    bool p2_trigger_on_host_gray = false; ///< selective 模式下 CPU/host gray 需求触发 P2
    bool p2_trigger_on_bgr = false; ///< selective 模式下 BGR/color 需求触发 P2
    bool p2_trigger_on_pair_quality = false; ///< selective 模式下 direct pair 质量差触发 P2
    bool p2_trigger_on_no_valid_direct_pair = false; ///< selective 模式下左右检测存在但无有效 direct pair 时触发 P2
    float p2_pair_quality_min_shifted_iou = 0.0f; ///< >0 时低于该 shifted IoU 触发 P2
    float p2_pair_quality_max_epipolar_dy = 0.0f; ///< >0 时超过该 y 偏差触发 P2
    float p2_pair_quality_min_confidence = 0.0f; ///< >0 时低于该 direct pair 语义置信度触发 P2
    int p2_diagnostic_stride = 10; ///< diagnostic lane 每 N 帧尝试一次
    int p2_diagnostic_max_in_flight = 1; ///< diagnostic lane 最大在途 job 数
    float p2_realtime_deadline_ms = 10.0f; ///< P2 realtime lane deadline
    float p2_diagnostic_deadline_ms = 50.0f; ///< P2 diagnostic lane 软 deadline
    bool p2_diagnostic_results_enabled = false; ///< diagnostic lane 迟到结果独立落盘开关
    std::string p2_diagnostic_results_path; ///< diagnostic 结果 CSV; 空则不写
    bool p2_diagnostic_point_debug_enabled = false; ///< diagnostic/inline 特征点级 debug CSV
    std::string p2_diagnostic_point_debug_path; ///< 点级 debug CSV; 空则由 results_path 派生
    bool p2_diagnostic_artifacts_enabled = false; ///< diagnostic lane 输出点对/采样点 PNG
    std::string p2_diagnostic_artifacts_dir; ///< diagnostic PNG 输出目录; 空则从 CSV 路径派生
    int p2_diagnostic_artifacts_max = 20; ///< 最多输出多少张 diagnostic artifact

    // VPI TNR (时域降噪)
    bool tnr_enabled = false;              ///< 是否启用 VPI TNR
    VPITNRPreset tnr_preset = VPI_TNR_PRESET_OUTDOOR_MEDIUM_LIGHT;
    float tnr_strength = 0.6f;             ///< 降噪强度 0.0~1.0
    VPITNRVersion tnr_version = VPI_TNR_DEFAULT;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PIPELINE_CONFIG_H_
