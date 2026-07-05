#ifndef STEREO_3D_PIPELINE_OBJECT3D_TYPES_H_
#define STEREO_3D_PIPELINE_OBJECT3D_TYPES_H_

#include <cstdint>

namespace stereo3d {

/**
 * @brief 3D 定位结果
 */
struct Object3D {
    float x, y, z;         ///< 3D 坐标 (米)
    float vx, vy, vz;      ///< 3D 速度 (m/s)
    float ax, ay, az;       ///< 3D 加速度 (m/s²)
    float raw_x, raw_y, raw_z; ///< 未经 Kalman 的当前帧观测, raw_mode 录制用
    int raw_observation_valid; ///< 1=raw_x/raw_y/raw_z 有效
    float predicted_z;      ///< Kalman update 前的 z 预测, -1=无有效先验
    float innovation_z;     ///< z 观测创新: raw_z - predicted_z
    float innovation_norm;  ///< z 创新归一化值: innovation_z / sqrt(Pzz_prior + Rz)
    float kalman_sigma_z;   ///< Kalman update 后 z 标准差估计
    float z_mono;          ///< 单目测距 (m), 校准用
    float z_stereo;        ///< 双目测距 (m), -1=无效
    float z_bbox_center;   ///< bbox 中心视差测距, -1=无效
    float z_bbox_left_edge;///< bbox 左边缘视差测距, -1=无效
    float z_bbox_right_edge;///< bbox 右边缘视差测距, -1=无效
    float z_circle_center; ///< ROI 圆心视差测距, -1=无效
    float z_circle_left_edge; ///< ROI 圆左边缘视差测距, -1=无效
    float z_circle_right_edge;///< ROI 圆右边缘视差测距, -1=无效
    float z_roi_edge_centroid;///< ROI 边缘梯度质心视差测距, -1=无效
    float z_roi_radial_center;///< ROI 径向梯度中心视差测距, -1=无效
    float z_roi_edge_pair_center;///< ROI 左右边缘成对中心视差测距, -1=无效
    float z_roi_corner_points;///< ROI 角点特征点视差测距, -1=无效
    float z_roi_texture_points;///< ROI 纹理/梯度特征点视差测距, -1=无效
    float z_roi_binary_points;///< ROI 二值描述子特征点视差测距, -1=无效
    float z_roi_orb_points;///< ROI ORB 特征点视差测距, -1=无效
    float z_roi_brisk_points;///< ROI BRISK 特征点视差测距, -1=无效
    float z_roi_akaze_points;///< ROI AKAZE 特征点视差测距, -1=无效
    float z_roi_sift_points;///< ROI SIFT 特征点视差测距, -1=无效
    float z_roi_iou_region_color_patch;///< ROI 彩色区域 IoU/patch 视差测距, -1=无效
    float z_roi_patch_iou_color_edge;///< ROI 彩色边缘 IoU/patch 视差测距, -1=无效
    float z_roi_cuda_template_match;///< 自研 CUDA Template/NCC ROI 极线视差测距, -1=无效
    float z_roi_cuda_stereo_bm;///< OpenCV CUDA StereoBM 小 ROI 视差测距, -1=无效
    float z_roi_cuda_stereo_sgm;///< OpenCV CUDA StereoSGM 小 ROI 视差测距, -1=无效
    float z_roi_ring_edge_profile;///< CUDA ROI ring/edge profile 小范围极线视差测距, -1=无效
    float z_roi_neural_feature;///< ROI TensorRT 神经特征匹配视差测距, -1=无效
    float z_roi_center_patch; ///< ROI 中心 patch ZNCC 视差测距, -1=无效
    float z_roi_multi_point;  ///< ROI 多点 ZNCC 亚像素视差测距, -1=无效
    float z_yolo_bbox_pair; ///< 左右 YOLO bbox 中心视差测距, -1=无效
    float z_circle;        ///< 左右圆心视差测距, -1=无效
    float z_subpixel;      ///< ROI 多点亚像素视差测距, -1=无效
    float z_fallback;      ///< fallback 测距兼容汇总, -1=无效
    float z_fallback_epipolar; ///< 极线 fallback 测距, -1=无效
    float z_fallback_template; ///< 极线模板搜索 fallback 测距, -1=无效
    float z_fallback_feature_points; ///< 极线特征点 fallback 测距, -1=无效
    float disparity_bbox_center; ///< bbox 中心视差, -1=无效
    float disparity_bbox_left_edge; ///< bbox 左边缘视差, -1=无效
    float disparity_bbox_right_edge;///< bbox 右边缘视差, -1=无效
    float disparity_circle_center; ///< ROI 圆心视差, -1=无效
    float disparity_circle_left_edge; ///< ROI 圆左边缘视差, -1=无效
    float disparity_circle_right_edge;///< ROI 圆右边缘视差, -1=无效
    float disparity_roi_edge_centroid;///< ROI 边缘梯度质心视差, -1=无效
    float disparity_roi_radial_center;///< ROI 径向梯度中心视差, -1=无效
    float disparity_roi_edge_pair_center;///< ROI 左右边缘成对中心视差, -1=无效
    float disparity_roi_corner_points;///< ROI 角点特征点视差, -1=无效
    float disparity_roi_texture_points;///< ROI 纹理/梯度特征点视差, -1=无效
    float disparity_roi_binary_points;///< ROI 二值描述子特征点视差, -1=无效
    float disparity_roi_orb_points;///< ROI ORB 特征点视差, -1=无效
    float disparity_roi_brisk_points;///< ROI BRISK 特征点视差, -1=无效
    float disparity_roi_akaze_points;///< ROI AKAZE 特征点视差, -1=无效
    float disparity_roi_sift_points;///< ROI SIFT 特征点视差, -1=无效
    float disparity_roi_iou_region_color_patch;///< ROI 彩色区域 IoU/patch 视差, -1=无效
    float disparity_roi_patch_iou_color_edge;///< ROI 彩色边缘 IoU/patch 视差, -1=无效
    float disparity_roi_cuda_template_match;///< 自研 CUDA Template/NCC ROI 极线视差, -1=无效
    float disparity_roi_cuda_stereo_bm;///< OpenCV CUDA StereoBM 小 ROI 视差, -1=无效
    float disparity_roi_cuda_stereo_sgm;///< OpenCV CUDA StereoSGM 小 ROI 视差, -1=无效
    float disparity_roi_ring_edge_profile;///< CUDA ROI ring/edge profile 小范围极线视差, -1=无效
    float disparity_roi_neural_feature;///< ROI TensorRT 神经特征匹配视差, -1=无效
    float disparity_roi_center_patch; ///< ROI 中心 patch ZNCC 视差, -1=无效
    float disparity_roi_multi_point;  ///< ROI 多点 ZNCC 亚像素视差, -1=无效
    float disparity_fallback_epipolar; ///< 极线 fallback 视差, -1=无效
    float disparity_fallback_template; ///< 极线模板搜索 fallback 视差, -1=无效
    float disparity_fallback_feature_points; ///< 极线特征点 fallback 视差, -1=无效
    float disparity_yolo;  ///< 左右 YOLO bbox 中心视差, -1=无效
    float disparity_circle;///< 圆心视差, -1=无效
    float disparity_subpixel; ///< ROI 亚像素视差, -1=无效
    float left_bbox_cx, left_bbox_cy, left_bbox_w, left_bbox_h, left_bbox_conf;
    float right_bbox_cx, right_bbox_cy, right_bbox_w, right_bbox_h, right_bbox_conf;
    float left_circle_cx, left_circle_cy, left_circle_r;
    float right_circle_cx, right_circle_cy, right_circle_r;
    int left_circle_source; ///< 0=无,1=bbox代理,2=ROI圆拟合,3=极线搜索,4=模板搜索,5=特征fallback预测窗口
    int right_circle_source;///< 0=无,1=bbox代理,2=ROI圆拟合,3=极线搜索,4=模板搜索,5=特征fallback预测窗口
    float epipolar_dy;     ///< 左右圆心 y 差 (px), -1=无效
    float size_ratio;      ///< 左右圆半径比例, -1=无效
    float left_circle_conf;///< 左目圆拟合置信度
    float right_circle_conf;///< 右目圆拟合置信度
    int subpixel_valid;    ///< 1=亚像素测距被采用
    int subpixel_attempted;///< 1=尝试过亚像素测距
    int subpixel_support;  ///< 亚像素有效采样点数
    float subpixel_std_px; ///< 亚像素多点视差标准差
    float subpixel_confidence; ///< 亚像素测距置信度
    float subpixel_gate_px;///< 当前亚像素视差动态门限
    int roi_corner_points_support; ///< ROI 角点特征有效匹配点数
    float roi_corner_points_std_px; ///< ROI 角点特征视差标准差
    float roi_corner_points_confidence; ///< ROI 角点特征匹配置信度
    int roi_texture_points_support; ///< ROI 纹理特征有效匹配点数
    float roi_texture_points_std_px; ///< ROI 纹理特征视差标准差
    float roi_texture_points_confidence; ///< ROI 纹理特征匹配置信度
    int roi_binary_points_support; ///< ROI 二值描述子有效匹配点数
    float roi_binary_points_std_px; ///< ROI 二值描述子视差标准差
    float roi_binary_points_confidence; ///< ROI 二值描述子匹配置信度
    int roi_orb_points_support; ///< ROI ORB 有效匹配点数
    float roi_orb_points_std_px; ///< ROI ORB 视差标准差
    float roi_orb_points_confidence; ///< ROI ORB 匹配置信度
    int roi_brisk_points_support; ///< ROI BRISK 有效匹配点数
    float roi_brisk_points_std_px; ///< ROI BRISK 视差标准差
    float roi_brisk_points_confidence; ///< ROI BRISK 匹配置信度
    int roi_akaze_points_support; ///< ROI AKAZE 有效匹配点数
    float roi_akaze_points_std_px; ///< ROI AKAZE 视差标准差
    float roi_akaze_points_confidence; ///< ROI AKAZE 匹配置信度
    int roi_sift_points_support; ///< ROI SIFT 有效匹配点数
    float roi_sift_points_std_px; ///< ROI SIFT 视差标准差
    float roi_sift_points_confidence; ///< ROI SIFT 匹配置信度
    int roi_iou_region_color_patch_support; ///< ROI 彩色区域有效匹配点数
    float roi_iou_region_color_patch_std_px; ///< ROI 彩色区域视差标准差
    float roi_iou_region_color_patch_confidence; ///< ROI 彩色区域匹配置信度
    int roi_patch_iou_color_edge_support; ///< ROI 彩色边缘有效匹配点数
    float roi_patch_iou_color_edge_std_px; ///< ROI 彩色边缘视差标准差
    float roi_patch_iou_color_edge_confidence; ///< ROI 彩色边缘匹配置信度
    int roi_cuda_template_match_support; ///< 自研 CUDA Template/NCC 有效匹配点数
    float roi_cuda_template_match_std_px; ///< 自研 CUDA Template/NCC 视差标准差
    float roi_cuda_template_match_confidence; ///< 自研 CUDA Template/NCC 匹配置信度
    int roi_cuda_stereo_bm_support; ///< OpenCV CUDA StereoBM 有效采样点数
    float roi_cuda_stereo_bm_std_px; ///< OpenCV CUDA StereoBM 视差标准差
    float roi_cuda_stereo_bm_confidence; ///< OpenCV CUDA StereoBM 匹配置信度
    int roi_cuda_stereo_sgm_support; ///< OpenCV CUDA StereoSGM 有效采样点数
    float roi_cuda_stereo_sgm_std_px; ///< OpenCV CUDA StereoSGM 视差标准差
    float roi_cuda_stereo_sgm_confidence; ///< OpenCV CUDA StereoSGM 匹配置信度
    int roi_ring_edge_profile_support; ///< CUDA ring/edge profile 有效采样点数
    float roi_ring_edge_profile_std_px; ///< CUDA ring/edge profile 视差标准差
    float roi_ring_edge_profile_confidence; ///< CUDA ring/edge profile 匹配置信度
    int roi_neural_feature_support; ///< ROI 神经特征有效匹配点数
    float roi_neural_feature_std_px; ///< ROI 神经特征视差标准差
    float roi_neural_feature_confidence; ///< ROI 神经特征匹配置信度
    int fallback_feature_points_support; ///< fallback 特征有效匹配点数
    float fallback_feature_points_std_px; ///< fallback 特征视差标准差
    float fallback_feature_points_confidence; ///< fallback 特征匹配置信度
    float pair_initial_disparity; ///< 左右 YOLO pair 初始中心视差, -1=非直接 pair
    float pair_epipolar_dy; ///< 左右 YOLO pair bbox 中心 y 差, -1=非直接 pair
    float pair_y_tolerance; ///< 当前 pair gate y 容差, -1=非直接 pair
    float pair_size_ratio; ///< 左右 YOLO pair bbox 宽高最大比例, -1=非直接 pair
    float pair_shifted_iou; ///< 右框平移到左目坐标后的 bbox IoU, -1=非直接 pair
    float pair_score; ///< pair gate 排序分数, 已含 bbox 物理视差惩罚
    float pair_bbox_prior_penalty; ///< bbox 物理视差一致性排序惩罚
    int pair_positive_disparity; ///< 1=直接 pair 正视差且未超过 max_disparity
    int stereo_match_source; ///< 0=无,1=左右YOLO,2=左到右fallback,3=右到左fallback
    int stereo_depth_source; ///< 0=无,1=圆心/搜索,2=ROI多点,3=bbox中心,4=中心patch,5=边缘质心,6=bbox边缘,7=模板fallback,8=径向中心,9=边缘成对中心,10=角点特征,11=纹理特征,12=特征fallback,13=二值特征,14=ORB,15=BRISK,16=AKAZE,17=SIFT,18=彩色区域IoU,19=彩色边缘IoU,20=神经特征,21=CUDA模板匹配,22=CUDA StereoBM,23=CUDA StereoSGM,24=CUDA ring/edge profile
    uint64_t left_timestamp_us;  ///< 左目 SDK 时间戳原值, 海康 USB 当前实测为 ns
    uint64_t right_timestamp_us; ///< 右目 SDK 时间戳原值, 海康 USB 当前实测为 ns
    uint32_t left_frame_number;  ///< 左目 SDK 帧号
    uint32_t right_frame_number; ///< 右目 SDK 帧号
    uint32_t left_frame_counter; ///< 左目水印帧计数
    uint32_t right_frame_counter;///< 右目水印帧计数
    uint32_t left_trigger_index; ///< 左目水印外触发计数
    uint32_t right_trigger_index;///< 右目水印外触发计数
    int64_t frame_counter_delta; ///< left_frame_counter - right_frame_counter
    int64_t frame_number_delta;  ///< left_frame_number - right_frame_number
    int64_t timestamp_delta_us;  ///< 左右 SDK 时间戳差, 当前海康 USB 按 ns/1000 记录
    float confidence;      ///< 定位置信度
    int class_id;          ///< 类别 ID
    int track_id;          ///< 跟踪 ID (-1 = 未跟踪)
    int depth_method;      ///< 0=单目, 1=双目, 2=融合

    Object3D() : x(0), y(0), z(0), vx(0), vy(0), vz(0),
                 ax(0), ay(0), az(0),
                 raw_x(0), raw_y(0), raw_z(0), raw_observation_valid(0),
                 predicted_z(-1), innovation_z(0), innovation_norm(0),
                 kalman_sigma_z(-1),
                 z_mono(0), z_stereo(-1),
                 z_bbox_center(-1), z_bbox_left_edge(-1), z_bbox_right_edge(-1),
                 z_circle_center(-1), z_circle_left_edge(-1), z_circle_right_edge(-1),
                 z_roi_edge_centroid(-1), z_roi_radial_center(-1),
                 z_roi_edge_pair_center(-1), z_roi_corner_points(-1),
                 z_roi_texture_points(-1), z_roi_binary_points(-1),
                 z_roi_orb_points(-1), z_roi_brisk_points(-1),
                 z_roi_akaze_points(-1), z_roi_sift_points(-1),
                 z_roi_iou_region_color_patch(-1),
                 z_roi_patch_iou_color_edge(-1),
                 z_roi_cuda_template_match(-1),
                 z_roi_cuda_stereo_bm(-1),
                 z_roi_cuda_stereo_sgm(-1),
                 z_roi_ring_edge_profile(-1),
                 z_roi_neural_feature(-1),
                 z_roi_center_patch(-1),
                 z_roi_multi_point(-1),
                 z_yolo_bbox_pair(-1), z_circle(-1), z_subpixel(-1),
                 z_fallback(-1), z_fallback_epipolar(-1),
                 z_fallback_template(-1), z_fallback_feature_points(-1),
                 disparity_bbox_center(-1), disparity_bbox_left_edge(-1),
                 disparity_bbox_right_edge(-1), disparity_circle_center(-1),
                 disparity_circle_left_edge(-1), disparity_circle_right_edge(-1),
                 disparity_roi_edge_centroid(-1), disparity_roi_radial_center(-1),
                 disparity_roi_edge_pair_center(-1), disparity_roi_corner_points(-1),
                 disparity_roi_texture_points(-1), disparity_roi_binary_points(-1),
                 disparity_roi_orb_points(-1), disparity_roi_brisk_points(-1),
                 disparity_roi_akaze_points(-1), disparity_roi_sift_points(-1),
                 disparity_roi_iou_region_color_patch(-1),
                 disparity_roi_patch_iou_color_edge(-1),
                 disparity_roi_cuda_template_match(-1),
                 disparity_roi_cuda_stereo_bm(-1),
                 disparity_roi_cuda_stereo_sgm(-1),
                 disparity_roi_ring_edge_profile(-1),
                 disparity_roi_neural_feature(-1),
                 disparity_roi_center_patch(-1),
                 disparity_roi_multi_point(-1), disparity_fallback_epipolar(-1),
                 disparity_fallback_template(-1),
                 disparity_fallback_feature_points(-1),
                 disparity_yolo(-1), disparity_circle(-1), disparity_subpixel(-1),
                 left_bbox_cx(-1), left_bbox_cy(-1), left_bbox_w(-1),
                 left_bbox_h(-1), left_bbox_conf(0),
                 right_bbox_cx(-1), right_bbox_cy(-1), right_bbox_w(-1),
                 right_bbox_h(-1), right_bbox_conf(0),
                 left_circle_cx(-1), left_circle_cy(-1), left_circle_r(-1),
                 right_circle_cx(-1), right_circle_cy(-1), right_circle_r(-1),
                 left_circle_source(0), right_circle_source(0),
                 epipolar_dy(-1), size_ratio(-1),
                 left_circle_conf(0), right_circle_conf(0),
                 subpixel_valid(0), subpixel_attempted(0), subpixel_support(0),
                 subpixel_std_px(-1), subpixel_confidence(0), subpixel_gate_px(0),
                 roi_corner_points_support(0), roi_corner_points_std_px(-1),
                 roi_corner_points_confidence(0),
                 roi_texture_points_support(0), roi_texture_points_std_px(-1),
                 roi_texture_points_confidence(0),
                 roi_binary_points_support(0), roi_binary_points_std_px(-1),
                 roi_binary_points_confidence(0),
                 roi_orb_points_support(0), roi_orb_points_std_px(-1),
                 roi_orb_points_confidence(0),
                 roi_brisk_points_support(0), roi_brisk_points_std_px(-1),
                 roi_brisk_points_confidence(0),
                 roi_akaze_points_support(0), roi_akaze_points_std_px(-1),
                 roi_akaze_points_confidence(0),
                 roi_sift_points_support(0), roi_sift_points_std_px(-1),
                 roi_sift_points_confidence(0),
                 roi_iou_region_color_patch_support(0),
                 roi_iou_region_color_patch_std_px(-1),
                 roi_iou_region_color_patch_confidence(0),
                 roi_patch_iou_color_edge_support(0),
                 roi_patch_iou_color_edge_std_px(-1),
                 roi_patch_iou_color_edge_confidence(0),
                 roi_cuda_template_match_support(0),
                 roi_cuda_template_match_std_px(-1),
                 roi_cuda_template_match_confidence(0),
                 roi_cuda_stereo_bm_support(0),
                 roi_cuda_stereo_bm_std_px(-1),
                 roi_cuda_stereo_bm_confidence(0),
                 roi_cuda_stereo_sgm_support(0),
                 roi_cuda_stereo_sgm_std_px(-1),
                 roi_cuda_stereo_sgm_confidence(0),
                 roi_ring_edge_profile_support(0),
                 roi_ring_edge_profile_std_px(-1),
                 roi_ring_edge_profile_confidence(0),
                 roi_neural_feature_support(0), roi_neural_feature_std_px(-1),
                 roi_neural_feature_confidence(0),
                 fallback_feature_points_support(0), fallback_feature_points_std_px(-1),
                 fallback_feature_points_confidence(0),
                 pair_initial_disparity(-1), pair_epipolar_dy(-1),
                 pair_y_tolerance(-1), pair_size_ratio(-1),
                 pair_shifted_iou(-1), pair_score(-1),
                 pair_bbox_prior_penalty(-1), pair_positive_disparity(0),
                 stereo_match_source(0), stereo_depth_source(0),
                 left_timestamp_us(0), right_timestamp_us(0),
                 left_frame_number(0), right_frame_number(0),
                 left_frame_counter(0), right_frame_counter(0),
                 left_trigger_index(0), right_trigger_index(0),
                 frame_counter_delta(0), frame_number_delta(0), timestamp_delta_us(0),
                 confidence(0), class_id(0), track_id(-1), depth_method(0) {}
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_OBJECT3D_TYPES_H_
