/**
 * @file hybrid_depth.h
 * @brief 单目+双目混合测距 + Kalman 滤波
 *
 * 策略：
 *   - Z < 4m: 单目 (BBox 宽度 → 深度, 排球直径=0.22m)
 *   - Z > 5m: 双目 ROI SAD
 *   - 3-5m:   加权融合 (线性过渡)
 *   - Kalman 滤波: 平滑 + 速度估计 + 丢帧预测
 */

#ifndef STEREO_3D_PIPELINE_HYBRID_DEPTH_H_
#define STEREO_3D_PIPELINE_HYBRID_DEPTH_H_

#include "../pipeline/frame_slot.h"
#include "../stereo/roi_stereo_matcher.h"
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>

namespace stereo3d {

struct HybridDepthConfig {
    // 单目参数
    float object_diameter = 0.22f;     ///< 排球直径 (m)
    float bbox_scale      = 0.95f;     ///< BBox vs 实际球体比例补偿

    // 方法切换阈值
    float mono_max_z      = 5.0f;      ///< 单目最远有效距离 (m)
    float stereo_min_z    = 3.0f;      ///< 双目最近有效距离 (m)
    // 过渡带 = [stereo_min_z, mono_max_z] = [3, 5]m

    // Kalman 参数
    float dt              = 0.01f;     ///< 帧间隔 (s), 100Hz
    float process_accel   = 50.0f;     ///< 过程噪声: 最大加速度 (m/s^2)
    float R_mono          = 0.25f;     ///< 单目观测噪声方差
    float R_stereo        = 0.01f;     ///< 双目观测噪声方差

    // 跟踪管理
    int   lost_predict_frames = 5;     ///< 丢失后纯预测帧数
    int   lost_degrade_frames = 20;    ///< 降级帧数
    int   lost_delete_frames  = 40;    ///< 删除帧数
    float min_confidence      = 0.1f;  ///< 最低输出置信度

    // 深度限制
    float min_depth = 0.3f;
    float max_depth = 15.0f;
};

/**
 * @brief 单目标 Kalman 跟踪状态
 */
struct DepthTrack {
    // Kalman 状态: [z, vz]
    float z  = 0.0f;              ///< 滤波后距离 (m)
    float vz = 0.0f;              ///< 径向速度 (m/s)
    float P[2][2] = {{1,0},{0,1}};///< 协方差

    // 元数据
    int   track_id   = -1;
    int   age        = 0;         ///< 总帧数
    int   lost_count = 0;         ///< 连续丢失帧数
    float confidence = 0.0f;
    float last_raw_z = 0.0f;      ///< 上一次原始测量
    int   method     = 0;         ///< 0=mono, 1=stereo, 2=blend

    // IoU 跟踪用: 上一帧 BBox
    float last_cx = 0.0f;
    float last_cy = 0.0f;
    float last_w  = 0.0f;
    float last_h  = 0.0f;

    void predict(float dt, float sigma_a);
    void update(float z_obs, float R);
    void updateBBox(float cx, float cy, float w, float h);
};


class HybridDepthEstimator {
public:
    HybridDepthEstimator() = default;
    ~HybridDepthEstimator() = default;

    /**
     * @brief 初始化
     * @param focal    焦距 (px)
     * @param baseline 基线 (m)
     * @param cx       主点 x
     * @param cy       主点 y
     * @param config   参数
     */
    void init(float focal, float baseline, float cx, float cy,
              const HybridDepthConfig& config = HybridDepthConfig());

    /**
     * @brief 单帧混合深度估计
     *
     * @param detections  YOLO 检测结果
     * @param roi_results ROI 双目匹配结果 (可为空, 纯单目时)
     * @return 3D 定位结果 (含 Kalman 滤波)
     */
    std::vector<Object3D> estimate(
        const std::vector<Detection>& detections,
        const std::vector<Object3D>& roi_results,
        double actual_dt = 0.0);

    /**
     * @brief 处理丢帧 (无检测时调用, 纯 Kalman 预测)
     * @return 预测的 3D 结果
     */
    std::vector<Object3D> predictOnly();

    /**
     * @brief 重置所有跟踪
     */
    void reset();

    /**
     * @brief 获取活跃跟踪数
     */
    int activeTrackCount() const;

private:
    float focal_    = 0.0f;
    float baseline_ = 0.0f;
    float cx_       = 0.0f;
    float cy_       = 0.0f;
    HybridDepthConfig config_;

    // 跟踪列表 (IoU 贪心匹配)
    std::vector<DepthTrack> tracks_;
    int next_track_id_ = 0;

    // 内部方法
    float monoDepth(const Detection& det) const;
    float blendDepth(float z_mono, float z_stereo, float z_pred) const;
    float getObsNoise(float z, int method) const;
    void  pruneDeadTracks();

    // IoU 匹配
    static float computeIoU(float cx1, float cy1, float w1, float h1,
                            float cx2, float cy2, float w2, float h2);
    std::vector<int> greedyIoUMatch(
        const std::vector<Detection>& detections, float iou_threshold = 0.2f);
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_HYBRID_DEPTH_H_
