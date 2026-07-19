/**
 * @file hybrid_depth.h
 * @brief 单目+双目混合测距 + Kalman 滤波
 *
 * 策略：
 *   - Z < 3m: 单目 (BBox 宽度 → 深度, 排球直径=0.23m)
 *   - Z > 5m: 双目 Circle-Fit (Sobel + Kåsa 圆拟合)
 *   - 3-5m:   加权融合 (线性过渡)
 *   - 9D Kalman 滤波: [x,y,z,vx,vy,vz,ax,ay,az] 恒加速模型
 */

#ifndef STEREO_3D_PIPELINE_HYBRID_DEPTH_H_
#define STEREO_3D_PIPELINE_HYBRID_DEPTH_H_

#include "../pipeline/detection_types.h"
#include "../pipeline/object3d_types.h"
#include <vector>

namespace stereo3d {

struct HybridDepthConfig {
    // 单目参数
    float object_diameter = 0.215f;    ///< 排球直径 (m), YAML 可覆盖
    float bbox_scale      = 0.95f;     ///< BBox vs 实际球体比例补偿

    // 方法切换阈值
    float mono_max_z      = 5.0f;      ///< 单目最远有效距离 (m)
    float stereo_min_z    = 3.0f;      ///< 双目最近有效距离 (m)
    // 过渡带 = [stereo_min_z, mono_max_z] = [3, 5]m

    // Kalman 参数
    float dt              = 0.01f;     ///< 帧间隔 (s), 100Hz
    float process_accel   = 50.0f;     ///< 过程噪声: 最大加速度 (m/s^2)
    float R_mono          = 0.003f;    ///< 单目观测噪声基值 R(z)=R_mono*z², 实测σ/z≈3-5%
    float R_stereo        = 0.020f;    ///< 双目观测噪声基值 R(z)=R_stereo*z², σ≈单目2.5倍(方差≈6倍)

    // 跟踪管理
    int   lost_predict_frames = 5;     ///< 丢失后纯预测帧数
    int   lost_degrade_frames = 20;    ///< 降级帧数
    int   lost_delete_frames  = 40;    ///< 删除帧数
    int   max_tracks          = 64;    ///< 轨迹数量硬上限，防止误检导致状态无限增长
    float min_confidence      = 0.1f;  ///< 最低输出置信度
    float track_confidence_alpha = 0.25f; ///< 检测置信度 EMA 更新系数

    // 2D/3D 关联与异常观测门控
    float match_iou_threshold = 0.10f;
    float match_center_gate   = 2.5f;  ///< 预测中心距离，以平均 bbox 尺寸归一化
    float innovation_gate_sigma = 6.0f; ///< 单轴归一化创新门限；<=0 表示关闭
    int   innovation_gate_min_age = 3; ///< 新轨迹先积累速度，再启用创新门控

    // 深度限制
    float min_depth = 0.3f;
    float max_depth = 15.0f;

    // 自适应偏差校正 (EMA)。准确双目标定下应关闭；仅用于明确做过
    // 单目/双目尺度联合标定的历史配置。
    bool  stereo_bias_correction_enabled = false;
    float stereo_bias_initial = 1.0f;
    float stereo_bias_alpha = 0.05f;  ///< EMA 平滑因子 (加快偏差收敛)

    // IVW 融合权重 (与 Kalman R 分离)
    float ivw_R_mono   = 0.004f;   ///< IVW 单目噪声方差 (σ≈0.063m)
    float ivw_R_stereo = 0.025f;   ///< IVW 双目噪声方差 (实测方差≈单目6倍)

    // fallback 深度只作为低权重在线观测, 避免单侧漏检误匹配强拉轨迹。
    float fallback_stereo_weight_scale = 0.35f; ///< IVW 中 fallback stereo 权重缩放
    float fallback_obs_noise_scale = 4.0f;      ///< Kalman 中 fallback Rz/Rxy 放大倍数
};

/**
 * @brief 单目标 Kalman 跟踪状态 (9维恒加速模型)
 *
 * 状态向量: [x, y, z, vx, vy, vz, ax, ay, az]
 * 观测向量: [x, y, z] (3D位置, 从像素反投影+混合深度得到)
 */
struct DepthTrack {
    static constexpr int N = 9;  ///< 状态维度
    static constexpr int M = 3;  ///< 观测维度

    // Kalman 状态: [x, y, z, vx, vy, vz, ax, ay, az]
    float state[N] = {};
    float P[N][N]  = {};          ///< 协方差 9×9

    // 快捷访问
    float& x()  { return state[0]; }
    float& y()  { return state[1]; }
    float& z()  { return state[2]; }
    float& vx() { return state[3]; }
    float& vy() { return state[4]; }
    float& vz() { return state[5]; }
    float& ax() { return state[6]; }
    float& ay() { return state[7]; }
    float& az() { return state[8]; }

    float x()  const { return state[0]; }
    float y()  const { return state[1]; }
    float z()  const { return state[2]; }
    float vx() const { return state[3]; }
    float vy() const { return state[4]; }
    float vz() const { return state[5]; }
    float ax() const { return state[6]; }
    float ay() const { return state[7]; }
    float az() const { return state[8]; }

    // 元数据
    int   track_id   = -1;
    int   age        = 0;         ///< 总帧数
    int   lost_count = 0;         ///< 连续丢失帧数
    float confidence = 0.0f;
    int   class_id   = -1;
    float last_raw_x = 0.0f;
    float last_raw_y = 0.0f;
    float last_raw_z = 0.0f;      ///< 上一次原始 z 测量
    int   method     = 0;         ///< 0=mono, 1=stereo, 2=blend

    // IoU 跟踪用: 上一帧 BBox
    float last_cx = 0.0f;
    float last_cy = 0.0f;
    float last_w  = 0.0f;
    float last_h  = 0.0f;

    void init(float x0, float y0, float z0);
    void predict(float dt, float sigma_a);
    void update(float obs_x, float obs_y, float obs_z,
                float Rxy, float Rz);
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
    std::vector<Object3D> predictOnly(double actual_dt = 0.0);

    /**
     * @brief 重置所有跟踪
     */
    void reset();

    /**
     * @brief 获取活跃跟踪数
     */
    int activeTrackCount() const;

    /**
     * @brief 根据上一帧 bbox track 估计当前检测的预测深度
     * @return 匹配到活跃 track 时返回 z, 否则返回 -1
     */
    float predictDepthForDetection(
        const Detection& det,
        float iou_threshold = 0.2f) const;

    /**
     * @brief 返回当前最可信活跃目标的预测深度
     *
     * 用于右目单检 fallback: 右目 bbox 不在左目 track 坐标系内,
     * 不能直接做 IoU 匹配; 单排球场景下可用全局主 track 作为视差先验。
     */
    float predictPrimaryDepth() const;

private:
    float focal_    = 0.0f;
    float baseline_ = 0.0f;
    float cx_       = 0.0f;
    float cy_       = 0.0f;
    HybridDepthConfig config_;

    // 跟踪列表 (IoU 贪心匹配)
    std::vector<DepthTrack> tracks_;
    int next_track_id_ = 0;

    // 自适应偏差校正: EMA 跟踪 zs/zm 比例
    float stereo_bias_ = 1.0f;

    // 内部方法
    float monoDepth(const Detection& det) const;
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
