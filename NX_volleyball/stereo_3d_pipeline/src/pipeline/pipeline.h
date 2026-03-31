/**
 * @file pipeline.h
 * @brief 四级流水线 Pipeline 类
 *
 * 流水线架构:
 *   Stage 0 [Grab + Rectify]  → CPU + PVA    ~3ms
 *   Stage 1 [Detect / DLA]    → NVDLA         ~12-15ms  ┐
 *   Stage 2 [Stereo / GPU]    → GPU CUDA      ~10-12ms  ┘ 串行执行, 帧级流水线重叠
 *   Stage 3 [Fuse + Output]   → GPU/CPU       ~1ms
 *
 * 使用三缓冲实现帧间流水线重叠:
 *   Frame N:   Stage 3 (Fuse)
 *   Frame N+1: Stage 1+2 (Detect+Stereo)
 *   Frame N+2: Stage 0 (Grab+Rect)
 *
 * 吞吐量 = 1 / max(Stage_i latency) → 60-100 FPS
 */

#ifndef STEREO_3D_PIPELINE_PIPELINE_H_
#define STEREO_3D_PIPELINE_PIPELINE_H_

#include "frame_slot.h"
#include "sync.h"
#ifdef HIK_CAMERA_ENABLED
#include "../capture/hikvision_camera.h"
#else
namespace stereo3d { class HikvisionCamera; struct CameraConfig; }
#endif
#include "../calibration/stereo_calibration.h"
#include "../rectify/vpi_rectifier.h"
#include "../detect/trt_detector.h"
#include "../stereo/vpi_stereo.h"
#include "../fusion/coordinate_3d.h"
#include "../utils/profiler.h"

#include <atomic>
#include <functional>
#include <string>
#include <thread>

namespace stereo3d {

/**
 * @brief 视差策略
 */
enum class DisparityStrategy {
    FULL_FRAME,        ///< 全帧视差 (默认, 1280x720)
    HALF_RESOLUTION,   ///< 半分辨率 (640x360)
    ROI_ONLY           ///< 仅计算 ROI 区域
};

/**
 * @brief Pipeline 运行时参数
 */
struct PipelineConfig {
    // 图像尺寸 (分离原始/校正分辨率)
    int raw_width   = 1440;    ///< 相机原始分辨率
    int raw_height  = 1080;
    int rect_width  = 1280;    ///< 校正后分辨率
    int rect_height = 720;

    // 相机
    int cam_left_index  = 0;
    int cam_right_index = 1;
    std::string cam_left_serial  = "";
    std::string cam_right_serial = "";
    float exposure_us  = 9867.0f;
    float gain_db      = 10.0f;
    bool  use_trigger  = true;
    std::string trigger_source     = "Line0";
    std::string trigger_activation = "RisingEdge";
    int trigger_freq_hz = 100;

    // 标定
    std::string calibration_file = "config/intrinsics.yaml";

    // 检测
    std::string engine_file = "models/yolov8n_int8.engine";
    int  input_size      = 320;    ///< 模型输入尺寸
    float conf_threshold = 0.5f;
    float nms_threshold  = 0.4f;
    int  max_detections  = 10;
    bool use_dla = true;           ///< 使用 NVDLA (否则 GPU)
    int  dla_core = 0;             ///< DLA 核心 ID (0 或 1)

    // 视差
    int max_disparity = 128;
    int window_size   = 5;
    int stereo_quality = 6;
    DisparityStrategy disparity_strategy = DisparityStrategy::FULL_FRAME;

    // 深度
    float min_depth = 0.3f;        ///< 最小深度 (m)
    float max_depth = 15.0f;       ///< 最大深度 (m)

    // 性能
    int stats_interval = 100;      ///< 每 N 帧打印统计
};

/**
 * @brief 结果回调
 */
using ResultCallback = std::function<void(int frame_id, const std::vector<Object3D>& results)>;

/**
 * @brief 四级流水线主类
 */
class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    /**
     * @brief 初始化 Pipeline 所有组件
     * @return true 全部初始化成功
     */
    bool init(const PipelineConfig& config);

    /**
     * @brief 启动 Pipeline (在独立线程中运行)
     */
    void start();

    /**
     * @brief 停止 Pipeline 并等待线程退出
     */
    void stop();

    /**
     * @brief 设置结果回调
     */
    void setResultCallback(ResultCallback cb) { result_callback_ = std::move(cb); }

    /**
     * @brief 获取当前帧率 (吞吐量)
     */
    float getCurrentFPS() const { return current_fps_.load(); }

    /**
     * @brief 打印性能报告
     */
    void printPerfReport() const;

private:
    // ===== Pipeline 主循环 =====
    void pipelineLoop();

    // ===== Stage 函数 =====
    void stage0_grab_and_rectify(FrameSlot& slot);
    void stage1_detect(FrameSlot& slot);
    void stage2_stereo(FrameSlot& slot);
    void stage3_fuse(FrameSlot& slot);

    // ===== 组件 =====
    PipelineConfig config_;
    PipelineStreams streams_;
    FrameSlot slots_[RING_BUFFER_SIZE];

    std::unique_ptr<HikvisionCamera> camera_;    ///< 双目相机 (单实例管理左右)
    std::unique_ptr<StereoCalibration> calibration_;
    std::unique_ptr<VPIRectifier> rectifier_;
    std::unique_ptr<TRTDetector> detector_;
    std::unique_ptr<VPIStereo> stereo_;
    std::unique_ptr<Coordinate3D> fusion_;

    // ===== 状态 =====
    std::atomic<bool> running_{false};
    std::atomic<float> current_fps_{0.0f};
    std::thread pipeline_thread_;              ///< Pipeline 工作线程

    ResultCallback result_callback_;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PIPELINE_H_
