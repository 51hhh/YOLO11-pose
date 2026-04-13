/**
 * @file sot_tracker.h
 * @brief SOT (Single Object Tracking) 抽象接口
 *
 * 在 YOLO 检测间隙帧产生 bbox，使检测层恢复全帧输出。
 * 具体实现: NanoTrackTRT, MixFormerTRT
 */

#ifndef STEREO_3D_PIPELINE_SOT_TRACKER_H_
#define STEREO_3D_PIPELINE_SOT_TRACKER_H_

#include <cuda_runtime.h>
#include <string>
#include "../pipeline/frame_slot.h"

namespace stereo3d {

class SOTTracker {
public:
    virtual ~SOTTracker() = default;

    /**
     * @brief 初始化 TRT 引擎
     * @param engine_path 主引擎路径 (NanoTrack: backbone, MixFormer: 单引擎)
     * @param head_engine_path NanoTrack head 引擎路径 (MixFormer 不使用)
     * @param stream CUDA stream
     */
    virtual bool init(const std::string& engine_path,
                      const std::string& head_engine_path,
                      cudaStream_t stream) = 0;

    /**
     * @brief 设置/刷新跟踪模板 (YOLO 检测到目标时调用)
     * @param gpu_image 校正后图像 GPU 指针 (U8 灰度或 BGR)
     * @param pitch 行字节跨度
     */
    virtual void setTarget(const void* gpu_image, int pitch,
                           int img_width, int img_height,
                           const Detection& det) = 0;

    /**
     * @brief 在当前帧执行跟踪 (同步)
     */
    virtual SOTResult track(const void* gpu_image, int pitch,
                            int img_width, int img_height) = 0;

    virtual void reset() = 0;
    virtual bool hasTarget() const = 0;
    virtual int getTemplateSize() const = 0;
    virtual int getSearchSize() const = 0;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_SOT_TRACKER_H_
