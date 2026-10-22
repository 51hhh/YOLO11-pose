#ifndef STEREO_3D_PIPELINE_DETECTION_TYPES_H_
#define STEREO_3D_PIPELINE_DETECTION_TYPES_H_

namespace stereo3d {

/**
 * @brief 单个检测结果
 */
struct Detection {
    float cx, cy;          ///< 检测框中心 (像素坐标)
    float width, height;   ///< 检测框尺寸
    float confidence;      ///< 置信度
    int class_id;          ///< 类别 ID

    Detection() : cx(0), cy(0), width(0), height(0), confidence(0), class_id(0) {}
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_DETECTION_TYPES_H_
