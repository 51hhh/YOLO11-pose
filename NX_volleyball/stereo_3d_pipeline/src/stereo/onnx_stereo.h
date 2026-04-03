/**
 * @file onnx_stereo.h
 * @brief ONNX Runtime 深度学习立体匹配推理 (CREStereo / HITNet)
 *
 * 在 Jetson NX 上通过 ONNX Runtime C++ API 推理 DL 立体匹配模型,
 * 支持 CUDAExecutionProvider (如有) 或 CPUExecutionProvider 回退.
 *
 * 使用:
 *   OnnxStereo model;
 *   model.load("crestereo_init_iter10_480x640.onnx", OnnxStereo::Model::CREStereo);
 *   cv::Mat disp = model.compute(rectL_gray, rectR_gray);
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

#ifdef HAS_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>

namespace stereo3d {

class OnnxStereo {
public:
    enum class Model {
        CREStereo,   // 输入: 2×(1,3,H,W) float, 输出: (1,2,H,W) 或 (1,H,W) 视差
        HITNet,      // 输入: (1,C,H,W) C=2(灰度)/6(RGB), 输出: (1,1,H,W) 视差
        Generic      // 自动探测输入输出形状
    };

    OnnxStereo();
    ~OnnxStereo();

    // 不可拷贝
    OnnxStereo(const OnnxStereo&) = delete;
    OnnxStereo& operator=(const OnnxStereo&) = delete;

    /**
     * @brief 加载 ONNX 模型
     * @param onnxPath  ONNX 文件路径
     * @param modelType 模型类型 (自动适配输入预处理)
     * @return 成功返回 true
     */
    bool load(const std::string& onnxPath, Model modelType = Model::Generic);

    /**
     * @brief 推理计算视差图
     * @param leftGray  校正后左图 (CV_8U 灰度)
     * @param rightGray 校正后右图 (CV_8U 灰度)
     * @return CV_32F 视差图 (像素单位), 空则表示失败
     */
    cv::Mat compute(const cv::Mat& leftGray, const cv::Mat& rightGray);

    /** @brief 模型是否已加载 */
    bool isLoaded() const { return session_ != nullptr; }

    /** @brief 模型名称 */
    const std::string& modelName() const { return modelName_; }

    /** @brief 模型期望输入尺寸 (H,W), 推理前自动缩放 */
    cv::Size inputSize() const { return inputSize_; }

private:
    // 预处理: 灰度/RGB → 模型输入 tensor
    std::vector<float> preprocess(const cv::Mat& leftGray, const cv::Mat& rightGray,
                                  int inputC, int inputH, int inputW);

    // 从输出 tensor 提取视差图
    cv::Mat extractDisparity(const std::vector<Ort::Value>& outputs,
                             int origH, int origW);

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    Model modelType_ = Model::Generic;
    std::string modelName_;
    cv::Size inputSize_{0, 0};

    // 缓存 I/O 名称
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamePtrs_;
    std::vector<const char*> outputNamePtrs_;

    // 缓存输入形状
    struct InputInfo {
        std::string name;
        std::vector<int64_t> shape;  // e.g. {1,3,480,640} or {1,2,480,640}
    };
    std::vector<InputInfo> inputs_;
};

} // namespace stereo3d

#else  // !HAS_ONNXRUNTIME

// Stub: 当环境无 ONNX Runtime 时编译通过, 但 load() 总返回 false
namespace stereo3d {

class OnnxStereo {
public:
    enum class Model { CREStereo, HITNet, Generic };
    OnnxStereo() = default;
    bool load(const std::string&, Model = Model::Generic) { return false; }
    cv::Mat compute(const cv::Mat&, const cv::Mat&) { return {}; }
    bool isLoaded() const { return false; }
    const std::string& modelName() const { static std::string s = "N/A"; return s; }
    cv::Size inputSize() const { return {0, 0}; }
};

} // namespace stereo3d

#endif // HAS_ONNXRUNTIME
