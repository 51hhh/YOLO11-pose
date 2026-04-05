/**
 * @file onnx_stereo.cpp
 * @brief ONNX Runtime 深度学习立体匹配推理实现
 */

#ifdef HAS_ONNXRUNTIME

#include "stereo/onnx_stereo.h"
#include "utils/logger.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace stereo3d {

OnnxStereo::OnnxStereo()
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxStereo")
{
}

OnnxStereo::~OnnxStereo() = default;

bool OnnxStereo::load(const std::string& onnxPath, Model modelType)
{
    modelType_ = modelType;
    modelName_ = onnxPath.substr(onnxPath.find_last_of("/\\") + 1);

    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 尝试 CUDA EP
        try {
            OrtCUDAProviderOptions cuda_opts;
            cuda_opts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
            LOG_INFO("OnnxStereo: CUDA EP requested");
        } catch (...) {
            LOG_INFO("OnnxStereo: CUDA EP unavailable, using CPU");
        }

        session_ = std::make_unique<Ort::Session>(env_, onnxPath.c_str(), opts);

        // 解析输入
        size_t numInputs = session_->GetInputCount();
        inputs_.resize(numInputs);
        inputNames_.resize(numInputs);
        inputNamePtrs_.resize(numInputs);

        for (size_t i = 0; i < numInputs; ++i) {
            auto namePtr = session_->GetInputNameAllocated(i, allocator_);
            inputNames_[i] = namePtr.get();
            inputNamePtrs_[i] = inputNames_[i].c_str();

            auto typeInfo = session_->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            inputs_[i].name = inputNames_[i];
            inputs_[i].shape = tensorInfo.GetShape();

            // 替换动态维度 (-1) 为默认值
            for (auto& d : inputs_[i].shape) {
                if (d <= 0) d = 1;
            }

            LOG_INFO("  Input[%zu] '%s': [%lld,%lld,%lld,%lld]",
                     i, inputNames_[i].c_str(),
                     inputs_[i].shape.size() > 0 ? inputs_[i].shape[0] : 0,
                     inputs_[i].shape.size() > 1 ? inputs_[i].shape[1] : 0,
                     inputs_[i].shape.size() > 2 ? inputs_[i].shape[2] : 0,
                     inputs_[i].shape.size() > 3 ? inputs_[i].shape[3] : 0);
        }

        // 解析输出
        size_t numOutputs = session_->GetOutputCount();
        outputNames_.resize(numOutputs);
        outputNamePtrs_.resize(numOutputs);
        for (size_t i = 0; i < numOutputs; ++i) {
            auto namePtr = session_->GetOutputNameAllocated(i, allocator_);
            outputNames_[i] = namePtr.get();
            outputNamePtrs_[i] = outputNames_[i].c_str();

            auto typeInfo = session_->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            auto shape = tensorInfo.GetShape();
            LOG_INFO("  Output[%zu] '%s': [%lld,%lld,%lld,%lld]",
                     i, outputNames_[i].c_str(),
                     shape.size() > 0 ? shape[0] : 0,
                     shape.size() > 1 ? shape[1] : 0,
                     shape.size() > 2 ? shape[2] : 0,
                     shape.size() > 3 ? shape[3] : 0);
        }

        // 推断模型输入尺寸(H,W)
        if (!inputs_.empty() && inputs_[0].shape.size() == 4) {
            int h = static_cast<int>(inputs_[0].shape[2]);
            int w = static_cast<int>(inputs_[0].shape[3]);
            if (h > 1 && w > 1) {
                inputSize_ = cv::Size(w, h);
            }
        }

        LOG_INFO("OnnxStereo: loaded '%s' (%zu inputs, %zu outputs, %dx%d)",
                 modelName_.c_str(), numInputs, numOutputs,
                 inputSize_.width, inputSize_.height);
        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("OnnxStereo load failed: %s", e.what());
        session_.reset();
        return false;
    }
}

std::vector<float> OnnxStereo::preprocess(const cv::Mat& leftGray, const cv::Mat& rightGray,
                                           int inputC, int inputH, int inputW)
{
    // 缩放到模型期望尺寸
    cv::Mat left, right;
    if (leftGray.rows != inputH || leftGray.cols != inputW) {
        cv::resize(leftGray, left, cv::Size(inputW, inputH), 0, 0, cv::INTER_LINEAR);
        cv::resize(rightGray, right, cv::Size(inputW, inputH), 0, 0, cv::INTER_LINEAR);
    } else {
        left = leftGray;
        right = rightGray;
    }

    // 归一化到 [0,1] float
    cv::Mat leftF, rightF;
    left.convertTo(leftF, CV_32F, 1.0 / 255.0);
    right.convertTo(rightF, CV_32F, 1.0 / 255.0);

    std::vector<float> tensor;

    if (modelType_ == Model::HITNet) {
        // HITNet: 通道拼接 left+right → (1, C, H, W)
        // C=2: 左右各1灰度通道; C=6: 左右各3 RGB 通道
        if (inputC == 2) {
            // 2 通道: 灰度 L + 灰度 R
            tensor.resize(1 * 2 * inputH * inputW);
            for (int y = 0; y < inputH; ++y) {
                for (int x = 0; x < inputW; ++x) {
                    tensor[0 * inputH * inputW + y * inputW + x] = leftF.at<float>(y, x);
                    tensor[1 * inputH * inputW + y * inputW + x] = rightF.at<float>(y, x);
                }
            }
        } else if (inputC == 6) {
            // 6 通道: RGB_L(3) + RGB_R(3)
            cv::Mat leftRGB, rightRGB;
            cv::cvtColor(leftF, leftRGB, cv::COLOR_GRAY2RGB);
            cv::cvtColor(rightF, rightRGB, cv::COLOR_GRAY2RGB);
            tensor.resize(1 * 6 * inputH * inputW);
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < inputH; ++y) {
                    for (int x = 0; x < inputW; ++x) {
                        tensor[(c) * inputH * inputW + y * inputW + x] =
                            leftRGB.at<cv::Vec3f>(y, x)[c];
                        tensor[(c + 3) * inputH * inputW + y * inputW + x] =
                            rightRGB.at<cv::Vec3f>(y, x)[c];
                    }
                }
            }
        } else {
            // 通用: 假设 C=1 灰度 L+R 交替
            tensor.resize(1 * inputC * inputH * inputW, 0.0f);
            for (int y = 0; y < inputH; ++y) {
                for (int x = 0; x < inputW; ++x) {
                    tensor[0 * inputH * inputW + y * inputW + x] = leftF.at<float>(y, x);
                    if (inputC > 1)
                        tensor[1 * inputH * inputW + y * inputW + x] = rightF.at<float>(y, x);
                }
            }
        }
    } else {
        // CREStereo / Generic: 灰度→RGB 复制3通道, (1,3,H,W), 两个独立输入
        // 此函数只返回第一个输入，第二个在 compute() 中处理
        tensor.resize(1 * 3 * inputH * inputW);
        for (int y = 0; y < inputH; ++y) {
            for (int x = 0; x < inputW; ++x) {
                float v = leftF.at<float>(y, x);
                tensor[0 * inputH * inputW + y * inputW + x] = v;
                tensor[1 * inputH * inputW + y * inputW + x] = v;
                tensor[2 * inputH * inputW + y * inputW + x] = v;
            }
        }
    }

    return tensor;
}

cv::Mat OnnxStereo::extractDisparity(const std::vector<Ort::Value>& outputs,
                                      int origH, int origW)
{
    if (outputs.empty()) return {};

    // 遍历输出找视差图 (选最接近 H×W 的 2D slice)
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto info = outputs[i].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        const float* data = outputs[i].GetTensorData<float>();

        int64_t total = 1;
        for (auto d : shape) total *= d;
        if (total <= 0) continue;

        int h = 0, w = 0;
        int offset = 0;

        if (shape.size() == 4) {
            // (N, C, H, W) — 取第一个 batch, 选通道
            h = static_cast<int>(shape[2]);
            w = static_cast<int>(shape[3]);
            int c = static_cast<int>(shape[1]);
            // 对于 flow-based 模型, 视差在 channel 0
            offset = 0;
            if (c == 2) {
                // CREStereo: output (1,2,H,W) — 取 channel 0 (水平视差)
                offset = 0;
            }
        } else if (shape.size() == 3) {
            // (N, H, W)
            h = static_cast<int>(shape[1]);
            w = static_cast<int>(shape[2]);
        } else if (shape.size() == 2) {
            h = static_cast<int>(shape[0]);
            w = static_cast<int>(shape[1]);
        } else {
            continue;
        }

        if (h <= 0 || w <= 0) continue;

        cv::Mat disp(h, w, CV_32F);
        std::memcpy(disp.data, data + offset * h * w, h * w * sizeof(float));

        // CREStereo 输出为光流形式, 需要取绝对值
        if (modelType_ == Model::CREStereo) {
            for (int y = 0; y < h; ++y) {
                float* p = disp.ptr<float>(y);
                for (int x = 0; x < w; ++x) {
                    p[x] = std::abs(p[x]);
                }
            }
        }

        // 缩放回原始尺寸
        if (h != origH || w != origW) {
            float scaleX = static_cast<float>(origW) / w;
            cv::Mat resized;
            cv::resize(disp, resized, cv::Size(origW, origH), 0, 0, cv::INTER_LINEAR);
            resized *= scaleX;  // 视差随宽度缩放
            return resized;
        }

        return disp;
    }

    return {};
}

cv::Mat OnnxStereo::compute(const cv::Mat& leftGray, const cv::Mat& rightGray)
{
    if (!session_) return {};
    if (leftGray.empty() || rightGray.empty()) return {};

    int origH = leftGray.rows;
    int origW = leftGray.cols;

    try {
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> inputTensors;

        if (modelType_ == Model::HITNet && inputs_.size() == 1) {
            // HITNet: 单输入 (1, C, H, W), L+R 通道拼接
            auto& inp = inputs_[0];
            int inputC = static_cast<int>(inp.shape[1]);
            int inputH = static_cast<int>(inp.shape[2]);
            int inputW = static_cast<int>(inp.shape[3]);

            auto tensor = preprocess(leftGray, rightGray, inputC, inputH, inputW);
            inputTensors.push_back(
                Ort::Value::CreateTensor<float>(memInfo, tensor.data(), tensor.size(),
                                                inp.shape.data(), inp.shape.size()));
        } else {
            // CREStereo / Generic: 多输入, 每个是 (1,3,H,W)
            // 或单输入形状自动处理
            // 注意: 所有输入 buffer 必须在 Run() 前保持存活
            std::vector<std::vector<float>> inputBuffers(inputs_.size());
            for (size_t i = 0; i < inputs_.size(); ++i) {
                auto& inp = inputs_[i];
                int inputC = inp.shape.size() > 1 ? static_cast<int>(inp.shape[1]) : 3;
                int inputH = inp.shape.size() > 2 ? static_cast<int>(inp.shape[2]) : origH;
                int inputW = inp.shape.size() > 3 ? static_cast<int>(inp.shape[3]) : origW;

                const cv::Mat& src = (i == 0) ? leftGray : rightGray;
                cv::Mat scaled;
                if (src.rows != inputH || src.cols != inputW) {
                    cv::resize(src, scaled, cv::Size(inputW, inputH));
                } else {
                    scaled = src;
                }

                cv::Mat scaledF;
                scaled.convertTo(scaledF, CV_32F, 1.0 / 255.0);

                // 灰度 → 3 通道
                auto& data = inputBuffers[i];
                data.resize(1 * inputC * inputH * inputW, 0.0f);
                for (int y = 0; y < inputH; ++y) {
                    for (int x = 0; x < inputW; ++x) {
                        float v = scaledF.at<float>(y, x);
                        for (int c = 0; c < inputC; ++c) {
                            data[c * inputH * inputW + y * inputW + x] = v;
                        }
                    }
                }

                inputTensors.push_back(
                    Ort::Value::CreateTensor<float>(memInfo, data.data(), data.size(),
                                                    inp.shape.data(), inp.shape.size()));
            }
        }

        // 推理
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            inputNamePtrs_.data(), inputTensors.data(), inputTensors.size(),
            outputNamePtrs_.data(), outputNamePtrs_.size());

        return extractDisparity(outputs, origH, origW);

    } catch (const Ort::Exception& e) {
        LOG_ERROR("OnnxStereo::compute failed: %s", e.what());
        return {};
    }
}

} // namespace stereo3d

#endif // HAS_ONNXRUNTIME
