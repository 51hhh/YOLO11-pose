/**
 * @file trt_detector.cpp
 * @brief TensorRT 检测器实现 (NVDLA + GPU Fallback + INT8)
 *
 * Engine 构建时通过 trtexec 指定 DLA:
 *   trtexec --onnx=model.onnx --int8 --useDLACore=0 --allowGPUFallback \
 *           --saveEngine=model_dla_int8.engine
 *
 * 运行时通过反序列化 engine 直接推理。
 */

#include "trt_detector.h"
#include "volleyball/adapters/tensorrt/cuda_kernels.hpp"
#include "volleyball/adapters/tensorrt/engine_contract.hpp"
#include "volleyball/adapters/tensorrt/postprocess.hpp"
#include "volleyball/logging/log.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

namespace stereo3d {

using volleyball::adapters::tensorrt::DetectorEngineContract;
using volleyball::adapters::tensorrt::DetectorTensorSpec;
using volleyball::adapters::tensorrt::DflScaleView;
using volleyball::adapters::tensorrt::PostprocessConfig;
using volleyball::adapters::tensorrt::SingleOutputView;
using volleyball::adapters::tensorrt::decodeDflOutputs;
using volleyball::adapters::tensorrt::decodeSingleOutput;
using volleyball::adapters::tensorrt::nonMaxSuppression;
using volleyball::adapters::tensorrt::validateDetectorEngineContract;

// TensorRT Logger
static class : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING)
            LOG_WARN("[TRT] %s", msg);
    }
} sTrtLogger;

bool checkCudaSubmit(cudaError_t err, const char* op) {
    if (err == cudaSuccess) return true;
    LOG_ERROR("[TRT] %s failed: %s", op, cudaGetErrorString(err));
    return false;
}

TRTDetector::TRTDetector() = default;

TRTDetector::~TRTDetector() {
    freeBuffers();
#if NV_TENSORRT_MAJOR >= 10
    // TensorRT 10.x: 使用 delete
    for (auto& context : contexts_) {
        if (context) {
            delete context;
            context = nullptr;
        }
    }
    if (engine_)  { delete engine_;  engine_ = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
#else
    // TensorRT 8.x: 使用 deprecated destroy()
    for (auto& context : contexts_) {
        if (context) {
            context->destroy();
            context = nullptr;
        }
    }
    if (engine_)  { engine_->destroy();  engine_ = nullptr; }
    if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }
#endif
}

bool TRTDetector::init(const std::string& engineFile, bool useDLA, int dlaCore,
                       float confThreshold, float nmsThreshold,
                       const std::string& inputFormat) {
    useDLA_        = useDLA;
    dlaCore_       = dlaCore;
    confThreshold_ = confThreshold;
    nmsThreshold_  = nmsThreshold;

    std::string fmt = inputFormat;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (fmt == "bayer") {
        inputFormat_ = InputFormat::BAYER;
    } else if (fmt == "bgr") {
        inputFormat_ = InputFormat::BGR;
    } else {
        inputFormat_ = InputFormat::GRAY;
    }

    if (!loadEngine(engineFile)) {
        LOG_ERROR("Failed to load TRT engine: %s", engineFile.c_str());
        return false;
    }

    if (!allocateBuffers()) {
        LOG_ERROR("Failed to allocate TRT buffers");
        return false;
    }

    LOG_INFO("TRT Detector initialized: %s", engineFile.c_str());
    LOG_INFO("  Input: %dx%d, DLA=%d (core %d)", inputSize_, inputSize_, useDLA_, dlaCore_);
    LOG_INFO("  Output elements: %d", outputSize_);
    LOG_INFO("  Input format: %s", inputFormat_ == InputFormat::BAYER ? "BAYER" :
                                 (inputFormat_ == InputFormat::BGR ? "BGR" : "GRAY"));

    return true;
}

bool TRTDetector::loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open engine file: %s", path.c_str());
        return false;
    }

    const std::streampos end = file.tellg();
    if (end <= 0) {
        LOG_ERROR("TensorRT engine is empty or unreadable: %s", path.c_str());
        return false;
    }
    const size_t fileSize = static_cast<size_t>(end);
    file.seekg(0);
    std::vector<char> engineData(fileSize);
    if (!file.read(engineData.data(), static_cast<std::streamsize>(fileSize))) {
        LOG_ERROR("Failed to read complete TensorRT engine: %s", path.c_str());
        return false;
    }
    file.close();

    runtime_ = nvinfer1::createInferRuntime(sTrtLogger);
    if (!runtime_) return false;

    // 如果 Engine 是 DLA 编译的，Runtime 会自动识别
    engine_ = runtime_->deserializeCudaEngine(engineData.data(), fileSize);
    if (!engine_) return false;

    // 先把 TensorRT 元数据投影为无 SDK 依赖的严格契约。旧实现只取 d[2]
    // 并假定正方形静态输入，遇到动态/矩形/多输入 engine 会按错误尺寸分配缓冲。
    const int nbBindings = engine_->getNbIOTensors();
    std::vector<DetectorTensorSpec> tensorSpecs;
    tensorSpecs.reserve(nbBindings > 0 ? static_cast<size_t>(nbBindings) : 0U);
    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (!name) {
            LOG_ERROR("TensorRT engine returned a null tensor name at index %d", i);
            return false;
        }
        auto mode = engine_->getTensorIOMode(name);
        if (mode != nvinfer1::TensorIOMode::kINPUT &&
            mode != nvinfer1::TensorIOMode::kOUTPUT) {
            LOG_ERROR("Tensor '%s' has unsupported I/O mode", name);
            return false;
        }
        auto dims = engine_->getTensorShape(name);
        DetectorTensorSpec spec;
        spec.name = name;
        spec.is_input = mode == nvinfer1::TensorIOMode::kINPUT;
        spec.is_float32 = engine_->getTensorDataType(name) ==
                          nvinfer1::DataType::kFLOAT;
        if (dims.nbDims > 0) {
            spec.dimensions.reserve(static_cast<size_t>(dims.nbDims));
            for (int d = 0; d < dims.nbDims; ++d) {
                spec.dimensions.push_back(static_cast<std::int64_t>(dims.d[d]));
            }
        }
        tensorSpecs.push_back(std::move(spec));
    }

    DetectorEngineContract contract;
    std::string contractError;
    if (!validateDetectorEngineContract(tensorSpecs, contract, contractError)) {
        LOG_ERROR("Unsupported TensorRT engine contract: %s",
                  contractError.c_str());
        return false;
    }
    inputTensorName_ = contract.input_name;
    outputTensorName_ = contract.single_output_name;
    inputSize_ = contract.input_size;
    numOutputTensors_ = contract.output_count;
    outputSize_ = contract.single_output_elements;
    singleOutputClassCount_ = contract.single_output_class_count;
    singleOutputLayout_ = contract.single_output_layout;
    multiScaleOutput_ = contract.multi_scale_output;

    for (const auto& spec : tensorSpecs) {
        std::ostringstream shape;
        shape << '[';
        for (size_t d = 0; d < spec.dimensions.size(); ++d) {
            if (d != 0) shape << ',';
            shape << spec.dimensions[d];
        }
        shape << ']';
        LOG_INFO("  %s tensor: %s %s float32",
                 spec.is_input ? "Input" : "Output",
                 spec.name.c_str(), shape.str().c_str());
    }
    if (multiScaleOutput_) {
        LOG_INFO("  Detected validated 6-tensor multi-scale DFL output format");
    }

    for (auto& context : contexts_) {
        context = engine_->createExecutionContext();
        if (!context) {
            LOG_ERROR("Failed to create TensorRT execution context");
            return false;
        }
    }

    return true;
}

bool TRTDetector::allocateBuffers() {
    size_t inputBytes  = 3 * inputSize_ * inputSize_ * sizeof(float);

    cudaError_t err;
    auto fail = [&](cudaError_t e, const char* what) {
        LOG_ERROR("[TRT] %s allocation failed: %s", what, cudaGetErrorString(e));
        freeBuffers();
        return false;
    };
    for (auto& b : buffers_) {
        err = cudaMalloc(&b.inputDevice, inputBytes);
        if (err != cudaSuccess) return fail(err, "input");

        if (multiScaleOutput_) {
            // 6-tensor mode: allocate per-scale buffers
            b.scaleOutputs.clear();
            int nbBindings = engine_->getNbIOTensors();
            for (int i = 0; i < nbBindings; ++i) {
                const char* name = engine_->getIOTensorName(i);
                auto mode = engine_->getTensorIOMode(name);
                if (mode == nvinfer1::TensorIOMode::kINPUT) continue;

                auto dims = engine_->getTensorShape(name);
                // dims: [1, H, W, C] for NHWC format
                BufferSet::ScaleOutput so;
                so.name = name;
                so.h = dims.d[1];
                so.w = dims.d[2];
                so.channels = dims.d[3];
                so.isCls = (so.channels <= 4);  // nc=1 → cls, 64 → bbox DFL
                // Infer stride from spatial dims: 80→8, 40→16, 20→32
                so.stride = inputSize_ / so.h;

                size_t bytes = 1;
                for (int d = 0; d < dims.nbDims; ++d) bytes *= dims.d[d];
                bytes *= sizeof(float);

                err = cudaMalloc(&so.device, bytes);
                if (err != cudaSuccess) return fail(err, "scale output device");
                err = cudaHostAlloc(reinterpret_cast<void**>(&so.host),
                                    bytes, cudaHostAllocDefault);
                if (err != cudaSuccess) {
                    cudaFree(so.device);
                    so.device = nullptr;
                    return fail(err, "scale output host");
                }

                LOG_INFO("  Scale output: %s [%d,%d,%d] stride=%d %s",
                         name, so.h, so.w, so.channels, so.stride,
                         so.isCls ? "CLS" : "BBOX");
                b.scaleOutputs.push_back(so);
            }
            // Allocate GPU DFL decode buffers
            constexpr int MAX_DFL_DET = 512;
            err = cudaMalloc(&b.dflOutDevice, MAX_DFL_DET * 6 * sizeof(float));
            if (err != cudaSuccess) return fail(err, "DFL output device");
            err = cudaHostAlloc(reinterpret_cast<void**>(&b.dflOutHost),
                                MAX_DFL_DET * 6 * sizeof(float), cudaHostAllocDefault);
            if (err != cudaSuccess) return fail(err, "DFL output host");
            err = cudaMalloc(reinterpret_cast<void**>(&b.dflCountDevice),
                             sizeof(int));
            if (err != cudaSuccess) return fail(err, "DFL count device");
            err = cudaHostAlloc(reinterpret_cast<void**>(&b.dflCountHost),
                                sizeof(int), cudaHostAllocDefault);
            if (err != cudaSuccess) return fail(err, "DFL count host");
        } else {
            // Single-output mode
            size_t outputBytes = outputSize_ * sizeof(float);
            err = cudaMalloc(&b.outputDevice, outputBytes);
            if (err != cudaSuccess) return fail(err, "output device");
            err = cudaHostAlloc(reinterpret_cast<void**>(&b.outputHost),
                                outputBytes, cudaHostAllocDefault);
            if (err != cudaSuccess) return fail(err, "output host");
        }
    }

    return true;
}

void TRTDetector::freeBuffers() {
    for (auto& b : buffers_) {
        if (b.inputDevice)  { cudaFree(b.inputDevice);   b.inputDevice = nullptr; }
        if (b.outputDevice) { cudaFree(b.outputDevice);  b.outputDevice = nullptr; }
        if (b.outputHost)   { cudaFreeHost(b.outputHost); b.outputHost = nullptr; }
        for (auto& so : b.scaleOutputs) {
            if (so.device) { cudaFree(so.device); so.device = nullptr; }
            if (so.host)   { cudaFreeHost(so.host); so.host = nullptr; }
        }
        b.scaleOutputs.clear();
        // Free GPU DFL decode buffers
        if (b.dflOutDevice)    { cudaFree(b.dflOutDevice);        b.dflOutDevice = nullptr; }
        if (b.dflOutHost)      { cudaFreeHost(b.dflOutHost);      b.dflOutHost = nullptr; }
        if (b.dflCountDevice)  { cudaFree(b.dflCountDevice);      b.dflCountDevice = nullptr; }
        if (b.dflCountHost)    { cudaFreeHost(b.dflCountHost);    b.dflCountHost = nullptr; }
    }
}

std::vector<Detection> TRTDetector::detect(const void* gpuImageU8, int pitch,
                                           int width, int height,
                                           cudaStream_t stream)
{
    // 兼容接口: 使用 slot 0 同步执行
    if (!enqueue(0, gpuImageU8, pitch, width, height, stream)) {
        return {};
    }
    cudaError_t err = cudaStreamSynchronize(stream);
    if (!checkCudaSubmit(err, "detect cudaStreamSynchronize")) {
        return {};
    }
    return collect(0, width, height);
}

bool TRTDetector::enqueue(int slotId,
                          const void* gpuImageU8, int pitch,
                          int width, int height,
                          cudaStream_t stream)
{
    if (slotId < 0 || slotId >= RING_BUFFER_SIZE) {
        LOG_ERROR("[TRT] Invalid ring slot id: %d (expected 0..%d)",
                  slotId, RING_BUFFER_SIZE - 1);
        return false;
    }
    int sid = slotId;
    auto& b = buffers_[sid];
    auto* context = contexts_[sid];
    if (!gpuImageU8 || pitch <= 0 || width <= 0 || height <= 0) {
        LOG_ERROR("[TRT] Invalid enqueue input: ptr=%p pitch=%d size=%dx%d",
                  gpuImageU8, pitch, width, height);
        return false;
    }
    if (!context) {
        LOG_ERROR("[TRT] Invalid execution context for slot %d", sid);
        return false;
    }

    // 1. GPU 预处理: U8 灰度 → float32 RGB CHW
    preprocessGPU(gpuImageU8, pitch, width, height, b.inputDevice, stream);
    if (!checkCudaSubmit(cudaGetLastError(), "preprocessGPU launch")) {
        return false;
    }

    // 2. 绑定本次推理缓冲并异步推理
    if (!context->setTensorAddress(inputTensorName_.c_str(), b.inputDevice)) {
        LOG_ERROR("[TRT] setTensorAddress input '%s' failed",
                  inputTensorName_.c_str());
        return false;
    }

    if (multiScaleOutput_) {
        // Bind all 6 output tensors
        for (auto& so : b.scaleOutputs) {
            if (!context->setTensorAddress(so.name.c_str(), so.device)) {
                LOG_ERROR("[TRT] setTensorAddress output '%s' failed",
                          so.name.c_str());
                return false;
            }
        }
    } else {
        if (!context->setTensorAddress(outputTensorName_.c_str(), b.outputDevice)) {
            LOG_ERROR("[TRT] setTensorAddress output '%s' failed",
                      outputTensorName_.c_str());
            return false;
        }
    }
    if (!context->enqueueV3(stream)) {
        LOG_ERROR("[TRT] enqueueV3 failed");
        return false;
    }

    // 3. 异步后处理 + 拷回 host
    if (multiScaleOutput_) {
        if (useGPUPostprocess_) {
            // GPU DFL decode: launch kernel per scale on device buffers,
            // then copy only the small result to host (saves ~100KB D2H)
            constexpr int MAX_DFL_DET = 512;
            if (!checkCudaSubmit(
                    cudaMemsetAsync(b.dflCountDevice, 0, sizeof(int), stream),
                    "reset DFL detection count")) {
                return false;
            }
            for (size_t i = 0; i < b.scaleOutputs.size(); ++i) {
                const auto& so = b.scaleOutputs[i];
                if (!so.isCls) continue;
                // Find matching bbox tensor
                const BufferSet::ScaleOutput* bboxOut = nullptr;
                for (size_t j = 0; j < b.scaleOutputs.size(); ++j) {
                    if (j == i) continue;
                    const auto& other = b.scaleOutputs[j];
                    if (!other.isCls && other.h == so.h && other.w == so.w) {
                        bboxOut = &other;
                        break;
                    }
                }
                if (!bboxOut) continue;
                launchDFLDecodeKernel(
                    static_cast<const float*>(so.device),
                    static_cast<const float*>(bboxOut->device),
                    b.dflOutDevice, b.dflCountDevice,
                    so.h, so.w, so.channels, regMax_,
                    so.stride, confThreshold_, MAX_DFL_DET, stream);
                if (!checkCudaSubmit(cudaGetLastError(),
                                     "DFL decode launch")) {
                    return false;
                }
            }
            // Copy small result: count + detections
            if (!checkCudaSubmit(
                    cudaMemcpyAsync(b.dflCountHost, b.dflCountDevice, sizeof(int),
                                    cudaMemcpyDeviceToHost, stream),
                    "copy DFL count to host")) {
                return false;
            }
            if (!checkCudaSubmit(
                    cudaMemcpyAsync(b.dflOutHost, b.dflOutDevice,
                                    MAX_DFL_DET * 6 * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream),
                    "copy DFL detections to host")) {
                return false;
            }
        } else {
            // CPU path: copy all tensors to host
            for (auto& so : b.scaleOutputs) {
                size_t bytes = so.h * so.w * so.channels * sizeof(float);
                if (!checkCudaSubmit(
                        cudaMemcpyAsync(so.host, so.device, bytes,
                                        cudaMemcpyDeviceToHost, stream),
                        "copy output tensor to host")) {
                    return false;
                }
            }
        }
    } else {
        size_t outputBytes = outputSize_ * sizeof(float);
        if (!checkCudaSubmit(
                cudaMemcpyAsync(b.outputHost, b.outputDevice, outputBytes,
                                cudaMemcpyDeviceToHost, stream),
                "copy output to host")) {
            return false;
        }
    }
    return true;
}

std::vector<Detection> TRTDetector::collect(int slotId, int width, int height) {
    if (slotId < 0 || slotId >= RING_BUFFER_SIZE) {
        LOG_ERROR("[TRT] Invalid collect ring slot id: %d (expected 0..%d)",
                  slotId, RING_BUFFER_SIZE - 1);
        return {};
    }
    int sid = slotId;
    auto& b = buffers_[sid];

    if (useLetterbox_) {
        float scale = std::min(static_cast<float>(inputSize_) / width,
                               static_cast<float>(inputSize_) / height);
        float padX = (inputSize_ - width * scale) / 2.0f;
        float padY = (inputSize_ - height * scale) / 2.0f;
        float invScale = 1.0f / scale;

        if (multiScaleOutput_) {
            std::vector<Detection> dets;
            if (useGPUPostprocess_) {
                dets = collectGPUDFLResults(b);
            } else {
                dets = postprocessMultiScale(b, 1.0f, 1.0f);
            }
            for (auto& d : dets) {
                d.cx     = (d.cx - padX) * invScale;
                d.cy     = (d.cy - padY) * invScale;
                d.width  = d.width  * invScale;
                d.height = d.height * invScale;
            }
            return dets;
        } else {
            auto dets = postprocess(b.outputHost, 1.0f, 1.0f);
            for (auto& d : dets) {
                d.cx     = (d.cx - padX) * invScale;
                d.cy     = (d.cy - padY) * invScale;
                d.width  = d.width  * invScale;
                d.height = d.height * invScale;
            }
            return dets;
        }
    } else {
        float scaleX = static_cast<float>(width) / inputSize_;
        float scaleY = static_cast<float>(height) / inputSize_;
        if (multiScaleOutput_) {
            if (useGPUPostprocess_) {
                auto dets = collectGPUDFLResults(b);
                for (auto& d : dets) {
                    d.cx *= scaleX; d.cy *= scaleY;
                    d.width *= scaleX; d.height *= scaleY;
                }
                return dets;
            }
            return postprocessMultiScale(b, scaleX, scaleY);
        } else {
            return postprocess(b.outputHost, scaleX, scaleY);
        }
    }
}

void TRTDetector::preprocessGPU(const void* gpuImageU8, int pitch,
                                int srcWidth, int srcHeight,
                                void* inputBufferDevice,
                                cudaStream_t stream)
{
    if (useLetterbox_) {
        // Letterbox: 等比缩放 + 灰色填充, 匹配 YOLO 训练时的预处理
        float scale = std::min(static_cast<float>(inputSize_) / srcWidth,
                               static_cast<float>(inputSize_) / srcHeight);
        int newW = static_cast<int>(srcWidth * scale);
        int newH = static_cast<int>(srcHeight * scale);
        int padX = (inputSize_ - newW) / 2;
        int padY = (inputSize_ - newH) / 2;

        switch (inputFormat_) {
            case InputFormat::BAYER:
                launchBayerToRGBLetterboxKernel(
                    static_cast<const unsigned char*>(gpuImageU8),
                    static_cast<float*>(inputBufferDevice),
                    srcWidth, srcHeight, pitch,
                    inputSize_, inputSize_,
                    newW, newH, padX, padY,
                    stream);
                break;
            case InputFormat::BGR:
                launchBGRToRGBLetterboxKernel(
                    static_cast<const unsigned char*>(gpuImageU8),
                    static_cast<float*>(inputBufferDevice),
                    srcWidth, srcHeight, pitch,
                    inputSize_, inputSize_,
                    newW, newH, padX, padY,
                    stream);
                break;
            case InputFormat::GRAY:
            default:
                launchGrayToRGBLetterboxKernel(
                    static_cast<const unsigned char*>(gpuImageU8),
                    static_cast<float*>(inputBufferDevice),
                    srcWidth, srcHeight, pitch,
                    inputSize_, inputSize_,
                    newW, newH, padX, padY,
                    stream);
                break;
        }
    } else {
        // Direct resize (不保持宽高比)
        switch (inputFormat_) {
            case InputFormat::BAYER:
                launchBayerToRGBKernel(
                    static_cast<const unsigned char*>(gpuImageU8),
                    static_cast<float*>(inputBufferDevice),
                    srcWidth, srcHeight, pitch,
                    inputSize_, inputSize_,
                    stream);
                break;
            case InputFormat::BGR:
                launchBGRToRGBLetterboxKernel(
                    static_cast<const unsigned char*>(gpuImageU8),
                    static_cast<float*>(inputBufferDevice),
                    srcWidth, srcHeight, pitch,
                    inputSize_, inputSize_,
                    inputSize_, inputSize_, 0, 0,
                    stream);
                break;
            case InputFormat::GRAY:
            default:
                launchGrayToRGBKernel(
                    static_cast<const unsigned char*>(gpuImageU8),
                    static_cast<float*>(inputBufferDevice),
                    srcWidth, srcHeight, pitch,
                    inputSize_, inputSize_,
                    stream);
                break;
        }
    }
}

std::vector<Detection> TRTDetector::postprocess(const float* outputBufferHost,
                                                float scaleX, float scaleY) {
    const auto dimensions = engine_->getTensorShape(outputTensorName_.c_str());
    const SingleOutputView output{
        outputBufferHost,
        static_cast<std::size_t>(std::max(0, outputSize_)),
        dimensions.nbDims >= 2 ? dimensions.d[1] : 0,
        dimensions.nbDims >= 3 ? dimensions.d[2] : 0,
        singleOutputClassCount_,
        singleOutputLayout_,
    };
    const auto result = decodeSingleOutput(
        output,
        scaleX,
        scaleY,
        PostprocessConfig{confThreshold_, nmsThreshold_});
    if (!result.success) {
        LOG_ERROR("[TRT] Single-output postprocess failed: %s",
                  result.message.c_str());
        return {};
    }
    return result.detections;
}

std::vector<Detection> TRTDetector::collectGPUDFLResults(const BufferSet& buf) {
    // GPU DFL kernel already ran in enqueue(), results in dflOutHost
    constexpr int MAX_DFL_DET = 512;
    int count = std::clamp(*buf.dflCountHost, 0, MAX_DFL_DET);

    std::vector<Detection> raw_dets;
    raw_dets.reserve(count);
    for (int i = 0; i < count; ++i) {
        const float* row = buf.dflOutHost + i * 6;
        const bool valid =
            std::isfinite(row[0]) && std::isfinite(row[1]) &&
            std::isfinite(row[2]) && std::isfinite(row[3]) &&
            std::isfinite(row[4]) && std::isfinite(row[5]) &&
            row[2] > 0.0F && row[3] > 0.0F &&
            row[4] >= confThreshold_ && row[5] >= 0.0F &&
            row[5] <= static_cast<float>(std::numeric_limits<int>::max());
        if (!valid) {
            LOG_ERROR("[TRT] GPU DFL output contains an invalid detection");
            return {};
        }
        Detection det;
        det.cx         = row[0];
        det.cy         = row[1];
        det.width      = row[2];
        det.height     = row[3];
        det.confidence = row[4];
        det.class_id   = static_cast<int>(row[5]);
        raw_dets.push_back(det);
    }
    return nonMaxSuppression(raw_dets, nmsThreshold_);
}

std::vector<Detection> TRTDetector::postprocessMultiScale(
    const BufferSet& buf, float scaleX, float scaleY)
{
    std::vector<DflScaleView> scales;
    for (std::size_t i = 0; i < buf.scaleOutputs.size(); ++i) {
        const auto& so = buf.scaleOutputs[i];
        if (!so.isCls) continue;

        const BufferSet::ScaleOutput* bboxOut = nullptr;
        for (std::size_t j = 0; j < buf.scaleOutputs.size(); ++j) {
            if (j == i) continue;
            const auto& other = buf.scaleOutputs[j];
            if (!other.isCls && other.h == so.h && other.w == so.w) {
                bboxOut = &other;
                break;
            }
        }
        if (!bboxOut) {
            LOG_ERROR("[TRT] DFL classification tensor '%s' has no box pair",
                      so.name.c_str());
            return {};
        }
        const auto cells = static_cast<std::size_t>(so.h) *
                           static_cast<std::size_t>(so.w);
        scales.push_back({
            so.host,
            cells * static_cast<std::size_t>(so.channels),
            bboxOut->host,
            cells * static_cast<std::size_t>(bboxOut->channels),
            so.h,
            so.w,
            so.channels,
            so.stride,
        });
    }

    const auto result = decodeDflOutputs(
        scales,
        regMax_,
        scaleX,
        scaleY,
        PostprocessConfig{confThreshold_, nmsThreshold_});
    if (!result.success) {
        LOG_ERROR("[TRT] DFL postprocess failed: %s", result.message.c_str());
        return {};
    }
    return result.detections;
}

}  // namespace stereo3d
