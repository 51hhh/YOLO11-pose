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
#include "../utils/logger.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cmath>
#include <numeric>

// CUDA 预处理 kernel (灰度 U8 → RGB float32 CHW)
extern "C" void launchGrayToRGBKernel(const unsigned char* gray, float* dst,
                                       int srcW, int srcH, int srcPitch,
                                       int dstW, int dstH,
                                       cudaStream_t stream);

// Letterbox 预处理 kernel (保持宽高比 + 灰色填充)
extern "C" void launchGrayToRGBLetterboxKernel(const unsigned char* gray, float* dst,
                                                int srcW, int srcH, int srcPitch,
                                                int dstW, int dstH,
                                                int newW, int newH,
                                                int padX, int padY,
                                                cudaStream_t stream);

extern "C" void launchBayerToRGBKernel(const unsigned char* bayer, float* dst,
                                        int srcW, int srcH, int srcPitch,
                                        int dstW, int dstH,
                                        cudaStream_t stream);

extern "C" void launchBayerToRGBLetterboxKernel(const unsigned char* bayer, float* dst,
                                                 int srcW, int srcH, int srcPitch,
                                                 int dstW, int dstH,
                                                 int newW, int newH,
                                                 int padX, int padY,
                                                 cudaStream_t stream);

extern "C" void launchBGRToRGBLetterboxKernel(const unsigned char* bgr, float* dst,
                                               int srcW, int srcH, int srcStep,
                                               int dstW, int dstH,
                                               int newW, int newH,
                                               int padX, int padY,
                                               cudaStream_t stream);

// DFL 后处理 kernel (GPU 加速 sigmoid + softmax + dist2bbox)
extern "C" void launchDFLDecodeKernel(
    const float* cls_data,
    const float* bbox_data,
    float* out_boxes,
    int* out_count,
    int H, int W, int nc, int reg_max,
    int stride, float conf_thresh, int max_det,
    cudaStream_t stream);

namespace stereo3d {

// TensorRT Logger
static class : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING)
            LOG_WARN("[TRT] %s", msg);
    }
} sTrtLogger;

TRTDetector::TRTDetector() = default;

TRTDetector::~TRTDetector() {
    freeBuffers();
#if NV_TENSORRT_MAJOR >= 10
    // TensorRT 10.x: 使用 delete
    if (context_) { delete context_; context_ = nullptr; }
    if (engine_)  { delete engine_;  engine_ = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
#else
    // TensorRT 8.x: 使用 deprecated destroy()
    if (context_) { context_->destroy(); context_ = nullptr; }
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

    size_t fileSize = file.tellg();
    file.seekg(0);
    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(sTrtLogger);
    if (!runtime_) return false;

    // 如果 Engine 是 DLA 编译的，Runtime 会自动识别
    engine_ = runtime_->deserializeCudaEngine(engineData.data(), fileSize);
    if (!engine_) return false;

    context_ = engine_->createExecutionContext();
    if (!context_) return false;

    // 获取 tensor 信息
    int nbBindings = engine_->getNbIOTensors();
    numOutputTensors_ = 0;
    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = engine_->getTensorShape(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputTensorName_ = name;
            // dims: [batch, channels, H, W]
            if (dims.nbDims >= 4) {
                inputSize_ = dims.d[2];  // H == W
            }
            LOG_INFO("  Input tensor: %s [%d,%d,%d,%d]",
                     name, dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
        } else {
            numOutputTensors_++;
            if (numOutputTensors_ == 1) {
                outputTensorName_ = name;
                outputSize_ = 1;
                for (int d = 0; d < dims.nbDims; ++d) {
                    outputSize_ *= dims.d[d];
                }
            }
            LOG_INFO("  Output tensor %d: %s [%d dims]", numOutputTensors_, name,
                     dims.nbDims);
        }
    }

    // Detect 6-tensor multi-scale output format
    // Format: cls[1,H,W,nc], bbox[1,H,W,64] alternating per scale
    if (numOutputTensors_ == 6) {
        multiScaleOutput_ = true;
        LOG_INFO("  Detected 6-tensor multi-scale DFL output format");
    } else {
        multiScaleOutput_ = false;
    }

    return true;
}

bool TRTDetector::allocateBuffers() {
    size_t inputBytes  = 3 * inputSize_ * inputSize_ * sizeof(float);

    cudaError_t err;
    for (auto& b : buffers_) {
        err = cudaMalloc(&b.inputDevice, inputBytes);
        if (err != cudaSuccess) return false;

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
                if (err != cudaSuccess) return false;
                err = cudaHostAlloc(reinterpret_cast<void**>(&so.host),
                                    bytes, cudaHostAllocDefault);
                if (err != cudaSuccess) return false;

                LOG_INFO("  Scale output: %s [%d,%d,%d] stride=%d %s",
                         name, so.h, so.w, so.channels, so.stride,
                         so.isCls ? "CLS" : "BBOX");
                b.scaleOutputs.push_back(so);
            }
            // Allocate GPU DFL decode buffers
            constexpr int MAX_DFL_DET = 512;
            err = cudaMalloc(&b.dflOutDevice, MAX_DFL_DET * 6 * sizeof(float));
            if (err != cudaSuccess) return false;
            err = cudaHostAlloc(reinterpret_cast<void**>(&b.dflOutHost),
                                MAX_DFL_DET * 6 * sizeof(float), cudaHostAllocDefault);
            if (err != cudaSuccess) return false;
            err = cudaMalloc(&b.dflCountDevice, sizeof(int));
            if (err != cudaSuccess) return false;
            err = cudaHostAlloc(reinterpret_cast<void**>(&b.dflCountHost),
                                sizeof(int), cudaHostAllocDefault);
            if (err != cudaSuccess) return false;
        } else {
            // Single-output mode
            size_t outputBytes = outputSize_ * sizeof(float);
            err = cudaMalloc(&b.outputDevice, outputBytes);
            if (err != cudaSuccess) return false;
            err = cudaHostAlloc(reinterpret_cast<void**>(&b.outputHost),
                                outputBytes, cudaHostAllocDefault);
            if (err != cudaSuccess) return false;
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
    enqueue(0, gpuImageU8, pitch, width, height, stream);
    cudaStreamSynchronize(stream);
    return collect(0, width, height);
}

void TRTDetector::enqueue(int slotId,
                          const void* gpuImageU8, int pitch,
                          int width, int height,
                          cudaStream_t stream)
{
    int sid = slotId % RING_BUFFER_SIZE;
    auto& b = buffers_[sid];

    // 1. GPU 预处理: U8 灰度 → float32 RGB CHW
    preprocessGPU(gpuImageU8, pitch, width, height, b.inputDevice, stream);

    // 2. 绑定本次推理缓冲并异步推理
    context_->setTensorAddress(inputTensorName_.c_str(), b.inputDevice);

    if (multiScaleOutput_) {
        // Bind all 6 output tensors
        for (auto& so : b.scaleOutputs) {
            context_->setTensorAddress(so.name.c_str(), so.device);
        }
    } else {
        context_->setTensorAddress(outputTensorName_.c_str(), b.outputDevice);
    }
    context_->enqueueV3(stream);

    // 3. 异步后处理 + 拷回 host
    if (multiScaleOutput_) {
        if (useGPUPostprocess_) {
            // GPU DFL decode: launch kernel per scale on device buffers,
            // then copy only the small result to host (saves ~100KB D2H)
            constexpr int MAX_DFL_DET = 512;
            cudaMemsetAsync(b.dflCountDevice, 0, sizeof(int), stream);
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
            }
            // Copy small result: count + detections
            cudaMemcpyAsync(b.dflCountHost, b.dflCountDevice, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(b.dflOutHost, b.dflOutDevice,
                            MAX_DFL_DET * 6 * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
        } else {
            // CPU path: copy all tensors to host
            for (auto& so : b.scaleOutputs) {
                size_t bytes = so.h * so.w * so.channels * sizeof(float);
                cudaMemcpyAsync(so.host, so.device, bytes,
                                cudaMemcpyDeviceToHost, stream);
            }
        }
    } else {
        size_t outputBytes = outputSize_ * sizeof(float);
        cudaMemcpyAsync(b.outputHost, b.outputDevice, outputBytes,
                        cudaMemcpyDeviceToHost, stream);
    }
}

std::vector<Detection> TRTDetector::collect(int slotId, int width, int height) {
    int sid = slotId % RING_BUFFER_SIZE;
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
    // YOLOv8/v11 输出格式自动检测:
    //
    // 格式A - Pre-NMS 列优先 (YOLOv8 默认):
    //   [1, 4+nc, N]  如 [1, 5, 8400] → nc=1, 有 8400 个候选框
    //
    // 格式B - Pre-NMS 行优先:
    //   [1, N, 4+nc]  如 [1, 8400, 5] → cx,cy,w,h + per-class scores
    //
    // 格式C - Post-NMS (ultralytics v11 带内置 NMS 的导出):
    //   [1, N, 6]  如 [1, 300, 6] → x1,y1,x2,y2,conf,class_id
    //   特征: dim2==6, 已经做完 NMS, 坐标是 xyxy 绝对值

    auto outDims = engine_->getTensorShape(outputTensorName_.c_str());
    int dim1 = (outDims.nbDims >= 2) ? outDims.d[1] : 0;
    int dim2 = (outDims.nbDims >= 3) ? outDims.d[2] : 0;

    std::vector<Detection> raw_dets;
    const float* out = outputBufferHost;

    // 格式C: Post-NMS [1, N, 6] — ultralytics v11 导出 (x1,y1,x2,y2,conf,class_id)
    if (dim2 == 6 && dim1 > 0 && dim1 <= 1000) {
        int numDets = dim1;
        for (int i = 0; i < numDets; ++i) {
            const float* row = out + i * 6;
            float x1   = row[0];
            float y1   = row[1];
            float x2   = row[2];
            float y2   = row[3];
            float conf = row[4];
            int   cls  = static_cast<int>(row[5]);

            if (conf < confThreshold_) continue;

            Detection det;
            det.cx         = (x1 + x2) / 2.0f * scaleX;
            det.cy         = (y1 + y2) / 2.0f * scaleY;
            det.width      = (x2 - x1) * scaleX;
            det.height     = (y2 - y1) * scaleY;
            det.confidence = conf;
            det.class_id   = cls;
            raw_dets.push_back(det);
        }
        return raw_dets;  // 已经是 NMS 后的结果, 直接返回
    }

    // 格式A vs B 判断
    bool transposed = (dim1 > 0 && dim2 > 0 && dim1 < dim2);

    if (transposed) {
        // 列优先: [1, 4+nc, N]  dim1=5, dim2=8400
        int channels = dim1;    // 4 + nc
        int numDets  = dim2;    // N
        int nc = channels - 4;

        for (int i = 0; i < numDets; ++i) {
            // 每列 i: out[row * numDets + i]
            float cx = out[0 * numDets + i];
            float cy = out[1 * numDets + i];
            float w  = out[2 * numDets + i];
            float h  = out[3 * numDets + i];

            // 找最大类别置信度
            float maxConf = 0;
            int maxClass = 0;
            for (int c = 0; c < nc; ++c) {
                float score = out[(4 + c) * numDets + i];
                if (score > maxConf) { maxConf = score; maxClass = c; }
            }

            if (maxConf < confThreshold_) continue;

            Detection det;
            det.cx         = cx * scaleX;
            det.cy         = cy * scaleY;
            det.width      = w  * scaleX;
            det.height     = h  * scaleY;
            det.confidence = maxConf;
            det.class_id   = maxClass;
            raw_dets.push_back(det);
        }
    } else {
        // 行优先: [1, N, 4+nc]  dim1=8400, dim2=5
        int numDets  = dim1;
        int channels = dim2;    // 4 + nc
        int nc = channels - 4;

        for (int i = 0; i < numDets; ++i) {
            const float* row = out + i * channels;
            float cx = row[0];
            float cy = row[1];
            float w  = row[2];
            float h  = row[3];

            float maxConf = 0;
            int maxClass = 0;
            for (int c = 0; c < nc; ++c) {
                if (row[4 + c] > maxConf) { maxConf = row[4 + c]; maxClass = c; }
            }

            if (maxConf < confThreshold_) continue;

            Detection det;
            det.cx         = cx * scaleX;
            det.cy         = cy * scaleY;
            det.width      = w  * scaleX;
            det.height     = h  * scaleY;
            det.confidence = maxConf;
            det.class_id   = maxClass;
            raw_dets.push_back(det);
        }
    }

    return nms(raw_dets);
}

std::vector<Detection> TRTDetector::nms(const std::vector<Detection>& dets) {
    if (dets.empty()) return {};

    // 按置信度排序
    std::vector<int> indices(dets.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return dets[a].confidence > dets[b].confidence; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;

    for (int idx : indices) {
        if (suppressed[idx]) continue;
        result.push_back(dets[idx]);
        for (int j : indices) {
            if (j == idx || suppressed[j]) continue;
            if (computeIoU(dets[idx], dets[j]) > nmsThreshold_) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

float TRTDetector::computeIoU(const Detection& a, const Detection& b) {
    float ax1 = a.cx - a.width / 2, ay1 = a.cy - a.height / 2;
    float ax2 = a.cx + a.width / 2, ay2 = a.cy + a.height / 2;
    float bx1 = b.cx - b.width / 2, by1 = b.cy - b.height / 2;
    float bx2 = b.cx + b.width / 2, by2 = b.cy + b.height / 2;

    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);

    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    return inter / (areaA + areaB - inter + 1e-6f);
}

std::vector<Detection> TRTDetector::collectGPUDFLResults(const BufferSet& buf) {
    // GPU DFL kernel already ran in enqueue(), results in dflOutHost
    constexpr int MAX_DFL_DET = 512;
    int count = std::min(*buf.dflCountHost, MAX_DFL_DET);

    std::vector<Detection> raw_dets;
    raw_dets.reserve(count);
    for (int i = 0; i < count; ++i) {
        const float* row = buf.dflOutHost + i * 6;
        Detection det;
        det.cx         = row[0];
        det.cy         = row[1];
        det.width      = row[2];
        det.height     = row[3];
        det.confidence = row[4];
        det.class_id   = static_cast<int>(row[5]);
        raw_dets.push_back(det);
    }
    return nms(raw_dets);
}

std::vector<Detection> TRTDetector::postprocessMultiScale(
    const BufferSet& buf, float scaleX, float scaleY)
{
    // 6-tensor output from yolov11n_dla:
    //   cls_s8  [1,80,80,nc], bbox_s8  [1,80,80,64]
    //   cls_s16 [1,40,40,nc], bbox_s16 [1,40,40,64]
    //   cls_s32 [1,20,20,nc], bbox_s32 [1,20,20,64]
    //
    // Output tensors are interleaved: cls, bbox, cls, bbox, cls, bbox
    // Detect by channels: nc (1-4) = cls, 64 = bbox DFL

    std::vector<Detection> raw_dets;

    // Group outputs by scale: find cls+bbox pairs
    for (size_t i = 0; i < buf.scaleOutputs.size(); ++i) {
        const auto& so = buf.scaleOutputs[i];
        if (!so.isCls) continue;

        // Find matching bbox tensor for same scale (same H,W)
        const BufferSet::ScaleOutput* bboxOut = nullptr;
        for (size_t j = 0; j < buf.scaleOutputs.size(); ++j) {
            if (j == i) continue;
            const auto& other = buf.scaleOutputs[j];
            if (!other.isCls && other.h == so.h && other.w == so.w) {
                bboxOut = &other;
                break;
            }
        }
        if (!bboxOut) continue;

        int H = so.h;
        int W = so.w;
        int nc = so.channels;
        int stride = so.stride;
        const float* clsData = so.host;
        const float* bboxData = bboxOut->host;

        // Process each grid cell
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int offset = y * W + x;

                // Find max class score (sigmoid)
                float maxConf = 0;
                int maxClass = 0;
                for (int c = 0; c < nc; ++c) {
                    float logit = clsData[offset * nc + c];
                    float score = 1.0f / (1.0f + std::exp(-logit));  // sigmoid
                    if (score > maxConf) {
                        maxConf = score;
                        maxClass = c;
                    }
                }
                if (maxConf < confThreshold_) continue;

                // DFL decode: bbox [H*W, 4*reg_max] → 4 offsets
                // Data layout: [H, W, 4*reg_max] where reg_max=16
                const float* bboxRow = bboxData + offset * (4 * regMax_);
                float offsets[4];

                for (int edge = 0; edge < 4; ++edge) {
                    const float* dfl = bboxRow + edge * regMax_;

                    // Softmax across reg_max bins
                    float maxVal = dfl[0];
                    for (int k = 1; k < regMax_; ++k)
                        maxVal = std::max(maxVal, dfl[k]);

                    float sumExp = 0;
                    float weighted = 0;
                    for (int k = 0; k < regMax_; ++k) {
                        float e = std::exp(dfl[k] - maxVal);
                        sumExp += e;
                        weighted += e * k;
                    }
                    offsets[edge] = weighted / (sumExp + 1e-9f);
                }

                // dist2bbox: anchor = (x+0.5, y+0.5), offsets = (left, top, right, bottom)
                float anchorX = (x + 0.5f) * stride;
                float anchorY = (y + 0.5f) * stride;
                float x1 = anchorX - offsets[0] * stride;
                float y1 = anchorY - offsets[1] * stride;
                float x2 = anchorX + offsets[2] * stride;
                float y2 = anchorY + offsets[3] * stride;

                Detection det;
                det.cx         = (x1 + x2) / 2.0f * scaleX;
                det.cy         = (y1 + y2) / 2.0f * scaleY;
                det.width      = (x2 - x1) * scaleX;
                det.height     = (y2 - y1) * scaleY;
                det.confidence = maxConf;
                det.class_id   = maxClass;
                raw_dets.push_back(det);
            }
        }
    }

    return nms(raw_dets);
}

}  // namespace stereo3d
