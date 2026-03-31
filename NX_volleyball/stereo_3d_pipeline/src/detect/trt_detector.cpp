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
#include <cstring>
#include <numeric>

// CUDA 预处理 kernel (灰度 U8 → RGB float32 CHW)
extern "C" void launchGrayToRGBKernel(const unsigned char* gray, float* dst,
                                       int srcW, int srcH, int srcPitch,
                                       int dstW, int dstH,
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
                       float confThreshold, float nmsThreshold) {
    useDLA_        = useDLA;
    dlaCore_       = dlaCore;
    confThreshold_ = confThreshold;
    nmsThreshold_  = nmsThreshold;

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
            outputTensorName_ = name;
            outputSize_ = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                outputSize_ *= dims.d[d];
            }
            LOG_INFO("  Output tensor: %s (%d elements)", name, outputSize_);
        }
    }

    return true;
}

bool TRTDetector::allocateBuffers() {
    size_t inputBytes  = 3 * inputSize_ * inputSize_ * sizeof(float);
    size_t outputBytes = outputSize_ * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&inputBufferDevice_, inputBytes);
    if (err != cudaSuccess) return false;

    err = cudaMalloc(&outputBufferDevice_, outputBytes);
    if (err != cudaSuccess) return false;

    // Host 输出缓冲 (pinned memory)
    err = cudaHostAlloc(reinterpret_cast<void**>(&outputBufferHost_),
                        outputBytes, cudaHostAllocDefault);
    if (err != cudaSuccess) return false;

    // 绑定 tensor 地址
    context_->setTensorAddress(inputTensorName_.c_str(), inputBufferDevice_);
    context_->setTensorAddress(outputTensorName_.c_str(), outputBufferDevice_);

    return true;
}

void TRTDetector::freeBuffers() {
    if (inputBufferDevice_)  { cudaFree(inputBufferDevice_);  inputBufferDevice_ = nullptr; }
    if (outputBufferDevice_) { cudaFree(outputBufferDevice_); outputBufferDevice_ = nullptr; }
    if (outputBufferHost_)   { cudaFreeHost(outputBufferHost_); outputBufferHost_ = nullptr; }
}

std::vector<Detection> TRTDetector::detect(const void* gpuImageU8, int pitch,
                                           int width, int height,
                                           cudaStream_t stream)
{
    // 1. GPU 预处理: U8 灰度 → float32 RGB CHW
    preprocessGPU(gpuImageU8, pitch, width, height, stream);

    // 2. TRT 异步推理 (DLA/GPU)
    context_->enqueueV3(stream);

    // 3. D2H 拷贝输出 + 同步 (后处理需要 CPU 数据)
    size_t outputBytes = outputSize_ * sizeof(float);
    cudaMemcpyAsync(outputBufferHost_, outputBufferDevice_, outputBytes,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // 仅此一次同步 (后处理在 CPU)

    // 4. 后处理
    float scaleX = static_cast<float>(width) / inputSize_;
    float scaleY = static_cast<float>(height) / inputSize_;
    return postprocess(scaleX, scaleY);
}

void TRTDetector::preprocessGPU(const void* gpuImageU8, int pitch,
                                int srcWidth, int srcHeight,
                                cudaStream_t stream)
{
    launchGrayToRGBKernel(
        static_cast<const unsigned char*>(gpuImageU8),
        static_cast<float*>(inputBufferDevice_),
        srcWidth, srcHeight, pitch,
        inputSize_, inputSize_,
        stream);
}

std::vector<Detection> TRTDetector::postprocess(float scaleX, float scaleY) {
    // YOLOv8/v11 输出格式自动检测:
    //   行优先 (row-major): [1, N, 4+nc]  → stride = 4+nc, N 条检测, 每行是一个检测
    //   列优先 (col-major): [1, 4+nc, N]  → nc 类, N 列, 每列是一个检测 (YOLOv8 默认导出)
    //
    // 区分方法:
    //   获取 output tensor shape, 若 dims[1] < dims[2] → 列优先 (转置)
    //   例如 [1, 5, 8400] → 列优先;  [1, 8400, 5] → 行优先

    auto outDims = engine_->getTensorShape(outputTensorName_.c_str());
    int dim1 = (outDims.nbDims >= 2) ? outDims.d[1] : 0;
    int dim2 = (outDims.nbDims >= 3) ? outDims.d[2] : 0;

    std::vector<Detection> raw_dets;
    const float* out = outputBufferHost_;

    // 单类别排球检测: nc=1, 所以 4+nc=5
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

}  // namespace stereo3d
