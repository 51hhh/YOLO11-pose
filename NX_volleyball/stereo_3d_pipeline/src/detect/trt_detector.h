/**
 * @file trt_detector.h
 * @brief TensorRT 检测器 (支持 NVDLA + GPU Fallback + INT8)
 *
 * 在 Xavier NX 上:
 *   - 使用 NVDLA Core 0 (独立于 GPU) 运行推理
 *   - 不支持的层自动 Fallback 到 GPU
 *   - 解放 GPU 给 VPI 视差计算
 *
 * 架构:
 *   VPIImage (GPU ptr) → CUDA 预处理 → TRT Engine (DLA) → 后处理 → Detections
 */

#ifndef STEREO_3D_PIPELINE_TRT_DETECTOR_H_
#define STEREO_3D_PIPELINE_TRT_DETECTOR_H_

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <array>
#include <string>
#include <vector>

#include "../pipeline/frame_slot.h"

namespace stereo3d {

class TRTDetector {
public:
    enum class InputFormat {
        GRAY,   ///< 单通道灰度
        BAYER,  ///< BayerRG8 单通道马赛克
        BGR     ///< 三通道 BGR
    };

    TRTDetector();
    ~TRTDetector();

    TRTDetector(const TRTDetector&) = delete;
    TRTDetector& operator=(const TRTDetector&) = delete;

    /**
     * @brief 初始化检测器
     * @param engineFile TensorRT Engine 文件路径
     * @param useDLA 是否使用 NVDLA
     * @param dlaCore DLA 核心 ID (0 或 1)
     * @param confThreshold 置信度阈值
     * @param nmsThreshold NMS 阈值
    * @param inputFormat 输入格式: "gray" | "bayer" | "bgr"
     * @return true 初始化成功
     */
    bool init(const std::string& engineFile, bool useDLA, int dlaCore,
            float confThreshold, float nmsThreshold,
            const std::string& inputFormat = "gray");

    /**
    * @brief 从 GPU 图像执行检测
    * @param gpuImageU8 GPU 图像指针 (格式由 inputFormat 决定)
     * @param pitch 图像行字节跨度
     * @param width 图像宽度
     * @param height 图像高度
     * @param stream CUDA Stream (用于异步推理)
     * @return 检测结果列表
     */
    std::vector<Detection> detect(const void* gpuImageU8, int pitch,
                                  int width, int height,
                                  cudaStream_t stream);

    /**
     * @brief 异步提交一次推理 (不阻塞)
     * @param slotId RingBuffer 槽位 ID [0, RING_BUFFER_SIZE)
     */
    void enqueue(int slotId,
                 const void* gpuImageU8, int pitch,
                 int width, int height,
                 cudaStream_t stream);

    /**
     * @brief 从指定槽位回收推理结果并做后处理
     *
     * 调用方需保证对应 stream 的推理和 D2H 已完成 (例如通过 cudaEvent 等待)。
     */
    std::vector<Detection> collect(int slotId, int width, int height);

    int getInputSize() const { return inputSize_; }
    bool isDLA() const { return useDLA_; }

private:
    struct BufferSet {
        void* inputDevice = nullptr;
        void* outputDevice = nullptr;
        float* outputHost = nullptr;
        // Multi-scale outputs (6-tensor format: cls+bbox per scale)
        struct ScaleOutput {
            void* device = nullptr;
            float* host = nullptr;
            int h = 0, w = 0, channels = 0;
            int stride = 0;
            bool isCls = false;  // true=classification, false=bbox DFL
            std::string name;
        };
        std::vector<ScaleOutput> scaleOutputs;
        // GPU DFL decode buffers (shared across scales)
        float* dflOutDevice = nullptr;   // [max_det * 6] on GPU
        float* dflOutHost   = nullptr;   // [max_det * 6] pinned
        int*   dflCountDevice = nullptr; // atomic counter on GPU
        int*   dflCountHost   = nullptr; // pinned
    };

    // TensorRT 组件
    nvinfer1::IRuntime* runtime_   = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // 每个 RingSlot 一套 I/O 缓冲, 避免异步覆盖
    std::array<BufferSet, RING_BUFFER_SIZE> buffers_;

    // 参数
    int inputSize_   = 640;     ///< 模型输入正方形尺寸
    int outputSize_  = 0;       ///< 输出 tensor 元素数 (single-output mode)
    float confThreshold_ = 0.5f;
    float nmsThreshold_  = 0.4f;
    bool useDLA_    = true;
    int  dlaCore_   = 0;
    bool useLetterbox_ = true;  ///< 使用 letterbox 预处理 (保持宽高比)
    InputFormat inputFormat_ = InputFormat::GRAY;  ///< 输入像素格式
    bool multiScaleOutput_ = false;  ///< 6-tensor DFL output mode
    int  numClasses_ = 1;       ///< Number of classes (volleyball=1)
    int  regMax_ = 16;          ///< DFL regression max bins

    // TensorRT 10.x tensor 名称
    std::string inputTensorName_;
    std::string outputTensorName_;    ///< Single-output mode only
    int numOutputTensors_ = 0;

    bool loadEngine(const std::string& path);
    bool allocateBuffers();
    void freeBuffers();

    /**
    * @brief CUDA 预处理: GRAY/BAYER/BGR -> RGB float32 CHW
     */
    void preprocessGPU(const void* gpuImageU8, int pitch,
                       int srcWidth, int srcHeight,
                       void* inputBufferDevice,
                       cudaStream_t stream);

    /**
     * @brief 后处理: 从 TRT 输出提取检测框
     */
    std::vector<Detection> postprocess(const float* outputBufferHost,
                                       float scaleX, float scaleY);

    /**
     * @brief Multi-scale DFL postprocess (CPU fallback)
     */
    std::vector<Detection> postprocessMultiScale(const BufferSet& buf,
                                                  float scaleX, float scaleY);

    /**
     * @brief Multi-scale DFL postprocess via CUDA kernel (GPU accelerated)
     *
     * GPU DFL kernel runs in enqueue(). This method reads the pre-computed
     * results from dflOutHost and applies NMS.
     */
    std::vector<Detection> collectGPUDFLResults(const BufferSet& buf);
    bool useGPUPostprocess_ = true;   ///< Use CUDA DFL decode (vs CPU)

    /**
     * @brief NMS
     */
    std::vector<Detection> nms(const std::vector<Detection>& dets);
    float computeIoU(const Detection& a, const Detection& b);
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_TRT_DETECTOR_H_
