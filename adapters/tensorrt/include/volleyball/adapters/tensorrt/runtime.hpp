/**
 * @file runtime.hpp
 * @brief TensorRT 检测运行时（支持 NVDLA、GPU fallback 和三槽异步执行）。
 *
 * 输入是调用方持有的 CUDA 图像地址；adapter 管理 TensorRT runtime、engine、
 * execution context、I/O buffer 和 detector CUDA kernel。
 */

#ifndef VOLLEYBALL_ADAPTERS_TENSORRT_RUNTIME_HPP_
#define VOLLEYBALL_ADAPTERS_TENSORRT_RUNTIME_HPP_

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <array>
#include <string>
#include <vector>

#include "volleyball/adapters/tensorrt/engine_contract.hpp"
#include "volleyball/adapters/tensorrt/postprocess.hpp"

namespace volleyball::adapters::tensorrt {

class TensorRtRuntime {
public:
    static constexpr int kBufferCount = 3;

    enum class InputFormat {
        GRAY,   ///< 单通道灰度
        BAYER,  ///< BayerRG8 单通道马赛克
        BGR     ///< 三通道 BGR
    };

    TensorRtRuntime();
    ~TensorRtRuntime();

    TensorRtRuntime(const TensorRtRuntime&) = delete;
    TensorRtRuntime& operator=(const TensorRtRuntime&) = delete;

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
    bool init(const std::string& engineFile,
              bool useDLA,
              int dlaCore,
              float confThreshold,
              float nmsThreshold,
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

    /** 同步执行并区分合法零检测与运行时失败。 */
    [[nodiscard]] PostprocessResult detectChecked(
        const void* gpuImageU8,
        int pitch,
        int width,
        int height,
        cudaStream_t stream);

    /**
     * @brief 异步提交一次推理 (不阻塞)
     * @param slotId 调用方异步槽位 ID `[0, kBufferCount)`
     */
    bool enqueue(int slotId,
                 const void* gpuImageU8, int pitch,
                 int width, int height,
                 cudaStream_t stream);

    /**
     * @brief 从指定槽位回收推理结果并做后处理
     *
     * 调用方需保证对应 stream 的推理和 D2H 已完成 (例如通过 cudaEvent 等待)。
     */
    std::vector<Detection> collect(int slotId, int width, int height);

    /** 回收异步结果，并保留后处理失败语义。 */
    [[nodiscard]] PostprocessResult collectChecked(
        int slotId,
        int width,
        int height);

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
    // One execution context per ring slot. TensorRT tensor addresses are
    // context state, so sharing one context across asynchronous slot enqueues
    // can race when the next frame updates bindings before the previous one
    // has finished.
    std::array<nvinfer1::IExecutionContext*, kBufferCount> contexts_{};

    // 每个 RingSlot 一套 I/O 缓冲, 避免异步覆盖
    std::array<BufferSet, kBufferCount> buffers_;

    // 参数
    int inputSize_   = 640;     ///< 模型输入正方形尺寸
    int outputSize_  = 0;       ///< 输出 tensor 元素数 (single-output mode)
    float confThreshold_ = 0.5f;
    float nmsThreshold_  = 0.4f;
    bool useDLA_    = true;
    int  dlaCore_   = 0;
    bool useLetterbox_ = true;  ///< 使用 letterbox 预处理 (保持宽高比)
    InputFormat inputFormat_ = InputFormat::GRAY;  ///< 输入像素格式
    int singleOutputClassCount_ = 0;
    SingleOutputLayout singleOutputLayout_ = SingleOutputLayout::none;
    bool multiScaleOutput_ = false;  ///< 6-tensor DFL output mode
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
    PostprocessResult postprocess(
        const float* outputBufferHost,
        float scaleX,
        float scaleY);

    /**
     * @brief Multi-scale DFL postprocess (CPU fallback)
     */
    PostprocessResult postprocessMultiScale(
        const BufferSet& buf,
        float scaleX,
        float scaleY);

    /**
     * @brief Multi-scale DFL postprocess via CUDA kernel (GPU accelerated)
     *
     * GPU DFL kernel runs in enqueue(). This method reads the pre-computed
     * results from dflOutHost and applies NMS.
     */
    PostprocessResult collectGPUDFLResults(const BufferSet& buf);
    bool useGPUPostprocess_ = true;   ///< Use CUDA DFL decode (vs CPU)

};

}  // namespace volleyball::adapters::tensorrt

#endif  // VOLLEYBALL_ADAPTERS_TENSORRT_RUNTIME_HPP_
