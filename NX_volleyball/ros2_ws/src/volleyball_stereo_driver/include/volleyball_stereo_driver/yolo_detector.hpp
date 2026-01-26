/**
 * @file yolo_detector.hpp
 * @brief YOLO11n TensorRT 检测器 (支持双流并行推理)
 */

#ifndef VOLLEYBALL_STEREO_DRIVER__YOLO_DETECTOR_HPP_
#define VOLLEYBALL_STEREO_DRIVER__YOLO_DETECTOR_HPP_

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <utility>  // for std::pair
#include <cuda_runtime.h>

namespace volleyball {

struct Detection {
  float cx, cy;
  float width, height;
  float confidence;
  bool valid;

  Detection()
      : cx(0), cy(0), width(0), height(0), confidence(0), valid(false) {}
};

class YOLODetector {
public:
  YOLODetector(const std::string &engine_path, float conf_threshold = 0.5f,
               float nms_threshold = 0.4f);
  ~YOLODetector();

  Detection detectGlobal(const cv::Mat &image, int target_size = 640);
  Detection detectROI(const cv::Mat &roi, const cv::Point2f &offset,
                      int target_size = 320);
  
  // ✅ 双路检测 (自动选择 batch=2 或双流模式)
  std::pair<Detection, Detection> detectDual(
      const cv::Mat &image_left, const cv::Mat &image_right, int target_size = 640);
  
  // ✅ Batch=2 推理 (一次调用处理两张图)
  std::pair<Detection, Detection> detectBatch2(
      const cv::Mat &image_left, const cv::Mat &image_right, int target_size = 640);

  void setConfidenceThreshold(float threshold) { conf_threshold_ = threshold; }
  void setNMSThreshold(float threshold) { nms_threshold_ = threshold; }
  
  // 获取模型信息
  int getBatchSize() const { return batch_size_; }
  bool isBatch2Model() const { return batch_size_ >= 2; }

private:
  nvinfer1::IRuntime *runtime_;
  nvinfer1::ICudaEngine *engine_;
  nvinfer1::IExecutionContext *context_;
  nvinfer1::IExecutionContext *context2_;  // 第二个执行上下文 (双流模式备用)

  void *input_buffer_device_;
  void *output_buffer_device_;
  float *input_buffer_host_;
  float *output_buffer_host_;
  
  // 第二路缓冲区 (双流模式备用)
  void *input_buffer_device2_;
  void *output_buffer_device2_;
  float *output_buffer_host2_;
  
  // CUDA流和GPU缓冲
  cudaStream_t cuda_stream_;
  cudaStream_t cuda_stream2_;
  void *gpu_resize_buffer_;
  void *gpu_rgb_buffer_;
  
  // 预分配的GPU源图像缓冲区
  void *gpu_src_buffer_;
  void *gpu_src_buffer2_;
  size_t gpu_src_buffer_size_;
  
  // TensorRT 10.x tensor 名称
  std::string input_tensor_name_;
  std::string output_tensor_name_;

  int input_size_;
  int output_size_;       // 单batch输出大小
  int batch_size_;        // ✅ 模型batch大小
  float conf_threshold_;
  float nms_threshold_;

  bool loadEngine(const std::string &engine_path);
  bool allocateBuffers();
  void freeBuffers();
  void preprocess(const cv::Mat &image, int target_size);
  void preprocessGPU(const cv::Mat &image, int target_size);
  void preprocessGPUStream(const cv::Mat &image, int target_size, 
                           void* gpu_src, void* input_device, cudaStream_t stream);
  // ✅ Batch预处理: 两张图写入连续GPU内存
  void preprocessBatch2(const cv::Mat &image1, const cv::Mat &image2, int target_size);
  Detection postprocess(float scale_x, float scale_y);
  Detection postprocessBuffer(float* buffer, float scale_x, float scale_y);
  // ✅ Batch后处理: 从batch输出中提取第N个结果
  Detection postprocessBatchIndex(float* buffer, int batch_idx, float scale_x, float scale_y);
  std::vector<Detection> nms(const std::vector<Detection> &detections);
  float computeIoU(const Detection &a, const Detection &b);
};

} // namespace volleyball

#endif // VOLLEYBALL_STEREO_DRIVER__YOLO_DETECTOR_HPP_

