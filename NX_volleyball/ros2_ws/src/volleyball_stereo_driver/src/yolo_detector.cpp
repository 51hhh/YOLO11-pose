/**
 * @file yolo_detector.cpp
 * @brief YOLO11n TensorRT 检测器实现（目标检测版本）
 */

#include "volleyball_stereo_driver/yolo_detector.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

// 外部CUDA函数声明
extern "C" void launchPreprocessKernel(const unsigned char* src, float* dst,
                                        int src_w, int src_h, int dst_w, int dst_h,
                                        cudaStream_t stream);

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TensorRT] " << msg << std::endl;
    }
  }
};

static Logger gLogger;

namespace volleyball {

YOLODetector::YOLODetector(const std::string &engine_path, float conf_threshold,
                           float nms_threshold)
    : runtime_(nullptr), engine_(nullptr), context_(nullptr), context2_(nullptr),
      input_buffer_device_(nullptr), output_buffer_device_(nullptr),
      input_buffer_host_(nullptr), output_buffer_host_(nullptr),
      input_buffer_device2_(nullptr), output_buffer_device2_(nullptr),
      output_buffer_host2_(nullptr),
      cuda_stream_(nullptr), cuda_stream2_(nullptr),
      gpu_resize_buffer_(nullptr), gpu_rgb_buffer_(nullptr),
      gpu_src_buffer_(nullptr), gpu_src_buffer2_(nullptr), gpu_src_buffer_size_(0),
      input_size_(640), output_size_(0), batch_size_(1),
      conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
  std::cout << "🎯 初始化 YOLO 检测器..." << std::endl;
  std::cout << "   模型: " << engine_path << std::endl;
  std::cout << "   置信度阈值: " << conf_threshold_ << std::endl;
  std::cout << "   NMS 阈值: " << nms_threshold_ << std::endl;
  
  // 创建两个CUDA流用于双路并行处理
  cudaStreamCreate(&cuda_stream_);
  cudaStreamCreate(&cuda_stream2_);
  std::cout << "✅ 双CUDA流已创建 (并行推理)" << std::endl;

  if (loadEngine(engine_path)) {
    std::cout << "✅ YOLO 检测器初始化成功" << std::endl;
  } else {
    std::cerr << "❌ YOLO 检测器初始化失败" << std::endl;
  }
}

YOLODetector::~YOLODetector() {
  freeBuffers();
  
  if (cuda_stream_) {
    cudaStreamSynchronize(cuda_stream_);
    cudaStreamDestroy(cuda_stream_);
  }
  if (cuda_stream2_) {
    cudaStreamSynchronize(cuda_stream2_);
    cudaStreamDestroy(cuda_stream2_);
  }

  if (context_) {
    delete context_;
  }
  if (context2_) {
    delete context2_;
  }
  if (engine_) {
    delete engine_;
  }
  if (runtime_) {
    delete runtime_;
  }
}

bool YOLODetector::loadEngine(const std::string &engine_path) {
  // 读取 Engine 文件
  std::ifstream file(engine_path, std::ios::binary);
  if (!file.good()) {
    std::cerr << "❌ 无法打开引擎文件: " << engine_path << std::endl;
    return false;
  }

  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  std::cout << "📦 加载引擎文件: " << size / (1024.0 * 1024.0) << " MB"
            << std::endl;

  // 创建 Runtime
  runtime_ = nvinfer1::createInferRuntime(gLogger);
  if (!runtime_) {
    std::cerr << "❌ 创建 Runtime 失败" << std::endl;
    return false;
  }

  // 反序列化 Engine
  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
  if (!engine_) {
    std::cerr << "❌ 反序列化 Engine 失败" << std::endl;
    return false;
  }

  // 创建执行上下文
  context_ = engine_->createExecutionContext();
  if (!context_) {
    std::cerr << "❌ 创建执行上下文失败" << std::endl;
    return false;
  }

  // 获取输入输出 tensor 名称 (TensorRT 10.x)
  int num_io_tensors = engine_->getNbIOTensors();
  std::cout << "   IO Tensors: " << num_io_tensors << std::endl;
  
  for (int i = 0; i < num_io_tensors; i++) {
    const char* name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    auto dims = engine_->getTensorShape(name);
    
    std::cout << "     [" << i << "] " << name << " - ";
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      std::cout << "INPUT";
      input_tensor_name_ = name;
      // ✅ 从输入维度获取batch大小
      if (dims.nbDims >= 1 && dims.d[0] > 0) {
        batch_size_ = dims.d[0];
      } else {
        // 动态 batch 或维度为 -1，默认为 1
        std::cout << " (动态batch，默认使用 batch=1)";
        batch_size_ = 1;
      }
    } else {
      std::cout << "OUTPUT";
      output_tensor_name_ = name;
    }
    std::cout << " [";
    for (int d = 0; d < dims.nbDims; d++) {
      std::cout << dims.d[d];
      if (d < dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  // 获取输入输出维度
  // YOLOv11n: 输入 [batch, 3, 640, 640], 输出 [batch, 5, 8400]
  input_size_ = 640;
  output_size_ = 5 * 8400; // 单batch: [cx, cy, w, h, conf] × 8400

  // 分配缓冲区
  allocateBuffers();

  std::cout << "✅ Engine 加载成功" << std::endl;
  std::cout << "   Batch 大小: " << batch_size_ << std::endl;
  std::cout << "   输入尺寸: " << input_size_ << "x" << input_size_ << std::endl;
  std::cout << "   输出尺寸: 5x8400 (单类别模型)" << std::endl;
  
  if (batch_size_ >= 2) {
    std::cout << "🚀 检测到固定 Batch=" << batch_size_ << " 模型，启用批量推理模式" << std::endl;
  } else {
    std::cout << "⚡ Batch=1 模型，启用双流并行模式" << std::endl;
  }

  return true;
}

void YOLODetector::allocateBuffers() {
  // ✅ 确保 batch_size 至少为 1
  if (batch_size_ <= 0) {
    std::cout << "⚠️  检测到动态 batch (size=" << batch_size_ << ")，强制设为 1" << std::endl;
    batch_size_ = 1;
  }
  
  // 单张图输入大小
  size_t single_input_bytes = 3 * input_size_ * input_size_ * sizeof(float);
  size_t single_output_bytes = output_size_ * sizeof(float);
  
  // ✅ 根据batch大小分配缓冲区
  size_t input_size_bytes = batch_size_ * single_input_bytes;
  size_t output_size_bytes = batch_size_ * single_output_bytes;
  
  cudaMalloc(&input_buffer_device_, input_size_bytes);
  cudaMallocHost(&input_buffer_host_, input_size_bytes);  // pinned memory

  cudaMalloc(&output_buffer_device_, output_size_bytes);
  cudaMallocHost(&output_buffer_host_, output_size_bytes);  // pinned memory
  
  // ========== 双流并行推理备用：第二套缓冲区 (仅 batch=1 时使用) ==========
  if (batch_size_ == 1) {
    cudaMalloc(&input_buffer_device2_, single_input_bytes);
    cudaMalloc(&output_buffer_device2_, single_output_bytes);
    cudaMallocHost(&output_buffer_host2_, single_output_bytes);
    
    // 创建第二个执行上下文
    context2_ = engine_->createExecutionContext();
    if (!context2_) {
      std::cerr << "❌ 创建第二执行上下文失败" << std::endl;
    } else {
      std::cout << "✅ 双执行上下文已创建 (双流并行模式)" << std::endl;
    }
  }
  
  // GPU预处理缓冲 (resize + RGB)
  size_t resize_bytes = input_size_ * input_size_ * 3 * sizeof(unsigned char);
  cudaMalloc(&gpu_resize_buffer_, resize_bytes);
  cudaMalloc(&gpu_rgb_buffer_, resize_bytes);
  
  // ✅ 预分配GPU源图像缓冲区 (1440x1080x3 = 4.67MB, 预留6MB)
  gpu_src_buffer_size_ = 6 * 1024 * 1024;  // 6MB足够大多数分辨率
  cudaMalloc(&gpu_src_buffer_, gpu_src_buffer_size_);
  cudaMalloc(&gpu_src_buffer2_, gpu_src_buffer_size_);  // 第二路源图像缓冲

  std::cout << "📊 缓冲区分配 (batch=" << batch_size_ << "):" << std::endl;
  std::cout << "   输入: " << input_size_bytes / (1024.0 * 1024.0) << " MB (pinned)" << std::endl;
  std::cout << "   输出: " << output_size_bytes / (1024.0 * 1024.0) << " MB (pinned)" << std::endl;
  std::cout << "   GPU源图像缓冲x2: " << gpu_src_buffer_size_ * 2 / (1024.0 * 1024.0) << " MB" << std::endl;
}

void YOLODetector::freeBuffers() {
  if (input_buffer_device_) {
    cudaFree(input_buffer_device_);
    input_buffer_device_ = nullptr;
  }
  if (output_buffer_device_) {
    cudaFree(output_buffer_device_);
    output_buffer_device_ = nullptr;
  }
  // 第二套缓冲区
  if (input_buffer_device2_) {
    cudaFree(input_buffer_device2_);
    input_buffer_device2_ = nullptr;
  }
  if (output_buffer_device2_) {
    cudaFree(output_buffer_device2_);
    output_buffer_device2_ = nullptr;
  }
  if (output_buffer_host2_) {
    cudaFreeHost(output_buffer_host2_);
    output_buffer_host2_ = nullptr;
  }
  if (gpu_resize_buffer_) {
    cudaFree(gpu_resize_buffer_);
    gpu_resize_buffer_ = nullptr;
  }
  if (gpu_rgb_buffer_) {
    cudaFree(gpu_rgb_buffer_);
    gpu_rgb_buffer_ = nullptr;
  }
  if (gpu_src_buffer_) {
    cudaFree(gpu_src_buffer_);
    gpu_src_buffer_ = nullptr;
    gpu_src_buffer_size_ = 0;
  }
  if (gpu_src_buffer2_) {
    cudaFree(gpu_src_buffer2_);
    gpu_src_buffer2_ = nullptr;
  }
  if (input_buffer_host_) {
    cudaFreeHost(input_buffer_host_);  // 释放pinned memory
    input_buffer_host_ = nullptr;
  }
  if (output_buffer_host_) {
    cudaFreeHost(output_buffer_host_);  // 释放pinned memory
    output_buffer_host_ = nullptr;
  }
}

// CPU预处理 (保留备用)
void YOLODetector::preprocess(const cv::Mat &image, int target_size) {
  // Resize
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(target_size, target_size));

  // BGR -> RGB
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  // 归一化到 [0, 1]
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  // 转换为 CHW 格式
  int idx = 0;
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < target_size; ++h) {
      for (int w = 0; w < target_size; ++w) {
        input_buffer_host_[idx++] = rgb.at<cv::Vec3f>(h, w)[c];
      }
    }
  }
}

Detection YOLODetector::postprocess(float scale_x, float scale_y) {
  Detection best_det;
  best_det.valid = false;

  // 单类别 YOLOv11 输出格式: [5, 8400]
  // 每列: [cx, cy, w, h, conf]
  // 内存布局: [所有8400个cx, 所有8400个cy, 所有8400个w, 所有8400个h, 所有8400个conf]

  float max_conf = 0.0f;
  int best_idx = -1;

  // 遍历所有 8400 个检测框
  for (int i = 0; i < 8400; ++i) {
    // 获取置信度 (第 5 行，即索引 4*8400 + i)
    float conf = output_buffer_host_[4 * 8400 + i];

    if (conf > conf_threshold_ && conf > max_conf) {
      max_conf = conf;
      best_idx = i;
    }
  }

  if (best_idx >= 0) {
    // 提取 bbox (按行存储: [cx行, cy行, w行, h行, conf行])
    float cx = output_buffer_host_[0 * 8400 + best_idx] * scale_x;
    float cy = output_buffer_host_[1 * 8400 + best_idx] * scale_y;
    float w = output_buffer_host_[2 * 8400 + best_idx] * scale_x;
    float h = output_buffer_host_[3 * 8400 + best_idx] * scale_y;

    best_det.cx = cx;
    best_det.cy = cy;
    best_det.width = w;
    best_det.height = h;
    best_det.confidence = max_conf;
    best_det.valid = true;
  }

  return best_det;
}

// GPU预处理实现 (调用CUDA kernel)
void YOLODetector::preprocessGPU(const cv::Mat &image, int target_size) {
  size_t src_bytes = image.rows * image.cols * 3 * sizeof(unsigned char);
  
  // ✅ 使用预分配缓冲区，避免每帧malloc/free
  // 检查缓冲区是否足够大
  if (src_bytes > gpu_src_buffer_size_) {
    // 需要扩展缓冲区 (罕见情况)
    cudaFree(gpu_src_buffer_);
    gpu_src_buffer_size_ = src_bytes * 2;  // 分配2倍大小避免频繁扩展
    cudaMalloc(&gpu_src_buffer_, gpu_src_buffer_size_);
    std::cout << "⚠️  GPU源缓冲区扩展至: " << gpu_src_buffer_size_ / (1024.0 * 1024.0) << " MB" << std::endl;
  }
  
  // 异步上传原始图像到预分配的GPU缓冲区
  cudaMemcpyAsync(gpu_src_buffer_, image.data, src_bytes, cudaMemcpyHostToDevice, cuda_stream_);
  
  // 调用CUDA kernel: 直接写入input_buffer_device_
  launchPreprocessKernel(
    (unsigned char*)gpu_src_buffer_, (float*)input_buffer_device_,
    image.cols, image.rows, target_size, target_size,
    cuda_stream_
  );
  
  // 不需要同步 - 后续推理会在同一stream上进行，自动保证顺序
}

Detection YOLODetector::detectGlobal(const cv::Mat &image, int target_size) {
  if (image.empty() || !context_) {
    return Detection();
  }

  // 性能分析：预处理
  auto t1 = std::chrono::high_resolution_clock::now();
  
  // GPU加速预处理 (直接写入input_buffer_device_，使用预分配缓冲区)
  preprocessGPU(image, target_size);
  
  // 同步预处理完成，以便准确测量推理时间
  cudaStreamSynchronize(cuda_stream_);
  auto t2 = std::chrono::high_resolution_clock::now();

  // 设置输入输出tensor地址 (TensorRT 10.x API)
  context_->setTensorAddress(input_tensor_name_.c_str(), input_buffer_device_);
  context_->setTensorAddress(output_tensor_name_.c_str(), output_buffer_device_);
  
  // 异步推理 (TensorRT 10.x 使用 enqueueV3)
  context_->enqueueV3(cuda_stream_);
  
  // 异步拷贝回CPU (使用pinned memory)
  size_t output_size_bytes = output_size_ * sizeof(float);
  cudaMemcpyAsync(output_buffer_host_, output_buffer_device_, output_size_bytes,
                  cudaMemcpyDeviceToHost, cuda_stream_);
  
  // 同步等待推理和D2H完成
  cudaStreamSynchronize(cuda_stream_);
  
  auto t3 = std::chrono::high_resolution_clock::now();

  // 后处理
  float scale_x = static_cast<float>(image.cols) / target_size;
  float scale_y = static_cast<float>(image.rows) / target_size;

  Detection result = postprocess(scale_x, scale_y);
  
  auto t4 = std::chrono::high_resolution_clock::now();
  
  // 每100帧打印一次性能统计
  static int call_count = 0;
  static double total_preprocess = 0, total_inference = 0, total_postprocess = 0;
  
  double preprocess_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  double inference_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();  // 包含推理+D2H
  double postprocess_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
  
  total_preprocess += preprocess_ms;
  total_inference += inference_ms;
  total_postprocess += postprocess_ms;
  call_count++;
  
  if (call_count % 200 == 0) {  // 每200帧打印一次 (左右相机共200次调用=100帧)
    double avg_preprocess = total_preprocess / call_count;
    double avg_inference = total_inference / call_count;
    double avg_postprocess = total_postprocess / call_count;
    double avg_total = avg_preprocess + avg_inference + avg_postprocess;
    
    std::cout << "\n🔍 YOLO检测性能分析 [" << call_count << "次调用]:" << std::endl;
    std::cout << "   预处理:     " << avg_preprocess << "ms (" 
              << (avg_preprocess/avg_total*100) << "%)" << std::endl;
    std::cout << "   推理+D2H:   " << avg_inference << "ms (" 
              << (avg_inference/avg_total*100) << "%)" << std::endl;
    std::cout << "   后处理:     " << avg_postprocess << "ms (" 
              << (avg_postprocess/avg_total*100) << "%)" << std::endl;
    std::cout << "   单路总计:   " << avg_total << "ms" << std::endl;
    std::cout << "   双路总计:   " << avg_total * 2 << "ms (可优化为batch推理)" << std::endl;
    
    // 重置统计
    call_count = 0;
    total_preprocess = 0;
    total_inference = 0;
    total_postprocess = 0;
  }

  return result;
}

Detection YOLODetector::detectROI(const cv::Mat &roi, const cv::Point2f &offset,
                                  int target_size) {
  Detection det = detectGlobal(roi, target_size);

  if (det.valid) {
    // 坐标还原到原图
    det.cx += offset.x;
    det.cy += offset.y;
  }

  return det;
}

std::vector<Detection>
YOLODetector::nms(const std::vector<Detection> &detections) {
  std::vector<Detection> result;

  if (detections.empty()) {
    return result;
  }

  // 按置信度排序
  std::vector<Detection> sorted = detections;
  std::sort(sorted.begin(), sorted.end(),
            [](const Detection &a, const Detection &b) {
              return a.confidence > b.confidence;
            });

  // NMS
  std::vector<bool> suppressed(sorted.size(), false);

  for (size_t i = 0; i < sorted.size(); ++i) {
    if (suppressed[i])
      continue;

    result.push_back(sorted[i]);

    for (size_t j = i + 1; j < sorted.size(); ++j) {
      if (suppressed[j])
        continue;

      float iou = computeIoU(sorted[i], sorted[j]);
      if (iou > nms_threshold_) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

float YOLODetector::computeIoU(const Detection &a, const Detection &b) {
  float x1_a = a.cx - a.width / 2;
  float y1_a = a.cy - a.height / 2;
  float x2_a = a.cx + a.width / 2;
  float y2_a = a.cy + a.height / 2;

  float x1_b = b.cx - b.width / 2;
  float y1_b = b.cy - b.height / 2;
  float x2_b = b.cx + b.width / 2;
  float y2_b = b.cy + b.height / 2;

  float x1_inter = std::max(x1_a, x1_b);
  float y1_inter = std::max(y1_a, y1_b);
  float x2_inter = std::min(x2_a, x2_b);
  float y2_inter = std::min(y2_a, y2_b);

  float inter_area =
      std::max(0.0f, x2_inter - x1_inter) * std::max(0.0f, y2_inter - y1_inter);

  float area_a = a.width * a.height;
  float area_b = b.width * b.height;
  float union_area = area_a + area_b - inter_area;

  return inter_area / (union_area + 1e-6f);
}

// ==================== 双流并行推理实现 ====================

// GPU预处理到指定流和缓冲区
void YOLODetector::preprocessGPUStream(const cv::Mat &image, int target_size,
                                        void* gpu_src, void* gpu_input,
                                        cudaStream_t stream) {
  size_t src_bytes = image.rows * image.cols * 3 * sizeof(unsigned char);
  
  // 异步上传原始图像到GPU缓冲区
  cudaMemcpyAsync(gpu_src, image.data, src_bytes, cudaMemcpyHostToDevice, stream);
  
  // 调用CUDA kernel: 直接写入指定的input缓冲区
  launchPreprocessKernel(
    (unsigned char*)gpu_src, (float*)gpu_input,
    image.cols, image.rows, target_size, target_size,
    stream
  );
}

// ==================== Batch=2 预处理 (双流并行) ====================
// 两张图并行上传+处理，写入连续GPU内存: [img1, img2] → input_buffer_device_
void YOLODetector::preprocessBatch2(const cv::Mat &image1, const cv::Mat &image2, int target_size) {
  size_t single_input_size = 3 * target_size * target_size * sizeof(float);
  
  // 图1 → 流1, 偏移0
  float* dst1 = (float*)input_buffer_device_;
  preprocessGPUStream(image1, target_size, gpu_src_buffer_, dst1, cuda_stream_);
  
  // 图2 → 流2, 偏移 single_input_size (并行执行!)
  float* dst2 = (float*)((char*)input_buffer_device_ + single_input_size);
  preprocessGPUStream(image2, target_size, gpu_src_buffer2_, dst2, cuda_stream2_);
  
  // ✅ 等待两个流都完成预处理
  cudaStreamSynchronize(cuda_stream_);
  cudaStreamSynchronize(cuda_stream2_);
}

// 从指定缓冲区后处理
Detection YOLODetector::postprocessBuffer(float* buffer, float scale_x, float scale_y) {
  Detection best_det;
  best_det.valid = false;

  float max_conf = 0.0f;
  int best_idx = -1;

  // 遍历所有 8400 个检测框
  for (int i = 0; i < 8400; ++i) {
    float conf = buffer[4 * 8400 + i];
    if (conf > conf_threshold_ && conf > max_conf) {
      max_conf = conf;
      best_idx = i;
    }
  }

  if (best_idx >= 0) {
    float cx = buffer[0 * 8400 + best_idx] * scale_x;
    float cy = buffer[1 * 8400 + best_idx] * scale_y;
    float w = buffer[2 * 8400 + best_idx] * scale_x;
    float h = buffer[3 * 8400 + best_idx] * scale_y;

    best_det.cx = cx;
    best_det.cy = cy;
    best_det.width = w;
    best_det.height = h;
    best_det.confidence = max_conf;
    best_det.valid = true;
  }

  return best_det;
}

// ==================== Batch后处理 ====================
// 从batch输出中提取第N个结果
// 输出格式: [batch, 5, 8400] → 内存布局: batch0_所有数据, batch1_所有数据
Detection YOLODetector::postprocessBatchIndex(float* buffer, int batch_idx, float scale_x, float scale_y) {
  // 每个batch的输出大小: 5 * 8400
  size_t batch_offset = batch_idx * output_size_;
  float* batch_buffer = buffer + batch_offset;
  
  return postprocessBuffer(batch_buffer, scale_x, scale_y);
}

// ==================== Batch=2 推理 ====================
std::pair<Detection, Detection> YOLODetector::detectBatch2(
    const cv::Mat &image_left, const cv::Mat &image_right, int target_size) {
  
  if (image_left.empty() || image_right.empty() || !context_) {
    return {Detection(), Detection()};
  }
  
  if (batch_size_ < 2) {
    std::cerr << "❌ 模型不支持 batch=2，请使用 detectDual" << std::endl;
    return {Detection(), Detection()};
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  // ========== 阶段1: Batch预处理 (两张图 → 连续GPU内存) ==========
  // 同一 stream 内顺序执行，不需要同步，后续 enqueueV3 会自动串联
  preprocessBatch2(image_left, image_right, target_size);
  auto t_preprocess = std::chrono::high_resolution_clock::now();

  // ========== 阶段2: 单次Batch=2推理 ==========
  context_->setTensorAddress(input_tensor_name_.c_str(), input_buffer_device_);
  context_->setTensorAddress(output_tensor_name_.c_str(), output_buffer_device_);
  context_->enqueueV3(cuda_stream_);
  
  // ========== 阶段3: D2H传输 ==========
  size_t output_size_bytes = batch_size_ * output_size_ * sizeof(float);
  cudaMemcpyAsync(output_buffer_host_, output_buffer_device_, output_size_bytes,
                  cudaMemcpyDeviceToHost, cuda_stream_);
  
  cudaStreamSynchronize(cuda_stream_);
  auto t_gpu = std::chrono::high_resolution_clock::now();

  // ========== 阶段4: 后处理两个batch ==========
  float scale_x_left = static_cast<float>(image_left.cols) / target_size;
  float scale_y_left = static_cast<float>(image_left.rows) / target_size;
  float scale_x_right = static_cast<float>(image_right.cols) / target_size;
  float scale_y_right = static_cast<float>(image_right.rows) / target_size;

  Detection det_left = postprocessBatchIndex(output_buffer_host_, 0, scale_x_left, scale_y_left);
  Detection det_right = postprocessBatchIndex(output_buffer_host_, 1, scale_x_right, scale_y_right);

  auto t_end = std::chrono::high_resolution_clock::now();

  // 性能统计
  static int batch_call_count = 0;
  static double total_preprocess = 0, total_inference = 0, total_post = 0;
  
  double preprocess_ms = std::chrono::duration<double, std::milli>(t_preprocess - t_start).count();
  double inference_ms = std::chrono::duration<double, std::milli>(t_gpu - t_preprocess).count();
  double post_ms = std::chrono::duration<double, std::milli>(t_end - t_gpu).count();
  
  total_preprocess += preprocess_ms;
  total_inference += inference_ms;
  total_post += post_ms;
  batch_call_count++;
  
  if (batch_call_count % 100 == 0) {
    double avg_preprocess = total_preprocess / batch_call_count;
    double avg_inference = total_inference / batch_call_count;
    double avg_post = total_post / batch_call_count;
    double avg_total = avg_preprocess + avg_inference + avg_post;
    
    std::cout << "\n🚀 Batch=2 推理性能 [" << batch_call_count << "帧]:" << std::endl;
    std::cout << "   预处理x2:   " << avg_preprocess << "ms" << std::endl;
    std::cout << "   推理+D2H:   " << avg_inference << "ms" << std::endl;
    std::cout << "   后处理x2:   " << avg_post << "ms" << std::endl;
    std::cout << "   双路总计:   " << avg_total << "ms" << std::endl;
    std::cout << "   理论FPS:    " << (1000.0 / avg_total) << " Hz" << std::endl;
    
    batch_call_count = 0;
    total_preprocess = 0;
    total_inference = 0;
    total_post = 0;
  }

  return {det_left, det_right};
}

// ==================== 双路检测 (自动选择模式) ====================
std::pair<Detection, Detection> YOLODetector::detectDual(
    const cv::Mat &image_left, const cv::Mat &image_right, int target_size) {
  
  // ✅ 如果是 batch=2 模型，使用批量推理
  if (batch_size_ >= 2) {
    return detectBatch2(image_left, image_right, target_size);
  }
  
  // ========== 以下是 batch=1 的双流并行模式 ==========
  if (image_left.empty() || image_right.empty() || !context_ || !context2_) {
    return {Detection(), Detection()};
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  // 阶段1: 双路并行预处理 (异步)
  preprocessGPUStream(image_left, target_size, 
                      gpu_src_buffer_, input_buffer_device_, cuda_stream_);
  preprocessGPUStream(image_right, target_size,
                      gpu_src_buffer2_, input_buffer_device2_, cuda_stream2_);

  // 阶段2: 双路并行推理 (异步)
  context_->setTensorAddress(input_tensor_name_.c_str(), input_buffer_device_);
  context_->setTensorAddress(output_tensor_name_.c_str(), output_buffer_device_);
  context_->enqueueV3(cuda_stream_);
  
  context2_->setTensorAddress(input_tensor_name_.c_str(), input_buffer_device2_);
  context2_->setTensorAddress(output_tensor_name_.c_str(), output_buffer_device2_);
  context2_->enqueueV3(cuda_stream2_);

  // 阶段3: 双路并行D2H传输 (异步)
  size_t output_size_bytes = output_size_ * sizeof(float);
  cudaMemcpyAsync(output_buffer_host_, output_buffer_device_, output_size_bytes,
                  cudaMemcpyDeviceToHost, cuda_stream_);
  cudaMemcpyAsync(output_buffer_host2_, output_buffer_device2_, output_size_bytes,
                  cudaMemcpyDeviceToHost, cuda_stream2_);

  // 阶段4: 等待两个流完成
  cudaStreamSynchronize(cuda_stream_);
  cudaStreamSynchronize(cuda_stream2_);

  auto t_gpu = std::chrono::high_resolution_clock::now();

  // 阶段5: 双路后处理
  float scale_x_left = static_cast<float>(image_left.cols) / target_size;
  float scale_y_left = static_cast<float>(image_left.rows) / target_size;
  float scale_x_right = static_cast<float>(image_right.cols) / target_size;
  float scale_y_right = static_cast<float>(image_right.rows) / target_size;

  Detection det_left = postprocessBuffer(output_buffer_host_, scale_x_left, scale_y_left);
  Detection det_right = postprocessBuffer(output_buffer_host2_, scale_x_right, scale_y_right);

  auto t_end = std::chrono::high_resolution_clock::now();

  // 性能统计
  static int dual_call_count = 0;
  static double total_gpu_time = 0, total_post_time = 0;
  
  double gpu_ms = std::chrono::duration<double, std::milli>(t_gpu - t_start).count();
  double post_ms = std::chrono::duration<double, std::milli>(t_end - t_gpu).count();
  
  total_gpu_time += gpu_ms;
  total_post_time += post_ms;
  dual_call_count++;
  
  if (dual_call_count % 100 == 0) {
    double avg_gpu = total_gpu_time / dual_call_count;
    double avg_post = total_post_time / dual_call_count;
    double avg_total = avg_gpu + avg_post;
    
    std::cout << "\n🚀 双流并行推理性能 [" << dual_call_count << "帧]:" << std::endl;
    std::cout << "   GPU并行(预处理+推理+D2H): " << avg_gpu << "ms" << std::endl;
    std::cout << "   后处理x2:                " << avg_post << "ms" << std::endl;
    std::cout << "   双路总计:                " << avg_total << "ms" << std::endl;
    
    dual_call_count = 0;
    total_gpu_time = 0;
    total_post_time = 0;
  }

  return {det_left, det_right};
}

} // namespace volleyball
