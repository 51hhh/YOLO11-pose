# ⚡ 性能优化参考指南

> 整合自：NX性能优化指南.md + 性能优化详解.md  
> 更新时间：2026-01-26

---

## 🎯 性能优化总结（已验证 ✅）

### 当前性能指标
- **采集频率**: 100 Hz (PWM触发)
- **同步率**: 100% (1700帧对零失配)
- **推理延迟**: 9.5-12ms (Batch=2模式)
- **实际FPS**: 55-76 Hz
- **丢帧率**: 0.06% (近乎完美)

### 已实施的优化
1. ✅ **Batch=2 批量推理**: 节省 ~40% GPU调度开销
2. ✅ **回调模式采集**: 零等待延迟，CPU占用极低
3. ✅ **双缓冲零拷贝**: 避免数据竞争与内存分配
4. ✅ **PWM时间戳同步**: 帧号差≤3 且 时间差<25ms
5. ✅ **降低图像发布频率**: 原始10Hz，检测20Hz（减少80%带宽）

---

## 📊 性能瓶颈分析

### 延迟分解（实测）
```
预处理 (双路):  1.69 - 2.70 ms  (14-22%)
TensorRT推理:   7.84 - 9.49 ms  (65-78%)
后处理 (双路):  0.02 - 0.04 ms  (<1%)
立体匹配:       0.03 - 0.09 ms  (<1%)
卡尔曼追踪:     0.01 - 0.04 ms  (<1%)
─────────────────────────────────────────
总延迟:         9.55 - 12.20 ms
理论FPS:        82 - 105 Hz
```

### 瓶颈识别
1. **主要瓶颈**: TensorRT推理（占65-78%）
2. **次要瓶颈**: CPU预处理（占14-22%）
3. **非瓶颈**: 立体匹配、卡尔曼追踪（<1%）

---

## 🚀 进一步优化方向

### 1. TensorRT FP16 优化（预期提升30-50%）

**当前**: FP32 推理 7.84-9.49ms  
**预期**: FP16 推理 5-7ms

**实施步骤**:
```bash
# 重新导出ONNX模型
python export.py --weights yolo11n.pt --include onnx --simplify

# 转换为FP16 TensorRT引擎
trtexec --onnx=yolo11n.onnx \
        --fp16 \
        --workspace=4096 \
        --saveEngine=yolo11n_fp16_batch2.engine \
        --minShapes=images:2x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:2x3x640x640
```

**注意事项**:
- Jetson Orin NX 支持 FP16 Tensor Cores
- 精度损失通常 <1%
- 需要验证检测精度不下降

---

### 2. CUDA 预处理优化（预期节省1-2ms）

**当前**: CPU resize/normalize 1.69-2.70ms  
**预期**: GPU预处理 <1ms

**实施方案**:
```cuda
// yolo_preprocessor.cu 扩展
__global__ void preprocessKernel(
    const uint8_t* src,      // BGR8输入
    float* dst,              // NCHW float输出
    int width, int height,
    float mean[3], float std[3])
{
    // GPU并行执行: resize + normalize + layout转换
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Bilinear resize + normalize + BGR2RGB + HWC2CHW
        // 单次kernel完成所有预处理
    }
}
```

**优势**:
- 减少CPU↔GPU数据传输
- 利用GPU并行能力
- 零拷贝优化

---

### 3. CUDA Stream 异步流水线（预期提升15-25%）

**原理**: 重叠预处理、推理、后处理

```cpp
// 创建多个CUDA流
cudaStream_t stream_preprocess, stream_inference, stream_postprocess;
cudaStreamCreate(&stream_preprocess);
cudaStreamCreate(&stream_inference);
cudaStreamCreate(&stream_postprocess);

// 流水线执行
while (running_) {
    // 阶段1: 预处理帧N+1 (stream_preprocess)
    preprocessAsync(frame_n1, stream_preprocess);
    
    // 阶段2: 推理帧N (stream_inference)
    context_->enqueueV2(buffers, stream_inference, nullptr);
    
    // 阶段3: 后处理帧N-1 (stream_postprocess)
    postprocessAsync(result_n_1, stream_postprocess);
    
    // 同步关键点
    cudaStreamSynchronize(stream_postprocess);
}
```

**效果**:
- 三阶段并行执行
- 减少GPU空闲时间
- 预期总延迟降低15-25%

---

### 4. 动态 Batch 自适应（节省无目标时的开销）

**策略**:
```cpp
if (no_detection_for_10_frames) {
    // 降低为 Batch=1，只检测左相机
    batch_size = 1;
} else {
    // 恢复 Batch=2 双目检测
    batch_size = 2;
}
```

**优势**:
- 无目标时节省50%推理时间
- 有目标时保持双目精度
- 自适应负载

---

### 5. 多级 ROI 策略（减少计算量）

**三级检测**:
```
Level 1: 全图检测 640x640 (无目标时)
         ↓
Level 2: 粗ROI 320x320 (低置信度时)
         ↓
Level 3: 精细ROI 160x160 (高置信度追踪)
```

**效果**:
- Level 3 相比 Level 1 计算量减少 93.75%
- 追踪模式下推理时间可降至 2-3ms
- 需要可靠的追踪状态机

---

## 🔌 系统层面优化

### 电源管理

**问题**: 过流警告 `throttled duo over-current`

**解决方案**:
```bash
# 1. 使用官方电源适配器（19V 5A 90W+）

# 2. 设置性能模式
sudo nvpmodel -m 0  # MAXN模式
sudo jetson_clocks   # 锁定最高频率

# 3. 监控功耗
sudo tegrastats
```

**推荐电源**:
- 电压: 19V
- 电流: ≥5A
- 功率: ≥90W

---

### 散热优化

```bash
# 检查温度
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# 风扇控制（如果有）
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
```

**建议**:
- 添加主动散热风扇
- 确保通风良好
- 避免长时间满负载运行

---

### 系统调优

```bash
# 1. 增加 USB 缓冲区
sudo sh -c 'echo 1024 > /sys/module/usbcore/parameters/usbfs_memory_mb'

# 2. 关闭桌面环境（降低负载）
sudo systemctl stop gdm3

# 3. 实时优先级（需要root）
sudo chrt -f 50 ros2 run volleyball_stereo_driver volleyball_tracker_node

# 4. CPU亲和性绑定
taskset -c 4,5 ros2 run volleyball_stereo_driver volleyball_tracker_node
```

---

## 📈 优化优先级建议

### 短期（立即可做）
1. ✅ **FP16推理**: 最高ROI，30-50%性能提升
2. ✅ **降低图像发布频率**: 已实施，减少80%带宽
3. ✅ **系统调优**: 电源、散热、性能模式

### 中期（需要1-2周）
1. ⏳ **CUDA预处理**: 需要编写CUDA kernel
2. ⏳ **动态Batch**: 需要状态机逻辑
3. ⏳ **多级ROI**: 需要完善追踪策略

### 长期（需要重构）
1. ⏳ **CUDA Stream流水线**: 需要重构推理架构
2. ⏳ **端到端GPU**: 需要重写整个数据流

---

## 🎯 预期性能目标

### 当前性能（已达成）
- 端到端延迟: 9.5-12ms
- 实际FPS: 55-76 Hz
- 同步率: 100%

### FP16优化后（预期）
- 端到端延迟: 6-8ms
- 理论FPS: 120-165 Hz
- 实际FPS: 100 Hz（受PWM限制）

### 完全优化后（理论上限）
- 端到端延迟: 4-6ms
- 理论FPS: 165-250 Hz
- 实际FPS: 100 Hz（受PWM限制）

---

## ⚠️ 注意事项

1. **PWM频率上限**: 当前100Hz，受GPIO软件PWM限制
2. **USB带宽**: 双相机1440×1080@100Hz接近USB3.0上限
3. **内存带宽**: 图像拷贝开销不可忽略
4. **热节流**: 长时间高负载会触发温度保护

---

*整合文档 - 基于实测数据更新*  
*最后更新: 2026-01-26*
