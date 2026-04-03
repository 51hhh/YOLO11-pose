# 问题速查：ONNX 模型兼容性

> 模型: CREStereo 480×640 · HITNet eth3d 480×640

## DL 模型输入格式

| 模型 | 输入 | 形状 | 说明 |
|---|---|---|---|
| CREStereo | left, right | 2×(1,3,480,640) | 双输入，RGB float32 |
| HITNet | input | (1,2,480,640) 或 (1,6,480,640) | 左右 concat，灰度或RGB |

## CREStereo 常见问题

### Resize 导致崩溃
输入图像必须 resize 到 480×640，不匹配会触发 ONNX shape 推理失败：
```python
left = cv2.resize(left, (640, 480))  # 注意 (W, H) 顺序
```

### 推理超慢 (12s)
PyTorch ONNX Runtime CPU 推理约 12s，**必须 TensorRT 加速**：
```bash
trtexec --onnx=crestereo.onnx --saveEngine=crestereo_fp16.engine --fp16
```

## HITNet 常见问题

### Channel mismatch
HITNet 模型有两种变体：
- 2 通道输入 → 灰度左右 concat
- 6 通道输入 → RGB 左右 concat

检测方式：
```python
session = ort.InferenceSession("hitnet.onnx")
channels = session.get_inputs()[0].shape[1]  # 2 or 6
```

### 深度范围偏差
HITNet 输出 disparity 单位依赖模型训练分辨率。若模型在 480×640 训练但推理用 720×1280，需缩放：
```python
scale = actual_width / model_width
disparity = raw_disparity * scale
```

## C++ ONNX Runtime 集成

### 条件编译
```cmake
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/local/include/onnxruntime)
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    add_definitions(-DHAS_ONNXRUNTIME)
endif()
```

### CUDA EP 降级
NX 上 CUDA EP 可能不可用（版本不匹配），代码需 fallback：
```cpp
try {
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(opts, 0));
} catch (...) {
    // 自动 fallback 到 CPU
}
```
