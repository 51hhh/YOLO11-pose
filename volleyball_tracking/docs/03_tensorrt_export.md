# ⚡ TensorRT 转换与优化指南

## TensorRT 简介

TensorRT 是 NVIDIA 的高性能深度学习推理优化器，可将模型推理速度提升 2-10 倍。

### 优化技术

- **层融合**: 合并卷积+BN+激活
- **精度校准**: FP16/INT8 量化
- **内核自动调优**: 针对硬件优化
- **动态张量内存**: 减少内存占用

---

## 环境准备

### Orin NX 环境

```bash
# 检查 JetPack 版本
sudo apt-cache show nvidia-jetpack

# 推荐: JetPack 5.1.2 (TensorRT 8.5.2)

# 检查 TensorRT 版本
dpkg -l | grep TensorRT
```

### Python 依赖

```bash
pip install --upgrade pip
pip install nvidia-tensorrt==8.5.2
pip install pycuda
pip install onnx==1.14.0
pip install onnxruntime-gpu==1.15.1
```

---

## 转换流程

### 方法 1: Ultralytics 直接导出 (推荐)

```python
#!/usr/bin/env python3
"""
使用 Ultralytics 直接导出 TensorRT 引擎
"""
from ultralytics import YOLO

def export_tensorrt(
    weights_path: str,
    imgsz: int = 640,
    device: int = 0,
    half: bool = True,
    workspace: int = 4,
    verbose: bool = True
):
    """
    导出 TensorRT 引擎
    
    Args:
        weights_path: PyTorch 模型路径
        imgsz: 输入图像尺寸
        device: GPU 设备 ID
        half: 是否使用 FP16
        workspace: 工作空间大小 (GB)
        verbose: 详细输出
    """
    # 加载模型
    model = YOLO(weights_path)
    
    # 导出 TensorRT
    model.export(
        format='engine',        # TensorRT 格式
        imgsz=imgsz,           # 输入尺寸
        device=device,         # GPU 设备
        half=half,             # FP16 精度
        workspace=workspace,   # 工作空间
        verbose=verbose,       # 详细输出
        simplify=True,         # 简化 ONNX
        dynamic=False,         # 静态 batch
        batch=1,               # batch size
    )
    
    engine_path = weights_path.replace('.pt', '.engine')
    print(f"\n✅ TensorRT 引擎已保存: {engine_path}")
    return engine_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='PyTorch 模型路径')
    parser.add_argument('--imgsz', type=int, default=640, help='输入尺寸')
    parser.add_argument('--device', type=int, default=0, help='GPU 设备')
    parser.add_argument('--fp16', action='store_true', help='使用 FP16')
    parser.add_argument('--workspace', type=int, default=4, help='工作空间 (GB)')
    
    args = parser.parse_args()
    
    export_tensorrt(
        weights_path=args.weights,
        imgsz=args.imgsz,
        device=args.device,
        half=args.fp16,
        workspace=args.workspace,
    )
```

**使用示例**:
```bash
python export_tensorrt.py \
    --weights ../models/volleyball_best.pt \
    --imgsz 640 \
    --fp16 \
    --workspace 4
```

---

### 方法 2: 两步转换 (PyTorch → ONNX → TensorRT)

#### 步骤 1: 导出 ONNX

```python
from ultralytics import YOLO

model = YOLO('volleyball_best.pt')

model.export(
    format='onnx',
    imgsz=640,
    opset=11,           # ONNX opset 版本
    simplify=True,      # 简化图
    dynamic=False,      # 静态形状
)
```

#### 步骤 2: ONNX → TensorRT

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace: int = 4
):
    """
    从 ONNX 构建 TensorRT 引擎
    """
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # 解析 ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX 解析失败")
    
    # 配置构建器
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * (1 << 30)  # GB to bytes
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ 启用 FP16 精度")
    
    # 构建引擎
    print("正在构建 TensorRT 引擎...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        raise RuntimeError("引擎构建失败")
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"✅ 引擎已保存: {engine_path}")
    return engine_path

# 使用
build_engine(
    onnx_path='volleyball_best.onnx',
    engine_path='volleyball.engine',
    fp16=True,
    workspace=4
)
```

---

## INT8 量化 (可选)

### 校准数据准备

```python
import cv2
import numpy as np
from pathlib import Path

class CalibrationDataset:
    """INT8 校准数据集"""
    
    def __init__(self, image_dir: str, num_images: int = 500):
        self.images = list(Path(image_dir).glob('*.jpg'))[:num_images]
        self.batch_size = 1
        self.current = 0
    
    def get_batch(self):
        """获取一个 batch 的数据"""
        if self.current >= len(self.images):
            return None
        
        img_path = self.images[self.current]
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dim
        
        self.current += 1
        return img
    
    def reset(self):
        self.current = 0

# 使用
calib_dataset = CalibrationDataset('../data/images/train')
```

### INT8 校准器

```python
import tensorrt as trt

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 熵校准器"""
    
    def __init__(self, dataset, cache_file='calibration.cache'):
        super().__init__()
        self.dataset = dataset
        self.cache_file = cache_file
        self.device_input = cuda.mem_alloc(1 * 3 * 640 * 640 * 4)  # FP32
    
    def get_batch_size(self):
        return 1
    
    def get_batch(self, names):
        batch = self.dataset.get_batch()
        if batch is None:
            return None
        
        # 复制到 GPU
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

# 在构建引擎时使用
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = Int8Calibrator(calib_dataset)
```

---

## 推理引擎封装

### TensorRT 推理类

```python
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

class TRTInference:
    """TensorRT 推理引擎"""
    
    def __init__(self, engine_path: str):
        """
        初始化推理引擎
        
        Args:
            engine_path: TensorRT 引擎文件路径
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 分配内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配 host 和 device 内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer(self, image: np.ndarray):
        """
        执行推理
        
        Args:
            image: 输入图像 (640, 640, 3)
        
        Returns:
            detections: (N, 6+5*3) [x1, y1, x2, y2, conf, cls, kpt...]
        """
        # 预处理
        img = cv2.resize(image, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        
        # 复制到 host 内存
        np.copyto(self.inputs[0]['host'], img.ravel())
        
        # 传输到 device
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 传输回 host
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        
        # 同步
        self.stream.synchronize()
        
        # 后处理
        output = self.outputs[0]['host']
        return self.postprocess(output)
    
    def postprocess(self, output: np.ndarray):
        """后处理输出"""
        # 根据 YOLO 输出格式解析
        # 输出形状: (1, 25200, 21)  # 21 = 5 + 1 + 5*3
        output = output.reshape(1, -1, 21)
        
        # NMS 等后处理
        # ... (使用 Ultralytics 的后处理函数)
        
        return output
    
    def __del__(self):
        """清理资源"""
        del self.context
        del self.engine
```

---

## 性能基准测试

### 基准测试脚本

```python
import time
import numpy as np
from tqdm import tqdm

def benchmark(engine_path: str, num_iterations: int = 1000):
    """
    性能基准测试
    
    Args:
        engine_path: TensorRT 引擎路径
        num_iterations: 测试迭代次数
    """
    # 初始化推理引擎
    inference = TRTInference(engine_path)
    
    # 创建随机输入
    dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 预热
    print("预热中...")
    for _ in range(10):
        inference.infer(dummy_input)
    
    # 基准测试
    print(f"运行 {num_iterations} 次推理...")
    times = []
    
    for _ in tqdm(range(num_iterations)):
        start = time.perf_counter()
        inference.infer(dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    # 统计
    times = np.array(times)
    print("\n" + "="*50)
    print("性能统计:")
    print("="*50)
    print(f"平均延迟: {times.mean():.2f} ms")
    print(f"中位数延迟: {np.median(times):.2f} ms")
    print(f"最小延迟: {times.min():.2f} ms")
    print(f"最大延迟: {times.max():.2f} ms")
    print(f"标准差: {times.std():.2f} ms")
    print(f"平均 FPS: {1000 / times.mean():.1f}")
    print(f"P95 延迟: {np.percentile(times, 95):.2f} ms")
    print(f"P99 延迟: {np.percentile(times, 99):.2f} ms")

# 运行基准测试
benchmark('../models/volleyball.engine', num_iterations=1000)
```

**预期结果** (Orin NX):
```
平均延迟: 6.02 ms
中位数延迟: 5.98 ms
最小延迟: 5.85 ms
最大延迟: 7.21 ms
标准差: 0.18 ms
平均 FPS: 166.1
P95 延迟: 6.35 ms
P99 延迟: 6.58 ms
```

---

## 优化技巧

### 1. 使用 CUDA Graph

```python
# 在 TRTInference 类中添加
def enable_cuda_graph(self):
    """启用 CUDA Graph 加速"""
    # 预热
    for _ in range(10):
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
    
    # 捕获 graph
    self.graph = cuda.Graph()
    self.graph.begin_capture(self.stream)
    self.context.execute_async_v2(
        bindings=self.bindings,
        stream_handle=self.stream.handle
    )
    self.graph.end_capture(self.stream)
    
    # 实例化
    self.graph_exec = self.graph.instantiate()

def infer_with_graph(self, image: np.ndarray):
    """使用 CUDA Graph 推理"""
    # 预处理 (同上)
    # ...
    
    # 执行 graph
    self.graph_exec.launch(self.stream)
    self.stream.synchronize()
    
    # 后处理 (同上)
    # ...
```

### 2. 零拷贝内存

```python
# 使用 pinned memory 避免拷贝
input_host = cuda.pagelocked_empty((3, 640, 640), dtype=np.float32)
```

### 3. 批处理推理

```python
# 修改引擎支持动态 batch
config.set_optimization_profile(profile)
profile.set_shape('input', (1, 3, 640, 640), (4, 3, 640, 640), (8, 3, 640, 640))
```

---

## 常见问题

### Q1: 转换失败 "Unsupported ONNX operator"?

**解决**:
```bash
# 降低 ONNX opset 版本
model.export(format='onnx', opset=11)

# 或使用 TensorRT 插件
```

### Q2: FP16 精度损失过大?

**检查**:
```python
# 对比 FP32 和 FP16 输出
output_fp32 = model_fp32.infer(image)
output_fp16 = model_fp16.infer(image)

diff = np.abs(output_fp32 - output_fp16).mean()
print(f"平均差异: {diff}")
```

### Q3: Orin NX 上性能不如预期?

**优化清单**:
- [ ] 锁定 CPU/GPU 频率到最大
- [ ] 使用 `jetson_clocks` 工具
- [ ] 关闭 GUI 和后台进程
- [ ] 使用 FP16 精度
- [ ] 启用 CUDA Graph

```bash
# 最大化性能
sudo jetson_clocks
sudo nvpmodel -m 0  # 最大功耗模式
```

---

## 下一步

TensorRT 引擎准备完成后，进入 [Orin NX 部署](04_deployment.md) 阶段。
