# INT8 量化与校准指南

> 平台：Jetson Orin NX Super 16GB | TensorRT 10.3 | JetPack 6.2  
> 模型：yolo26.onnx (YOLO11n-pose, 640×640, Pre-NMS)

## 1. 为什么需要校准

### 1.1 INT8 量化原理

```
FP32 范围: [-∞, +∞] → 32 bit, 高精度但慢
FP16 范围: [-65504, +65504] → 16 bit, 2× 加速
INT8 范围: [-128, +127] → 8 bit, 4× 加速 (理论)

量化公式: x_int8 = round(x_fp32 / scale)
反量化:   x_fp32 ≈ x_int8 × scale

scale = max(|x|) / 127  ← 需要从真实数据统计获得
```

### 1.2 TensorRT 校准流程

```
ONNX 模型 (FP32)
      ↓
TensorRT Builder
      ↓  --int8 标志
需要校准数据集 (Calibration Dataset)
      ↓
对每一层运行代表性输入
      ↓
收集激活值分布 (histograms)
      ↓
计算最优 scale/zero-point (KL 散度最小化)
      ↓
生成 Calibration Cache (校准缓存文件)
      ↓
INT8 Engine (.engine)
```

### 1.3 不校准的后果

目前我们构建 INT8 engine 时：

```bash
trtexec --onnx=yolo26.onnx --int8 --fp16 --saveEngine=...
```

**trtexec 使用随机输入数据**作为校准输入。这意味着：
- 激活值分布可能不正确 → 量化 scale 不最优
- 精度损失不可控（可能 mAP 下降 2-10%）
- 检测阈值可能需要重新调整
- 排球等小目标/白色目标可能漏检

**正确做法**：提供真实排球场景的图像作为校准数据集。

## 2. 校准数据集要求

### 2.1 基本要求

| 参数 | 要求 | 说明 |
|------|------|------|
| **图片数量** | **200-1000 张** | 推荐 500 张。太少→分布不完整，太多→校准时间长 |
| **分辨率** | 任意（会自动 resize 到 640×640） | 原始分辨率不限 |
| **格式** | JPG / PNG | 标准图片格式 |
| **内容代表性** | **必须包含实际应用场景** | 球场、排球、人物、不同光照 |

### 2.2 推荐数据组成

```
校准数据集 (约 500 张):
├── 有排球的场景 (~200 张, 40%)
│   ├── 排球在不同位置 (近/远/边角)
│   ├── 排球不同大小 (占画面 1%~20%)
│   ├── 排球被遮挡场景
│   └── 多个排球
├── 人物场景 (~150 张, 30%)
│   ├── 站立/跑动/跳跃
│   ├── 不同距离
│   └── 关键点可见/部分遮挡
├── 球场环境 (~100 张, 20%)
│   ├── 不同光照 (日光/灯光/背光)
│   ├── 不同角度
│   └── 观众/裁判
└── 边界情况 (~50 张, 10%)
    ├── 空场景 (无目标)
    ├── 运动模糊
    └── 极端光照
```

### 2.3 数据来源选项

| 来源 | 优先级 | 说明 |
|------|--------|------|
| **训练数据集的子集** | **最推荐** | 已标注，分布与训练一致 |
| 现场采集的原始图像 | 推荐 | 最真实，但可能未标注 |
| COCO/VOC 排球子类 | 可用 | 通用性好但场景不匹配 |
| 随机抽帧 | 最低 | 缺乏代表性 |

## 3. 校准实施方法

### 3.1 方法一: trtexec 内置校准 (推荐)

```bash
# 准备校准图像目录
# 图像必须预处理为 NCHW 格式的 binary 文件

# Step 1: 预处理校准图像
python3 prepare_calibration_data.py \
    --input-dir /path/to/calibration_images/ \
    --output-dir /home/nvidia/NX_volleyball/calibration_data/ \
    --size 640 \
    --count 500

# Step 2: 构建 INT8 engine with 校准
trtexec --onnx=yolo26.onnx \
    --int8 --fp16 \
    --calib=/home/nvidia/NX_volleyball/calibration_data/ \
    --saveEngine=yolo26_gpu_int8_calibrated.engine \
    --memPoolSize=workspace:4096MiB
```

### 3.2 方法二: Python API 自定义校准器

```python
import tensorrt as trt
import numpy as np
from PIL import Image
import os

class YOLOCalibrator(trt.IInt8EntropyCalibrator2):
    """使用真实图像的 INT8 校准器"""
    
    def __init__(self, image_dir, cache_file, batch_size=1, 
                 input_size=640):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_size = input_size
        
        # 收集图像路径
        self.images = sorted([
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.current_index = 0
        
        # 分配 GPU 内存
        import pycuda.driver as cuda
        import pycuda.autoinit
        self.device_input = cuda.mem_alloc(
            batch_size * 3 * input_size * input_size * 4  # FP32
        )
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index >= len(self.images):
            return None
        
        batch = np.zeros(
            (self.batch_size, 3, self.input_size, self.input_size), 
            dtype=np.float32
        )
        
        for i in range(self.batch_size):
            if self.current_index >= len(self.images):
                break
            img = Image.open(self.images[self.current_index])
            img = self._letterbox(img, self.input_size)
            img = np.array(img, dtype=np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            batch[i] = img
            self.current_index += 1
        
        import pycuda.driver as cuda
        cuda.memcpy_htod(self.device_input, batch.tobytes())
        return [int(self.device_input)]
    
    def _letterbox(self, img, size):
        """Letterbox resize 保持宽高比"""
        w, h = img.size
        scale = min(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        
        new_img = Image.new('RGB', (size, size), (114, 114, 114))
        new_img.paste(img, ((size - nw) // 2, (size - nh) // 2))
        return new_img
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"Calibration cache saved to {self.cache_file}")


def build_int8_engine(onnx_path, engine_path, calibrator, 
                      use_dla=False, dla_core=0):
    """构建校准后的 INT8 engine"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            return False
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator
    
    if use_dla:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    
    engine = builder.build_serialized_network(network, config)
    if engine:
        with open(engine_path, 'wb') as f:
            f.write(engine)
        print(f"Engine saved to {engine_path}")
        return True
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="yolo26.onnx")
    parser.add_argument("--images", required=True, 
                        help="Directory of calibration images")
    parser.add_argument("--engine", default="yolo26_int8_calibrated.engine")
    parser.add_argument("--cache", default="yolo26_calibration.cache")
    parser.add_argument("--dla", type=int, default=-1, 
                        help="DLA core (-1=GPU, 0=DLA0, 1=DLA1)")
    parser.add_argument("--count", type=int, default=500,
                        help="Number of calibration images to use")
    args = parser.parse_args()
    
    calibrator = YOLOCalibrator(
        args.images, args.cache, batch_size=1, input_size=640
    )
    
    build_int8_engine(
        args.onnx, args.engine, calibrator,
        use_dla=(args.dla >= 0), dla_core=max(0, args.dla)
    )
```

### 3.3 方法三: 使用 calibration cache (最快)

如果已有 calibration cache（在相同模型上一次性生成），
后续构建 engine（包括不同配置：GPU/DLA/Hybrid）可复用：

```bash
# 先生成 cache (一次)
python3 build_calibrated_engine.py \
    --onnx yolo26.onnx \
    --images /path/to/calibration_images/ \
    --engine yolo26_gpu_int8.engine \
    --cache yolo26_calibration.cache

# 复用 cache 构建 DLA engine (快速, 无需再次校准)
trtexec --onnx=yolo26.onnx \
    --int8 --fp16 \
    --calib=yolo26_calibration.cache \
    --useDLACore=0 --allowGPUFallback \
    --saveEngine=yolo26_dla0_int8_calibrated.engine
```

## 4. 校准前后精度对比流程

### 4.1 验证步骤

```bash
# 1. 准备验证集 (100-200张有标注的图像)
# 2. 分别在 FP16 和 INT8(有/无校准) engine 上推理
# 3. 对比 mAP / 检测率 / 漏检率

python3 validate_engine.py \
    --engine-fp16 yolo26_gpu_fp16.engine \
    --engine-int8-uncalib yolo26_gpu_int8.engine \
    --engine-int8-calib yolo26_gpu_int8_calibrated.engine \
    --images /path/to/validation_images/ \
    --labels /path/to/labels/ \
    --iou-threshold 0.5
```

### 4.2 预期精度对比

| 配置 | 延迟 | mAP@0.5 预期 | 备注 |
|------|------|-------------|------|
| GPU FP16 | 3.57ms | ~98% of FP32 | 基线参考 |
| GPU INT8 (无校准) | 2.93ms | ~90-95% of FP32 | 随机数据校准，不确定 |
| GPU INT8 (校准后) | 2.93ms | ~96-98% of FP32 | 延迟不变，精度提升 |
| DLA0 INT8 (无校准) | 15.67ms | ~85-92% of FP32 | DLA 对量化更敏感 |
| DLA0 INT8 (校准后) | 15.67ms | ~92-96% of FP32 | 校准对 DLA 帮助更大 |

> 注意: 以上 mAP 为经验估计，实际值需用真实标注数据验证。

## 5. 注意事项

### 5.1 校准缓存的有效性

- **模型变化**: 模型结构或权重变化 → 必须重新校准
- **跨设备**: GPU 生成的 cache **不能**直接用于 DLA（量化范围不同）
- **TRT 版本**: TRT 大版本更新（如 10.x → 11.x）→ 重新校准

### 5.2 DLA 特殊要求

DLA INT8 的校准与 GPU INT8 不同：
- DLA 使用固定缩放因子，精度对 scale 更敏感
- 建议分别为 GPU 和 DLA 生成独立的 calibration cache
- DLA FP16 不需要校准

### 5.3 Hybrid Engine 校准

Hybrid A/B engine (DLA + GPU 混合) 的校准：
- DLA 部分和 GPU 部分使用各自的量化参数
- 需要同一个 calibration dataset
- 建议用 Python API 方式构建，可精确控制 DLA/GPU 各层的 calibrator

## 6. 文件清单

| 文件 | 说明 |
|------|------|
| `yolo26_calibration.cache` | 校准缓存（生成后可复用） |
| `build_calibrated_engine.py` | 校准构建脚本 |
| `prepare_calibration_data.py` | 图像预处理脚本 |
| `validate_engine.py` | 精度验证脚本 |
| `calibration_images/` | 校准图像目录（用户提供） |
