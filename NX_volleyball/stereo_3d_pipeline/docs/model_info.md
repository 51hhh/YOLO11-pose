# YOLO26 模型信息

## 正确模型

| 项目 | 值 |
|------|-----|
| **权重文件** | `NX_volleyball/model/yolo26n.pt` |
| **ONNX 文件** | `NX_volleyball/model/yolo26.onnx` (9.7MB) |
| **导出工具** | `D:\robotmaster\ObjectDetectionYOLO26\scripts\export_onnx.py` |
| **导出参数** | `ultralytics.YOLO.export(format="onnx", imgsz=640, opset=17)` |
| **输入** | `images` [1, 3, 640, 640] (static) |
| **输出** | `output0` [1, 5, 8400] — Pre-NMS, 5 = (x,y,w,h,conf), 8400 候选框 |
| **NMS** | 已移除 (直接输出置信框，后处理在 C++ 端完成) |
| **TopK** | 官方导出工具已优化 TopK，无 TopK 算子 |

## 错误模型 (已废弃)

| 项目 | 值 |
|------|-----|
| **ONNX 文件** | `NX_volleyball/model/yolo26n.onnx` (9.6MB) |
| **来源** | 在 NX 上本地导出，**未使用官方导出工具** |
| **输入** | `images` [1, 3, 320, 320] |
| **输出** | `output0` [1, 300, 6] — Post-NMS, 包含 2× TopK 算子 |
| **问题** | 320 尺寸非目标分辨率；TopK 不被 DLA 原生支持导致 GPU 回退 |

> **重要**: `yolo26n.onnx` 和相关的所有 `*_320.engine` 文件不应用于生产。
> 正确的流程: `yolo26n.pt` → 通过 `export_onnx.py` 导出 → `yolo26.onnx` (640×640)

## 导出脚本说明

`D:\robotmaster\ObjectDetectionYOLO26\scripts\export_onnx.py`:
- 默认权重路径: `ObjectDetectionYOLO26/weight/best.pt`
- 默认输出路径: `ObjectDetectionYOLO26/deploy/agx_zed/models/{stem}.onnx`
- 关键参数: `--imgsz 640 --opset 17`
- 内部调用 `ultralytics.YOLO(weights).export()`

## TensorRT 引擎构建

模型输入固定为 640×640。如需缩放分辨率，在 TensorRT 构建时通过 `--shapes` 参数指定 (但通常保持 640)。

### 有效引擎 (基于 yolo26.onnx)

```
yolo26_gpu_fp16_640.engine     GPU FP16  8.2MB
yolo26_gpu_int8_640.engine     GPU INT8  4.7MB
yolo26_dla0_fp16_640.engine    DLA0 FP16 7.0MB
yolo26_dla1_fp16_640.engine    DLA1 FP16 7.1MB
yolo26_dla0_int8_640.engine    DLA0 INT8 3.9MB
yolo26_dla1_int8_640.engine    DLA1 INT8 3.9MB
yolo26_dla_fp16.engine         DLA0 FP16 7.0MB (旧构建)
yolo26_fp16.engine             GPU FP16  8.6MB (旧构建)
```

### 废弃引擎 (基于 yolo26n.onnx，不应使用)

```
yolo26n_gpu_fp16_320.engine
yolo26n_gpu_int8_320.engine
yolo26n_dla0_int8_320.engine / yolo26n_dla1_int8_320.engine
yolo26n_dla0_fp16_320.engine / yolo26n_dla1_fp16_320.engine
yolo26n_dla0_int8.engine / yolo26n_dla1_int8.engine
yolo26n.engine
```

---

*2026-04-04 创建*
