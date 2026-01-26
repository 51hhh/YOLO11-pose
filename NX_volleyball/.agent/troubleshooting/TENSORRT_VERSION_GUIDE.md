# 🔧 TensorRT 版本兼容性说明

## 📋 版本信息

### TensorRT 10.x (最新版本)
- **参数变更**: `--workspace` → `--memPoolSize`
- **格式**: `--memPoolSize=workspace:4096` (单位: MB)
- **Jetson NX**: 通常预装 TensorRT 10.x

### TensorRT 8.x/9.x (旧版本)
- **参数**: `--workspace=4096`
- **格式**: 直接指定大小 (单位: MB)

---

## 🔍 检查 TensorRT 版本

```bash
# 方法 1: 使用 dpkg
dpkg -l | grep tensorrt

# 方法 2: 使用 trtexec
trtexec --help | head -20

# 方法 3: Python
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

---

## ✅ 已更新的脚本

`scripts/convert_yolo_to_tensorrt.py` 已更新为使用 TensorRT 10.x 的新参数：

```python
# TensorRT 10.x 使用 --memPoolSize 替代 --workspace
cmd += " --memPoolSize=workspace:4096"  # 4GB workspace
```

---

## 🚀 使用方法

### 转换模型

```bash
cd ~/desktop/yolo/yoloProject/NX_volleyball

# 运行转换脚本（自动使用正确的参数）
python3 scripts/convert_yolo_to_tensorrt.py \
  --model ros2_ws/src/volleyball_stereo_driver/model/best.pt \
  --output ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine \
  --fp16
```

### 预期输出

```
📦 步骤 1: 转换 best.pt → best.onnx
✅ ONNX 模型已生成: best.onnx

🚀 步骤 2: 转换 best.onnx → yolo11n.engine
执行命令: trtexec --onnx=best.onnx --saveEngine=yolo11n.engine --fp16 --memPoolSize=workspace:4096 --verbose --dumpLayerInfo --exportLayerInfo=layer_info.json
⏳ 转换中，这可能需要几分钟...
✅ TensorRT Engine 已生成: yolo11n.engine
   文件大小: XX.XX MB
```

---

## 📊 参数说明

### --memPoolSize

指定 TensorRT 构建时使用的内存池大小。

**格式**: `--memPoolSize=<pool_type>:<size_in_MB>`

**示例**:
- `--memPoolSize=workspace:4096` - 4GB workspace
- `--memPoolSize=workspace:8192` - 8GB workspace (如果内存充足)

### --fp16

启用 FP16 精度，可以显著加速推理并减小模型大小。

**优点**:
- 推理速度提升 2-3 倍
- 模型大小减半
- 内存占用减少

**缺点**:
- 精度略有下降（通常可忽略）

### --verbose

输出详细的转换日志，便于调试。

### --dumpLayerInfo

输出每一层的详细信息。

### --exportLayerInfo

将层信息导出为 JSON 文件，便于分析。

---

## 🐛 常见问题

### 问题 1: Unknown option: --workspace

**原因**: 使用了旧版本的参数  
**解决**: 更新脚本使用 `--memPoolSize`（已修复）

### 问题 2: Out of memory

**原因**: workspace 设置过大  
**解决**: 减小 memPoolSize

```bash
# 从 4096 减小到 2048
--memPoolSize=workspace:2048
```

### 问题 3: 转换时间过长

**原因**: 模型复杂或 workspace 过大  
**解决**: 
- 耐心等待（YOLOv11n 通常需要 5-10 分钟）
- 使用 `--verbose` 查看进度

---

## 📝 完整转换命令参考

### 基础转换
```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \
        --memPoolSize=workspace:4096
```

### 带详细输出
```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \
        --memPoolSize=workspace:4096 \
        --verbose \
        --dumpLayerInfo \
        --exportLayerInfo=layer_info.json
```

### 指定输入尺寸（如果需要）
```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \
        --memPoolSize=workspace:4096 \
        --shapes=images:1x3x640x640
```

---

## ✨ 总结

- ✅ 脚本已更新为 TensorRT 10.x 兼容
- ✅ 使用 `--memPoolSize=workspace:4096` 替代 `--workspace 4096`
- ✅ 添加了更多有用的参数
- ✅ 提供了详细的错误提示

**现在可以正常转换模型了！** 🚀
