# YOLO 模型文件目录

## 📁 文件说明

- `best.pt` - PyTorch 训练模型（原始）
- `best.onnx` - ONNX 格式（中间格式）
- `yolo11n.engine` - TensorRT 引擎（运行时使用）⭐

## ✅ 使用方法

系统运行时需要 `yolo11n.engine` 文件。

### 文件查找优先级

节点会按以下顺序搜索模型文件：

1. **Install 目录**（编译后自动复制）
   ```
   ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/yolo11n.engine
   ```

2. **Source 目录**（开发时）
   ```
   ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine
   ```

3. **工作空间根目录**
   ```
   ~/NX_volleyball/ros2_ws/model/yolo11n.engine
   ```

### ⚙️ 编译后自动安装

运行 `colcon build` 后，模型文件会自动复制到 install 目录：

```bash
cd ~/NX_volleyball/ros2_ws
colcon build --packages-select volleyball_stereo_driver
# 模型文件会自动安装到:
# install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/
```

### 🔍 验证模型文件

```bash
# 检查 src 目录
ls -lh ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine

# 检查 install 目录（编译后）
ls -lh ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/yolo11n.engine
```

## 🔄 模型转换

如需重新生成 TensorRT 引擎：

```bash
cd ~/NX_volleyball
python3 scripts/convert_yolo_to_tensorrt.py \
  --model ros2_ws/src/volleyball_stereo_driver/model/best.pt \
  --output ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine \
  --imgsz 640 \
  --fp16
```

## ⚠️ 注意事项

1. **TensorRT 引擎是平台相关的**
   - 在 x86 生成的 engine 不能在 ARM (Jetson) 上使用
   - 必须在目标设备上重新生成

2. **文件大小**
   - `best.pt`: ~6 MB
   - `yolo11n.engine`: ~12 MB（FP16）

3. **不要提交到 Git**
   - `.engine` 文件已在 `.gitignore` 中
   - 只需传输到部署设备

## 🐛 故障排查

### 问题：运行时找不到模型

**症状**:
```
[WARN] ⚠️  未找到模型文件: model/yolo11n.engine
```

**解决方案**:

1. **检查文件是否存在**:
```bash
ls -lh ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine
```

2. **重新编译安装**:
```bash
cd ~/NX_volleyball/ros2_ws
rm -rf build/ install/ log/
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash
```

3. **手动复制**（临时方案）:
```bash
mkdir -p ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/
cp ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine \
   ~/NX_volleyball/ros2_ws/install/volleyball_stereo_driver/share/volleyball_stereo_driver/model/
```

4. **使用绝对路径**（测试用）:
修改配置文件使用绝对路径：
```yaml
detector:
  model_path: "/home/nvidia/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/yolo11n.engine"
```
