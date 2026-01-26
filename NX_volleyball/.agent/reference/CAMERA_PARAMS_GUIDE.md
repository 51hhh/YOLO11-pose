# 📸 相机参数配置说明

## ✅ 已优化的参数

通过实际测试，确定的最佳相机参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| **曝光时间** | 9867.0 us | 约 9.9ms，适合 100Hz 采集 |
| **增益** | 10.9854 dB | 平衡亮度和噪声 |
| **触发模式** | On (外部触发) | 由 PWM 控制 |
| **触发源** | Line0 | 连接到 GPIO Line 7 |
| **触发激活** | RisingEdge | 上升沿触发 |

---

## 📊 参数分析

### 曝光时间: 9867 us (9.9ms)

**为什么选择这个值？**

- 100Hz 采集周期 = 10ms
- 曝光时间 9.9ms，留 0.1ms 余量
- 最大化光线采集，同时避免运动模糊
- 适合室内排球场照明条件

**注意事项**:
- 如果球速过快导致模糊，可适当降低曝光时间
- 如果图像过暗，优先调整增益而非曝光

### 增益: 10.9854 dB

**为什么选择这个值？**

- 提供足够的图像亮度
- 噪声水平可接受
- 适合当前照明环境

**注意事项**:
- 增益过高会增加噪声
- 如果照明改善，可以降低增益
- 建议范围: 0-15 dB

---

## 🔧 如何修改参数

### 方法 1: 修改配置文件 (推荐)

编辑 `config/camera_params.yaml`:

```yaml
stereo_camera_node:
  ros__parameters:
    exposure_time: 9867.0   # 修改这里
    gain: 10.9854           # 修改这里
```

然后重新运行节点:

```bash
ros2 run volleyball_stereo_driver stereo_camera_node
```

### 方法 2: 命令行参数

```bash
ros2 run volleyball_stereo_driver stereo_camera_node \
    --ros-args \
    -p exposure_time:=9867.0 \
    -p gain:=10.9854
```

---

## 🎯 不同场景的参数建议

### 场景 1: 高速运动 (快速扣球)

```yaml
exposure_time: 5000.0   # 降低曝光，减少模糊
gain: 15.0              # 提高增益补偿亮度
```

### 场景 2: 低照明环境

```yaml
exposure_time: 9867.0   # 保持最大曝光
gain: 15.0              # 提高增益
```

### 场景 3: 高照明环境 (室外)

```yaml
exposure_time: 3000.0   # 降低曝光
gain: 5.0               # 降低增益
```

---

## ⚠️ 重要提示

### 不要修改的参数

以下参数已经过优化，**不建议修改**：

- `trigger_mode: true` - 必须使用外部触发
- `trigger_source: "Line0"` - 硬件连接固定
- `trigger_activation: "RisingEdge"` - 与 PWM 匹配

### 可以调整的参数

根据实际情况可以调整：

- `exposure_time` - 根据光照和运动速度
- `gain` - 根据图像亮度
- `left_camera_index` / `right_camera_index` - 如果相机顺序不同

---

## 📈 参数验证

### 检查曝光时间

```bash
# 运行节点后，查看日志
# 应该看到: ✅ 曝光时间: 9867 us
```

### 检查图像质量

```bash
# 查看图像
ros2 run image_tools showimage --ros-args --remap image:=/stereo/left/image_raw

# 检查:
# - 亮度是否合适
# - 是否有运动模糊
# - 噪声水平
```

### 检查帧率

```bash
ros2 topic hz /stereo/left/image_raw

# 应该稳定在 100 Hz
```

---

## 🔄 参数调优流程

1. **从默认参数开始** (已完成)
   - 曝光: 9867 us
   - 增益: 10.9854 dB

2. **检查图像质量**
   - 太暗 → 增加增益
   - 太亮 → 降低增益或曝光
   - 模糊 → 降低曝光时间

3. **验证帧率稳定性**
   - 确保 100 Hz 稳定

4. **记录最终参数**
   - 更新配置文件

---

**当前配置已经过实际测试优化，可以直接使用！**
