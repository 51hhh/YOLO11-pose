# ⚡ Jetson NX 过流和性能优化指南

## 问题诊断

### 症状
```
throttled duo over-current
```

### 原因
1. **高负载** - YOLO推理 + 图像发布 + 100Hz采集
2. **供电不足** - 使用的电源适配器功率不够
3. **散热不良** - 长时间高负载导致温度过高

---

## 🔧 已实施的优化

### 1️⃣ 降低图像发布频率
- **原始图像**: 从100Hz降低到10Hz（每10帧发布一次）
- **检测图像**: 从100Hz降低到20Hz（每5帧发布一次）
- **效果**: 减少约80%的图像编码和传输开销

### 2️⃣ 修复统计bug
- 修复检测率溢出问题（`18446744073709551616%`）
- 在立体匹配失败时正确计入`lost_frames_`
- 统计重置时同时重置`lost_frames_`

### 3️⃣ 默认配置优化
- 关闭原始图像发布（`publish_images: false`）
- 只保留检测可视化图像（用于调试）

---

## 🔌 电源建议

### 推荐电源规格
- **电压**: 19V
- **电流**: ≥5A
- **功率**: ≥90W
- **接口**: DC 5.5x2.5mm 或官方电源适配器

### 检查当前电源
```bash
# 查看当前功耗
sudo tegrastats

# 查看电源状态
sudo nvpmodel -q
```

---

## 🌡️ 散热优化

### 检查温度
```bash
# 实时监控温度
watch -n 1 'cat /sys/devices/virtual/thermal/thermal_zone*/temp'

# 或使用
sudo tegrastats
```

### 散热建议
1. **添加散热风扇** - 主动散热
2. **改善通风** - 确保设备周围空气流通
3. **降低环境温度** - 避免阳光直射

---

## ⚙️ 性能模式调整

### 查看当前模式
```bash
sudo nvpmodel -q
```

### 切换到低功耗模式（如果过流严重）
```bash
# 模式0: MAXN (最高性能)
sudo nvpmodel -m 0

# 模式1: 15W (平衡模式)
sudo nvpmodel -m 1

# 模式2: 10W (省电模式)
sudo nvpmodel -m 2
```

### 查看CPU频率
```bash
# 查看当前频率
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# 设置固定频率（避免动态调频）
sudo jetson_clocks --show
sudo jetson_clocks  # 锁定到最高频率
```

---

## 📊 性能监控

### 启用性能监控
```bash
# 终端1: 运行追踪节点
sudo -E bash -c "source /opt/ros/humble/setup.bash && source ~/NX_volleyball/ros2_ws/install/setup.bash && ros2 run volleyball_stereo_driver volleyball_tracker_node"

# 终端2: 监控系统状态
sudo tegrastats --interval 1000
```

### 关键指标
- **CPU使用率**: 应 <80%
- **GPU使用率**: 应 <90%
- **温度**: 应 <80°C
- **功耗**: 应在电源规格内

---

## 🚀 进一步优化（如仍有过流）

### 1. 降低采集频率
修改 `tracker_params.yaml`:
```yaml
pwm:
  frequency: 50.0  # 从100Hz降低到50Hz
```

### 2. 降低检测尺寸
```yaml
detector:
  global_size: 416  # 从640降低到416
```

### 3. 提高置信度阈值
```yaml
detector:
  confidence_threshold: 0.6  # 从0.5提高到0.6，减少后处理
```

### 4. 完全关闭图像发布
```yaml
debug:
  publish_images: false
  publish_detection_image: false  # 也关闭检测图像
```

### 5. 增加统计间隔
```yaml
debug:
  log_interval: 200  # 从100增加到200
```

---

## ✅ 验证优化效果

运行后检查：
1. ✅ 检测率正常显示（0-100%）
2. ✅ 位置持续更新（不固定）
3. ✅ 无 "over-current" 警告
4. ✅ 温度稳定在安全范围
5. ✅ FPS稳定在15-20

---

## 🔍 调试命令

```bash
# 查看话题发布频率
ros2 topic hz /volleyball/pose_3d

# 查看检测图像发布频率
ros2 topic hz /volleyball/detection_image

# 查看话题列表
ros2 topic list

# 查看节点信息
ros2 node info /volleyball_tracker
```

---

**更新时间**: 2026-01-25
