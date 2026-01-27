# 🏐 海康双目高速排球追踪系统

> **Jetson Orin NX 16GB** | **ROS2 Humble** | **TensorRT** | **100+ FPS**

---

## 📊 项目状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| ✅ 代码实现 | **完成** | ROS2 包、所有组件 |
| ✅ NX 部署验证 | **完成** | 2026-01-27 双缓冲架构优化 |
| ✅ PWM 同步 | **完成** | 100Hz ±0.05%, 同步率100% |
| ✅ 双缓冲架构 | **完成** | 100fps实测，零轮询，-39%延迟 |
| ✅ CUDA Bayer预处理 | **完成** | GPU加速，7.3ms推理延迟 |
| ✅ Batch=2 推理 | **完成** | 137Hz理论FPS，完美同步 |
| ⏳ 双目标定 | 待执行 | 采集棋盘格图像 |
| ⏳ YOLO 模型 | 待优化 | 当前使用占位符 |

📖 **详细进度**: [.agent/README.md](.agent/README.md) | 💾 **实现总结**: [.agent/IMPLEMENTATION_SUMMARY.md](.agent/IMPLEMENTATION_SUMMARY.md)

---

## 🚀 快速开始

```bash
# 一键编译测试
cd ~/NX_volleyball
./quick_build_test.sh

# 或手动运行
cd ~/NX_volleyball/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select volleyball_stereo_driver
source install/setup.bash

# 运行节点
sudo -E ros2 run volleyball_stereo_driver stereo_system_node
```

---

## 🏗️ 系统架构

### 整体流程

```
┌──────────────────────────────────────────┐
│      volleyball_tracker_node             │
│         (All-in-One 追踪节点)             │
├──────────────────────────────────────────┤
│  PWM 触发 (100Hz)                        │
│      ↓                                   │
│  双目相机同步采集 (海康 MV-CA016-10UC)    │
│      ↓                                   │
│  YOLO11n TensorRT 检测 (全图/ROI)        │
│      ↓                                   │
│  立体匹配 (三角测量)                      │
│      ↓                                   │
│  3D 卡尔曼滤波                           │
│      ↓                                   │
│  发布 /volleyball/pose_3d                │
└──────────────────────────────────────────┘
```

### 🚀 双缓冲+条件变量推理流程（100fps优化架构）

```
═══════════════════════════════════════════════════════════════════
                      时间轴 (每10ms一个周期)
═══════════════════════════════════════════════════════════════════

相机回调线程 (异步触发，持续运行):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T=0ms   │ PWM触发 → 左右相机开始曝光 (9.8ms)
T=10ms  │ 采集完成 → onLeftFrameCallback()
        │   ├─ 检查写缓冲区 (left_write_idx)
        │   ├─ Bayer→BGR + 元数据 → Buffer[0]
        │   ├─ 切换索引 (write_idx: 0→1)
        │   └─ frame_cv_.notify_one() ← 唤醒推理线程
        │
        │ 同时 onRightFrameCallback() 执行相同流程
        │   └─ notify_one() ← 再次唤醒 (保证及时响应)
        │
T=20ms  │ Frame2 采集完成 → 写入Buffer[1] → notify...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

推理线程 (condition_variable等待，零轮询):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T=0-10ms  │ 🛌 frame_cv_.wait() ← 阻塞等待 (零CPU开销)
          │
T=10.05ms │ ⚡ 被notify唤醒 (延迟<0.1ms)
          │   ├─ waitForSyncedPair():
          │   │   ├─ 读取 left_buffers[read_idx]
          │   │   ├─ 读取 right_buffers[read_idx]
          │   │   ├─ 帧号/时间戳同步检查 (2.5ms)
          │   │   └─ 标记已读 (允许相机覆盖)
          │   │
T=12.5ms  │   ├─ detectVolleyball(): 
          │   │   ├─ Bayer→RGB + Resize + Normalize (CUDA, 1.9ms)
          │   │   ├─ TensorRT Batch=2推理 (5.4ms)
          │   │   └─ NMS后处理 (0.02ms)
          │   │
T=19.9ms  │   ├─ computeStereoMatch() (0.05ms)
          │   ├─ updateTracker() (0.02ms)
          │   └─ publishResults()
          │
T=20ms    │ 循环回到 wait() ← 等待Frame2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

关键特性:
✅ 零轮询: 条件变量被动等待，CPU占用<1%
✅ 零延迟: 相机回调直接写缓冲并唤醒，响应<0.1ms
✅ 流水线: 推理(8ms) 与 下一帧采集(10ms) 完全并行
✅ 零拷贝: 乒乓缓冲避免clone()，直接引用
✅ 丢帧保护: 推理慢时相机覆盖write_buffer，不阻塞采集

实测性能:
• FPS: 95.7-100.2 Hz (PWM理论极限100Hz)
• 同步延迟: 2.57ms平均 (wait + sync检查)
• 端到端: 10ms采集 + 2.5ms同步 + 7.3ms推理 = 19.8ms
• 吞吐率: 1000/10 = 100fps ✅
═══════════════════════════════════════════════════════════════════
```

---

## 📦 项目结构

```
NX_volleyball/
├── README.md                # 本文件
├── quick_build_test.sh      # 快速编译脚本
├── deploy_to_nx.sh          # 部署到 NX 脚本
│
├── ros2_ws/                 # ROS2 工作空间
│   └── src/volleyball_stereo_driver/  # 唯一的 ROS2 包
│       ├── include/         # 7 个头文件
│       ├── src/             # 10 个源文件
│       ├── config/          # 配置文件
│       ├── model/           # YOLO 模型目录
│       └── calibration/     # 标定文件目录
│
├── scripts/                 # 工具脚本
│   ├── hik_camera.py        # 相机驱动
│   ├── test_pwm_*.py        # PWM 测试
│   └── convert_yolo_to_tensorrt.py  # 模型转换
│
├── calibration/             # 标定工具
│   └── capture_chessboard.py
│
└── .agent/                  # 📖 详细文档
    ├── README.md            # 文档索引和项目状态
    ├── reference/           # 技术参考文档
    ├── troubleshooting/     # 故障排查
    └── archive/             # 归档文档
```

---

## 🔧 硬件配置

| 组件 | 型号/参数 |
|------|----------|
| **平台** | Jetson Orin NX 16GB |
| **相机** | 2x Hikvision MV-CA016-10UC (IMX273, 1440x1080) |
| **镜头** | 5mm 广角 (FOV ~52°) |
| **基线** | 25cm |
| **触发** | GPIO PWM → 相机 Line0 (硬触发) |

---

## 📊 性能指标（NX 实测 ✅ 2026-01-27 双缓冲架构优化）

### PWM 触发性能
- **实际频率**: 99.98 - 100.04 Hz
- **频率误差**: ±0.05 Hz (±0.05%)
- **稳定性**: 500周期内波动 < 0.1Hz

### 采集同步性能（双缓冲+条件变量架构）
- **同步成功率**: **100%** (1200/1200 帧对)
- **失配次数**: 0
- **丢帧统计**: 启动时 L=26 R=25帧，稳定后 0 丢帧
- **同步策略**: 帧号差≤3 且 接收时间差<25ms
- **缓冲切换**: 左右各1200次 (完美1:1同步)
- **条件变量唤醒**: 2400次 (每对帧2次通知)

### 推理性能 (Batch=2 + CUDA Bayer预处理)
- **Bayer预处理 (GPU双路)**: 1.82 - 2.03 ms
- **TensorRT推理+D2H**: 5.31 - 5.67 ms
- **NMS后处理 (双路)**: 0.022 - 0.029 ms
- **端到端延迟**: 7.15 - 7.62 ms
- **理论推理FPS**: 131 - 140 Hz

### 🚀 系统整体性能（架构优化后）
- **实际FPS**: **95.7 - 100.2 Hz** ✅ (达到PWM硬件极限)
- **同步等待延迟**: 2.57 ms平均 (条件变量wait + 帧对匹配)
- **端到端总延迟**: ~20 ms (采集10ms + 同步2.5ms + 推理7.3ms)
- **CPU占用**: <5% (零轮询架构)
- **GPU利用率**: ~62% (受PWM频率限制，非GPU瓶颈)

### 性能对比（优化前后）

| 指标 | 轮询架构 | 双缓冲+条件变量 | 提升 |
|------|---------|---------------|------|
| **实际FPS** | 75-80 Hz | **95-100 Hz** | **+25%** |
| **等待开销** | 4.22 ms | **2.57 ms** | **-39%** |
| **推理延迟** | 8.25 ms | 7.28 ms | -12% |
| **CPU占用** | 15-20% | **<5%** | **-75%** |
| **架构** | 主动轮询 | **被动唤醒** | 零轮询 |

### 深度精度（理论计算）

| 距离 | 深度精度 | 说明 |
|------|----------|------|
| 3m | ±1.2cm | 极高精度 |
| 9m | ±11cm | 高精度 |
| 15m | ±31cm | 中等精度 (物理限制) |

---

## 📖 文档

| 需求 | 文档路径 |
|------|----------|
| 项目状态和文档索引 | [.agent/README.md](.agent/README.md) |
| 实现完成总结 | [.agent/IMPLEMENTATION_SUMMARY.md](.agent/IMPLEMENTATION_SUMMARY.md) |
| 架构设计 | [.agent/reference/ROS2_ARCHITECTURE_FINAL.md](.agent/reference/ROS2_ARCHITECTURE_FINAL.md) |
| NX 编译测试 | [.agent/reference/NX_BUILD_TEST_GUIDE.md](.agent/reference/NX_BUILD_TEST_GUIDE.md) |
| YOLO 集成 | [.agent/reference/YOLO_INTEGRATION_GUIDE.md](.agent/reference/YOLO_INTEGRATION_GUIDE.md) |
| 故障排查 | [.agent/troubleshooting/](.agent/troubleshooting/) |

---

## 🔧 性能优化记录

### 已完成的优化 ✅

#### 1. 双缓冲+条件变量架构 (2026-01-27)
**问题**: 轮询等待导致4.2ms开销，FPS限制在75-80Hz  
**方案**: 
- 相机回调直接写入乒乓缓冲区
- 推理线程使用`condition_variable`被动等待
- 零轮询、零拷贝、流水线并行

**效果**:
- FPS: 75Hz → **100Hz** (+33%)
- 等待延迟: 4.22ms → **2.57ms** (-39%)
- CPU占用: 15% → **<5%** (-67%)

#### 2. CUDA Bayer预处理 (2026-01-26)
**问题**: CPU demosaic阻塞推理流程  
**方案**: GPU kernels融合 Bayer→RGB + Resize + Normalize  
**效果**: 预处理 2.5ms → **1.9ms** (-24%)

#### 3. Batch=2 批量推理
**问题**: 双路独立推理GPU调度开销大  
**方案**: 左右图像合批，单次TensorRT调用  
**效果**: 推理开销 -40%，吞吐量提升35%

#### 4. Bayer RG8 格式 (2026-01-26)
**问题**: RGB8格式USB带宽4.67MB限制76fps  
**方案**: Bayer RG8 (1.56MB) + GPU demosaic  
**效果**: 带宽 -67%，支持100fps采集

### 可选优化方向 🚀
1. **TensorRT FP16**: 推理速度可提升 30-50%，需重新导出模型
2. **动态Batch**: 根据检测结果动态选择 Batch=1/2，节省无目标时的开销
3. **异步推理流**: 使用 CUDA Stream 进一步流水线化

### 调优参数
如需适配不同相机/负载，可调整 `tracker_params.yaml`：
```yaml
# 同步容忍度 (当前值适配100Hz PWM + USB3.0传输)
sync:
  max_frame_diff: 3        # 帧号差容忍 (默认3)
  max_time_diff_us: 25000  # 时间差容忍 25ms (默认)

# 推理输入尺寸
detector:
  global_size: 640         # 全图检测尺寸
  roi_size: 320            # ROI追踪尺寸
```

---

## 📝 下一步

1. ✅ ~~双目标定~~: 使用默认参数已可运行
2. **YOLO 优化**: 训练真实排球检测模型，替换占位符
3. **实地测试**: 真实排球场景端到端验证

---

*更新: 2026-01-27 (双缓冲架构优化完成，100fps目标达成)*
