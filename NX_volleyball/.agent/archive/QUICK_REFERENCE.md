# 🏐 排球追踪系统 - 快速参考

## 🚀 快速开始 (在 Jetson NX 上)

```bash
# 1. 一键编译测试
cd ~/desktop/yolo/yoloProject/NX_volleyball
./quick_build_test.sh

# 2. 运行基础节点 (自动加载配置)
ros2 run volleyball_stereo_driver stereo_system_node

# 3. 运行追踪节点 (自动加载配置)
ros2 run volleyball_stereo_driver volleyball_tracker_node
```

💡 **提示**: 配置文件会自动从安装目录加载，无需手动指定路径！

---

## 📂 项目结构

```
NX_volleyball/
├── quick_build_test.sh          # 快速编译脚本
├── NX_BUILD_TEST_GUIDE.md       # 编译测试指南
├── IMPLEMENTATION_SUMMARY.md    # 实现总结
├── ROS2_ARCHITECTURE_FINAL.md   # 架构设计
└── ros2_ws/
    └── src/volleyball_stereo_driver/  # 唯一的 ROS2 包
        ├── include/                    # 7 个头文件
        ├── src/                        # 9 个源文件
        ├── config/                     # 4 个配置文件
        ├── scripts/                    # 3 个脚本
        ├── model/                      # YOLO 模型目录
        └── calibration/                # 标定文件目录
```

---

## 🎯 核心组件

| 组件 | 文件 | 状态 | 说明 |
|------|------|------|------|
| ROI 管理器 | `roi_manager.cpp` | ✅ | ROI 裁切和坐标还原 |
| 卡尔曼滤波 | `kalman_filter_3d.cpp` | ✅ | 9 维状态估计 |
| 立体匹配 | `stereo_matcher.cpp` | ✅ | 三角测量 |
| YOLO 检测 | `yolo_detector.cpp` | ⚠️  | 占位符 (待完善) |
| 主追踪节点 | `volleyball_tracker_node.cpp` | ✅ | All-in-One 集成 |

---

## 📊 ROS 话题

### 基础节点 (stereo_system_node)
- `/camera_trigger` - PWM 触发时间戳 (100Hz)
- `/stereo/left/image_raw` - 左相机图像
- `/stereo/right/image_raw` - 右相机图像

### 追踪节点 (volleyball_tracker_node)
- `/volleyball/pose_3d` - 3D 位置
- `/volleyball/velocity` - 3D 速度
- `/volleyball/debug_info` - 调试信息 (JSON)

---

## ⚙️ 配置文件

| 文件 | 用途 |
|------|------|
| `config/system_params.yaml` | 基础节点参数 |
| `config/tracker_params.yaml` | 追踪节点参数 |
| `config/pwm_params.yaml` | PWM 参数 |
| `config/camera_params.yaml` | 相机参数 |

---

## 🔧 常用命令

```bash
# 编译
colcon build --packages-select volleyball_stereo_driver

# 查看话题
ros2 topic list
ros2 topic hz /camera_trigger
ros2 topic echo /volleyball/pose_3d

# 查看节点
ros2 node list
ros2 node info /volleyball_tracker

# 查看参数
ros2 param list /volleyball_tracker
ros2 param get /volleyball_tracker detector.confidence_threshold
```

---

## 🐛 常见问题

| 问题 | 解决方案 |
|------|----------|
| OpenCV 未找到 | `sudo apt install libopencv-dev` |
| 相机无法打开 | 检查 USB 连接和权限 |
| GPIO 权限不足 | `sudo usermod -a -G gpio $USER` |
| TensorRT 未找到 | 可选依赖，节点会自动跳过 |

---

## 📝 待完成

- [ ] 双目标定 (生成 `stereo_calib.yaml`)
- [ ] YOLO 模型 (训练/获取 `yolo11n.engine`)
- [ ] YOLO TensorRT 实现 (完善 `yolo_detector.cpp`)
- [ ] 端到端测试和优化

---

## 📖 详细文档

- 📘 编译测试: `NX_BUILD_TEST_GUIDE.md`
- 📘 实现总结: `IMPLEMENTATION_SUMMARY.md`
- 📘 架构设计: `ROS2_ARCHITECTURE_FINAL.md`
- 📘 进度追踪: `PROGRESS.md`

---

**版本**: v1.0  
**更新**: 2026-01-25  
**状态**: 代码实现完成，准备测试 ✅
