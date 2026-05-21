# 开发流程与约束

## 远程开发流程

本项目在 **本地开发** + **AGX Orin 远程编译运行**。

### 设备信息

| 角色 | IP | 用户 | 密码 |
|------|-----|------|------|
| AGX Orin (运行) | 192.168.31.223 | nvidia | nvidia |
| 开发机 (编辑) | localhost | rick | - |

### 工作流

```
1. 本地修改代码 (IDE 工具, 不用终端写代码)
2. rsync 同步到远程
3. 远程 cmake + make 编译
4. 远程运行 / 检查 topic
```

### 命令模板

```bash
# 同步代码到 AGX (排除 build/ 和 models/)
rsync -avz --delete --exclude='build/' --exclude='models/' \
  /home/rick/mid360/YOLO11-pose/NX_volleyball/stereo_3d_pipeline/ \
  nvidia@192.168.31.223:~/stereo_3d_pipeline/ \
  --rsh="sshpass -p 'nvidia' ssh"

# 远程编译 (带 ROS2)
sshpass -p 'nvidia' ssh nvidia@192.168.31.223 \
  "cd ~/stereo_3d_pipeline/build && source /opt/ros/humble/setup.bash && \
   cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_ROS2=ON && make -j6"

# 远程运行
sshpass -p 'nvidia' ssh nvidia@192.168.31.223 \
  "cd ~/stereo_3d_pipeline && source /opt/ros/humble/setup.bash && \
   ./build/stereo_pipeline -c config/pipeline_zed.yaml"

# 检查 ROS2 topic
sshpass -p 'nvidia' ssh nvidia@192.168.31.223 \
  "source /opt/ros/humble/setup.bash && ros2 topic list"
```

## 编码约束

1. **配置唯一源**: 所有运行时参数来自 YAML 文件，代码中不重复定义默认值
2. **免编译调参**: 修改 `config/pipeline_zed.yaml` 后重启即生效
3. **不做兜底**: 不在代码中插入防御性默认值，不做多层 fallback
4. **精简设计**: 每个模块职责单一、接口清晰，不添加"以防万一"的逻辑

## 异常处理规则

- 连续 3 次编译/运行失败 → 使用提问工具询问用户
- SSH 连接失败 / IP 不通 → 使用提问工具询问用户
- 需要物理操作 (接显示器、插USB等) → 使用提问工具
- 执行危险操作 (删除文件、格式化等) → 事先确认

## 项目构建选项

| CMake 选项 | 说明 |
|-----------|------|
| `-DUSE_ROS2=ON` | 启用 ROS2 bridge (需 source /opt/ros/humble/setup.bash) |
| `-DCMAKE_BUILD_TYPE=Release` | 优化编译 |
| `-DCUDA_ARCH="87"` | Orin 架构 (SM87) |

## 关键文件

| 文件 | 作用 |
|------|------|
| `config/pipeline_zed.yaml` | ZED 配置 (当前使用) |
| `src/main.cpp` | 入口: 配置加载 + pipeline + ROS2 集成 |
| `src/pipeline/pipeline.h` | PipelineConfig + Ros2BridgeConfig 定义 |
| `src/ros/goal_pose_bridge.h/cpp` | ROS2 bridge 实现 |
| `models/best_960_fp16.engine` | TRT 引擎 (56MB, 不入 git) |
