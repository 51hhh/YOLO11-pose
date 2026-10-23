# 实机与标定辅助工具

主程序、正式采集和回放统一经根目录的 `volleyball` CLI 进入。仓库只保留三个
受测辅助程序：`hik_frame_metadata_probe`、`capture_chessboard` 和
`stereo_calibrate`。它们由根工程构建并复用正式 MVS/GPIO 适配器；一次性脚本
不进入仓库。

普通主机只做离线双目标定时，不需要安装 MVS、CUDA、TensorRT、ROS2 或
libgpiod：

```bash
cmake --preset host-calibration-release
cmake --build --preset host-calibration-release --parallel
build/host-calibration-release/tools/hardware/stereo_calibrate --help
```

NX 预设仍会一次构建主程序和全部三个现场工具：

```bash
cmake --preset nx-mainline-release
cmake --build --preset nx-mainline-release --parallel
```

## 推荐顺序

1. 用 `hik_frame_metadata_probe` 验证相机角色、FrameCounter、GPIO 上升沿与 USB 到达乱序。
2. 用 `capture_chessboard` 通过正式硬触发链采集标定图。
3. 用 `stereo_calibrate` 计算标定；人工复核重投影误差、基线方向和已知距离误差。
4. 将通过复核的 YAML 放入 `configs/calibration/`，更新 profile 与哈希后运行 `volleyball doctor`。

相机工具与生产主线共享同一个 `libgpiod` 触发适配器。硬件构建缺少
`gpiod.h`/`libgpiod` 会在 CMake 阶段失败，运行时无法请求配置的 chip/line 也会直接
失败；不再使用已废弃的 sysfs GPIO，也不会根据硬编码 base 猜测全局 GPIO 编号。
5. 只有 doctor 和[验收门禁](acceptance.md)通过后，才运行 `volleyball capture` 采集正式 trial。

## 帧元数据探针

```bash
build/nx-mainline-release/tools/hardware/hik_frame_metadata_probe --help
build/nx-mainline-release/tools/hardware/hik_frame_metadata_probe \
  --left-sn <左相机序列号> --right-sn <右相机序列号> \
  --frames 1000 --trigger-chip gpiochip2 --trigger-line 7 --pwm-hz 90
```

探针用于确认 payload 自带的 `FrameCounter` 是否连续、左右是否同号，以及 USB 到达顺序是否会交换。它会拒绝左右角色解析到同一物理相机，并在打开设备后强制读取 `DeviceFirmwareVersion`；启动日志中的序列号、枚举型号和固件版本应原样填入实机清单，固件节点缺失或为空时探针直接失败。探针与正式取流保持相同的最小水印策略：先关闭持久化的多余 FrameSpecInfo，再只启用并读回确认 `Timestamp`、`Framecounter`、`ExtTriggerCount`。固定曝光、增益、上升沿触发的关键读回不匹配，或 SDK 缓冲归还失败，都会以非零状态退出，不能把不完整元数据当作通过。GPIO 上升沿是曝光开始估计的权威时基；当前相机曝光延迟按 0 ns 假设并明确标记为未测量，后续示波器测量不得覆盖原始厂商时间戳或主机到达时间。

## 标定图采集

```bash
build/nx-mainline-release/tools/hardware/capture_chessboard --help
build/nx-mainline-release/tools/hardware/capture_chessboard \
  --serial-left <左相机序列号> --serial-right <右相机序列号> \
  -o /path/to/new_calibration_images -e 9867 -g 11.9906 \
  --trigger-chip gpiochip2 --trigger-line 7 --pwm-hz 90
```

正式标定应使用与运行相同的分辨率、镜头、光圈、焦距、对焦、全局快门、曝光和硬触发方式。`--free-run`、`--no-pwm` 仅用于诊断，不应用于生成生产标定。
输出目录必须不存在或为空，工具不会覆盖已有标定图或 metadata；需要重采时请使用新的目录，旧数据由操作者显式归档或删除。
每对图片先写入同目录临时文件，左右两侧和 metadata 均成功后才发布；编码、重命名或 metadata 写入失败会清理本对图片并返回非零。`-n` 自动采集若未达到请求数量，同样视为失败，不能把不完整目录误当作成功标定集。

## 双目标定

```bash
build/host-calibration-release/tools/hardware/stereo_calibrate --help
build/host-calibration-release/tools/hardware/stereo_calibrate \
  --images-dir /path/to/calibration_images \
  --square-size <方格边长毫米> \
  --output /tmp/new_stereo_calib.yaml
```

在 NX 上也可使用 `build/nx-mainline-release/tools/hardware/stereo_calibrate`，两者
执行同一份纯 OpenCV 实现；主机预设只是去掉无关的相机 SDK 依赖。

输出文件必须不存在，工具不会覆盖已有候选或权威标定。先保留候选输出和原始图像，完成静态已知位置、不同深度和视场边缘验证，再通过普通代码评审替换 `configs/calibration/stereo_calib.yaml`。
角点检测和标定在 CPU 上执行，可用 `--jobs` 控制检测线程；它不启动相机，也不
借用实时 CUDA 图像流水线。
角点检测与左右单目标定的工作线程异常会使工具明确失败；最终 YAML 先写入同目录 `.partial` 文件，完整关闭并确认非空后再以无覆盖方式发布。残留 `.partial` 会被拒绝并要求人工检查，不会被下一次运行静默复用。

## 标定资产归档

`configs/calibration/vehicle_stereo_main.yaml` 是 profile 引用的资产描述，固定标定
ID、创建时间、源文件和坐标系；`configs/calibration/stereo_calib.yaml` 保存 OpenCV
原始内参、畸变、左右外参和校正投影矩阵。两者职责不同，不能合并成一份可随意
覆盖的运行配置。

正式采集时，`python/volleyball_vision/calibration.py` 从校正投影矩阵生成符合
v2.1 契约的 `calibrations/<calibration_id>.json`。当前资产只证明左目校正光心
坐标系和实测双目基线；rig 到 `base_link` 的变换尚未独立测量，在测量完成前必须
保持缺失，不能用猜测值补齐。

## 主流水线

不要手工运行可执行文件或编辑生成 YAML。使用统一入口：

```bash
volleyball config render-mainline --profile nx-rdk
volleyball doctor --profile nx-rdk
volleyball run --profile nx-rdk --dry-run
volleyball run --profile nx-rdk
```

自由飞行数据使用 `volleyball capture --profile free-flight ...`，以便生成完整 v2.1 元数据、逐帧原始引用、哈希和原子发布记录。

## 不进入仓库的内容

- 标定原图、视频、TensorRT engine 和现场日志；
- 临时性能结果、截图和人工分析 CSV；
- 只执行一次的转换、同步或诊断脚本；
- SDK、上游研究仓库和本地构建目录。

若某个诊断过程需要长期复现，应把它实现为有参数校验、有测试的 CLI 子命令或明确的 C++ 工具，并同步更新本页和验收文档。
