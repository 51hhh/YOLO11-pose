# 双目相机标定工具（Legacy Python）

> 当前正式标定入口已经迁移到 C++ 工具：
> `NX_volleyball/stereo_3d_pipeline/build_standalone/capture_chessboard`
> 和 `NX_volleyball/stereo_3d_pipeline/build_standalone/stereo_calibrate`。
>
> 原因：C++ 工具复用主进程的海康 SDK 配置、FrameSpecInfo 水印同步、PWM 触发顺序和 Bayer 处理逻辑，更适合现在的 USB3 双目长基线标定。
> 本目录 Python 脚本仅保留为历史参考和离线验证，不建议用于新的正式标定。

新流程文档见：

- `NX_volleyball/stereo_3d_pipeline/docs/相机标定实施手册.md`
- `NX_volleyball/stereo_3d_pipeline/docs/04_标定与相机.md`
- `NX_volleyball/stereo_3d_pipeline/README.md`

下面内容是旧 Python 流水线说明，仅用于复现历史标定或离线对比。新标定请优先阅读上面的 C++ 文档入口。

## Legacy 流程概览

```
采集棋盘格图像 → 立体标定 → 深度测试验证
capture_chessboard.py → stereo_calibration.py → stereo_depth_test.py
```

## Legacy 快速开始

### 1. 采集标定图像（不推荐用于新标定）

```bash
# PWM硬件触发模式(默认)
python3 capture_chessboard.py

# 自由运行模式(无PWM)
python3 capture_chessboard.py --free-run
```

操作: **空格**=采集 | **q/ESC**=退出 | **c**=清空

图像保存到 `calibration_images/left/` 和 `calibration_images/right/`。

> **采集建议:**
> - 采集 15~25 对，尽量覆盖画面四个象限
> - 棋盘格尽量铺满画面，包含倾斜角度
> - 保持棋盘格平整，避免弯曲
> - 图像自动保存为无损 PNG

### 2. 运行标定（仅历史复现）

```bash
# -s 方格边长(mm)，必须准确测量
python3 stereo_calibration.py -s 24.5
```

输出:
- `stereo_calib.yaml` — OpenCV FileStorage 格式 (C++/Python 通用)
- `stereo_calib.npz` — NumPy 格式 (Python 快速加载)

**标定质量判断:**
- 单目 RMS < 0.5 px → 优秀
- 立体 RMS < 1.0 px → 合格
- 焦距 fx/fy 不应超过 ~4000

### 3. 验证深度（仅历史复现）

```bash
# 交互浏览所有图像对，鼠标点击测距
python3 stereo_depth_test.py -c stereo_calib.yaml

# 指定单对图像
python3 stereo_depth_test.py -c stereo_calib.yaml --left l.png --right r.png
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `capture_chessboard.py` | 双目图像采集 (PWM触发/自由运行) |
| `stereo_calibration.py` | 标定流水线 (单目→剔除→立体→校正) |
| `stereo_depth_test.py` | 深度验证 (视差图/深度图/点击测距) |

## 默认参数

在各脚本顶部可修改:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BOARD_WIDTH` | 5 | 棋盘格内角点列数 |
| `BOARD_HEIGHT` | 8 | 棋盘格内角点行数 |
| `SQUARE_SIZE` | 30.0 | 方格边长 (mm) |
| `EXPOSURE_TIME` | 2000 | 曝光时间 (μs) |
| `GPIOCHIP` | gpiochip2 | GPIO芯片 (libgpiod) |
| `LINE_OFFSET` | 7 | GPIO引脚 (gpiochip2 line 7) |

## 关键修正 (参考知乎教程)

1. **`CALIB_CB_FILTER_QUADS`** — 角点检测时过滤假四边形
2. **`CALIB_USE_INTRINSIC_GUESS`** — 立体标定以单目结果为初值 (非 `flags=0`)
3. **旧式重投影筛查** — Python 历史脚本包含重投影误差筛查逻辑；当前正式 C++ 采集工具保持纯采集，质量判断在标定和验收阶段完成
4. **PNG无损** — 避免JPEG压缩伪影影响亚像素精度
5. **视差÷16** — StereoSGBM返回定点数，需除以16再算深度

## 部署到 ROS2

```bash
cp stereo_calib.yaml ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/calibration/
```
