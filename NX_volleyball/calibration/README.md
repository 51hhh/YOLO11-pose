# 双目标定离线目录

本目录只保留离线标定、深度验证和算法对比脚本。棋盘图像采集统一使用
`stereo_3d_pipeline` 的 C++ 工具 `capture_chessboard`，Python 采集脚本已删除。

## 标准流程

### 1. 在 NX 上采集

在 NX 的 `~/NX_volleyball/stereo_3d_pipeline/build_standalone` 下执行：

```bash
./capture_chessboard \
  -o calibration_images \
  -g 17.0 \
  --serial-left 00D39342665 \
  --serial-right 00219471413
```

GUI 模式下按空格保存当前同步帧，`q` 或 `ESC` 退出。正式标定只使用硬触发采集，
不要用 `--free-run`。

### 2. 直接在 NX 上用 C++ 求解

```bash
./stereo_calibrate -s 26.0 --board-w 5 --board-h 8
```

默认读取 `calibration_images/left` 和 `calibration_images/right`，默认输出
`stereo_calib.yaml`。

### 3. 拷贝到本机用 Python 求解

在仓库根目录执行：

```bash
scp -r nvidia@10.42.0.148:~/NX_volleyball/stereo_3d_pipeline/build_standalone/calibration_images ./NX_volleyball/calibration
cd NX_volleyball/calibration
python3 stereo_calibration.py -s 26.0
```

Python 脚本只做离线求解，不负责相机采集。它按左右同名文件配对，默认使用与 C++
工具一致的 `CALIB_FIX_INTRINSIC`，报告最差重投影误差，不自动剔除图像对；如果 RMS
或校正 ROI 明显异常，会拒绝保存标定文件。需要联合优化内参时显式加
`--optimize-intrinsics`。

## 输出文件

| 文件 | 说明 |
|------|------|
| `stereo_calib.yaml` | OpenCV FileStorage 格式，C++ pipeline 直接读取 |
| `stereo_calib.npz` | Python 快速加载格式 |
| `calibration_images/` | 从 NX 下载的原始左右图像 |
| `calibration_images/capture_metadata.csv` | C++ 采集工具保存的同步水印记录 |

## 质量门槛

| 指标 | 要求 |
|------|------|
| 单目 RMS | 目标 `< 0.5 px`，良好 `< 0.3 px` |
| 立体 RMS | 必须 `< 1.0 px`，良好 `< 0.5 px` |
| 有效图像对 | 正式长基线标定建议 `>= 20` |
| 校正 ROI | 左右 ROI 不能为空 |
| 同步水印 | `capture_metadata.csv` 中 `frame_counter_delta` 应稳定为 `0` |

如果质量不达标，重新采集，优先改善棋盘覆盖、姿态分布、清晰度、曝光和棋盘刚性。
不要靠求解阶段选择性忽略坏图来掩盖采集问题。

## 辅助脚本

| 文件 | 功能 |
|------|------|
| `stereo_calibration.py` | 本机离线双目标定求解 |
| `stereo_depth_test.py` | 加载标定文件做校正、视差和点击测距验证 |
| `depth_algo_compare.py` | 对比多种深度算法输出 |
