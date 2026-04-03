# PWM 频率抖动与 Bayer 颜色错误修复报告

## 一、问题描述

| # | 现象 | 根因 |
|---|------|------|
| 1 | 示波器测量 PWM 频率波动 | `sleep_for` 受系统调度干扰，无内存锁定/CPU 亲和性 |
| 2 | 蓝色变黄色、黄色变蓝色 | 海康 BayerRG8 对应 OpenCV `COLOR_BayerBG2BGR`，代码误用了 `COLOR_BayerRG2BGR` |

---

## 二、PWM 修复 — `pwm_trigger.h`

### 2.1 旧方案问题

```
accurateSleep()
  ├─ std::this_thread::sleep_for(dur - 0.5ms)   ← 内核调度抖动 50-200μs
  └─ busy-wait last 0.5ms                        ← 可被高优先级线程抢占
```

- SCHED_FIFO 优先级仅 50，低于部分内核线程
- 无 `mlockall` → 缺页中断导致偶发延迟峰值
- 无 CPU 亲和性 → 线程在核间迁移增加 jitter
- sysfs 后端完全没有 RT 调度

### 2.2 新方案

```
clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL)
  ├─ 内核 hrtimer 绝对时间唤醒，抖动 < 50μs (SCHED_FIFO 下)
  ├─ 绝对时间戳：即使某次唤醒延迟，下一次仍准确（无误差累积）
  └─ 比 sleep_for + busy-wait 更省 CPU
```

### 2.3 RT 线程配置

| 机制 | 说明 |
|------|------|
| `SCHED_FIFO` priority 80 | 高于大多数内核线程，确保不被抢占 |
| `mlockall(MCL_CURRENT \| MCL_FUTURE)` | 锁定所有页面，消除缺页中断 |
| `pthread_setaffinity_np` | 绑定到最后一个 CPU 核，避免迁移开销 |
| 绝对时间戳跟踪 | `clock_gettime` 获取初始时间，每半周期累加 |

### 2.4 代码变更

- 删除 `accurateSleep()` 函数
- 新增 `setupRealtimeThread()` — 统一设置 SCHED_FIFO + mlockall + CPU affinity
- 新增 `sleepUntil()` — 封装 `clock_nanosleep(TIMER_ABSTIME)`
- 新增 `timespecAdd()` — 纳秒级时间累加
- libgpiod 和 sysfs 两个后端均使用新定时机制
- 新增 `#include <time.h>`, `<sys/mman.h>`, `<pthread.h>`，`<sched.h>` 不再条件编译

---

## 三、Bayer 颜色修复

### 3.1 根因分析

海康 SDK 与 OpenCV 的 Bayer 命名规则不同：

| 海康 SDK 像素格式 | 含义 | OpenCV 转换标志 |
|-------------------|------|----------------|
| `PixelType_Gvsp_BayerRG8` | 传感器 (0,0) 为 R | `cv::COLOR_BayerBG2BGR` |

海康按传感器左上角像素命名；OpenCV 按 2×2 块的右下角像素命名。两者对**同一物理布局**的命名恰好**反转**。

使用错误的 `COLOR_BayerRG2BGR` 会导致 R↔B 通道互换 → 蓝色变黄色、黄色变蓝色。

> **参考**: 项目文档 `docs/海康相机配置手册.md` 第 262 行已记录正确映射关系

### 3.2 修改文件清单

| 文件 | 修改位置 | 变更 |
|------|---------|------|
| `src/calibration/capture_chessboard.cpp` | L171-172 | `COLOR_BayerRG2BGR` → `COLOR_BayerBG2BGR` (×2) |
| `src/calibration/stereo_calibrate.cpp` | L103, L375-376 | `COLOR_BayerRG2BGR` → `COLOR_BayerBG2BGR` (×3) |
| `calibration/stereo_calibration.py` | L48 | `cv2.COLOR_BayerRG2BGR` → `cv2.COLOR_BayerBG2BGR` |
| `calibration/stereo_depth_test.py` | L113 | `cv2.COLOR_BayerRG2BGR` → `cv2.COLOR_BayerBG2BGR` |
| `Bayer_CUDA加速说明.md` | L298 | 文档代码示例同步修正 |

### 3.3 不受影响的部分

- **主 pipeline** (`pipeline.h/cpp`, `main.cpp`): 直接将 BayerRG8 当灰度 U8 送入 VPI，不做颜色转换
- **TRT 检测器** (`detect_preprocess.cu`): 处理灰度输入 R=G=B=pixel/255.0
- **`docs/海康相机配置手册.md`**: 已是正确的 `COLOR_BayerBG2BGR`，无需修改

---

## 四、验证

### 编译

```
NX 编译: cmake + make -j4 → 0 error, 3 targets built
  ├─ capture_chessboard  ✓
  ├─ stereo_calibrate    ✓
  └─ stereo_pipeline     ✓
```

### 测试建议

| 测试项 | 方法 | 预期 |
|--------|------|------|
| PWM 稳定性 | 示波器测 100Hz，运行 `stress -c 6` 加载 | 频率波动 < ±0.5% |
| Bayer 颜色 | `capture_chessboard --headless=false`，观察实时画面 | 颜色还原正确 |
| 标定色彩 | `stereo_calibrate` 可视化窗口 | 棋盘格颜色正常 |
| Python 脚本 | `python3 stereo_calibration.py` | BGR 输出颜色正确 |

---

## 五、备忘

- PWM 需要 `sudo` 或 `root` 权限才能设置 `SCHED_FIFO` 和 `mlockall`
- 如无权限，程序会打印警告但仍能运行（精度降为内核默认调度级别）
- 若未来相机切换为其他 Bayer 格式，需重新对照海康与 OpenCV 命名映射
