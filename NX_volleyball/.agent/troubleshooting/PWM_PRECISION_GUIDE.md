# 🎯 PWM 精度优化指南

## 问题分析

你测量到的频率是 **93Hz**，而不是目标的 100Hz，误差约 **7%**。

### 误差来源

1. **Python `time.sleep()` 精度不足**
   - `sleep()` 依赖操作系统调度
   - 最小粒度通常 1-10ms
   - 累积误差导致频率偏低

2. **线程调度延迟**
   - Linux 不是实时操作系统
   - 线程可能被抢占
   - 上下文切换开销

3. **CPU 频率波动**
   - 动态调频 (DVFS) 影响定时精度
   - 节能模式降低性能

---

## 解决方案

### 方案 1: 高精度 PWM 脚本 (推荐)

使用新的 `test_pwm_precise.py`，采用以下优化：

#### 核心技术

1. **误差补偿算法**
   ```python
   # 不是每次都 sleep(period)
   # 而是计算到下一个边沿的精确时间
   next_edge_time += high_time
   wait_time = next_edge_time - time.perf_counter()
   sleep(wait_time)
   ```

2. **忙等待 (Busy-Wait)**
   ```python
   # 短时间 (<0.5ms) 使用忙等待
   target = time.perf_counter() + duration
   while time.perf_counter() < target:
       pass  # CPU 空转，但精度极高
   ```

3. **实时线程优先级**
   ```python
   # 使用 SCHED_FIFO 调度策略
   sched_setscheduler(0, SCHED_FIFO, priority=50)
   ```

#### 使用方法

```bash
# 1. 优化系统
cd ~/NX_volleyball/scripts
chmod +x optimize_system.sh
sudo ./optimize_system.sh

# 2. 运行高精度 PWM
sudo python3 test_pwm_precise.py
```

**预期输出**:
```
✅ 高精度 PWM 已启动: 100 Hz, 50% (忙等待)
  ✅ 线程优先级已提升 (SCHED_FIFO)

  周期: 500 | 实际频率: 99.98 Hz | 误差: -0.02 Hz | 抖动: 0.015 ms
  周期: 500 | 实际频率: 100.01 Hz | 误差: +0.01 Hz | 抖动: 0.012 ms

最终统计:
  目标频率: 100 Hz
  实际频率: 99.995 Hz
  频率误差: -0.005 Hz (-0.01%)
  周期抖动: 0.013 ms
```

---

### 方案 2: 系统优化

运行 `optimize_system.sh` 进行系统级优化：

```bash
sudo ./optimize_system.sh
```

**优化内容**:
1. ✅ 锁定 CPU/GPU 频率 (`jetson_clocks`)
2. ✅ 禁用 CPU 节能模式
3. ✅ 配置实时调度权限
4. ✅ 减少系统中断

---

### 方案 3: 硬件 PWM (终极方案)

如果软件 PWM 仍不满足要求，考虑使用硬件 PWM：

#### 选项 A: 外部 PWM 模块

使用专用 PWM 芯片 (如 PCA9685):
- 16 路硬件 PWM
- I2C 接口
- 精度 <0.1%

#### 选项 B: Arduino/STM32 辅助

使用微控制器生成精确 PWM:
```
Orin NX (串口) → Arduino → PWM 输出 → 相机触发
```

---

## 性能对比

| 方案 | 频率精度 | CPU 占用 | 抖动 | 复杂度 |
|------|----------|----------|------|--------|
| **原始 sleep** | ±7% (93Hz) | <1% | 高 | 低 |
| **高精度 + 优化** | ±0.1% (99.9Hz) | ~5% | 低 | 中 |
| **硬件 PWM** | ±0.01% | 0% | 极低 | 高 |

---

## 测试步骤

### 1. 运行高精度脚本

```bash
sudo python3 test_pwm_precise.py
```

### 2. 示波器验证

连接示波器到 gpiochip2 line 7:
- 测量频率
- 测量占空比
- 观察抖动

### 3. 调整参数

如果频率仍有偏差，可以微调:

```python
# 在 test_pwm_precise.py 中
PWM_FREQ = 100.5  # 补偿系统延迟
```

---

## 常见问题

### Q1: 为什么需要 sudo？

**A**: 提升线程优先级需要 root 权限。不用 sudo 也能运行，但精度会降低。

### Q2: CPU 占用太高？

**A**: 忙等待会占用 CPU。可以调整阈值:

```python
# 减少忙等待时间
BUSY_WAIT_THRESHOLD = 0.0001  # 0.1ms
```

### Q3: 频率还是不准？

**A**: 尝试以下方法:
1. 确认已运行 `optimize_system.sh`
2. 关闭其他程序
3. 使用 `taskset` 绑定到特定 CPU:
   ```bash
   sudo taskset -c 0 python3 test_pwm_precise.py
   ```

---

## 实测结果 (预期)

### 优化前
- 频率: 93 Hz (误差 -7%)
- 抖动: ~1 ms
- CPU: <1%

### 优化后
- 频率: 99.9-100.1 Hz (误差 <0.1%)
- 抖动: <0.05 ms
- CPU: ~5%

---

## 下一步

1. ✅ 运行 `sudo ./optimize_system.sh`
2. ✅ 测试 `sudo python3 test_pwm_precise.py`
3. ✅ 示波器验证频率
4. ⏳ 如果满意，更新相机测试脚本

---

**准备好测试了吗？运行命令开始优化！**
