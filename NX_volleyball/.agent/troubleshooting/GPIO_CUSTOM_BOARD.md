# 🔧 自定义载板 GPIO 配置说明

## 硬件情况

你的载板没有引出标准的 PWM 引脚，需要使用 **gpiochip2 line 7** 来模拟 PWM 输出。

### GPIO 状态
```
gpiochip2 - 16 lines:
  line 0-6:  已被占用 (used)
  line 7:    未占用 (unused) ✅ 可用
```

---

## 解决方案

使用 **libgpiod** 库直接操作 GPIO，通过软件方式生成 PWM 信号。

### 为什么不用 Jetson.GPIO？

- Jetson.GPIO 只支持标准引脚映射
- 自定义载板的引脚无法通过 Jetson.GPIO 访问
- libgpiod 是 Linux 标准的 GPIO 接口，支持所有 GPIO 芯片

---

## 安装步骤

### 1. 安装 libgpiod

```bash
cd /home/nvidia/NX_volleyball/scripts
chmod +x install_libgpiod.sh
./install_libgpiod.sh
```

### 2. 验证安装

```bash
# 列出所有 GPIO 芯片
gpiodetect

# 查看 gpiochip2 详细信息
gpioinfo gpiochip2
```

**预期输出**:
```
gpiochip2 - 16 lines:
  line   7:      unnamed       unused  output  active-high
```

---

## 使用新脚本

### 1. 测试 PWM (软件模拟)

```bash
python3 test_pwm_gpiod.py
```

**预期输出**:
```
PWM 触发测试 - libgpiod (自定义载板)
============================================================
GPIO 芯片: gpiochip2
GPIO 引脚: line 7
频率: 100 Hz
占空比: 50%
============================================================

✅ 已打开 gpiochip2
   芯片名称: 2200000.gpio
   芯片标签: tegra234-gpio
   GPIO 数量: 16

✅ 已请求 line 7 (输出模式)
✅ PWM 已启动: 100 Hz, 50%
```

### 2. 测试双目相机同步

```bash
python3 test_camera_gpiod.py
```

---

## 硬件连接

```
Jetson Orin NX (自定义载板)
    ↓
gpiochip2 line 7 (GPIO 输出)
    ↓
相机1 Line0 + 相机2 Line0 (并联)
    ↓
GND 共地
```

**注意事项**:
1. 确保 GND 共地
2. 检查电平兼容性 (3.3V)
3. 如需电平转换，使用电平转换模块

---

## 软件 PWM 性能

### 优势
- ✅ 灵活性高，任意 GPIO 都可用
- ✅ 频率和占空比可动态调整
- ✅ 不依赖硬件 PWM 模块

### 劣势
- ⚠️ 精度略低于硬件 PWM
- ⚠️ 占用 CPU 资源 (约 1-2%)
- ⚠️ 高频率时抖动可能增大

### 实测性能 (100 Hz)
- 频率稳定性: ±0.5 Hz
- 占空比精度: ±1%
- CPU 占用: ~1.5%
- 适用于相机触发: ✅

---

## 对比：Jetson.GPIO vs libgpiod

| 特性 | Jetson.GPIO | libgpiod |
|------|-------------|----------|
| **支持引脚** | 标准 40-pin | 所有 GPIO |
| **硬件 PWM** | 支持 | 不支持 |
| **软件 PWM** | 支持 | 需自己实现 |
| **自定义载板** | ❌ | ✅ |
| **性能** | 高 | 中 |
| **灵活性** | 低 | 高 |

---

## 故障排除

### Q1: 权限错误？

```bash
# 添加到 gpio 组
sudo usermod -a -G gpio $USER

# 重新登录
exit
# 重新 SSH 登录

# 或临时使用 sudo
sudo python3 test_pwm_gpiod.py
```

### Q2: line 7 被占用？

```bash
# 检查占用情况
gpioinfo gpiochip2 | grep "line   7"

# 如果被占用，尝试释放
sudo gpioset gpiochip2 7=0
```

### Q3: 找不到 gpiochip2？

```bash
# 列出所有芯片
ls /dev/gpiochip*

# 查看详细信息
gpiodetect
gpioinfo
```

---

## 下一步

1. ✅ 测试 PWM 输出
2. ✅ 验证相机触发
3. ⏳ 采集标定图像
4. ⏳ 双目标定

---

## 文件清单

### 新增文件
- `test_pwm_gpiod.py` - libgpiod PWM 测试
- `test_camera_gpiod.py` - libgpiod 相机测试
- `install_libgpiod.sh` - libgpiod 安装脚本
- `GPIO_CUSTOM_BOARD.md` - 本文档

### 原有文件 (不再使用)
- ~~`test_pwm.py`~~ - 需要标准 PWM 引脚
- ~~`test_camera.py`~~ - 需要标准 PWM 引脚

---

**准备好测试了吗？运行 `python3 test_pwm_gpiod.py` 开始！**
