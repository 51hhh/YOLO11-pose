#!/usr/bin/env python3
"""
高精度 PWM 触发脚本 (使用 libgpiod)
使用误差补偿算法，实现精确的 100Hz 输出

优化技术:
  1. 高精度时钟 (perf_counter)
  2. 累积误差补偿
  3. 实时线程优先级
  4. 忙等待 (busy-wait) 精确定时

适用于 gpiochip2 line 7
"""

import time
import signal
import sys
import threading
import os

try:
    import gpiod
except ImportError:
    print("❌ 错误: 未安装 libgpiod")
    print("安装方法: sudo apt install python3-libgpiod")
    sys.exit(1)

# ==================== 配置 ====================
GPIOCHIP = "gpiochip2"
LINE_OFFSET = 7
PWM_FREQ = 100  # 目标频率 100 Hz
PWM_DUTY = 50   # 占空比 50%

# 高精度模式配置
USE_BUSY_WAIT = True  # 使用忙等待提高精度
BUSY_WAIT_THRESHOLD = 0.0005  # 0.5ms 以下使用忙等待

# ==================== 全局变量 ====================
chip = None
line = None
pwm_thread = None

# ==================== 高精度软件 PWM ====================
class HighPrecisionPWM:
    """高精度软件 PWM 实现"""
    
    def __init__(self, line, frequency, duty_cycle, use_busy_wait=True):
        """
        初始化高精度 PWM
        
        Args:
            line: gpiod.Line 对象
            frequency: 频率 (Hz)
            duty_cycle: 占空比 (0-100)
            use_busy_wait: 是否使用忙等待
        """
        self.line = line
        self.frequency = frequency
        self.duty_cycle = duty_cycle
        self.use_busy_wait = use_busy_wait
        self.running = False
        self.thread = None
        
        # 计算时间参数
        self.period = 1.0 / frequency  # 周期 (秒)
        self.high_time = self.period * (duty_cycle / 100.0)
        self.low_time = self.period - self.high_time
        
        # 统计信息
        self.actual_periods = []
        self.max_periods = 1000  # 保留最近 1000 个周期
    
    def _accurate_sleep(self, duration):
        """
        高精度睡眠
        
        Args:
            duration: 睡眠时间 (秒)
        """
        if not self.use_busy_wait or duration > BUSY_WAIT_THRESHOLD:
            # 长时间睡眠: 使用 sleep 节省 CPU
            # 提前唤醒一点，然后用忙等待补偿
            if self.use_busy_wait and duration > BUSY_WAIT_THRESHOLD:
                time.sleep(duration - BUSY_WAIT_THRESHOLD)
                duration = BUSY_WAIT_THRESHOLD
            else:
                time.sleep(duration)
                return
        
        # 短时间睡眠: 使用忙等待
        target = time.perf_counter() + duration
        while time.perf_counter() < target:
            pass  # 忙等待
    
    def _pwm_loop(self):
        """PWM 循环 - 带误差补偿"""
        # 设置线程优先级 (需要 root 权限)
        try:
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            # SCHED_FIFO = 1, priority = 50
            class sched_param(ctypes.Structure):
                _fields_ = [('sched_priority', ctypes.c_int)]
            
            param = sched_param(50)
            libc.sched_setscheduler(0, 1, ctypes.byref(param))
            print("  ✅ 线程优先级已提升 (SCHED_FIFO)")
        except:
            print("  ⚠️  无法提升线程优先级 (需要 sudo)")
        
        # 初始化
        next_edge_time = time.perf_counter()
        cycle_count = 0
        last_stat_time = time.perf_counter()
        
        while self.running:
            cycle_start = time.perf_counter()
            
            # === 高电平 ===
            self.line.set_value(1)
            next_edge_time += self.high_time
            
            # 精确等待到下一个边沿
            wait_time = next_edge_time - time.perf_counter()
            if wait_time > 0:
                self._accurate_sleep(wait_time)
            
            # === 低电平 ===
            self.line.set_value(0)
            next_edge_time += self.low_time
            
            # 精确等待到下一个边沿
            wait_time = next_edge_time - time.perf_counter()
            if wait_time > 0:
                self._accurate_sleep(wait_time)
            
            # 统计实际周期
            cycle_end = time.perf_counter()
            actual_period = cycle_end - cycle_start
            self.actual_periods.append(actual_period)
            
            if len(self.actual_periods) > self.max_periods:
                self.actual_periods.pop(0)
            
            cycle_count += 1
            
            # 每秒打印一次统计
            if cycle_end - last_stat_time >= 5.0:
                self._print_stats(cycle_count, cycle_end - last_stat_time)
                cycle_count = 0
                last_stat_time = cycle_end
    
    def _print_stats(self, cycles, duration):
        """打印统计信息"""
        if not self.actual_periods:
            return
        
        import numpy as np
        periods = np.array(self.actual_periods)
        actual_freq = 1.0 / periods.mean()
        freq_std = periods.std() * 1000  # ms
        
        print(f"  周期: {cycles} | "
              f"实际频率: {actual_freq:.2f} Hz | "
              f"误差: {actual_freq - self.frequency:+.2f} Hz | "
              f"抖动: {freq_std:.3f} ms")
    
    def start(self):
        """启动 PWM"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._pwm_loop, daemon=True)
        self.thread.start()
        
        mode = "忙等待" if self.use_busy_wait else "sleep"
        print(f"✅ 高精度 PWM 已启动: {self.frequency} Hz, {self.duty_cycle}% ({mode})")
    
    def stop(self):
        """停止 PWM"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        self.line.set_value(0)
        print("✅ PWM 已停止")
        
        # 打印最终统计
        if self.actual_periods:
            import numpy as np
            periods = np.array(self.actual_periods)
            actual_freq = 1.0 / periods.mean()
            print(f"\n最终统计:")
            print(f"  目标频率: {self.frequency} Hz")
            print(f"  实际频率: {actual_freq:.3f} Hz")
            print(f"  频率误差: {actual_freq - self.frequency:+.3f} Hz ({(actual_freq - self.frequency)/self.frequency*100:+.2f}%)")
            print(f"  周期抖动: {periods.std()*1000:.3f} ms")
    
    def change_frequency(self, frequency):
        """改变频率"""
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.high_time = self.period * (self.duty_cycle / 100.0)
        self.low_time = self.period - self.high_time
        print(f"  频率已改变: {frequency} Hz")
    
    def change_duty_cycle(self, duty_cycle):
        """改变占空比"""
        self.duty_cycle = duty_cycle
        self.high_time = self.period * (duty_cycle / 100.0)
        self.low_time = self.period - self.high_time
        print(f"  占空比已改变: {duty_cycle}%")

# ==================== 信号处理 ====================
def signal_handler(sig, frame):
    """Ctrl+C 处理"""
    print("\n\n🛑 停止 PWM...")
    cleanup()
    sys.exit(0)

def cleanup():
    """清理资源"""
    global pwm_thread, line, chip
    
    try:
        if pwm_thread:
            pwm_thread.stop()
        
        if line:
            line.release()
        
        if chip:
            chip.close()
        
        print("✅ GPIO 已清理")
    except:
        pass

# ==================== 主程序 ====================
def main():
    global chip, line, pwm_thread
    
    print("="*60)
    print("高精度 PWM 触发 - libgpiod")
    print("="*60)
    print(f"GPIO 芯片: {GPIOCHIP}")
    print(f"GPIO 引脚: line {LINE_OFFSET}")
    print(f"目标频率: {PWM_FREQ} Hz")
    print(f"占空比: {PWM_DUTY}%")
    print(f"精度模式: {'忙等待' if USE_BUSY_WAIT else '标准 sleep'}")
    print("="*60)
    print()
    
    # 检查是否为 root (提升优先级需要)
    if os.geteuid() != 0:
        print("⚠️  建议使用 sudo 运行以提升线程优先级")
        print("   sudo python3 test_pwm_precise.py")
        print()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # 打开 GPIO 芯片
        chip = gpiod.Chip(GPIOCHIP)
        print(f"✅ 已打开 {GPIOCHIP}")
        
        # 获取 GPIO 线
        line = chip.get_line(LINE_OFFSET)
        
        # 请求输出模式
        line.request(consumer="pwm_trigger_hp", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
        print(f"✅ 已请求 line {LINE_OFFSET} (输出模式)")
        print()
        
        # 创建高精度 PWM
        pwm_thread = HighPrecisionPWM(line, PWM_FREQ, PWM_DUTY, use_busy_wait=USE_BUSY_WAIT)
        pwm_thread.start()
        print()
        
        # 长时间运行测试
        print("📊 运行中... (按 Ctrl+C 停止)")
        print("-" * 60)
        
        try:
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            pass
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    finally:
        # 清理
        cleanup()
    
    print()
    print("="*60)
    print("✅ 测试完成")
    print("="*60)
    print()
    print("提示:")
    print("  1. 使用示波器验证频率")
    print("  2. 如果频率仍不准确:")
    print("     - 使用 sudo 运行提升优先级")
    print("     - 关闭其他占用 CPU 的程序")
    print("     - 锁定 CPU 频率: sudo jetson_clocks")
    print()

if __name__ == "__main__":
    main()
