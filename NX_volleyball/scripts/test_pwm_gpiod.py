#!/usr/bin/env python3
"""
PWM 触发测试脚本 (使用 libgpiod)
适用于自定义载板，使用 gpiochip2 line 7

硬件配置:
  - gpiochip2 line 7 (未占用) → 相机1 Line0 + 相机2 Line0 (并联)
  - GND → 相机 GND

测试内容:
  1. 软件 PWM 频率准确性
  2. 占空比稳定性
  3. 长时间运行稳定性
"""

import time
import signal
import sys
import threading

try:
    import gpiod
except ImportError:
    print("❌ 错误: 未安装 libgpiod")
    print("安装方法:")
    print("  sudo apt install python3-libgpiod")
    sys.exit(1)

# ==================== 配置 ====================
GPIOCHIP = "gpiochip2"  # GPIO 芯片
LINE_OFFSET = 7         # Line 7 (未占用)
PWM_FREQ = 100          # 100 Hz (相机帧率)
PWM_DUTY = 50           # 50% 占空比

# ==================== 全局变量 ====================
chip = None
line = None
pwm_thread = None
pwm_running = False

# ==================== 软件 PWM 线程 ====================
class SoftwarePWM:
    """软件 PWM 实现"""
    
    def __init__(self, line, frequency, duty_cycle):
        """
        初始化软件 PWM
        
        Args:
            line: gpiod.Line 对象
            frequency: 频率 (Hz)
            duty_cycle: 占空比 (0-100)
        """
        self.line = line
        self.frequency = frequency
        self.duty_cycle = duty_cycle
        self.running = False
        self.thread = None
        
        # 计算时间参数
        self.period = 1.0 / frequency  # 周期 (秒)
        self.high_time = self.period * (duty_cycle / 100.0)
        self.low_time = self.period - self.high_time
    
    def _pwm_loop(self):
        """PWM 循环"""
        while self.running:
            # 高电平
            self.line.set_value(1)
            time.sleep(self.high_time)
            
            # 低电平
            self.line.set_value(0)
            time.sleep(self.low_time)
    
    def start(self):
        """启动 PWM"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._pwm_loop, daemon=True)
        self.thread.start()
        print(f"✅ PWM 已启动: {self.frequency} Hz, {self.duty_cycle}%")
    
    def stop(self):
        """停止 PWM"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        # 确保输出低电平
        self.line.set_value(0)
        print("✅ PWM 已停止")
    
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
    print("PWM 触发测试 - libgpiod (自定义载板)")
    print("="*60)
    print(f"GPIO 芯片: {GPIOCHIP}")
    print(f"GPIO 引脚: line {LINE_OFFSET}")
    print(f"频率: {PWM_FREQ} Hz")
    print(f"占空比: {PWM_DUTY}%")
    print("="*60)
    print()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # 打开 GPIO 芯片
        chip = gpiod.Chip(GPIOCHIP)
        print(f"✅ 已打开 {GPIOCHIP}")
        print(f"   芯片名称: {chip.name()}")
        print(f"   芯片标签: {chip.label()}")
        print(f"   GPIO 数量: {chip.num_lines()}")
        print()
        
        # 获取 GPIO 线
        line = chip.get_line(LINE_OFFSET)
        
        # 检查线是否已被占用
        if line.is_used():
            print(f"⚠️  警告: line {LINE_OFFSET} 已被占用")
            print(f"   消费者: {line.consumer()}")
            print("   尝试强制请求...")
        
        # 请求输出模式
        line.request(consumer="pwm_trigger", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
        print(f"✅ 已请求 line {LINE_OFFSET} (输出模式)")
        print()
        
        # 创建软件 PWM
        pwm_thread = SoftwarePWM(line, PWM_FREQ, PWM_DUTY)
        pwm_thread.start()
        print()
        
        # 测试不同频率
        print("📊 测试 1: 频率准确性")
        print("-" * 60)
        
        test_freqs = [50, 100, 150, 200]
        for freq in test_freqs:
            pwm_thread.change_frequency(freq)
            print(f"  当前频率: {freq} Hz (周期: {1000.0/freq:.2f} ms)")
            time.sleep(3)
        
        # 恢复到目标频率
        pwm_thread.change_frequency(PWM_FREQ)
        print()
        
        # 测试不同占空比
        print("📊 测试 2: 占空比调节")
        print("-" * 60)
        
        test_duties = [10, 30, 50, 70, 90]
        for duty in test_duties:
            pwm_thread.change_duty_cycle(duty)
            print(f"  当前占空比: {duty}% (高电平: {pwm_thread.high_time*1000:.2f} ms)")
            time.sleep(2)
        
        # 恢复到目标占空比
        pwm_thread.change_duty_cycle(PWM_DUTY)
        print()
        
        # 长时间运行测试
        print("📊 测试 3: 长时间稳定性")
        print("-" * 60)
        print(f"  运行中... (按 Ctrl+C 停止)")
        print()
        
        start_time = time.time()
        pulse_count = 0
        
        try:
            while True:
                elapsed = time.time() - start_time
                pulse_count = int(elapsed * PWM_FREQ)
                
                # 每 5 秒打印一次状态
                if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                    print(f"  运行时间: {int(elapsed)}s | "
                          f"预计脉冲数: {pulse_count} | "
                          f"频率: {PWM_FREQ} Hz")
                    time.sleep(1)  # 避免重复打印
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            pass
    
    except PermissionError:
        print("❌ 错误: 权限不足")
        print("解决方法:")
        print("  1. sudo usermod -a -G gpio $USER")
        print("  2. 重新登录")
        print("  3. 或使用 sudo 运行此脚本")
        return
    
    except FileNotFoundError:
        print(f"❌ 错误: 未找到 {GPIOCHIP}")
        print("可用的 GPIO 芯片:")
        import os
        for chip_name in os.listdir("/dev"):
            if chip_name.startswith("gpiochip"):
                print(f"  /dev/{chip_name}")
        return
    
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
    print("下一步:")
    print("  1. 使用示波器验证 PWM 波形")
    print("  2. 连接相机并测试触发")
    print("  3. 运行: python3 test_camera_gpiod.py")
    print()

if __name__ == "__main__":
    main()
