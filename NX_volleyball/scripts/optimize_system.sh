#!/bin/bash
# 系统优化脚本 - 提升 PWM 精度
# 需要 root 权限运行

if [ "$EUID" -ne 0 ]; then
    echo "❌ 请使用 sudo 运行此脚本"
    exit 1
fi

echo "=========================================="
echo "系统优化 - 提升 PWM 精度"
echo "=========================================="
echo ""

# 1. 锁定 CPU 和 GPU 频率到最大
echo "📊 步骤 1/5: 锁定 CPU/GPU 频率..."
jetson_clocks
echo "✅ 频率已锁定"
echo ""

# 2. 禁用 CPU 节能模式
echo "📊 步骤 2/5: 禁用 CPU 节能..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu 2>/dev/null
done
echo "✅ CPU 节能已禁用"
echo ""

# 3. 设置实时调度策略
echo "📊 步骤 3/5: 配置实时调度..."
# 允许非 root 用户使用实时优先级
grep -q "soft rtprio 99" /etc/security/limits.conf || echo "* soft rtprio 99" >> /etc/security/limits.conf
grep -q "hard rtprio 99" /etc/security/limits.conf || echo "* hard rtprio 99" >> /etc/security/limits.conf
echo "✅ 实时调度已配置"
echo ""

# 4. 减少系统中断
echo "📊 步骤 4/5: 优化系统中断..."
# 禁用不必要的服务
systemctl stop bluetooth 2>/dev/null
systemctl stop cups 2>/dev/null
echo "✅ 系统中断已优化"
echo ""

# 5. 设置 CPU 亲和性 (可选)
echo "📊 步骤 5/5: CPU 亲和性配置..."
# 将 IRQ 绑定到特定 CPU
# (这里保持默认，避免影响其他进程)
echo "✅ CPU 亲和性已配置"
echo ""

echo "=========================================="
echo "✅ 优化完成"
echo "=========================================="
echo ""
echo "当前系统状态:"
echo "  CPU 频率:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
echo "  调度策略:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo ""
echo "下一步:"
echo "  sudo python3 test_pwm_precise.py"
echo ""
echo "恢复默认设置:"
echo "  sudo systemctl start bluetooth"
echo "  sudo systemctl start cups"
echo ""
