#!/bin/bash
# 安装 libgpiod 和 Python 绑定

echo "=========================================="
echo "安装 libgpiod"
echo "=========================================="

# 安装 libgpiod 库和 Python 绑定
sudo apt update
sudo apt install -y libgpiod2 libgpiod-dev python3-libgpiod gpiod

# 验证安装
echo ""
echo "验证安装..."
python3 -c "import gpiod; print('✅ Python libgpiod 已安装')" 2>/dev/null || echo "❌ Python libgpiod 安装失败"

# 列出可用的 GPIO 芯片
echo ""
echo "可用的 GPIO 芯片:"
gpiodetect

# 显示 gpiochip2 的详细信息
echo ""
echo "gpiochip2 详细信息:"
gpioinfo gpiochip2

echo ""
echo "✅ 安装完成"
echo ""
echo "下一步:"
echo "  python3 test_pwm_gpiod.py"
