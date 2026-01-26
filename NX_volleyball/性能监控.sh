#!/bin/bash
# 性能对比监控脚本 - 对比优化前后性能

echo "📊 排球追踪系统性能监控"
echo "════════════════════════════════════════"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 性能基线 (优化前)
BASELINE_FPS=20.4
BASELINE_CAPTURE=14.0
BASELINE_DETECT=35.0
BASELINE_TOTAL=49.0

# 性能目标 (优化后)
TARGET_FPS=50.0
TARGET_CAPTURE=9.0
TARGET_DETECT=12.0
TARGET_TOTAL=21.0

echo "性能基线 (优化前):"
echo "  FPS: ${BASELINE_FPS}  |  采集: ${BASELINE_CAPTURE}ms  |  检测: ${BASELINE_DETECT}ms  |  总计: ${BASELINE_TOTAL}ms"
echo ""
echo "性能目标 (优化后):"
echo "  FPS: ${TARGET_FPS}+  |  采集: ${TARGET_CAPTURE}ms   |  检测: ${TARGET_DETECT}ms  |  总计: ${TARGET_TOTAL}ms"
echo ""
echo "════════════════════════════════════════"
echo ""
echo "🔍 实时监控 (每5秒更新)..."
echo "   监听话题: /volleyball_tracker/position"
echo ""

# 监控ROS2日志中的性能统计
ros2 topic echo /volleyball_tracker/position | while read line; do
    # 这里只是示例，实际应该解析日志输出
    # 真实监控请查看终端输出的统计信息
    :
done &

# 监控系统资源
watch -n 5 '
echo "════════════════════════════════════════"
echo "📊 系统资源监控"
echo "════════════════════════════════════════"
echo ""
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits | \
  awk -F, '\''{printf "GPU利用率: %s%%  |  内存: %s%%  |  温度: %s°C  |  功耗: %sW\n", $1, $2, $3, $4}'\''
echo ""
echo "CPU温度:"
cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -4 | awk '\''{printf "  %.1f°C ", $1/1000}'\''
echo ""
echo ""
echo "════════════════════════════════════════"
echo "💡 性能分析提示:"
echo "════════════════════════════════════════"
echo "1. 观察ROS2节点终端输出的统计信息"
echo "2. 重点关注: FPS, 采集时间, 检测时间"
echo "3. 对比基线值判断优化效果"
echo ""
echo "预期改进:"
echo "  ✓ FPS提升: 20.4 → 50+ (2.5倍)"
echo "  ✓ 检测加速: 35ms → 12ms (3倍)"
echo "  ✓ 采集优化: 14ms → 9ms (1.5倍)"
'
