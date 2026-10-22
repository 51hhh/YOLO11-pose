"""跨采集链路复用的整数纳秒时间运算。"""

from __future__ import annotations


def integer_midpoint_ns(left_ns: int, right_ns: int) -> int:
    """返回两个非负纳秒时间戳的下取整中点，避免浮点精度损失。

    C++ 主线使用同一公式。拆半后再相加还可避免固定宽度整数中的
    ``left_ns + right_ns`` 溢出；奇数和的 0.5 ns 按整数时间轴向下取整。
    """

    if left_ns < 0 or right_ns < 0:
        raise ValueError("纳秒时间戳必须是非负整数")
    return (
        left_ns // 2
        + right_ns // 2
        + (left_ns % 2 + right_ns % 2) // 2
    )
