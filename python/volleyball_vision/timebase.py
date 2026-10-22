"""跨采集链路复用的整数纳秒时间运算。"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


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


def fit_affine_clock_mapping(
    source_times_ns: list[int],
    reference_times_ns: list[int],
    *,
    source_clock: str,
    reference_clock: str,
    mapping_id: str,
    source: str,
    validity_source_times_ns: Iterable[int] = (),
) -> dict[str, Any] | None:
    """用中心化最小二乘拟合一个时钟到参考时钟的仿射映射。"""

    points = [
        (source_time, reference_time)
        for source_time, reference_time in zip(source_times_ns, reference_times_ns)
        if source_time > 0
    ]
    if not points:
        return None
    source_anchor, reference_anchor = points[0]
    if len(points) >= 2:
        xs = [source_time - source_anchor for source_time, _ in points]
        ys = [reference_time - reference_anchor for _, reference_time in points]
        denominator = sum(value * value for value in xs)
        scale = (
            sum(x_value * y_value for x_value, y_value in zip(xs, ys))
            / denominator
            if denominator
            else 1.0
        )
    else:
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    drift_ppm = (scale - 1.0) * 1.0e6
    residuals = [
        abs(
            reference_anchor
            + round((source_time - source_anchor) * scale)
            - reference_time
        )
        for source_time, reference_time in points
    ]
    all_source = [source_time for source_time, _ in points]
    all_source.extend(value for value in validity_source_times_ns if value > 0)
    return {
        "mapping_id": mapping_id,
        "source_clock_domain": source_clock,
        "reference_clock_domain": reference_clock,
        "source_anchor_ns": source_anchor,
        "reference_anchor_ns": reference_anchor,
        "drift_ppm": drift_ppm,
        "uncertainty_ns": max(100_000.0, float(max(residuals, default=0))),
        "valid_from_source_ns": min(all_source),
        "valid_to_source_ns": max(all_source),
        "source": source,
    }


def map_timestamp_ns(timestamp_ns: int, mapping: Mapping[str, Any]) -> int:
    """按 v2.1 时钟映射字段把一个整数纳秒时间戳映射到参考时钟。"""

    scale = 1.0 + float(mapping["drift_ppm"]) * 1.0e-6
    return int(mapping["reference_anchor_ns"]) + round(
        (timestamp_ns - int(mapping["source_anchor_ns"])) * scale
    )
