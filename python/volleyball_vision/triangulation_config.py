"""原生双目深度候选的配置契约。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SUPPORTED_LIVE_METHODS: frozenset[str] = frozenset(
    {
        "circle_center",
        "radial_center",
        "edge_pair_center",
        "edge_centroid",
        "bbox_center",
        "bbox_edges",
    }
)


def triangulation_semantic_errors(triangulation: Mapping[str, Any]) -> list[str]:
    """返回原生运行时无法忠实执行的三角化配置错误。"""

    backend = triangulation["backend"]
    primary = triangulation["primary_method"]
    candidates = tuple(triangulation["candidate_methods"])
    fallbacks = tuple(triangulation["fallback_methods"])
    candidate_set = set(candidates)
    fallback_set = set(fallbacks)
    errors: list[str] = []

    subpixel = triangulation.get("subpixel")
    if (
        isinstance(subpixel, Mapping)
        and "min_points" in subpixel
        and "max_points" in subpixel
        and subpixel["min_points"] > subpixel["max_points"]
    ):
        errors.append("subpixel.min_points 不得大于 subpixel.max_points")

    if backend == "recorded":
        if primary != "recorded":
            errors.append("recorded 三角化必须使用 primary_method=recorded")
        if candidates:
            errors.append("recorded 三角化不得声明 candidate_methods")
        if fallbacks:
            errors.append("recorded 三角化不得声明 fallback_methods")
        return errors

    unsupported = sorted(
        (candidate_set | fallback_set | {primary}) - SUPPORTED_LIVE_METHODS
    )
    if unsupported:
        errors.append("原生实时实现不支持这些深度方法: " + ", ".join(unsupported))
    if primary not in candidate_set:
        errors.append("primary_method 必须同时出现在 candidate_methods")
    if primary in fallback_set:
        errors.append("primary_method 不得重复出现在 fallback_methods")

    missing_fallbacks = [method for method in fallbacks if method not in candidate_set]
    if missing_fallbacks:
        errors.append(
            "fallback_methods 必须是 candidate_methods 的子集: "
            + ", ".join(missing_fallbacks)
        )

    unselectable = [
        method
        for method in candidates
        if method != primary and method not in fallback_set
    ]
    if unselectable:
        errors.append(
            "会计算的候选必须进入 primary/fallback 选择顺序: "
            + ", ".join(unselectable)
        )
    return errors
