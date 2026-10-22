"""双目深度候选集合与旧 NX 选择契约。

主线把“会计算的候选”与“最终回退顺序”分开声明。旧 NX 目前只支持
两种首选选择器，并按固定相对顺序扫描 legacy fallback；本模块集中描述该
兼容边界，避免 schema、语义校验和 YAML 投影各自维护一份隐含规则。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CANDIDATE_TO_LEGACY_DEPTH_MODE: Mapping[str, str] = {
    "bbox_center": "bbox_pair",
    "bbox_edges": "bbox_edges",
    "circle_center": "circle_center",
    "edge_centroid": "roi_edge_centroid",
    "radial_center": "roi_radial_center",
    "edge_pair_center": "roi_edge_pair_center",
    "center_patch": "roi_center_patch",
    "roi_multi_point": "roi_subpixel",
    "cuda_template_match": "roi_cuda_template_match",
    "epipolar_fallback": "epipolar_fallback",
}

# selectLegacyDepthOutputCandidate() 的有效相对顺序。配置可以取其子序列，
# 但不能声称一个旧 C++ 选择器无法实现的顺序。
LEGACY_FALLBACK_ORDER: tuple[str, ...] = (
    "epipolar_fallback",
    "circle_center",
    "radial_center",
    "edge_pair_center",
    "edge_centroid",
    "bbox_center",
    "bbox_edges",
)

LIVE_PRIMARY_METHODS: frozenset[str] = frozenset(
    {"roi_multi_point", "circle_center"}
)

_SELECTION_CAPABLE_CANDIDATES = frozenset(LEGACY_FALLBACK_ORDER)


def triangulation_semantic_errors(triangulation: Mapping[str, Any]) -> list[str]:
    """返回当前实现无法忠实执行的三角化配置错误。"""

    backend = triangulation["backend"]
    primary = triangulation["primary_method"]
    candidates = tuple(triangulation["candidate_methods"])
    fallbacks = tuple(triangulation["fallback_methods"])
    candidate_set = set(candidates)
    fallback_set = set(fallbacks)
    errors: list[str] = []

    if backend == "recorded":
        if primary != "recorded":
            errors.append("recorded 三角化必须使用 primary_method=recorded")
        if candidates:
            errors.append("recorded 三角化不得声明 candidate_methods")
        if fallbacks:
            errors.append("recorded 三角化不得声明 fallback_methods")
        return errors

    if primary not in LIVE_PRIMARY_METHODS:
        errors.append(
            "当前实时实现的 primary_method 只支持 roi_multi_point 或 circle_center"
        )
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

    configured_order = [
        method for method in LEGACY_FALLBACK_ORDER if method in fallback_set
    ]
    if list(fallbacks) != configured_order:
        errors.append(
            "当前旧 NX fallback 只支持固定相对顺序: "
            + " -> ".join(LEGACY_FALLBACK_ORDER)
        )

    undeclared_selection = [
        method
        for method in candidates
        if method in _SELECTION_CAPABLE_CANDIDATES
        and method != primary
        and method not in fallback_set
    ]
    if undeclared_selection:
        errors.append(
            "可被旧选择器发布的候选必须列入 fallback_methods: "
            + ", ".join(undeclared_selection)
        )

    # 旧实现只有 depth_solver=roi_subpixel_match 时才真正计算多点候选。
    if "roi_multi_point" in candidate_set and primary != "roi_multi_point":
        errors.append(
            "当前旧 NX 只有 primary_method=roi_multi_point 时才计算 roi_multi_point"
        )
    return errors


def legacy_depth_projection(
    triangulation: Mapping[str, Any],
    base_modes: Mapping[str, Any],
) -> dict[str, Any]:
    """把主线候选集合完整投影为旧 NX depth_modes。

    基础模板中的每个 mode 都先关闭，再只开启主线明确声明的候选。因此新增或
    遗留模板开关不会静默进入运行时。
    """

    candidates = tuple(triangulation["candidate_methods"])
    modes = {name: False for name in base_modes}
    for candidate in candidates:
        legacy_name = CANDIDATE_TO_LEGACY_DEPTH_MODE[candidate]
        if legacy_name not in modes:
            raise ValueError(f"旧 NX 模板缺少 depth_modes.{legacy_name}")
        modes[legacy_name] = True

    primary = triangulation["primary_method"]
    return {
        "depth_modes": modes,
        "depth_solver": (
            "roi_subpixel_match" if primary == "roi_multi_point" else "circle_center"
        ),
        "subpixel_enabled": "roi_multi_point" in candidates,
        "fallback_epipolar_search": "epipolar_fallback" in candidates,
        "center_refine": bool(
            {
                "circle_center",
                "edge_centroid",
                "center_patch",
                "roi_multi_point",
                "epipolar_fallback",
            }
            & set(candidates)
        ),
    }

