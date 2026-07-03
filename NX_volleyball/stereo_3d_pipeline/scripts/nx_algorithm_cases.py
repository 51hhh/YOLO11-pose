"""Compatibility facade for NX realtime algorithm matrix case definitions."""

from __future__ import annotations

from nx_algorithm_case_baseline import APPROX_CASES, CASES
from nx_algorithm_case_sweeps import (
    COLOR_OFFLINE_TUNED,
    COLOR_WIDE_SEARCH,
    DENSE_PATCH9_SWEEP,
    ORB_FAST_SWEEP,
    ORB_WIDE_Y_SWEEP,
    RELAXED_CASES,
    TEMPLATE_PATCH9_SWEEP,
)
from nx_algorithm_case_types import MODE_KEYS, Case


__all__ = [
    "APPROX_CASES",
    "CASES",
    "COLOR_OFFLINE_TUNED",
    "COLOR_WIDE_SEARCH",
    "DENSE_PATCH9_SWEEP",
    "MODE_KEYS",
    "ORB_FAST_SWEEP",
    "ORB_WIDE_Y_SWEEP",
    "RELAXED_CASES",
    "TEMPLATE_PATCH9_SWEEP",
    "Case",
]
