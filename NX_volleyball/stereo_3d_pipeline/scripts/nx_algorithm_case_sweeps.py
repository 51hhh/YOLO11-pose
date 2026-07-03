"""Compatibility facade for relaxed and P2 sweep NX algorithm cases."""

from __future__ import annotations

from nx_algorithm_case_p2_cuda import CUDA_P2_CASES
from nx_algorithm_case_p2_neural import NEURAL_P2_CASES
from nx_algorithm_case_sweep_scalars import (
    COLOR_OFFLINE_TUNED,
    COLOR_WIDE_SEARCH,
    DENSE_PATCH9_SWEEP,
    ORB_FAST_SWEEP,
    ORB_WIDE_Y_SWEEP,
    TEMPLATE_PATCH9_SWEEP,
)


RELAXED_CASES = CUDA_P2_CASES + NEURAL_P2_CASES

__all__ = [
    "COLOR_OFFLINE_TUNED",
    "COLOR_WIDE_SEARCH",
    "CUDA_P2_CASES",
    "DENSE_PATCH9_SWEEP",
    "NEURAL_P2_CASES",
    "ORB_FAST_SWEEP",
    "ORB_WIDE_Y_SWEEP",
    "RELAXED_CASES",
    "TEMPLATE_PATCH9_SWEEP",
]
