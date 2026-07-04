"""TensorRT neural-feature P2 NX algorithm sweep case aggregate."""

from __future__ import annotations

from nx_algorithm_case_p2_neural_aliked import ALIKED_NEURAL_P2_CASES
from nx_algorithm_case_p2_neural_superpoint import SUPERPOINT_NEURAL_P2_CASES
from nx_algorithm_case_p2_neural_xfeat import XFEAT_NEURAL_P2_CASES


NEURAL_P2_CASES = (
    *XFEAT_NEURAL_P2_CASES,
    *SUPERPOINT_NEURAL_P2_CASES,
    *ALIKED_NEURAL_P2_CASES,
)
