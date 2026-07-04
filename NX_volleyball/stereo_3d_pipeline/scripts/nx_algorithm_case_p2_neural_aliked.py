"""ALIKED TensorRT neural-feature P2 NX algorithm sweep cases."""

from __future__ import annotations

from nx_algorithm_case_types import Case


ALIKED_NEURAL_P2_CASES = (
    Case(
        "neural_aliked_160_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="aliked",
        neural_engine="aliked_extractor_160_top64.engine",
        roi_size=160,
        top_k=64,
        descriptor_dim=128,
        neural_min_matches=4,
        note="P2 sweep: ALIKED fixed extractor if a TensorRT engine is available",
    ),
    Case(
        "neural_aliked_224_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="aliked",
        neural_engine="aliked_extractor_224_top64.engine",
        roi_size=224,
        top_k=64,
        descriptor_dim=128,
        neural_min_matches=4,
        note="P2 sweep: ALIKED fixed extractor 224/top64 if engine exists",
    ),
)
