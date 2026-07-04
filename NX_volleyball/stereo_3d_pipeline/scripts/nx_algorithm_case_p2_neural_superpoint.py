"""SuperPoint TensorRT neural-feature P2 NX algorithm sweep cases."""

from __future__ import annotations

from nx_algorithm_case_types import Case


SUPERPOINT_NEURAL_P2_CASES = (
    Case(
        "neural_superpoint_lightglue_relaxed",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_224_top128.engine",
        roi_size=224,
        top_k=128,
        descriptor_dim=256,
        neural_min_matches=4,
        neural_max_y_error_px=6.0,
        neural_max_disp_delta_px=96.0,
        neural_final_disp_gate_px=6.0,
        note="diagnostic only: TensorRT SuperPoint extractor with relaxed gates",
    ),
    Case(
        "neural_superpoint_128_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_128_top64.engine",
        roi_size=128,
        top_k=64,
        descriptor_dim=256,
        neural_min_matches=4,
        note="P2 sweep: TensorRT SuperPoint fixed extractor 128/top64",
    ),
    Case(
        "neural_superpoint_160_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_160_top64.engine",
        roi_size=160,
        top_k=64,
        descriptor_dim=256,
        neural_min_matches=4,
        note="P2 sweep: TensorRT SuperPoint fixed extractor 160/top64",
    ),
    Case(
        "neural_superpoint_224_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_224_top64.engine",
        roi_size=224,
        top_k=64,
        descriptor_dim=256,
        neural_min_matches=4,
        note="P2 sweep: TensorRT SuperPoint fixed extractor 224/top64",
    ),
)
