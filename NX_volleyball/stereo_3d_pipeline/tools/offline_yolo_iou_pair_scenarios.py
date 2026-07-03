"""Synthetic pair scenarios for offline YOLO IoU fallback regression."""

from __future__ import annotations

from stereo_feature_matching.realtime_contract import Detection


def fake_right_detection(left: Detection, true_right: Detection, scale: float) -> Detection:
    true_disp = max(1.0, left.cx - true_right.cx)
    return Detection(
        cx=left.cx - true_disp * scale,
        cy=true_right.cy,
        width=true_right.width,
        height=true_right.height,
        confidence=true_right.confidence,
        class_id=true_right.class_id,
    )


def fake_left_detection(true_left: Detection, right: Detection, scale: float) -> Detection:
    true_disp = max(1.0, true_left.cx - right.cx)
    return Detection(
        cx=right.cx + true_disp * scale,
        cy=true_left.cy,
        width=true_left.width,
        height=true_left.height,
        confidence=true_left.confidence,
        class_id=true_left.class_id,
    )


def selected_pair_score(selected, left_index: int, right_index: int) -> float:
    for pair in selected:
        if pair.left_index == left_index and pair.right_index == right_index:
            return pair.score
    return 0.0
