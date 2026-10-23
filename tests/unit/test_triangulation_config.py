from __future__ import annotations

import unittest

from volleyball_vision.triangulation_config import triangulation_semantic_errors


class TriangulationConfigTests(unittest.TestCase):
    def test_surface_measurement_cannot_be_selected_as_ball_center(self) -> None:
        for method in ("roi_multi_point", "center_patch"):
            errors = triangulation_semantic_errors({
                "backend": "cuda", "primary_method": method,
                "candidate_methods": [method], "fallback_methods": [],
            })
            self.assertTrue(any("不支持" in error for error in errors))

    def test_every_computed_candidate_must_be_selectable(self) -> None:
        errors = triangulation_semantic_errors(
            {
                "backend": "cuda",
                "primary_method": "roi_multi_point",
                "candidate_methods": ["roi_multi_point", "bbox_center"],
                "fallback_methods": [],
            }
        )
        self.assertTrue(any("必须进入" in error for error in errors))

    def test_native_selector_accepts_explicit_order(self) -> None:
        errors = triangulation_semantic_errors(
            {
                "backend": "cuda",
                "primary_method": "bbox_center",
                "candidate_methods": ["bbox_center", "circle_center"],
                "fallback_methods": ["circle_center"],
            }
        )
        self.assertEqual(errors, [])

    def test_recorded_backend_has_no_live_candidates(self) -> None:
        errors = triangulation_semantic_errors(
            {
                "backend": "recorded",
                "primary_method": "recorded",
                "candidate_methods": [],
                "fallback_methods": [],
            }
        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
