from __future__ import annotations

import unittest

from volleyball_vision.triangulation_config import (
    legacy_depth_projection,
    triangulation_semantic_errors,
)


class TriangulationConfigTests(unittest.TestCase):
    def test_projection_disables_every_unlisted_template_mode(self) -> None:
        triangulation = {
            "backend": "cuda",
            "primary_method": "roi_multi_point",
            "candidate_methods": ["roi_multi_point", "circle_center"],
            "fallback_methods": ["circle_center"],
        }
        projection = legacy_depth_projection(
            triangulation,
            {
                "bbox_pair": True,
                "circle_center": True,
                "roi_subpixel": True,
                "roi_cuda_template_match": True,
                "future_template_default": True,
            },
        )
        self.assertEqual(
            projection["depth_modes"],
            {
                "bbox_pair": False,
                "circle_center": True,
                "roi_subpixel": True,
                "roi_cuda_template_match": False,
                "future_template_default": False,
            },
        )

    def test_selection_capable_diagnostic_cannot_bypass_fallback_contract(self) -> None:
        errors = triangulation_semantic_errors(
            {
                "backend": "cuda",
                "primary_method": "roi_multi_point",
                "candidate_methods": ["roi_multi_point", "bbox_center"],
                "fallback_methods": [],
            }
        )
        self.assertTrue(any("必须列入 fallback_methods" in error for error in errors))

    def test_compute_only_candidate_need_not_be_a_fallback(self) -> None:
        errors = triangulation_semantic_errors(
            {
                "backend": "cuda",
                "primary_method": "roi_multi_point",
                "candidate_methods": [
                    "roi_multi_point",
                    "cuda_template_match",
                    "center_patch",
                ],
                "fallback_methods": [],
            }
        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
