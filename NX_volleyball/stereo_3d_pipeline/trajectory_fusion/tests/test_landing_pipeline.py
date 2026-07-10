#!/usr/bin/env python3
"""Synthetic tests for bbox+d0 -> Student-t EKF -> RK4 landing pipeline."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory_fusion.landing_pipeline.config import LandingPipelineConfig
from trajectory_fusion.landing_pipeline.observation import BBoxObservationBuilder, StereoGeometry
from trajectory_fusion.landing_pipeline.physics import height_above_ground, rollout_landing
from trajectory_fusion.landing_pipeline.predictor import LandingPipeline
from trajectory_fusion.landing_pipeline.ekf import EkfConfig


class LandingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geom = StereoGeometry(
            fx=1600.0,
            fy=1600.0,
            cx=640.0,
            cy=480.0,
            fB=1480.0,
            d0=-5.0,
            baseline_m=0.925,
        )
        self.g_hat = np.array([0.0, 1.0, 0.0], dtype=float)  # camera Y down ~= gravity
        self.ground_h = -2.0  # height = -y - ground_h = -y + 2

    def test_d0_reprojection(self) -> None:
        builder = BBoxObservationBuilder(self.geom, enable_circle_fallback=False)
        # z = fB / (disp - d0) = 1480 / (153) ≈ 9.673
        row = {
            "timestamp": 1.0,
            "disparity_bbox_center": 148.0,
            "left_bbox_cx": 640.0,
            "left_bbox_cy": 480.0,
            "p0p1_bbox_center_trust": 1.0,
        }
        obs = builder.from_row(row)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.source, "bbox_center")
        self.assertAlmostEqual(obs.p[2], 1480.0 / (148.0 - (-5.0)), places=6)
        self.assertAlmostEqual(obs.p[0], 0.0, places=6)
        self.assertAlmostEqual(obs.p[1], 0.0, places=6)

    def test_circle_fallback_when_bbox_missing(self) -> None:
        builder = BBoxObservationBuilder(self.geom, prefer_bbox=True, enable_circle_fallback=True)
        row = {
            "timestamp": 1.0,
            "disparity_circle_center": 148.0,
            "left_circle_cx": 640.0,
            "left_circle_cy": 480.0,
            "p0p1_circle_center_trust": 0.8,
        }
        obs = builder.from_row(row)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.source, "circle_center")

    def test_rollout_hits_ground(self) -> None:
        p = np.array([0.0, 0.0, 8.0])  # height = -0 -(-2)=2m above ground? wait ground_h=-2 => height=-y -(-2)= -y+2; y=0 => 2m
        v = np.array([1.0, 4.0, 0.0])  # moving downward in +Y
        out = rollout_landing(p, v, cd=0.10, t_now=0.0, g_hat=self.g_hat, ground_h=self.ground_h)
        self.assertIsNotNone(out)
        h = height_above_ground(out.landing, self.g_hat, self.ground_h)
        self.assertLess(abs(h), 0.05)
        self.assertGreater(out.time_to_land, 0.05)

    def test_ekf_converges_on_ballistic_observations(self) -> None:
        cfg = LandingPipelineConfig(
            fx=self.geom.fx,
            fy=self.geom.fy,
            cx=self.geom.cx,
            cy=self.geom.cy,
            fB=self.geom.fB,
            d0=self.geom.d0,
            g_hat=self.g_hat.tolist(),
            ground_h=self.ground_h,
            residual_checkpoint=None,
        )
        cfg.residual.enabled = False
        cfg.ekf = EkfConfig(cd=0.10, nu=12.0, sigma_d_px=0.3, fB=self.geom.fB, q_vel=1.0)
        pipe = LandingPipeline(cfg)

        # Simulate a throw from height ~3m, downward.
        # height = -y - ground_h = -y + 2 => for height 3, y = -1
        p = np.array([0.0, -1.0, 10.0], dtype=float)
        v = np.array([2.0, 3.0, -1.0], dtype=float)
        dt = 0.02
        t = 0.0
        last = None
        for i in range(45):
            # simple integration for synthetic truth-ish obs
            a = 9.81 * self.g_hat - (0.5 * 1.225 * math.pi * 0.105 ** 2 / 0.270) * 0.10 * np.linalg.norm(v) * v
            p = p + v * dt + 0.5 * a * dt * dt
            v = v + a * dt
            t += dt
            # Convert to disparity/u/v using inverse projection.
            z = p[2]
            disp = self.geom.fB / z + self.geom.d0
            u = self.geom.cx + p[0] * self.geom.fx / z
            vpx = self.geom.cy + p[1] * self.geom.fy / z
            out = pipe.update_bbox(t, disparity=disp, u=u, v=vpx, quality={"trust": 1.0, "consistency": 1.0})
            if out is not None:
                last = out
        self.assertIsNotNone(last)
        # Landing x should be in front of current x and finite.
        self.assertTrue(np.all(np.isfinite(last.landing)))
        self.assertGreater(last.time_to_land, 0.0)
        # Physics residual path disabled => landing equals landing_physics
        np.testing.assert_allclose(last.landing, last.landing_physics, atol=1e-9)
        self.assertFalse(last.residual_applied)

    def test_student_t_downweights_outlier(self) -> None:
        cfg = LandingPipelineConfig(
            fx=self.geom.fx,
            fy=self.geom.fy,
            cx=self.geom.cx,
            cy=self.geom.cy,
            fB=self.geom.fB,
            d0=self.geom.d0,
            g_hat=self.g_hat.tolist(),
            ground_h=self.ground_h,
            residual_checkpoint=None,
        )
        cfg.residual.enabled = False
        cfg.ekf = EkfConfig(cd=0.10, nu=12.0, sigma_d_px=0.25, fB=self.geom.fB, q_vel=0.5)
        pipe = LandingPipeline(cfg)

        p = np.array([0.0, -1.0, 9.0], dtype=float)
        v = np.array([1.5, 2.5, 0.0], dtype=float)
        dt = 0.02
        t = 0.0
        for i in range(20):
            a = 9.81 * self.g_hat
            p = p + v * dt + 0.5 * a * dt * dt
            v = v + a * dt
            t += dt
            z = p[2]
            disp = self.geom.fB / z + self.geom.d0
            u = self.geom.cx + p[0] * self.geom.fx / z
            vpx = self.geom.cy + p[1] * self.geom.fy / z
            # Inject a huge outlier once.
            if i == 12:
                disp = self.geom.fB / (z + 3.0) + self.geom.d0
            out = pipe.update_bbox(t, disparity=disp, u=u, v=vpx, quality={"trust": 1.0, "consistency": 1.0})
            if i == 12 and out is not None:
                self.assertLess(out.student_w, 0.5)


if __name__ == "__main__":
    unittest.main()
