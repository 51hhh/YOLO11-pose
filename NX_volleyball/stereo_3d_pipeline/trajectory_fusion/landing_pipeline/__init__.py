"""Deployable causal landing pipeline.

Primary path:
  bbox_center disparity + d0 reproject
    -> Student-t EKF (gravity + drag)
    -> RK4 landing rollout
    -> optional TinyGRU landing residual

Circle is retained only as consistency/fallback, not as a second independent
observation that would double-count common-mode depth bias.
"""

from .predictor import LandingPipeline, LandingPipelineConfig, LandingResult
from .observation import BBoxObservationBuilder, Observation

__all__ = [
    "LandingPipeline",
    "LandingPipelineConfig",
    "LandingResult",
    "BBoxObservationBuilder",
    "Observation",
]
