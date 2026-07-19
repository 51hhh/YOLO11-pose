"""Filter registry for trajectory analysis."""

from .base import FilterBase, FilterState
from .raw_passthrough import RawPassthrough
from .const_accel_9d import ConstAccel9D
from .gravity_ekf_6d import GravityEKF6D
from .gravity_bounce import GravityBounce
from .gravity_ekf_calibrated import GravityEKFCalibrated
from .gravity_ekf_adaptive_r import GravityEKFAdaptiveR
from .gravity_ekf_fallback import GravityEKFFallback
from .gravity_drag_7d import GravityDrag7D
from .gravity_bounce_v2 import GravityBounceV2
from .one_euro import OneEuroFilter
from .adaptive_ekf import AdaptiveEKF
from .imm_filter import IMMFilter
from .gravity_one_euro import GravityOneEuroHybrid
from .gravity_one_euro_variants import GravityOneEuroV2, GravityOneEuroAdaptive, AEKFOneEuro
from .round4_filters import FallbackV2, AdaptiveQOneEuro, ConfidenceEKF, IMM2Model, RobustBounceEKF, ConfidenceV2, AdaptiveQV2
from .fast_gravity_ekf import FastGravityEKF
from .robust_wrapper import RobustWrapper

FILTER_REGISTRY = {
    'raw_passthrough': RawPassthrough,
    'const_accel_9d': ConstAccel9D,
    'gravity_ekf_6d': GravityEKF6D,
    'gravity_bounce': GravityBounce,
    'gravity_ekf_calibrated': GravityEKFCalibrated,
    'gravity_ekf_adaptive_r': GravityEKFAdaptiveR,
    'gravity_ekf_fallback': GravityEKFFallback,
    'gravity_drag_7d': GravityDrag7D,
    'gravity_bounce_v2': GravityBounceV2,
    'one_euro': OneEuroFilter,
    'adaptive_ekf': AdaptiveEKF,
    'imm': IMMFilter,
    'gravity_one_euro': GravityOneEuroHybrid,
    'gravity_one_euro_v2': GravityOneEuroV2,
    'gravity_one_euro_adaptive': GravityOneEuroAdaptive,
    'aekf_one_euro': AEKFOneEuro,
    'fallback_v2': FallbackV2,
    'adaptive_q_one_euro': AdaptiveQOneEuro,
    'confidence_ekf': ConfidenceEKF,
    'imm_2model': IMM2Model,
    'robust_bounce_ekf': RobustBounceEKF,
    'confidence_v2': ConfidenceV2,
    'adaptive_q_v2': AdaptiveQV2,
    'fast_gravity_ekf': FastGravityEKF,
}


def create_filter(name: str, **kwargs) -> FilterBase:
    """Create a filter instance by name.

    Supports 'robust_<name>' prefix to auto-wrap any filter with RobustWrapper.
    Robust wrapper params can be passed via 'robust_' prefixed kwargs.
    """
    # Handle robust_ prefix
    if name.startswith('robust_') and name[7:] in FILTER_REGISTRY:
        inner_name = name[7:]
        # Separate robust params from inner filter params
        robust_keys = {'max_speed', 'max_jump', 'median_window', 'coast_limit',
                       'init_frames', 'physics_init', 'gravity_vec'}
        robust_params = {k: v for k, v in kwargs.items() if k in robust_keys}
        inner_params = {k: v for k, v in kwargs.items() if k not in robust_keys}

        inner_filter = FILTER_REGISTRY[inner_name](**inner_params)
        return RobustWrapper(inner_filter=inner_filter, **robust_params)

    if name not in FILTER_REGISTRY:
        raise ValueError(f"Unknown filter: {name}. Available: {list(FILTER_REGISTRY.keys())}")
    return FILTER_REGISTRY[name](**kwargs)


def list_filters():
    """Return list of available filter names."""
    return list(FILTER_REGISTRY.keys())
