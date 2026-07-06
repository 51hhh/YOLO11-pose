#!/usr/bin/env python3
"""Loss functions for self-supervised trajectory reliability learning."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def student_t_nll(
    residual: torch.Tensor,
    log_sigma: torch.Tensor,
    valid: torch.Tensor,
    df: float = 4.0,
) -> torch.Tensor:
    """Student-t negative log likelihood up to a constant."""

    sigma2 = torch.exp(2.0 * log_sigma).clamp_min(1e-8)
    scaled = residual * residual / (df * sigma2)
    nll = log_sigma + 0.5 * (df + 1.0) * torch.log1p(scaled)
    weighted = nll * valid
    return weighted.sum() / valid.sum().clamp_min(1.0)


def huber_measurement_loss(
    residual: torch.Tensor,
    log_sigma: torch.Tensor,
    valid: torch.Tensor,
    delta: float = 2.0,
) -> torch.Tensor:
    """Huber loss on normalized residuals."""

    normalized = residual / torch.exp(log_sigma).clamp_min(1e-4)
    loss = F.huber_loss(normalized, torch.zeros_like(normalized), delta=delta, reduction="none")
    weighted = loss * valid
    return weighted.sum() / valid.sum().clamp_min(1.0)


def measurement_consistency_loss(
    state_depth: torch.Tensor,
    measurements: torch.Tensor,
    valid: torch.Tensor,
    log_sigma: torch.Tensor,
    bias: torch.Tensor,
    outlier_logit: torch.Tensor,
    df: float = 4.0,
) -> torch.Tensor:
    """Robust NLL between latent depth and all valid measurements."""

    residual = measurements.unsqueeze(-1) - bias - state_depth.unsqueeze(2)
    inlier_prob = 1.0 - torch.sigmoid(outlier_logit)
    effective_valid = valid.unsqueeze(-1) * inlier_prob
    return student_t_nll(residual, log_sigma, effective_valid, df=df)


def physics_depth_loss(depth: torch.Tensor, dt: torch.Tensor | float, weight_jerk: float = 0.1) -> torch.Tensor:
    """Constant-velocity/low-jerk prior for depth axis.

    Depth is not the gravity axis in the current camera model, so z'' should
    usually be small except during strong spin/measurement artifacts.
    """

    if depth.shape[1] < 4:
        return depth.new_tensor(0.0)
    z = depth.squeeze(-1)
    dt_tensor = _sequence_dt(dt, z).clamp(1e-4, 0.5)
    prev_dt = dt_tensor[:, 1:-1]
    next_dt = dt_tensor[:, 2:]
    prev_vel = (z[:, 1:-1] - z[:, :-2]) / prev_dt
    next_vel = (z[:, 2:] - z[:, 1:-1]) / next_dt
    local_dt = 0.5 * (prev_dt + next_dt)
    second = 2.0 * (next_vel - prev_vel) / (prev_dt + next_dt) * local_dt * local_dt
    accel_loss = F.huber_loss(second, torch.zeros_like(second), delta=0.05, reduction="mean")
    jerk = second[:, 1:] - second[:, :-1]
    jerk_loss = F.huber_loss(jerk, torch.zeros_like(jerk), delta=0.05, reduction="mean")
    return accel_loss + weight_jerk * jerk_loss


def _sequence_dt(dt: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    """Return [batch, time] time deltas on the same device/dtype as reference."""

    if not torch.is_tensor(dt):
        return reference.new_full(reference.shape, float(dt))
    dt_tensor = dt.to(device=reference.device, dtype=reference.dtype)
    if dt_tensor.ndim == 0:
        return reference.new_full(reference.shape, float(dt_tensor.item()))
    if dt_tensor.ndim == 3 and dt_tensor.shape[-1] == 1:
        dt_tensor = dt_tensor.squeeze(-1)
    if dt_tensor.ndim == 1:
        dt_tensor = dt_tensor.unsqueeze(0)
    if dt_tensor.shape[0] == 1 and reference.shape[0] > 1:
        dt_tensor = dt_tensor.expand(reference.shape[0], -1)
    if dt_tensor.shape != reference.shape:
        raise ValueError(f"dt shape {tuple(dt_tensor.shape)} does not match depth sequence {tuple(reference.shape)}")
    return dt_tensor


def known_z_loss(
    depth: torch.Tensor,
    known_z: torch.Tensor,
    valid: torch.Tensor,
    delta: float = 0.05,
) -> torch.Tensor:
    """Huber loss for clips with a known static/weak distance label."""

    residual = depth.squeeze(-1) - known_z
    loss = F.huber_loss(residual, torch.zeros_like(residual), delta=delta, reduction="none")
    weighted = loss * valid
    return weighted.sum() / valid.sum().clamp_min(1.0)


def known_z_range_loss(
    depth: torch.Tensor,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Penalty for weak labels that only constrain depth to a range."""

    z = depth.squeeze(-1)
    below = F.relu(z_min - z)
    above = F.relu(z - z_max)
    loss = (below * below + above * above) * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


def static_depth_jitter_loss(depth: torch.Tensor, static_valid: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    """Short-window jitter prior for clips marked static in metadata."""

    if depth.shape[1] < 2:
        return depth.new_tensor(0.0)
    dz = depth[:, 1:, 0] - depth[:, :-1, 0]
    valid = static_valid[:, 1:] * static_valid[:, :-1]
    loss = F.huber_loss(dz, torch.zeros_like(dz), delta=delta, reduction="none") * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


def ballistic_position_loss(
    positions: torch.Tensor,
    dt: torch.Tensor | float,
    gravity_y: float = 9.81,
    weight_jerk: float = 0.05,
) -> torch.Tensor:
    """Physics prior for [x, y, z] camera coordinates.

    If camera-y is not aligned with gravity, set gravity_y to 0 for early tests.
    """

    if positions.shape[1] < 4:
        return positions.new_tensor(0.0)
    if not torch.is_tensor(dt):
        dt_value = float(dt)
        dt2 = positions.new_tensor(dt_value * dt_value)
    else:
        dt2 = dt[:, 1:-1].unsqueeze(-1).to(positions).pow(2)

    second = positions[:, 2:, :] - 2.0 * positions[:, 1:-1, :] + positions[:, :-2, :]
    target = torch.zeros_like(second)
    target[..., 1] = gravity_y * dt2.squeeze(-1)
    accel_loss = F.huber_loss(second, target, delta=0.05, reduction="mean")
    jerk = second[:, 1:, :] - second[:, :-1, :]
    jerk_loss = F.huber_loss(jerk, torch.zeros_like(jerk), delta=0.05, reduction="mean")
    return accel_loss + weight_jerk * jerk_loss


def uncertainty_regularizer(
    log_sigma: torch.Tensor,
    outlier_logit: torch.Tensor,
    valid: torch.Tensor,
    target_log_sigma: float = -2.3,
    outlier_prior: float = 0.05,
) -> torch.Tensor:
    """Keep the unsupervised model away from all-large-sigma/all-outlier collapse."""

    valid_e = valid.unsqueeze(-1)
    sigma_loss = ((log_sigma - target_log_sigma) ** 2 * valid_e).sum() / valid_e.sum().clamp_min(1.0)
    outlier_prob = torch.sigmoid(outlier_logit)
    outlier_loss = ((outlier_prob - outlier_prior) ** 2 * valid_e).sum() / valid_e.sum().clamp_min(1.0)
    return 0.02 * sigma_loss + 0.10 * outlier_loss


def leave_one_method_loss(
    measurements: torch.Tensor,
    valid: torch.Tensor,
    predicted_depth: torch.Tensor,
    log_sigma: torch.Tensor,
    bias: torch.Tensor,
    method_index: int,
) -> torch.Tensor:
    """Evaluate whether a held-out method is statistically predictable."""

    held_valid = valid[..., method_index : method_index + 1].unsqueeze(-1)
    held_measurement = measurements[..., method_index : method_index + 1].unsqueeze(-1)
    held_sigma = log_sigma[..., method_index : method_index + 1, :]
    held_bias = bias[..., method_index : method_index + 1, :]
    residual = held_measurement - held_bias - predicted_depth.unsqueeze(2)
    return student_t_nll(residual, held_sigma, held_valid)
