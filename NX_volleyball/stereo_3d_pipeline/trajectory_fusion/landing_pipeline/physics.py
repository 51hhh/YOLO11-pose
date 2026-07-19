"""Shared volleyball physics helpers for landing prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

G = 9.81
MASS = 0.270
RADIUS = 0.105
RHO = 1.225
AREA = math.pi * RADIUS * RADIUS
DRAG_K = 0.5 * RHO * AREA / MASS  # a_drag = -DRAG_K * Cd * |v| * v


def as_unit(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return arr / n


def plane_basis(g_hat: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (g_hat, e1, e2) with e1/e2 spanning the ground plane."""
    g = as_unit(g_hat)
    tmp = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(g, tmp))) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)
    e1 = np.cross(g, tmp)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(g, e1)
    return g, e1, e2


def height_above_ground(p: Sequence[float], g_hat: Sequence[float], ground_h: float) -> float:
    """Positive above ground. ground_h is the plane offset used by landing_eval."""
    return -float(np.dot(np.asarray(g_hat, dtype=float), np.asarray(p, dtype=float))) - float(ground_h)


def inplane_coords(vec: Sequence[float], e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    return np.array([float(np.dot(v, e1)), float(np.dot(v, e2))], dtype=float)


@dataclass(frozen=True)
class RolloutResult:
    landing: np.ndarray
    t_impact: float
    time_to_land: float


def rollout_landing(
    p: Sequence[float],
    v: Sequence[float],
    cd: float,
    t_now: float,
    g_hat: Sequence[float],
    ground_h: float,
    dt: float = 0.008,
    t_max: float = 3.0,
) -> Optional[RolloutResult]:
    """RK4 ballistic rollout with quadratic drag until ground-plane crossing."""
    p = np.asarray(p, dtype=float).copy()
    v = np.asarray(v, dtype=float).copy()
    g = as_unit(g_hat)

    def acc(vv: np.ndarray) -> np.ndarray:
        return G * g - DRAG_K * cd * np.linalg.norm(vv) * vv

    t = 0.0
    h_prev = height_above_ground(p, g, ground_h)
    if h_prev <= 0.0:
        return None

    while t < t_max:
        k1v = acc(v)
        k1p = v
        k2v = acc(v + 0.5 * dt * k1v)
        k2p = v + 0.5 * dt * k1v
        k3v = acc(v + 0.5 * dt * k2v)
        k3p = v + 0.5 * dt * k2v
        k4v = acc(v + dt * k3v)
        k4p = v + dt * k3v
        p = p + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        t += dt
        h = height_above_ground(p, g, ground_h)
        if h <= 0.0 and h_prev > 0.0:
            frac = h_prev / (h_prev - h)
            p_land = p + (frac - 1.0) * dt * v
            t_impact = t_now + t - (1.0 - frac) * dt
            return RolloutResult(
                landing=p_land,
                t_impact=float(t_impact),
                time_to_land=float(t_impact - t_now),
            )
        h_prev = h
    return None
