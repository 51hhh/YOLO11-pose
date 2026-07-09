#!/usr/bin/env python3
"""Causal learned Kalman filter for 3D volleyball trajectory estimation.

Stage-2 core model. This is a KalmanNet-style recurrent filter, not a plain
regression network and not the two-stage "reliability -> external smoother"
design in ``models.py``. Key properties, matching the agreed requirements:

1. End-to-end 3D state output. ``forward``/``step`` return the filtered state
   ``[x, y, z, vx, vy, vz]`` directly. There is no separate hand-written
   smoother downstream.

2. Strictly causal, frame-by-frame. The same ``step()`` is used in training
   (unrolled over a sequence) and at inference (one frame at a time, history
   carried in the GRU hidden state and the previous state/covariance-proxy).
   Nothing looks at future frames, so training and deployment share one path.

3. Physics is a hard structural prior, not a soft loss. The predict step uses
   the constant-velocity + gravity transition (same shape as
   ``robust_smoother._transition``): p += v*dt (+ 0.5 g dt^2 on the gravity
   axis), v += g*dt. The network only learns the *correction gain* applied to
   the innovation, so it cannot drift away from ballistic motion the way a free
   regression head can.

4. XY-Z noise coupling is visible. Observations are the per-method metric
   ``[X, Y, Z]`` from ``reproject.py``; because every candidate shares the left
   pixel anchor, a noisy Z produces proportionally noisy X/Y. The gain network
   sees the per-method spread and the innovation, so it can down-weight frames
   whose XY jitter is really Z jitter.

Gravity axis: the camera-y alignment with gravity is not yet confirmed
(recording metadata writes ``gravity_y: 0.0``). Pass ``gravity=0.0`` until a
clean vertical-drop clip calibrates it; the filter then degrades to a
constant-velocity prior, which is still correct, just weaker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class FilterConfig:
    num_methods: int
    quality_dim: int  # per-method quality features (support/std/conf/trust...)
    hidden_dim: int = 64
    gru_layers: int = 1
    gravity: float = 0.0  # m/s^2 on the gravity axis; 0 until y-axis confirmed
    gravity_axis: int = 1  # 0=x,1=y,2=z; camera y is the candidate down-axis
    min_gain: float = 0.0
    max_gain: float = 1.0
    process_floor: float = 1e-3


def build_transition(dt: torch.Tensor, gravity: float, gravity_axis: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Constant-velocity + gravity transition, batched over dt.

    dt shape: [batch]. Returns (F [batch,6,6], control [batch,6]) with the same
    convention as robust_smoother._transition: state = [x,y,z,vx,vy,vz],
    p += v*dt, v stays; gravity adds 0.5 g dt^2 to position and g dt to velocity
    on the gravity axis.
    """

    b = dt.shape[0]
    F = torch.eye(6, dtype=dt.dtype, device=dt.device).unsqueeze(0).repeat(b, 1, 1)
    F[:, 0, 3] = dt
    F[:, 1, 4] = dt
    F[:, 2, 5] = dt
    control = torch.zeros(b, 6, dtype=dt.dtype, device=dt.device)
    if gravity != 0.0:
        pos_axis = gravity_axis
        vel_axis = gravity_axis + 3
        control[:, pos_axis] = 0.5 * gravity * dt * dt
        control[:, vel_axis] = gravity * dt
    return F, control


def fuse_observation(
    obs_xyz: torch.Tensor,
    obs_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse per-method metric points into one observation + spread.

    obs_xyz shape:   [batch, num_methods, 3]
    obs_valid shape: [batch, num_methods]
    Returns:
      mean_xyz   [batch, 3]   median-like robust mean of valid methods
      spread_xyz [batch, 3]   dispersion across valid methods (0 if <2 valid)
      any_valid  [batch]      1.0 if at least one method valid
    The mean is the plain valid-mean here; the network additionally receives the
    full per-method tensor, so it can learn a better combination. This fused
    value is only the innovation anchor for the physical update.
    """

    valid = obs_valid.unsqueeze(-1)  # [b, m, 1]
    count = valid.sum(dim=1).clamp_min(1.0)  # [b,1]
    mean_xyz = (obs_xyz * valid).sum(dim=1) / count
    diff = (obs_xyz - mean_xyz.unsqueeze(1)) * valid
    var = (diff * diff).sum(dim=1) / count
    spread_xyz = torch.sqrt(var.clamp_min(0.0))
    any_valid = (obs_valid.sum(dim=1) > 0.0).to(obs_xyz.dtype)
    return mean_xyz, spread_xyz, any_valid


class CausalKalmanNet(nn.Module):
    """Learned-gain causal Kalman filter with a ballistic transition.

    Input per frame:
      obs_xyz    [batch, num_methods, 3]  per-method reprojected metric points
      obs_valid  [batch, num_methods]     validity mask
      quality    [batch, quality_dim]     per-frame quality features (normalized)
      dt         [batch]                  seconds since previous frame

    Output per frame:
      state      [batch, 6]  filtered [x,y,z,vx,vy,vz]

    The gain network outputs a per-axis correction gain in [0,1] for position
    and velocity (6 gains), applied as: state = pred + gain * (H_obs - H_pred),
    where the "measurement" is the fused xyz for the position channels and a
    finite-difference velocity for the velocity channels. This is the KalmanNet
    idea: replace the analytic Kalman gain with a learned, history-conditioned
    gain, keeping the physical predict step exact.
    """

    def __init__(self, cfg: FilterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # gain-net input: fused xyz(3) + spread(3) + innovation(3) +
        # per-method flattened xyz (m*3) + validity (m) + quality(q) + dt(1)
        gain_in = 3 + 3 + 3 + cfg.num_methods * 3 + cfg.num_methods + cfg.quality_dim + 1
        self.input_proj = nn.Sequential(
            nn.Linear(gain_in, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, num_layers=cfg.gru_layers, batch_first=True)
        # 6 gains (3 position + 3 velocity) + 3 log-variance for output covariance
        self.gain_head = nn.Linear(cfg.hidden_dim, 6)
        self.logvar_head = nn.Linear(cfg.hidden_dim, 3)
        nn.init.zeros_(self.gain_head.bias)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> dict:
        return {
            "state": torch.zeros(batch, 6, device=device, dtype=dtype),
            "hidden": torch.zeros(self.cfg.gru_layers, batch, self.cfg.hidden_dim, device=device, dtype=dtype),
            "prev_pos": None,  # last fused position, for velocity innovation
            "initialized": torch.zeros(batch, dtype=torch.bool, device=device),
        }

    def step(
        self,
        carry: dict,
        obs_xyz: torch.Tensor,
        obs_valid: torch.Tensor,
        quality: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Advance one frame. Returns (state[b,6], logvar[b,3], new_carry)."""

        cfg = self.cfg
        b = obs_xyz.shape[0]
        device, dtype = obs_xyz.device, obs_xyz.dtype
        state = carry["state"]
        hidden = carry["hidden"]
        prev_pos = carry["prev_pos"]
        initialized = carry["initialized"]

        mean_xyz, spread_xyz, any_valid = fuse_observation(obs_xyz, obs_valid)

        # --- physical predict ---
        F, control = build_transition(dt, cfg.gravity, cfg.gravity_axis)
        pred = torch.bmm(F, state.unsqueeze(-1)).squeeze(-1) + control  # [b,6]

        # cold-start: first valid frame seeds position with the observation and
        # zero velocity so the filter does not chase from the origin.
        seed = any_valid.bool() & (~initialized)
        if seed.any():
            pred = pred.clone()
            pred[seed, 0:3] = mean_xyz[seed]
            pred[seed, 3:6] = 0.0

        pred_pos = pred[:, 0:3]
        innovation = (mean_xyz - pred_pos) * any_valid.unsqueeze(-1)

        # velocity pseudo-measurement from finite difference of fused positions
        if prev_pos is None:
            vel_meas = pred[:, 3:6]
        else:
            vel_meas = (mean_xyz - prev_pos) / dt.clamp_min(1e-4).unsqueeze(-1)
        vel_innovation = (vel_meas - pred[:, 3:6]) * any_valid.unsqueeze(-1)

        # --- gain network ---
        gain_in = torch.cat(
            [
                mean_xyz,
                spread_xyz,
                innovation,
                obs_xyz.reshape(b, -1),
                obs_valid,
                quality,
                dt.unsqueeze(-1),
            ],
            dim=-1,
        )
        x = self.input_proj(gain_in).unsqueeze(1)  # [b,1,h]
        encoded, hidden = self.gru(x, hidden)
        encoded = encoded.squeeze(1)
        raw_gain = self.gain_head(encoded)
        gain = cfg.min_gain + (cfg.max_gain - cfg.min_gain) * torch.sigmoid(raw_gain)
        logvar = self.logvar_head(encoded)

        # gate gains by observation validity so unobserved frames coast on physics
        gain = gain * any_valid.unsqueeze(-1)

        # --- learned update ---
        corr_pos = gain[:, 0:3] * innovation
        corr_vel = gain[:, 3:6] * vel_innovation
        new_state = pred.clone()
        new_state[:, 0:3] = pred[:, 0:3] + corr_pos
        new_state[:, 3:6] = pred[:, 3:6] + corr_vel

        new_prev_pos = torch.where(
            any_valid.unsqueeze(-1).bool(),
            mean_xyz,
            prev_pos if prev_pos is not None else mean_xyz,
        )
        new_carry = {
            "state": new_state,
            "hidden": hidden,
            "prev_pos": new_prev_pos,
            "initialized": initialized | any_valid.bool(),
            "last_gain": gain,
        }
        return new_state, logvar, new_carry

    def forward(
        self,
        obs_xyz: torch.Tensor,
        obs_valid: torch.Tensor,
        quality: torch.Tensor,
        dt: torch.Tensor,
        return_gains: bool = False,
    ):
        """Unroll the causal filter over a full sequence.

        Shapes:
          obs_xyz   [batch, time, num_methods, 3]
          obs_valid [batch, time, num_methods]
          quality   [batch, time, quality_dim]
          dt        [batch, time]
        Returns:
          states [batch, time, 6]
          logvar [batch, time, 3]
          gains  [batch, time, 6]   (only when return_gains=True)
        This is a plain unroll of ``step``; there is no bidirectional pass, so
        the exact same computation runs at inference one frame at a time.
        """

        b, t = obs_xyz.shape[0], obs_xyz.shape[1]
        carry = self.init_state(b, obs_xyz.device, obs_xyz.dtype)
        states = []
        logvars = []
        gains = []
        for i in range(t):
            state, logvar, carry = self.step(
                carry,
                obs_xyz[:, i],
                obs_valid[:, i],
                quality[:, i],
                dt[:, i],
            )
            states.append(state)
            logvars.append(logvar)
            if return_gains:
                gains.append(carry["last_gain"])
        states_out = torch.stack(states, dim=1)
        logvars_out = torch.stack(logvars, dim=1)
        if return_gains:
            return states_out, logvars_out, torch.stack(gains, dim=1)
        return states_out, logvars_out
