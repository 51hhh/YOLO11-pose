#!/usr/bin/env python3
"""Neural reliability models for multi-method depth fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ReliabilityOutput:
    log_sigma: torch.Tensor
    bias: torch.Tensor
    outlier_logit: torch.Tensor
    common_log_sigma: torch.Tensor


class MeasurementReliabilityNet(nn.Module):
    """Predict measurement uncertainty, not the final trajectory.

    Input shape:  [batch, time, input_dim]
    Output shape: [batch, time, num_methods, measurement_dim]
    """

    def __init__(
        self,
        input_dim: int,
        num_methods: int,
        measurement_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        min_sigma: float = 0.01,
        max_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_methods = num_methods
        self.measurement_dim = measurement_dim
        self.min_log_sigma = float(torch.log(torch.tensor(min_sigma)))
        self.max_log_sigma = float(torch.log(torch.tensor(max_sigma)))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        head_dim = num_methods * measurement_dim
        self.log_sigma_head = nn.Linear(hidden_dim, head_dim)
        self.bias_head = nn.Linear(hidden_dim, head_dim)
        self.outlier_head = nn.Linear(hidden_dim, head_dim)
        self.common_head = nn.Linear(hidden_dim, measurement_dim)

        nn.init.constant_(self.log_sigma_head.bias, -1.6)
        nn.init.zeros_(self.bias_head.bias)
        nn.init.constant_(self.outlier_head.bias, -3.0)
        nn.init.constant_(self.common_head.bias, -2.3)

    def forward(self, features: torch.Tensor) -> ReliabilityOutput:
        x = self.input_proj(features)
        encoded, _ = self.gru(x)
        shape = (*encoded.shape[:2], self.num_methods, self.measurement_dim)

        log_sigma = self.log_sigma_head(encoded).view(shape)
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        bias = self.bias_head(encoded).view(shape)
        outlier_logit = self.outlier_head(encoded).view(shape)
        common_log_sigma = torch.clamp(
            self.common_head(encoded),
            self.min_log_sigma,
            self.max_log_sigma,
        )
        return ReliabilityOutput(
            log_sigma=log_sigma,
            bias=bias,
            outlier_logit=outlier_logit,
            common_log_sigma=common_log_sigma,
        )


def reliability_weight(output: ReliabilityOutput, valid: torch.Tensor) -> torch.Tensor:
    """Convert predicted uncertainty/outlier probability to fusion weights."""

    sigma2 = torch.exp(2.0 * output.log_sigma)
    inlier_prob = 1.0 - torch.sigmoid(output.outlier_logit)
    return valid.unsqueeze(-1) * inlier_prob / (sigma2 + 1e-8)


def weighted_depth_consensus(
    measurements: torch.Tensor,
    valid: torch.Tensor,
    output: ReliabilityOutput,
    detach: bool = False,
) -> torch.Tensor:
    """Fuse scalar depth measurements with learned reliability.

    measurements shape: [batch, time, num_methods]
    valid shape:        [batch, time, num_methods]
    return shape:       [batch, time, 1]
    """

    corrected = measurements.unsqueeze(-1) - output.bias
    weights = reliability_weight(output, valid)
    numerator = (weights * corrected).sum(dim=2)
    denominator = weights.sum(dim=2).clamp_min(1e-6)
    consensus = numerator / denominator
    if detach:
        consensus = consensus.detach()
    return consensus


class TinyCausalFusionHead(nn.Module):
    """Optional real-time head for deployment experiments.

    This head avoids recurrence and consumes a short feature vector already
    containing temporal deltas/innovations. It is useful if GRU latency on NX
    is not acceptable.
    """

    def __init__(self, input_dim: int, num_methods: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.num_methods = num_methods
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_methods * 3),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.net(features).view(*features.shape[:-1], self.num_methods, 3)
        return {
            "log_sigma": F.softplus(raw[..., 0]) - 3.0,
            "bias": raw[..., 1],
            "outlier_logit": raw[..., 2],
        }
