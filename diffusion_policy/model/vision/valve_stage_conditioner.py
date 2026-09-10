"""Context-conditioned residual feature routing for the valve policy.

The legacy 10-D/v1 contract uses five phase experts.  The v2 contract uses
four A/T/R/E experts and a fifth ``context_valid`` input.  In both versions,
phase probabilities select a soft mixture of residual experts.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ValveResidualExpert(nn.Module):
    """One phase-specific residual expert with the checkpoint's key layout."""

    def __init__(self, feature_dim: int, bottleneck_dim: int):
        super().__init__()
        # Keep this Sequential layout: checkpoint names include
        # ``experts.<phase>.network.<layer>``.
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ValveStageConditioner(nn.Module):
    """Add classifier context and phase-routed residuals to fusion features."""

    def __init__(
        self,
        context_dim: int = 10,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        expert_bottleneck_dim: int = 128,
        num_phase_experts: int = 5,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.feature_dim = int(feature_dim)
        self.num_phase_experts = int(num_phase_experts)
        supported = {(10, 5), (5, 4)}
        if (self.context_dim, self.num_phase_experts) not in supported:
            raise ValueError(
                "supported valve context contracts are (dim=10, experts=5) "
                "and (dim=5, experts=4), got "
                f"(dim={self.context_dim}, experts={self.num_phase_experts})"
            )
        # Do not wrap this in another named module: the trained checkpoint has
        # ``stage_encoder.0`` and ``stage_encoder.2`` keys.
        self.stage_encoder = nn.Sequential(
            nn.Linear(self.context_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.feature_dim),
        )
        self.experts = nn.ModuleList(
            [
                _ValveResidualExpert(
                    self.feature_dim, int(expert_bottleneck_dim)
                )
                for _ in range(self.num_phase_experts)
            ]
        )

    def forward(
        self, fused_feature: torch.Tensor, valve_context: torch.Tensor
    ) -> torch.Tensor:
        if fused_feature.ndim != 2 or fused_feature.shape[-1] != self.feature_dim:
            raise ValueError(
                "fusion feature must be [B,%d], got %s"
                % (self.feature_dim, tuple(fused_feature.shape))
            )
        if valve_context.ndim == 3:
            if valve_context.shape[1] != 1:
                raise ValueError(
                    "valve_context horizon must be one, got %s"
                    % (tuple(valve_context.shape),)
                )
            valve_context = valve_context[:, 0]
        if (
            valve_context.ndim != 2
            or valve_context.shape[0] != fused_feature.shape[0]
            or valve_context.shape[1] != self.context_dim
        ):
            raise ValueError(
                "valve_context must be [B,%d] (or [B,1,%d]), got %s"
                % (self.context_dim, self.context_dim, tuple(valve_context.shape))
            )
        if not torch.isfinite(valve_context).all():
            raise ValueError("valve_context contains NaN or Inf")

        phase_weights = valve_context[:, : self.num_phase_experts]
        stage_embedding = self.stage_encoder(valve_context)
        expert_outputs = torch.stack(
            [expert(fused_feature) for expert in self.experts], dim=1
        )
        routed_residual = torch.sum(
            expert_outputs * phase_weights.unsqueeze(-1), dim=1
        )
        return fused_feature + stage_embedding + routed_residual
