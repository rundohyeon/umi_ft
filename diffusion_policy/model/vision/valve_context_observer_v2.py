"""Portable loader for the frozen four-state causal valve observer.

The checkpoint is deliberately kept outside the action-policy state.  This
module only reconstructs the exact model serialized by ``best_context.pt``;
the action policy consumes its four probabilities plus a runtime-valid flag.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder


OBSERVER_SCHEMA = "umi_causal_context_observer_v2_4state"
OBSERVER_PHASE_NAMES = ("approach", "turning", "recovery", "error")
OBSERVER_FORCE_HISTORY_SAMPLES = 50
OBSERVER_RGB_HORIZON = 2
OBSERVER_RGB_STRIDE = 3


def observer_shape_meta() -> dict:
    """Return the immutable observation contract used by the checkpoint."""

    return {
        "obs": {
            "camera0_rgb": {
                "shape": [3, 224, 224],
                "horizon": OBSERVER_RGB_HORIZON,
                "type": "rgb",
                "ignore_by_policy": False,
            },
            "robot0_eef_pos": {
                "shape": [3],
                "horizon": OBSERVER_RGB_HORIZON,
                "type": "low_dim",
                "ignore_by_policy": False,
            },
            # The historical key name is retained for checkpoint compatibility;
            # its actual tensor is the 6-D rotation representation.
            "robot0_eef_rot_axis_angle": {
                "shape": [6],
                "horizon": OBSERVER_RGB_HORIZON,
                "type": "low_dim",
                "ignore_by_policy": False,
            },
            "robot0_gripper_width": {
                "shape": [1],
                "horizon": OBSERVER_RGB_HORIZON,
                "type": "low_dim",
                "ignore_by_policy": False,
            },
            "robot0_ft_history": {
                "shape": [OBSERVER_FORCE_HISTORY_SAMPLES, 12],
                "horizon": OBSERVER_RGB_HORIZON,
                "type": "force_history",
                "ignore_by_policy": False,
            },
            "robot0_ft_history_valid": {
                "shape": [OBSERVER_FORCE_HISTORY_SAMPLES, 1],
                "horizon": OBSERVER_RGB_HORIZON,
                "type": "force_history_mask",
                "ignore_by_policy": False,
            },
        },
        # Present in the observer normalizer checkpoint, but there is no action
        # head and this field is never supplied to ``predict_context``.
        "action": {"shape": [10], "horizon": 16},
    }


def _group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(int(max_groups), int(channels))
    while channels % groups != 0:
        groups -= 1
    return groups


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, padding=0, **kwargs)
        self.left_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(value, (self.left_padding, 0)))


class ForceHistoryEncoder(nn.Module):
    """Encode one masked 12-D, 50-sample physical-wrench history."""

    def __init__(
        self,
        channels: int = 12,
        feature_dim: int = 128,
        base_channels: int = 32,
    ):
        super().__init__()

        def block(input_channels: int, output_channels: int) -> nn.Sequential:
            return nn.Sequential(
                CausalConv1d(
                    input_channels, output_channels, kernel_size=5, stride=2
                ),
                nn.GroupNorm(_group_count(output_channels), output_channels),
                nn.SiLU(),
            )

        # The thirteenth channel distinguishes missing startup history from a
        # physically valid zero-wrench sample.
        self.input_channels = int(channels) + 1
        self.network = nn.Sequential(
            block(self.input_channels, int(base_channels)),
            block(int(base_channels), int(base_channels) * 2),
            block(int(base_channels) * 2, int(base_channels) * 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(base_channels) * 4, int(feature_dim)),
            nn.SiLU(),
        )

    def forward(
        self, history: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError(f"expected [B,12,H] force history, got {history.shape}")
        if valid_mask is None:
            mask_channel = torch.ones(
                (history.shape[0], 1, history.shape[2]),
                dtype=history.dtype,
                device=history.device,
            )
        else:
            if valid_mask.shape != history.shape[:1] + history.shape[2:]:
                raise ValueError("force validity mask must have shape [B,H]")
            mask_channel = valid_mask[:, None, :].to(dtype=history.dtype)
            history = history * mask_channel
        return self.projection(
            self.network(torch.cat([history, mask_channel], dim=1))
        )


class TimmForceHistoryObsEncoder(TimmObsEncoder):
    """Frozen CLIP-ViT RGB/proprio path plus a gated causal F/T branch."""

    def __init__(
        self,
        shape_meta: dict,
        force_key: str = "robot0_ft_history",
        force_mask_key: str = "robot0_ft_history_valid",
        force_feature_dim: int = 128,
        force_base_channels: int = 32,
        force_gate_init: float = 0.1,
        append_force_feature: bool = True,
        **kwargs,
    ):
        self._visual_backbone_frozen = bool(kwargs.get("frozen", False))
        self.full_shape_meta = copy.deepcopy(shape_meta)
        self.force_key = str(force_key)
        self.force_mask_key = str(force_mask_key)
        force_shape = tuple(shape_meta["obs"][self.force_key]["shape"])
        mask_shape = tuple(shape_meta["obs"][self.force_mask_key]["shape"])
        if force_shape != (OBSERVER_FORCE_HISTORY_SAMPLES, 12):
            raise ValueError(f"unexpected observer force shape {force_shape}")
        if mask_shape != (OBSERVER_FORCE_HISTORY_SAMPLES, 1):
            raise ValueError(f"unexpected observer force-mask shape {mask_shape}")

        stock_shape_meta = copy.deepcopy(shape_meta)
        del stock_shape_meta["obs"][self.force_key]
        del stock_shape_meta["obs"][self.force_mask_key]
        super().__init__(shape_meta=stock_shape_meta, **kwargs)

        self.full_shape_meta = copy.deepcopy(shape_meta)
        self.force_history_samples = force_shape[0]
        self.force_encoder = ForceHistoryEncoder(
            channels=12,
            feature_dim=int(force_feature_dim),
            base_channels=int(force_base_channels),
        )
        self.force_gate = nn.Parameter(torch.tensor(float(force_gate_init)))
        self.force_feature_dim = int(force_feature_dim)
        self.append_force_feature = bool(append_force_feature)

    def train(self, mode: bool = True):
        result = super().train(mode)
        if self._visual_backbone_frozen:
            for visual_model in self.key_model_map.values():
                visual_model.eval()
        return result

    def encode_stock_observation(self, obs_dict: dict) -> torch.Tensor:
        if self._visual_backbone_frozen:
            with torch.no_grad():
                return super().forward(obs_dict)
        return super().forward(obs_dict)

    def encode_force_history(self, obs_dict: dict) -> torch.Tensor:
        history = obs_dict[self.force_key]
        mask = obs_dict[self.force_mask_key]
        if history.ndim != 4:
            raise ValueError(f"observer force history must be [B,2,50,12], got {history.shape}")
        batch, horizon, samples, channels = history.shape
        if (horizon, samples, channels) != (2, self.force_history_samples, 12):
            raise ValueError(f"unexpected observer force history shape {history.shape}")
        if mask.shape != (batch, horizon, samples, 1):
            raise ValueError(f"unexpected observer force mask shape {mask.shape}")
        flat_history = history.reshape(batch * horizon, samples, channels)
        flat_history = flat_history.transpose(1, 2).contiguous()
        flat_mask = mask.reshape(batch * horizon, samples)
        encoded = self.force_encoder(flat_history, flat_mask)
        return encoded.reshape(batch, horizon * self.force_feature_dim)

    def forward(self, obs_dict: dict) -> torch.Tensor:
        stock = self.encode_stock_observation(obs_dict)
        if not self.append_force_feature:
            return stock
        force = self.encode_force_history(obs_dict)
        return torch.cat([stock, self.force_gate * force], dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example = {}
        for key, attr in self.full_shape_meta["obs"].items():
            example[key] = torch.zeros(
                (1, int(attr["horizon"])) + tuple(attr["shape"]),
                dtype=self.dtype,
                device=self.device,
            )
        output = self.forward(example)
        if output.ndim != 2 or output.shape[0] != 1:
            raise AssertionError(f"unexpected observer output shape {output.shape}")
        return output.shape


class UmiCausalContextPolicy(nn.Module):
    """Four-state context observer; this model has no action head."""

    def __init__(
        self,
        obs_encoder: nn.Module,
        context_phase_names: Sequence[str] = OBSERVER_PHASE_NAMES,
        context_head_hidden_dim: int = 256,
        context_head_dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.context_phase_names = tuple(map(str, context_phase_names))
        if self.context_phase_names != OBSERVER_PHASE_NAMES:
            raise ValueError("observer phase order does not match checkpoint contract")
        feature_dim = int(np.prod(self.obs_encoder.output_shape()))
        if feature_dim != 1812:
            raise ValueError(f"observer feature dimension must be 1812, got {feature_dim}")
        self.context_head = nn.Sequential(
            nn.Linear(feature_dim, int(context_head_hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(context_head_dropout)),
            nn.Linear(int(context_head_hidden_dim), len(self.context_phase_names)),
        )
        self.normalizer = LinearNormalizer()
        self.register_buffer(
            "context_class_weight", torch.ones(len(self.context_phase_names))
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _logits(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.context_head(self.obs_encoder(self.normalizer.normalize(obs)))

    def predict_context(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = self._logits(obs)
        return {
            "context_logits": logits,
            "context_prob": torch.softmax(logits, dim=-1),
            "context_pred": torch.argmax(logits, dim=-1),
        }

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._logits(obs)


def _normalizer_from_state(state: dict[str, torch.Tensor]) -> LinearNormalizer:
    prefix = "normalizer.params_dict."
    names = sorted(
        {
            key[len(prefix):].split(".", 1)[0]
            for key in state
            if key.startswith(prefix)
        }
    )
    normalizer = LinearNormalizer()
    for name in names:
        base = f"{prefix}{name}."
        stats = {
            item: state[f"{base}input_stats.{item}"].detach().cpu()
            for item in ("min", "max", "mean", "std")
        }
        normalizer[name] = SingleFieldLinearNormalizer.create_manual(
            scale=state[f"{base}scale"].detach().cpu(),
            offset=state[f"{base}offset"].detach().cpu(),
            input_stats_dict=stats,
        )
    return normalizer


def load_frozen_context_observer(
    checkpoint_path: str | Path, device: str | torch.device = "cuda"
) -> tuple[UmiCausalContextPolicy, tuple[str, ...], dict]:
    """Load ``best_context.pt`` without the original training dataset."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"context observer checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema") != OBSERVER_SCHEMA:
        raise ValueError(f"unsupported context observer schema {checkpoint.get('schema')!r}")
    phases = tuple(checkpoint.get("context_phase_names", ()))
    if phases != OBSERVER_PHASE_NAMES:
        raise ValueError(f"unexpected observer phase order {phases}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("context observer checkpoint has no model_state_dict")

    encoder = TimmForceHistoryObsEncoder(
        shape_meta=observer_shape_meta(),
        model_name="vit_base_patch16_clip_224.openai",
        pretrained=False,
        frozen=False,
        global_pool="",
        transforms=None,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
        feature_aggregation=None,
        downsample_ratio=32,
        position_encording="sinusoidal",
        force_feature_dim=128,
        force_base_channels=32,
        force_gate_init=0.1,
    )
    model = UmiCausalContextPolicy(obs_encoder=encoder)
    model.set_normalizer(_normalizer_from_state(state))
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.to(torch.device(device)).eval()
    return model, phases, checkpoint


__all__ = [
    "OBSERVER_SCHEMA",
    "OBSERVER_PHASE_NAMES",
    "OBSERVER_FORCE_HISTORY_SAMPLES",
    "OBSERVER_RGB_HORIZON",
    "OBSERVER_RGB_STRIDE",
    "observer_shape_meta",
    "load_frozen_context_observer",
]
