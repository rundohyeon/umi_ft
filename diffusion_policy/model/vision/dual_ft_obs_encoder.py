from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.model.vision.valve_stage_conditioner import ValveStageConditioner


logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """Causal convolution whose outputs align to each strided window's end.

    For ``kernel_size=2, stride=2`` no padding is needed: output ``j`` is
    aligned with input ``2*j+1`` and covers inputs ``2*j`` through ``2*j+1``.
    Padding by ``kernel_size-1`` here would align the first output with input
    zero and, after five downsampling stages, discard every later sample.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1):
        super().__init__()
        kernel_size = int(kernel_size)
        stride = int(stride)
        self.left_padding = max(kernel_size - stride, 0)
        self.conv = nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class CausalFTEncoder(nn.Module):
    """Encode a causal ``[B,T,6]`` native-sensor wrench history to one token."""

    def __init__(
        self,
        input_dim=6,
        channel_dims=(16, 32, 64, 128),
        output_dim=768,
        kernel_size=2,
        stride=2,
        negative_slope=0.1,
    ):
        super().__init__()
        self.register_buffer(
            "temporal_contract_version",
            torch.tensor(1, dtype=torch.int64),
            persistent=True,
        )
        dimensions = [int(input_dim), *map(int, channel_dims), int(output_dim)]
        layers = []
        for input_channels, output_channels in zip(dimensions[:-1], dimensions[1:]):
            layers.extend(
                [
                    CausalConv1d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                    ),
                    nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                ]
            )
        self.network = nn.Sequential(*layers)

    def forward_sequence(self, history):
        if history.ndim != 3 or history.shape[-1] != 6:
            raise ValueError(
                f"F/T history must have shape [B,T,6], got {tuple(history.shape)}"
            )
        return self.network(history.transpose(1, 2)).transpose(1, 2)

    def forward(self, history):
        return self.forward_sequence(history)[:, -1]


class CausalFTDifference(nn.Module):
    """Causally smooth a wrench history and form its backward difference.

    The EMA starts from the first (possibly repeat-padded) sample, so padded
    episode prefixes produce exactly zero deltas.  Both training and real-time
    inference run this module *after* the policy's affine F/T normalizer.  EMA
    commutes with that affine transform and the offset cancels in the
    difference, so this is equivalent to smoothing physical wrench first and
    then scaling its difference.
    """

    def __init__(self, ema_alpha: float = 0.25):
        super().__init__()
        ema_alpha = float(ema_alpha)
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("F/T EMA alpha must be in (0, 1]")
        self.register_buffer(
            "difference_contract_version",
            torch.tensor(1, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "ema_alpha",
            torch.tensor(ema_alpha, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, history):
        if history.ndim != 3 or history.shape[-1] != 6:
            raise ValueError(
                f"F/T history must have shape [B,T,6], got {tuple(history.shape)}"
            )
        if history.shape[1] < 1:
            raise ValueError("F/T history must contain at least one sample")
        alpha = self.ema_alpha.to(dtype=history.dtype, device=history.device)
        previous = history[:, 0]
        smoothed = [previous]
        for index in range(1, history.shape[1]):
            previous = alpha * history[:, index] + (1.0 - alpha) * previous
            smoothed.append(previous)
        smoothed = torch.stack(smoothed, dim=1)
        difference = torch.cat(
            [
                torch.zeros_like(smoothed[:, :1]),
                smoothed[:, 1:] - smoothed[:, :-1],
            ],
            dim=1,
        )
        return smoothed, difference


class LatestAbsoluteFTEncoder(nn.Module):
    """Encode the latest causally smoothed absolute wrench as one token."""

    def __init__(self, input_dim=6, hidden_dim=128, output_dim=768):
        super().__init__()
        self.register_buffer(
            "absolute_contract_version",
            torch.tensor(1, dtype=torch.int64),
            persistent=True,
        )
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(int(hidden_dim), int(output_dim)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, smoothed_history):
        if smoothed_history.ndim != 3 or smoothed_history.shape[-1] != 6:
            raise ValueError(
                "smoothed F/T history must have shape [B,T,6], got "
                f"{tuple(smoothed_history.shape)}"
            )
        return self.network(smoothed_history[:, -1])


class DualFTObsEncoder(ModuleAttrMixin):
    """TARGET vision/pose encoder augmented by native F/T fusion tokens.

    The vision backbone, image transforms, and low-dimensional pose flattening
    are delegated to :class:`TimmObsEncoder`. Left and right histories never
    share parameters unless ``share_ft_encoder`` is explicitly enabled.
    """

    def __init__(
        self,
        shape_meta: dict,
        model_name: str,
        pretrained: bool,
        frozen: bool,
        global_pool: str,
        transforms: list,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        feature_aggregation: str = "spatial_embedding",
        downsample_ratio: int = 32,
        position_encording: str = "learnable",
        left_ft_key: str = "robot0_ft_left",
        right_ft_key: str = "robot0_ft_right",
        vision_feature_dim: int = 768,
        fusion_dim: int = 768,
        fusion_heads: int = 8,
        fusion_layers: int = 1,
        fusion_feedforward_dim: int = 2048,
        fusion_dropout: float = 0.0,
        fusion_position_encoding: str = "learnable",
        ft_channel_dims=(16, 32, 64, 128),
        share_ft_encoder: bool = False,
        ft_feature_mode: str = "raw_history",
        ft_delta_ema_alpha: float = 0.25,
        valve_context_key: str | None = None,
        valve_context_dim: int = 10,
        valve_context_hidden_dim: int = 128,
        valve_expert_bottleneck_dim: int = 128,
        valve_context_num_phase_experts: int = 5,
    ):
        super().__init__()
        ft_feature_mode = str(ft_feature_mode)
        supported_ft_modes = {
            "raw_history",
            "raw_history_plus_delta_history",
            "latest_absolute_plus_delta_history",
        }
        if ft_feature_mode not in supported_ft_modes:
            raise ValueError(
                f"unsupported F/T feature mode {ft_feature_mode!r}; "
                f"expected one of {sorted(supported_ft_modes)}"
            )
        if valve_context_key is None and ft_feature_mode == "raw_history":
            architecture_contract_version = 2
        elif (
            (int(valve_context_dim), int(valve_context_num_phase_experts)) == (10, 5)
            and ft_feature_mode == "raw_history"
        ):
            architecture_contract_version = 3
        elif (
            (int(valve_context_dim), int(valve_context_num_phase_experts)) == (5, 4)
            and ft_feature_mode == "raw_history"
        ):
            architecture_contract_version = 4
        elif (
            (int(valve_context_dim), int(valve_context_num_phase_experts)) == (5, 4)
            and ft_feature_mode == "raw_history_plus_delta_history"
        ):
            architecture_contract_version = 5
        elif (
            (int(valve_context_dim), int(valve_context_num_phase_experts)) == (5, 4)
            and ft_feature_mode == "latest_absolute_plus_delta_history"
        ):
            architecture_contract_version = 6
        else:
            raise ValueError(
                "unsupported valve-context/F/T encoder contract: "
                f"context_key={valve_context_key!r}, dim={valve_context_dim}, "
                f"experts={valve_context_num_phase_experts}, "
                f"ft_feature_mode={ft_feature_mode!r}"
            )
        self.register_buffer(
            "architecture_contract_version",
            torch.tensor(architecture_contract_version, dtype=torch.int64),
            persistent=True,
        )
        self.shape_meta = shape_meta
        self.left_ft_key = left_ft_key
        self.right_ft_key = right_ft_key
        self.fusion_dim = int(fusion_dim)
        self.valve_context_key = valve_context_key
        self.vision_backbone_frozen = bool(frozen)
        self.ft_feature_mode = ft_feature_mode

        obs_meta = shape_meta["obs"]
        for key in (left_ft_key, right_ft_key):
            if key not in obs_meta:
                raise ValueError(f"shape_meta is missing required F/T key {key!r}")
            if tuple(obs_meta[key]["shape"]) != (6,):
                raise ValueError(f"{key} must have six independent channels")
        if valve_context_key is not None:
            context_meta = obs_meta.get(valve_context_key)
            if context_meta is None:
                raise ValueError(
                    f"shape_meta is missing required valve context key {valve_context_key!r}"
                )
            if tuple(context_meta.get("shape", ())) != (int(valve_context_dim),):
                raise ValueError(
                    f"{valve_context_key} must have shape [{int(valve_context_dim)}]"
                )
            if int(context_meta.get("horizon", -1)) != 1:
                raise ValueError(f"{valve_context_key} horizon must be one")
            if not bool(context_meta.get("ignore_by_policy", False)):
                raise ValueError(
                    f"{valve_context_key} must be ignored by the legacy low-dim encoder"
                )

        # The legacy encoder sees exactly the original RGB and pose fields.
        # F/T is removed rather than marked as an ordinary low-dimensional
        # feature, because each stream has its own temporal encoder below.
        legacy_shape_meta = copy.deepcopy(shape_meta)
        del legacy_shape_meta["obs"][left_ft_key]
        del legacy_shape_meta["obs"][right_ft_key]
        self.vision_pose_encoder = TimmObsEncoder(
            shape_meta=legacy_shape_meta,
            model_name=model_name,
            pretrained=pretrained,
            frozen=frozen,
            global_pool=global_pool,
            transforms=transforms,
            use_group_norm=use_group_norm,
            share_rgb_model=share_rgb_model,
            imagenet_norm=imagenet_norm,
            feature_aggregation=feature_aggregation,
            downsample_ratio=downsample_ratio,
            position_encording=position_encording,
        )
        if len(self.vision_pose_encoder.rgb_keys) != 1:
            raise ValueError(
                "Dual-F/T policy allowlist requires exactly one RGB stream, got "
                f"{self.vision_pose_encoder.rgb_keys}"
            )
        if self.vision_backbone_frozen:
            if any(
                param.requires_grad
                for param in self.vision_pose_encoder.key_model_map.parameters()
            ):
                raise AssertionError("frozen vision backbone has trainable parameters")
            # policy.train() is called every epoch. Keep the frozen timm
            # backbone deterministic while leaving image augmentation and all
            # trainable F/T/fusion/context modules in training mode.
            self.vision_pose_encoder.key_model_map.eval()

        if int(vision_feature_dim) == self.fusion_dim:
            self.visual_projection = nn.Identity()
        else:
            self.visual_projection = nn.Linear(
                int(vision_feature_dim), self.fusion_dim
            )

        if self.ft_feature_mode in (
            "raw_history",
            "raw_history_plus_delta_history",
        ):
            self.left_ft_encoder = CausalFTEncoder(
                channel_dims=ft_channel_dims,
                output_dim=self.fusion_dim,
            )
            if share_ft_encoder:
                self.right_ft_encoder = self.left_ft_encoder
            else:
                self.right_ft_encoder = CausalFTEncoder(
                    channel_dims=ft_channel_dims,
                    output_dim=self.fusion_dim,
                )
        else:
            absolute_hidden_dim = int(tuple(ft_channel_dims)[-1])
            self.left_ft_absolute_encoder = LatestAbsoluteFTEncoder(
                hidden_dim=absolute_hidden_dim,
                output_dim=self.fusion_dim,
            )
            if share_ft_encoder:
                self.right_ft_absolute_encoder = self.left_ft_absolute_encoder
            else:
                self.right_ft_absolute_encoder = LatestAbsoluteFTEncoder(
                    hidden_dim=absolute_hidden_dim,
                    output_dim=self.fusion_dim,
                )

        if self.ft_feature_mode != "raw_history":
            self.ft_difference = CausalFTDifference(
                ema_alpha=float(ft_delta_ema_alpha)
            )
            self.left_ft_delta_encoder = CausalFTEncoder(
                channel_dims=ft_channel_dims,
                output_dim=self.fusion_dim,
            )
            if share_ft_encoder:
                self.right_ft_delta_encoder = self.left_ft_delta_encoder
            else:
                self.right_ft_delta_encoder = CausalFTEncoder(
                    channel_dims=ft_channel_dims,
                    output_dim=self.fusion_dim,
                )
        self.share_ft_encoder = bool(share_ft_encoder)

        rgb_horizon = sum(
            int(legacy_shape_meta["obs"][key]["horizon"])
            for key in self.vision_pose_encoder.rgb_keys
        )
        self.num_fusion_tokens = rgb_horizon + (
            2 if self.ft_feature_mode == "raw_history" else 4
        )
        if int(fusion_layers) != 1:
            raise ValueError(
                "official UMI-FT fusion contract requires fusion_layers=1"
            )
        if str(fusion_position_encoding) != "learnable":
            raise ValueError(
                "official UMI-FT fusion contract requires learnable position encoding"
            )
        self.position_embedding = nn.Parameter(
            torch.randn(self.num_fusion_tokens, self.fusion_dim)
        )
        self.fusion = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=int(fusion_heads),
            dim_feedforward=int(fusion_feedforward_dim),
            dropout=float(fusion_dropout),
            batch_first=True,
        )
        self.fusion_projection = nn.Linear(
            self.num_fusion_tokens * self.fusion_dim,
            self.fusion_dim,
        )
        self.valve_stage_conditioner = (
            ValveStageConditioner(
                context_dim=int(valve_context_dim),
                feature_dim=self.fusion_dim,
                hidden_dim=int(valve_context_hidden_dim),
                expert_bottleneck_dim=int(valve_expert_bottleneck_dim),
                num_phase_experts=int(valve_context_num_phase_experts),
            )
            if valve_context_key is not None
            else None
        )
        # Disabled unless the real-robot evaluator explicitly requests a
        # diagnostic capture.  Keeping this state out of the checkpoint makes
        # it impossible for an eval-only switch to alter training behavior.
        self.capture_fusion_attention = False
        self.last_fusion_attention = None

        self.low_dim_output_dim = sum(
            int(attr["horizon"]) * int(torch.tensor(attr["shape"]).prod())
            for key, attr in legacy_shape_meta["obs"].items()
            if attr.get("type", "low_dim") == "low_dim"
            and not attr.get("ignore_by_policy", False)
        )
        logger.info(
            "DualFTObsEncoder: visual tokens=%d, fusion tokens=%d, "
            "ft_mode=%s, fusion_dim=%d, low_dim_output=%d, shared_ft=%s, "
            "frozen_vision=%s",
            rgb_horizon,
            self.num_fusion_tokens,
            self.ft_feature_mode,
            self.fusion_dim,
            self.low_dim_output_dim,
            self.share_ft_encoder,
            self.vision_backbone_frozen,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_backbone_frozen:
            self.vision_pose_encoder.key_model_map.eval()
        return self

    def _visual_tokens(self, obs_dict):
        tokens = []
        batch_size = next(iter(obs_dict.values())).shape[0]
        encoder = self.vision_pose_encoder
        for key in encoder.rgb_keys:
            image = obs_dict[key]
            batch, horizon = image.shape[:2]
            if batch != batch_size or tuple(image.shape[2:]) != encoder.key_shape_map[key]:
                raise ValueError(f"unexpected image tensor shape for {key}: {image.shape}")
            image = image.reshape(batch * horizon, *image.shape[2:])
            image = encoder.key_transform_map[key](image)
            raw_feature = encoder.key_model_map[key](image)
            feature = encoder.aggregate_feature(raw_feature)
            if feature.ndim != 2 or feature.shape[0] != batch * horizon:
                raise ValueError(
                    f"vision backbone must produce one token per frame, got {feature.shape}"
                )
            feature = self.visual_projection(feature)
            tokens.append(feature.reshape(batch, horizon, self.fusion_dim))
        return torch.cat(tokens, dim=1)

    def _low_dim_features(self, obs_dict):
        features = []
        batch_size = next(iter(obs_dict.values())).shape[0]
        encoder = self.vision_pose_encoder
        for key in encoder.low_dim_keys:
            data = obs_dict[key]
            if data.shape[0] != batch_size or tuple(data.shape[2:]) != encoder.key_shape_map[key]:
                raise ValueError(f"unexpected low-dimensional shape for {key}: {data.shape}")
            features.append(data.reshape(batch_size, -1))
        if not features:
            return torch.empty(
                (batch_size, 0), device=self.device, dtype=self.dtype
            )
        return torch.cat(features, dim=-1)

    def set_fusion_attention_capture(self, enabled: bool) -> None:
        """Capture per-head fusion self-attention during the next forward pass.

        The returned attention is descriptive only: it is the fusion layer's
        query-to-key weight matrix, not a causal action attribution.
        """
        self.capture_fusion_attention = bool(enabled)
        self.last_fusion_attention = None

    def fusion_token_names(self) -> list[str]:
        """Stable labels for the query/key axes of ``last_fusion_attention``."""
        names = []
        for key in self.vision_pose_encoder.rgb_keys:
            horizon = int(self.shape_meta["obs"][key]["horizon"])
            names.extend(f"{key}[t={idx}]" for idx in range(horizon))
        if self.ft_feature_mode == "raw_history":
            names.extend([self.left_ft_key, self.right_ft_key])
        elif self.ft_feature_mode == "raw_history_plus_delta_history":
            names.extend(
                [
                    self.left_ft_key,
                    self.right_ft_key,
                    f"{self.left_ft_key}_delta",
                    f"{self.right_ft_key}_delta",
                ]
            )
        else:
            names.extend(
                [
                    f"{self.left_ft_key}_latest_absolute",
                    f"{self.right_ft_key}_latest_absolute",
                    f"{self.left_ft_key}_delta",
                    f"{self.right_ft_key}_delta",
                ]
            )
        if len(names) != self.num_fusion_tokens:
            raise AssertionError(
                f"fusion token labels {len(names)} != {self.num_fusion_tokens}"
            )
        return names

    def _fuse_tokens(self, tokens):
        """Run the fusion layer, optionally retaining its exact attention map."""
        src = tokens + self.position_embedding.unsqueeze(0)
        if not self.capture_fusion_attention:
            return self.fusion(src)

        # TransformerEncoderLayer normally calls MultiheadAttention with
        # need_weights=False. Reproduce that layer's forward exactly while
        # requesting its [B, heads, query, key] weights for eval diagnostics.
        # There is no mask/cross-attention in this fixed-size token fusion.
        fusion = self.fusion
        if fusion.norm_first:
            attn_src = fusion.norm1(src)
            attn_out, attn_weights = fusion.self_attn(
                attn_src,
                attn_src,
                attn_src,
                need_weights=True,
                average_attn_weights=False,
                is_causal=False,
            )
            fused = src + fusion.dropout1(attn_out)
            fused = fused + fusion._ff_block(fusion.norm2(fused))
        else:
            attn_out, attn_weights = fusion.self_attn(
                src,
                src,
                src,
                need_weights=True,
                average_attn_weights=False,
                is_causal=False,
            )
            fused = fusion.norm1(src + fusion.dropout1(attn_out))
            fused = fusion.norm2(fused + fusion._ff_block(fused))

        self.last_fusion_attention = attn_weights.detach().to(
            device="cpu", dtype=torch.float32
        )
        return fused

    def forward(self, obs_dict):
        visual = self._visual_tokens(obs_dict)
        left_history = obs_dict[self.left_ft_key]
        right_history = obs_dict[self.right_ft_key]
        if self.ft_feature_mode == "raw_history":
            ft_tokens = [
                self.left_ft_encoder(left_history),
                self.right_ft_encoder(right_history),
            ]
        else:
            left_smoothed, left_delta = self.ft_difference(left_history)
            right_smoothed, right_delta = self.ft_difference(right_history)
            if self.ft_feature_mode == "raw_history_plus_delta_history":
                absolute_tokens = [
                    self.left_ft_encoder(left_history),
                    self.right_ft_encoder(right_history),
                ]
            else:
                absolute_tokens = [
                    self.left_ft_absolute_encoder(left_smoothed),
                    self.right_ft_absolute_encoder(right_smoothed),
                ]
            ft_tokens = absolute_tokens + [
                self.left_ft_delta_encoder(left_delta),
                self.right_ft_delta_encoder(right_delta),
            ]
        ft_tokens = [token.unsqueeze(1) for token in ft_tokens]
        batch_size = visual.shape[0]
        tokens = torch.cat([visual, *ft_tokens], dim=1)
        if tokens.shape[1] != self.num_fusion_tokens:
            raise ValueError(
                f"unexpected fusion token count {tokens.shape[1]} != "
                f"{self.num_fusion_tokens}"
            )
        fused = self._fuse_tokens(tokens)
        fused_feature = self.fusion_projection(fused.reshape(batch_size, -1))
        if self.valve_stage_conditioner is not None:
            fused_feature = self.valve_stage_conditioner(
                fused_feature, obs_dict[self.valve_context_key]
            )
        return torch.cat([fused_feature, self._low_dim_features(obs_dict)], dim=-1)

    @torch.no_grad()
    def output_shape(self):
        return torch.Size((1, self.fusion_dim + self.low_dim_output_dim))
