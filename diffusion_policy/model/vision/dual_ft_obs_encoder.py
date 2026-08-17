from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder


logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """One-dimensional convolution with left-only temporal padding."""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1):
        super().__init__()
        self.left_padding = int(kernel_size) - 1
        self.conv = nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
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


class DualFTObsEncoder(ModuleAttrMixin):
    """Existing TARGET vision/pose encoder augmented by two native F/T tokens.

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
        ft_channel_dims=(16, 32, 64, 128),
        share_ft_encoder: bool = False,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.left_ft_key = left_ft_key
        self.right_ft_key = right_ft_key
        self.fusion_dim = int(fusion_dim)

        obs_meta = shape_meta["obs"]
        for key in (left_ft_key, right_ft_key):
            if key not in obs_meta:
                raise ValueError(f"shape_meta is missing required F/T key {key!r}")
            if tuple(obs_meta[key]["shape"]) != (6,):
                raise ValueError(f"{key} must have six independent channels")

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

        if int(vision_feature_dim) == self.fusion_dim:
            self.visual_projection = nn.Identity()
        else:
            self.visual_projection = nn.Linear(
                int(vision_feature_dim), self.fusion_dim
            )

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
        self.share_ft_encoder = bool(share_ft_encoder)

        rgb_horizon = sum(
            int(legacy_shape_meta["obs"][key]["horizon"])
            for key in self.vision_pose_encoder.rgb_keys
        )
        self.num_fusion_tokens = 1 + rgb_horizon + 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.fusion_dim))
        self.token_embedding = nn.Parameter(
            torch.zeros(1, self.num_fusion_tokens, self.fusion_dim)
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.token_embedding, std=0.02)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=int(fusion_heads),
            dim_feedforward=int(fusion_feedforward_dim),
            dropout=float(fusion_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            fusion_layer,
            num_layers=int(fusion_layers),
            enable_nested_tensor=False,
        )
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        self.low_dim_output_dim = sum(
            int(attr["horizon"]) * int(torch.tensor(attr["shape"]).prod())
            for key, attr in legacy_shape_meta["obs"].items()
            if attr.get("type", "low_dim") == "low_dim"
            and not attr.get("ignore_by_policy", False)
        )
        logger.info(
            "DualFTObsEncoder: visual tokens=%d, fusion_dim=%d, "
            "low_dim_output=%d, shared_ft=%s",
            rgb_horizon,
            self.fusion_dim,
            self.low_dim_output_dim,
            self.share_ft_encoder,
        )

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

    def forward(self, obs_dict):
        visual = self._visual_tokens(obs_dict)
        left = self.left_ft_encoder(obs_dict[self.left_ft_key]).unsqueeze(1)
        right = self.right_ft_encoder(obs_dict[self.right_ft_key]).unsqueeze(1)
        batch_size = visual.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, visual, left, right], dim=1)
        if tokens.shape[1] != self.num_fusion_tokens:
            raise ValueError(
                f"unexpected fusion token count {tokens.shape[1]} != "
                f"{self.num_fusion_tokens}"
            )
        fused = self.fusion(tokens + self.token_embedding)
        fused_token = self.fusion_norm(fused[:, 0])
        return torch.cat([fused_token, self._low_dim_features(obs_dict)], dim=-1)

    @torch.no_grad()
    def output_shape(self):
        return torch.Size((1, self.fusion_dim + self.low_dim_output_dim))
