"""Pretrained vision + causal native F/T CNNs for four-state context inference."""
from __future__ import annotations

import torch
from torch import nn
import timm

from diffusion_policy.model.vision.dual_ft_obs_encoder import CausalFTEncoder
from diffusion_policy.context.ft_features import causal_ft_features, ft_feature_contract


class RGBForceContextEncoder(nn.Module):
    input_keys = ('camera0_rgb', 'robot0_ft_left', 'robot0_ft_right')

    def __init__(self, model_name='vit_base_patch16_clip_224.openai',
                 hidden_dim=128, num_layers=2, num_heads=4,
                 feedforward_dim=256, dropout=0.1, rgb_horizon=2,
                 ft_horizon=32, num_classes=4, pretrained=True,
                 ft_feature_mode='raw', ft_mean_window=5, ft_delta_lag=5):
        super().__init__()
        if ft_horizon != 32 or rgb_horizon != 2 or num_classes != 4:
            raise ValueError('This context contract requires RGB=2, F/T=32, classes=4')
        self.config = dict(model_name=model_name, hidden_dim=hidden_dim,
            num_layers=num_layers, num_heads=num_heads, feedforward_dim=feedforward_dim,
            dropout=dropout, rgb_horizon=rgb_horizon, ft_horizon=ft_horizon,
            num_classes=num_classes)
        self.ft_contract = ft_feature_contract(ft_feature_mode, ft_mean_window, ft_delta_lag)
        self.ft_channels = self.ft_contract['channels_per_finger']
        self.required_ft_history = ft_horizon + self.ft_contract['prefix_samples']
        if ft_feature_mode != 'raw':
            # Preserve the exact config/state contract of old raw-only checkpoints.
            self.config.update(ft_feature_mode=ft_feature_mode,
                ft_mean_window=ft_mean_window, ft_delta_lag=ft_delta_lag)
        # num_classes=0 retains the 768-D image representation before CLIP's head.
        self.vision = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.vision.requires_grad_(False).eval()
        cfg = self.vision.pretrained_cfg
        self.register_buffer('rgb_mean', torch.tensor(cfg['mean']).reshape(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor(cfg['std']).reshape(1, 3, 1, 1))
        self.register_buffer('ft_mean', torch.zeros(2 * self.ft_channels))
        self.register_buffer('ft_std', torch.ones(2 * self.ft_channels))
        self.visual_projection = nn.Linear(self.vision.num_features, hidden_dim)
        self.left_force = CausalFTEncoder(input_dim=self.ft_channels, output_dim=hidden_dim)
        self.right_force = CausalFTEncoder(input_dim=self.ft_channels, output_dim=hidden_dim)
        self.modality = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.position = nn.Parameter(torch.randn(1, 5, hidden_dim) * 0.02)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim,
            dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        return self

    def set_force_statistics(self, mean, std):
        mean, std = torch.as_tensor(mean), torch.as_tensor(std)
        if mean.shape != self.ft_mean.shape or std.shape != self.ft_std.shape:
            raise ValueError(f'Expected {2 * self.ft_channels} F/T feature normalization channels')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid F/T normalization statistics')
        self.ft_mean.copy_(mean)
        self.ft_std.copy_(std)

    def force_features(self, history):
        """Same deterministic feature transform in training and checkpoint inference."""
        return causal_ft_features(history, mode=self.ft_contract['mode'],
            mean_window=self.ft_contract['mean_window'], delta_lag=self.ft_contract['delta_lag'])

    def forward(self, obs):
        rgb, left, right = (obs[k] for k in self.input_keys)
        batch = rgb.shape[0]
        if tuple(rgb.shape[1:]) != (2, 3, 224, 224):
            raise ValueError('RGB must be [B,2,3,224,224], in [0,1]')
        if tuple(left.shape) != (batch, self.required_ft_history, 6) or right.shape != left.shape:
            raise ValueError(f'Each native F/T history must be [B,{self.required_ft_history},6]')
        with torch.no_grad():
            images = (rgb.flatten(0, 1) - self.rgb_mean) / self.rgb_std
            visual = self.vision(images).reshape(batch, 2, -1)
        visual = self.visual_projection(visual) + self.modality[0]
        left = (self.force_features(left) - self.ft_mean[:self.ft_channels]) / self.ft_std[:self.ft_channels]
        right = (self.force_features(right) - self.ft_mean[self.ft_channels:]) / self.ft_std[self.ft_channels:]
        force_left = self.left_force(left)[:, None] + self.modality[1]
        force_right = self.right_force(right)[:, None] + self.modality[2]
        # CNNs see only past-to-anchor measurements. The final query can attend
        # to both RGB observations and both force-history summaries.
        tokens = torch.cat([visual, force_left, force_right,
                            self.query.expand(batch, -1, -1)], dim=1) + self.position
        mask = torch.ones(5, 5, dtype=torch.bool, device=tokens.device).triu(1)
        return self.classifier(self.transformer(tokens, mask=mask)[:, -1])

    @classmethod
    def from_checkpoint(cls, payload):
        # Full visual weights are saved; restoring never downloads a backbone.
        model = cls(**payload['model_config'], pretrained=False)
        model.load_state_dict(payload['state_dict'], strict=True)
        return model
