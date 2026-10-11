"""RGB/F/T pairs plus a GRU over past observation-model probabilities."""
from __future__ import annotations

import torch
from torch import nn

from models.rgb_force_context_encoder import RGBForceContextEncoder
from diffusion_policy.context.slip_timing import rgb_window_spec


SCHEMA = 'slip_rgb_ft_context_sequence_v1'
PHASE_NAMES = ('approach', 'error1', 'error2', 'turning')


class SlipTemporalContextEncoder(RGBForceContextEncoder):
    supported_rgb_horizons = (4, 8)
    input_keys = ('camera0_rgb', 'robot0_ft_left', 'robot0_ft_right',
                  'context_probabilities', 'context_valid', 'context_ages')

    def __init__(self, history_steps=20, history_period_s=.1001, history_hidden=64,
                 history_dropout=.2, history_noise=.1, pretrained=True, rgb_span_s=None, **kwargs):
        kwargs.setdefault('rgb_horizon', 4)
        kwargs.setdefault('ft_feature_mode', 'raw_mean_delta')
        super().__init__(pretrained=pretrained, **kwargs)
        self.rgb_span_s, self.rgb_offsets = rgb_window_spec(self.rgb_horizon, rgb_span_s)
        if self.required_ft_history != 41 or self.ft_channels != 18:
            raise ValueError('Slip temporal model requires 41 raw F/T per image')
        if self.ft_contract['mean_window'] != 5 or self.ft_contract['delta_lag'] != 5:
            raise ValueError('Slip data normalization requires the same 5-sample mean and lag')
        if self.vision.num_features != 768:
            raise ValueError('Slip feature-cache contract requires 768-D CLIP features')
        if history_steps < 1 or history_period_s <= 0 or history_hidden < 1:
            raise ValueError('Invalid context history settings')
        if not 0 <= history_dropout < 1 or not 0 <= history_noise < 1:
            raise ValueError('History corruption probabilities must be in [0,1)')
        self.history_steps = int(history_steps)
        self.history_period_s = float(history_period_s)
        self.history_hidden = int(history_hidden)
        self.history_dropout = float(history_dropout)
        self.history_noise = float(history_noise)
        hidden = self.config['hidden_dim']
        self.num_tokens = 3 * self.rgb_horizon + 1  # RGB/left F/T/right F/T triples + query.
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, hidden) * .02)
        self.history_gru = nn.GRUCell(5, self.history_hidden)  # Four probabilities and sample age.
        self.temporal_correction = nn.Sequential(nn.Linear(hidden + history_hidden, hidden),
                                                nn.GELU(), nn.Linear(hidden, 4))
        nn.init.zeros_(self.temporal_correction[-1].weight)
        nn.init.zeros_(self.temporal_correction[-1].bias)
        self.config.update(history_steps=history_steps, history_period_s=history_period_s,
                           history_hidden=history_hidden, history_dropout=history_dropout,
                           history_noise=history_noise, rgb_span_s=self.rgb_span_s)

    def encode_images(self, rgb):
        if tuple(rgb.shape[-3:]) != (3, 224, 224):
            raise ValueError('RGB must end in [3,224,224]')
        with torch.no_grad():
            flat = rgb.reshape(-1, 3, 224, 224)
            return self.vision((flat - self.rgb_mean) / self.rgb_std).reshape(*rgb.shape[:-3], -1)

    def observe(self, obs):
        """Observation-only predictions are the history source in training AND live use."""
        visual = obs.get('rgb_features')
        if visual is None:
            visual = self.encode_images(obs['camera0_rgb'])
        if visual.ndim != 3 or tuple(visual.shape[1:]) != (self.rgb_horizon, self.vision.num_features):
            raise ValueError(f'Expected RGB features [B,{self.rgb_horizon},768]')
        batch = len(visual)
        tokens = [self.visual_projection(visual) + self.modality[0]]
        for i, (name, encoder) in enumerate((('robot0_ft_left', self.left_force),
                                             ('robot0_ft_right', self.right_force))):
            raw = obs[name]
            if tuple(raw.shape) != (batch, self.rgb_horizon, 41, 6):
                raise ValueError(f'Each finger must have [B,{self.rgb_horizon},41,6] raw measurements aligned to RGB')
            feature = self.force_features(raw.reshape(-1, 41, 6))
            feature = ((feature - self.ft_mean[i * 18:(i + 1) * 18])
                       / self.ft_std[i * 18:(i + 1) * 18])
            tokens.append(encoder(feature).reshape(batch, self.rgb_horizon, -1) + self.modality[i + 1])
        paired = torch.stack(tokens, dim=2).reshape(batch, 3 * self.rgb_horizon, -1)
        sequence = torch.cat([paired, self.query.expand(batch, -1, -1)], 1) + self.position
        mask = torch.ones(self.num_tokens, self.num_tokens, dtype=torch.bool, device=sequence.device).triu(1)
        feature = self.transformer(sequence, mask=mask)[:, -1]
        return feature, self.classifier(feature)

    def fuse(self, feature, base_logits, probabilities, valid, ages):
        batch = len(feature)
        if probabilities.shape != (batch, self.history_steps, 4) or valid.shape != probabilities.shape[:2] or ages.shape != valid.shape:
            raise ValueError('History must contain past [B,N,4] probabilities, masks and ages')
        valid = valid.bool()
        if (not torch.isfinite(probabilities).all() or not torch.isfinite(ages).all()
                or ((ages <= 0) & valid).any()
                or ((ages > self.history_steps * self.history_period_s + .025) & valid).any()):
            raise ValueError('Context history must be finite, strictly past, and within the horizon')
        if ((probabilities < 0).any() or
                (valid & ((probabilities.sum(-1) - 1).abs() > 1e-4)).any()):
            raise ValueError('Context history must contain class probabilities')
        # No gradient through past probabilities: they remain interpretable class estimates.
        p = probabilities.detach().float()
        if self.training:
            keep_sequence = torch.rand(batch, 1, device=p.device) >= self.history_dropout / 2
            valid = valid & keep_sequence & (torch.rand(valid.shape, device=p.device) >= self.history_dropout)
            replace = torch.rand(valid.shape, device=p.device) < self.history_noise
            noise = torch.rand_like(p)
            noise = noise / noise.sum(-1, keepdim=True)
            p = torch.where(replace[..., None], noise, p)
        x = torch.cat([p, (ages / (self.history_steps * self.history_period_s))[..., None]], -1)
        # PyTorch 2.1's fused CUDA GRUCell does not implement BF16. This small
        # probability-history branch stays FP32 even when the observation model uses AMP.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            hidden = torch.zeros(batch, self.history_hidden, device=feature.device, dtype=torch.float32)
            for step in range(self.history_steps):
                proposed = self.history_gru(x[:, step].float(), hidden)
                hidden = torch.where(valid[:, step, None], proposed, hidden)
        correction = self.temporal_correction(torch.cat([feature, hidden], -1))
        return base_logits + correction * valid.any(-1, keepdim=True)

    def forward(self, obs):
        if 'sequence_valid' not in obs:
            feature, base = self.observe(obs)
            return self.fuse(feature, base, obs['context_probabilities'],
                             obs['context_valid'], obs['context_ages'])
        valid, ages = obs['sequence_valid'], obs['sequence_ages']
        batch, steps = valid.shape
        if steps != self.history_steps + 1 or not valid[:, -1].all() or not (ages[:, -1] == 0).all():
            raise ValueError('Sequence must end at a valid current observation')
        flat = {k: v.flatten(0, 1) for k, v in obs.items()
                if k in ('rgb_features', 'camera0_rgb', 'robot0_ft_left', 'robot0_ft_right')}
        feature, logits = self.observe(flat)
        feature, logits = feature.reshape(batch, steps, -1), logits.reshape(batch, steps, 4)
        final = self.fuse(feature[:, -1], logits[:, -1], logits[:, :-1].float().softmax(-1),
                          valid[:, :-1], ages[:, :-1])
        return dict(logits=final, observation_logits=logits)

    @classmethod
    def from_checkpoint(cls, payload):
        if payload.get('schema') != SCHEMA or tuple(payload.get('phase_names', ())) != PHASE_NAMES:
            raise ValueError('This is not a slip temporal checkpoint with the required class order')
        model = cls(**payload['model_config'], pretrained=False)
        model.load_state_dict(payload['state_dict'], strict=True)
        return model
