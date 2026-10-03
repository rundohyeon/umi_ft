"""Residual context conditioning of the existing UMI diffusion policy."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from models.context_encoder import ContextEncoder, ContextEmbedding, ContextFusion, context_loss


class ContextAwarePolicy(DiffusionUnetTimmPolicy):
    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self.context_config = dict(context)
        self.context_mode = str(context.get('mode', 'soft'))
        if self.context_mode not in ('none','oracle','hard','soft'):
            raise ValueError('Context mode must be none/oracle/hard/soft')
        encoder_config = dict(context.get('encoder', {}))
        self.num_context_classes = context.get('num_classes', encoder_config.get('num_classes', 5))
        if encoder_config.get('num_classes', self.num_context_classes) != self.num_context_classes:
            raise ValueError('Context encoder and embedding class counts differ')
        encoder_config['num_classes'] = self.num_context_classes
        self.context_encoder = ContextEncoder(shape_meta=kwargs['shape_meta'], **encoder_config)
        self.context_embedding = ContextEmbedding(int(context.get('embedding_dim',32)), self.num_context_classes)
        self.context_fusion = ContextFusion(int(self.obs_feature_dim), int(context.get('embedding_dim',32)),
            int(context.get('fusion_hidden_dim',128)), float(context.get('gate_init',-4)))
        self.lambda_context = float(context.get('lambda_context',.1))
        self.smoothing_alpha = context.get('smoothing_alpha')
        if self.smoothing_alpha is not None and not 0 < self.smoothing_alpha <= 1:
            raise ValueError('smoothing_alpha must be null or in (0,1]')
        self.curriculum = context.get('curriculum', 'adapter')
        if self.curriculum not in ('adapter','full'):
            raise ValueError('curriculum must be adapter or full')
        self.freeze_context = bool(context.get('freeze_encoder',True))
        self.register_buffer('context_definition_digest', torch.zeros(32, dtype=torch.uint8))
        self.last_context = None
        self.reset()
        self.apply_curriculum()

    def apply_curriculum(self):
        train_base = self.curriculum == 'full' or self.context_mode == 'none'
        self.model.requires_grad_(train_base)
        self.obs_encoder.requires_grad_(train_base)
        if getattr(self.obs_encoder, 'vision_backbone_frozen', False):
            self.obs_encoder.vision_pose_encoder.key_model_map.requires_grad_(False)
        enabled = self.context_mode != 'none'
        self.context_encoder.requires_grad_(enabled and not self.freeze_context)
        self.context_embedding.requires_grad_(enabled)
        self.context_fusion.requires_grad_(enabled)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self,'context_encoder'):
            if self.freeze_context: self.context_encoder.eval()
            if self.curriculum == 'adapter' and self.context_mode != 'none':
                self.model.eval()
                self.obs_encoder.eval()
        return self

    def reset(self):
        self._smoothed = None
        self.last_context = None

    def load_context_checkpoint(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        saved_config = dict(payload['encoder_config'])
        # Older five-class checkpoints omitted this field. Never resize their head silently.
        saved_config.setdefault('num_classes', 5)
        if saved_config['num_classes'] != self.num_context_classes:
            raise ValueError(f"Stage A checkpoint has {saved_config['num_classes']} classes; policy expects {self.num_context_classes}. Train a new context encoder.")
        if saved_config != self.context_encoder.config:
            raise ValueError('Stage A context architecture/observation contract differs')
        self.context_encoder.load_state_dict(payload['state_dict'], strict=True)
        self.context_definition_hash = payload.get('context_definition_hash')
        if self.context_definition_hash:
            self.context_definition_digest.copy_(torch.tensor(list(bytes.fromhex(self.context_definition_hash)), dtype=torch.uint8))
        return {key: payload[key] for key in ('context_definition_hash', 'definitions', 'episode_splits', 'source_fingerprint')}

    def encode_condition(self, nobs, raw_obs, batch=None, inference=False):
        robot = self.obs_encoder(nobs)
        if self.context_mode == 'none':
            return robot, None
        logits = self.context_encoder(raw_obs)
        raw = logits.softmax(-1)
        used = raw
        if self.context_mode == 'oracle':
            labels = None if batch is None else batch.get('context_label')
            if labels is None or torch.any((labels < 0) | (labels >= self.num_context_classes)):
                raise ValueError('Oracle ablation requires known labels for every sample; unavailable on a real robot')
            used = F.one_hot(labels,self.num_context_classes).to(raw)
        elif self.context_mode == 'hard':
            used = F.one_hot(raw.argmax(-1),self.num_context_classes).to(raw)
        smoothed = raw
        if inference and self.smoothing_alpha is not None:
            if raw.shape[0] != 1:
                raise ValueError('Stateful smoothing requires one streaming episode; reset at boundaries')
            if self._smoothed is not None:
                smoothed = self.smoothing_alpha * raw + (1-self.smoothing_alpha) * self._smoothed
            self._smoothed = smoothed.detach()
            if self.context_mode == 'soft': used = smoothed
        self.last_context = {'logits':logits.detach(), 'raw_probabilities':raw.detach(),
                             'smoothed_probabilities':smoothed.detach(), 'used_probabilities':used.detach()}
        return self.context_fusion(robot, self.context_embedding(used)), logits

    def auxiliary_loss(self, logits, batch):
        if logits is None:
            return 0.
        if 'sample_info' in batch:
            info = batch['sample_info']
            for key in ('rgb_timestamps','pose_timestamps','left_ft_timestamps','right_ft_timestamps'):
                if torch.any(info[key] > info['anchor_timestamp'].unsqueeze(-1)):
                    raise ValueError('Future observation in context training batch')
        return self.lambda_context * context_loss(logits, batch.get('context_label'), batch.get('context_weight'))

    def optimizer_groups(self, base_lr):
        rates = self.context_config.get('learning_rates', {})
        groups = []
        for name, module, default in [('action',self.model,base_lr), ('observations',self.obs_encoder,base_lr*.1),
                ('context_encoder',self.context_encoder,base_lr*.1), ('context_embedding',self.context_embedding,base_lr),
                ('context_fusion',self.context_fusion,base_lr)]:
            params = [p for p in module.parameters() if p.requires_grad]
            if params: groups.append(dict(name=name, params=params, lr=float(rates.get(name,default))))
        return groups
