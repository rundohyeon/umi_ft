"""Small causal multirate Transformer over the policy's existing history buffers."""
from __future__ import annotations
import math
import time
import torch
from torch import nn
from torch.nn import functional as F


class ContextEncoder(nn.Module):
    def __init__(self, shape_meta, hidden_dim=128, num_layers=2, num_heads=4,
                 dropout=.1, history_length=32, rgb_pool_size=4, feedforward_dim=256, num_classes=5,
                 input_keys=None):
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError('num_classes must be an integer >= 2')
        self.num_classes = num_classes
        if hidden_dim % num_heads or history_length < 1:
            raise ValueError('Invalid context dimensions/history')
        self.config = dict(shape_meta=shape_meta, hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, history_length=history_length,
            rgb_pool_size=rgb_pool_size, feedforward_dim=feedforward_dim, num_classes=num_classes)
        available_keys = {k for k,v in shape_meta['obs'].items() if not v.get('ignore_by_policy', False)}
        if input_keys is None:
            # Preserve the architecture/config contract of existing checkpoints.
            self.keys = sorted(available_keys)
        else:
            if isinstance(input_keys, str):
                raise ValueError('input_keys must be a nonempty list of observation names')
            selected = list(input_keys)
            if not selected or any(not isinstance(k, str) for k in selected):
                raise ValueError('input_keys must be a nonempty list of observation names')
            if len(set(selected)) != len(selected):
                raise ValueError('input_keys must not contain duplicates')
            unavailable = set(selected) - available_keys
            if unavailable:
                raise ValueError(f'Context inputs are missing or ignored by shape_meta: {sorted(unavailable)}')
            self.keys = sorted(selected)
            self.config['input_keys'] = list(self.keys)
        if not self.keys or any('action' in k or 'context' in k for k in self.keys):
            raise ValueError('Context inputs must be observations, never targets or labels')
        if history_length > max(int(shape_meta['obs'][k]['horizon']) for k in self.keys):
            raise ValueError('Context history cannot exceed the supplied observation buffers')
        self.history_length, self.rgb_pool_size = history_length, rgb_pool_size
        self.meta = {k: shape_meta['obs'][k] for k in self.keys}
        self.projections = nn.ModuleDict()
        for k, attr in self.meta.items():
            dim = attr['shape'][0] * rgb_pool_size**2 if attr.get('type') == 'rgb' else math.prod(attr['shape'])
            self.projections[k] = nn.Linear(dim, hidden_dim)
            # Frozen per-channel training-set statistics, serialized in every checkpoint.
            self.register_buffer('mean_' + k, torch.zeros(dim))
            self.register_buffer('std_' + k, torch.ones(dim))
        self.modality = nn.Parameter(torch.randn(len(self.keys), hidden_dim) * .02)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position = nn.Parameter(torch.randn(history_length, hidden_dim) * .02)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim,
            dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def features(self, key, x):
        expected = tuple(self.meta[key]['shape'])
        if x.ndim != len(expected)+2 or tuple(x.shape[2:]) != expected:
            raise ValueError(f'{key}: expected [B,T,{expected}], got {tuple(x.shape)}')
        if x.shape[1] != int(self.meta[key]['horizon']):
            raise ValueError(f'{key}: history differs from checkpoint contract')
        x = x[:, -self.history_length:]
        if self.meta[key].get('type') == 'rgb':
            b,t = x.shape[:2]
            x = F.adaptive_avg_pool2d(x.flatten(0,1), self.rgb_pool_size).reshape(b,t,-1)
        return x.flatten(2)

    def forward(self, observations):
        tokens = []
        for i, key in enumerate(self.keys):
            x = self.features(key, observations[key])
            x = (x-getattr(self, 'mean_'+key)) / getattr(self, 'std_'+key).clamp_min(1e-6)
            tokens.append(self.projections[key](x) + self.modality[i] + self.position[-x.shape[1]:])
        x = torch.cat(tokens + [self.query.expand(tokens[0].shape[0], -1, -1)], dim=1)
        # Streams retain native rates; token order is modality then past-to-present.
        # Every token is <= the anchor. The terminal query can see all causal histories.
        mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
        return self.classifier(self.transformer(x, mask=mask)[:, -1])

    @torch.no_grad()
    def fit_statistics(self, loader, max_batches=None):
        totals, squares, counts = {}, {}, {}
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            for key in self.keys:
                x = self.features(key, batch['obs'][key]).flatten(0,1).double()
                totals[key] = totals.get(key, 0) + x.sum(0)
                squares[key] = squares.get(key, 0) + x.square().sum(0)
                counts[key] = counts.get(key, 0) + len(x)
        for key in self.keys:
            if not counts.get(key):
                raise ValueError('No training observations for normalization')
            mean = totals[key] / counts[key]
            std = (squares[key] / counts[key] - mean.square()).clamp_min(1e-8).sqrt()
            getattr(self, 'mean_'+key).copy_(mean.float())
            getattr(self, 'std_'+key).copy_(std.float())


class ContextEmbedding(nn.Module):
    def __init__(self, context_dim=32, num_classes=5):
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError('num_classes must be an integer >= 2')
        self.num_classes = num_classes
        self.context_embedding_table = nn.Parameter(torch.randn(num_classes, context_dim) * .02)

    def forward(self, probabilities):
        if probabilities.ndim != 2 or probabilities.shape[-1] != self.num_classes:
            raise ValueError(f'Context probabilities must have shape [B,{self.num_classes}]')
        return probabilities @ self.context_embedding_table


class ContextFusion(nn.Module):
    def __init__(self, robot_dim, context_dim=32, hidden_dim=128, gate_init=-4.):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(robot_dim+context_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, robot_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.gate_parameter = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, robot_feature, context_feature):
        delta = self.mlp(torch.cat([robot_feature, context_feature], dim=-1))
        if delta.shape != robot_feature.shape:
            raise ValueError('Context adapter changed the action-model conditioning dimension')
        return robot_feature + self.gate_parameter.sigmoid() * delta


def context_loss(logits, labels, weights=None, class_weights=None):
    if labels is None:
        return logits.sum() * 0
    valid = labels >= 0
    if not valid.any():
        return logits.sum() * 0
    loss = F.cross_entropy(logits[valid], labels[valid], weight=class_weights, reduction='none')
    weights = torch.ones_like(loss) if weights is None else weights[valid]
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)


@torch.no_grad()
def benchmark_encoder(model, sample, iterations=50, warmup=5):
    result = {'parameter_count': sum(p.numel() for p in model.parameters()), 'cpu_inference_ms': None,
              'gpu_inference_ms': None, 'batch_size': next(iter(sample.values())).shape[0],
              'iterations': iterations, 'cpu_threads': torch.get_num_threads()}
    original_device = next(model.parameters()).device
    was_training = model.training
    for device, name in [('cpu','cpu_inference_ms'), ('cuda','gpu_inference_ms')]:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        model.to(device).eval()
        obs = {k:v.to(device) for k,v in sample.items()}
        for _ in range(warmup): model(obs)
        if device == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations): model(obs)
        if device == 'cuda': torch.cuda.synchronize()
        result[name] = (time.perf_counter()-start)*1000/iterations
    model.to(original_device).train(was_training)
    return result
