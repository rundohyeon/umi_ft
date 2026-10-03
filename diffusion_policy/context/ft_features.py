"""Shared causal F/T feature construction for training statistics and inference."""
from __future__ import annotations

import torch


def ft_feature_contract(mode='raw', mean_window=5, delta_lag=5):
    if mode not in ('raw', 'raw_mean_delta'):
        raise ValueError('F/T feature mode must be raw or raw_mean_delta')
    if type(mean_window) is not int or mean_window < 1:
        raise ValueError('F/T mean window must be a positive integer')
    if type(delta_lag) is not int or delta_lag < 1:
        raise ValueError('F/T delta lag must be a positive integer')
    return dict(mode=mode, mean_window=mean_window, delta_lag=delta_lag,
                prefix_samples=0 if mode == 'raw' else mean_window - 1 + delta_lag,
                channels_per_finger=6 if mode == 'raw' else 18)


def causal_ft_features(history, mode='raw', mean_window=5, delta_lag=5):
    """Return raw / trailing mean / lagged-mean difference in that channel order.

    ``history`` contains six physical channels [Fx,Fy,Fz,Tx,Ty,Tz]. For the
    default enhanced mode, [B,41,6] becomes [B,32,18]. The first nine samples
    are real preceding history, or episode-start repetitions supplied by the
    caller. This function never pads at an arbitrary window boundary. Differences
    retain N / Nm units; they are not derivatives divided by time.
    """
    contract = ft_feature_contract(mode, mean_window, delta_lag)
    if history.ndim != 3 or history.shape[-1] != 6:
        raise ValueError('Physical F/T history must have shape [B,T,6]')
    if history.shape[1] <= contract['prefix_samples']:
        raise ValueError('Insufficient past samples for causal F/T features')
    # Keep averaging/subtraction in FP32, even inside the model's AMP context.
    history = history.float()
    if mode == 'raw':
        return history
    means = history.unfold(1, mean_window, 1).mean(dim=-1)
    current_mean, previous_mean = means[:, delta_lag:], means[:, :-delta_lag]
    raw = history[:, contract['prefix_samples']:]
    return torch.cat([raw, current_mean, current_mean - previous_mean], dim=-1)
