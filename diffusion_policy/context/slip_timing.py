"""Shared observation timing for slip training and live inference."""
from __future__ import annotations

import numpy as np

RGB_FPS = 60000 / 1001


def rgb_window_spec(rgb_horizon=4, rgb_span_s=None):
    if rgb_horizon not in (4, 8):
        raise ValueError('Slip RGB horizon must be 4 or 8')
    span = float((.5 if rgb_horizon == 4 else 1.) if rgb_span_s is None else rgb_span_s)
    if not np.isfinite(span) or span <= 0:
        raise ValueError('RGB span must be positive and finite')
    offsets = np.rint(np.linspace(span * RGB_FPS, 0, rgb_horizon)).astype(np.int64)
    if np.any(np.diff(offsets) >= 0):
        raise ValueError('RGB span must allow distinct source frames')
    return span, offsets
