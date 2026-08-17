from __future__ import annotations

import numpy as np


def causal_ft_history(
    timestamps,
    combined_ft,
    *,
    anchor_timestamp,
    num_steps,
    stride=1,
    frequency=100.0,
):
    """Select a left/right F/T history without looking past an RGB anchor.

    History slots before the first sensor sample repeat that first sample, but
    only after a causal sample exists at the anchor. No interpolation is used.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    combined_ft = np.asarray(combined_ft)
    anchor_timestamp = float(anchor_timestamp)
    num_steps = int(num_steps)
    stride = int(stride)
    frequency = float(frequency)
    if combined_ft.ndim != 2 or combined_ft.shape != (len(timestamps), 12):
        raise ValueError(
            "combined_ft must be [N,12] ordered left[6], right[6], got "
            f"{combined_ft.shape} for {len(timestamps)} timestamps"
        )
    if num_steps <= 0 or stride <= 0 or frequency <= 0:
        raise ValueError("num_steps, stride, and frequency must be positive")
    if len(timestamps) == 0 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("F/T timestamps must be non-empty and strictly increasing")

    latest = np.searchsorted(timestamps, anchor_timestamp, side="right") - 1
    if latest < 0:
        raise RuntimeError(
            "policy anchor precedes the first causal left/right F/T sample"
        )
    target_timestamps = anchor_timestamp - (
        np.arange(num_steps - 1, -1, -1, dtype=np.float64)
        * stride
        / frequency
    )
    indices = np.searchsorted(timestamps, target_timestamps, side="right") - 1
    indices = np.maximum(indices, 0)
    selected_timestamps = timestamps[indices]
    if np.any(selected_timestamps > anchor_timestamp):
        raise AssertionError("future F/T sample selected for policy observation")
    selected = combined_ft[indices]
    return {
        "robot0_ft_left": selected[:, :6].copy(),
        "robot0_ft_right": selected[:, 6:].copy(),
        "robot0_ft_left_timestamps": selected_timestamps.copy(),
        "robot0_ft_right_timestamps": selected_timestamps.copy(),
        "robot0_ft_left_age": np.asarray(
            anchor_timestamp - selected_timestamps[-1], dtype=np.float64
        ),
        "robot0_ft_right_age": np.asarray(
            anchor_timestamp - selected_timestamps[-1], dtype=np.float64
        ),
    }


def prepare_rg2ft_policy_obs(env_obs: dict, shape_meta) -> dict:
    """Add the split RG2-FT channels requested by an RG2 policy checkpoint.

    UmiEnv intentionally keeps the collection/replay representation as one
    ``robot0_ft`` vector ordered as left-six then right-six.  RG2 policy
    checkpoints use two six-dimensional observation keys, so adapt only the
    shallow eval dictionary and leave the environment/recording API intact.
    """
    out = dict(env_obs)
    obs_meta = shape_meta["obs"]
    wants_left = "robot0_ft_left" in obs_meta
    wants_right = "robot0_ft_right" in obs_meta
    if not (wants_left or wants_right):
        return out

    # New UmiEnv versions produce histories directly from raw sensor
    # timestamps. Preserve them and only validate their contract here.
    if wants_left and "robot0_ft_left" in out:
        left = np.asarray(out["robot0_ft_left"])
        expected = int(obs_meta["robot0_ft_left"]["horizon"])
        if left.shape != (expected, 6):
            raise ValueError(
                f"robot0_ft_left must be [{expected},6], got {left.shape}"
            )
    if wants_right and "robot0_ft_right" in out:
        right = np.asarray(out["robot0_ft_right"])
        expected = int(obs_meta["robot0_ft_right"]["horizon"])
        if right.shape != (expected, 6):
            raise ValueError(
                f"robot0_ft_right must be [{expected},6], got {right.shape}"
            )
    if (not wants_left or "robot0_ft_left" in out) and (
        not wants_right or "robot0_ft_right" in out
    ):
        return out

    # Compatibility for old checkpoints/environments with a short combined
    # observation. This path only splits already selected values; it does not
    # invent a frame transform or mix left/right channels.
    if "robot0_ft" not in out:
        raise KeyError(
            "RG2 checkpoint requires robot0_ft_left/right, but UmiEnv did "
            "not provide the combined robot0_ft observation"
        )
    ft = np.asarray(out["robot0_ft"])
    if ft.ndim < 1 or ft.shape[-1] != 12:
        raise ValueError(
            "robot0_ft must have 12 channels ordered left[6], right[6]; "
            f"got shape {ft.shape}"
        )

    if wants_left and "robot0_ft_left" not in out:
        out["robot0_ft_left"] = ft[..., :6].copy()
    if wants_right and "robot0_ft_right" not in out:
        out["robot0_ft_right"] = ft[..., 6:].copy()
    return out
