from __future__ import annotations

import numpy as np


def _validate_ft_stream(name, timestamps, values):
    """Return a validated timestamped six-axis F/T stream.

    The live RG2-FT device returns the two finger wrenches in one atomic
    Modbus reply, so their timestamps are currently identical.  Keeping this
    helper side-specific is intentional: the training dataset has independent
    left/right arrays and a future device adapter must not silently reuse one
    finger's samples for the other.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(timestamps), 6):
        raise ValueError(
            f"{name} F/T must be [N,6], got {values.shape} for "
            f"{len(timestamps)} timestamps"
        )
    if len(timestamps) == 0 or np.any(~np.isfinite(timestamps)):
        raise ValueError(f"{name} F/T timestamps must be non-empty and finite")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{name} F/T timestamps must be strictly increasing")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} F/T contains NaN or Inf")
    return timestamps, values


def _causal_stream_history(
    name,
    timestamps,
    values,
    *,
    anchor_timestamp,
    num_steps,
    stride,
    frequency,
):
    timestamps, values = _validate_ft_stream(name, timestamps, values)
    latest = np.searchsorted(timestamps, anchor_timestamp, side="right") - 1
    if latest < 0:
        raise RuntimeError(
            f"policy anchor precedes the first causal {name} F/T sample"
        )
    target_timestamps = anchor_timestamp - (
        np.arange(num_steps - 1, -1, -1, dtype=np.float64)
        * stride
        / frequency
    )
    indices = np.searchsorted(timestamps, target_timestamps, side="right") - 1
    # This exactly matches UmiDualFTDataset._ft_history: after at least one
    # causal sample exists, history before stream start is repeat-first.
    indices = np.maximum(indices, 0)
    selected_timestamps = timestamps[indices]
    if np.any(selected_timestamps > anchor_timestamp):
        raise AssertionError(f"future {name} F/T sample selected")
    return values[indices].copy(), selected_timestamps.copy()


def causal_ft_history_from_streams(
    left_timestamps,
    left_ft,
    right_timestamps,
    right_ft,
    *,
    anchor_timestamp,
    num_steps,
    stride=1,
    frequency=100.0,
    max_age=None,
):
    """Build independent causal histories for the two RG2-FT fingers.

    Every selected timestamp is at or before ``anchor_timestamp``. ``max_age``
    is an optional fail-closed freshness limit in seconds; it is evaluated on
    each side's newest causal sample.
    """
    anchor_timestamp = float(anchor_timestamp)
    num_steps = int(num_steps)
    stride = int(stride)
    frequency = float(frequency)
    if not np.isfinite(anchor_timestamp):
        raise ValueError("policy anchor timestamp must be finite")
    if num_steps <= 0 or stride <= 0 or frequency <= 0:
        raise ValueError("num_steps, stride, and frequency must be positive")

    left, left_selected_timestamps = _causal_stream_history(
        "left", left_timestamps, left_ft,
        anchor_timestamp=anchor_timestamp,
        num_steps=num_steps,
        stride=stride,
        frequency=frequency,
    )
    right, right_selected_timestamps = _causal_stream_history(
        "right", right_timestamps, right_ft,
        anchor_timestamp=anchor_timestamp,
        num_steps=num_steps,
        stride=stride,
        frequency=frequency,
    )
    left_age = anchor_timestamp - left_selected_timestamps[-1]
    right_age = anchor_timestamp - right_selected_timestamps[-1]
    if max_age is not None:
        max_age = float(max_age)
        if not np.isfinite(max_age) or max_age < 0:
            raise ValueError("max_age must be a finite non-negative number")
        stale = []
        if left_age > max_age:
            stale.append(f"left age={left_age:.6f}s")
        if right_age > max_age:
            stale.append(f"right age={right_age:.6f}s")
        if stale:
            raise RuntimeError(
                "dual-F/T observation is stale (" + ", ".join(stale)
                + f"; limit={max_age:.6f}s)"
            )
    return {
        "robot0_ft_left": left,
        "robot0_ft_right": right,
        "robot0_ft_left_timestamps": left_selected_timestamps,
        "robot0_ft_right_timestamps": right_selected_timestamps,
        "robot0_ft_left_age": np.asarray(left_age, dtype=np.float64),
        "robot0_ft_right_age": np.asarray(right_age, dtype=np.float64),
    }


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
    combined_ft = np.asarray(combined_ft, dtype=np.float32)
    if combined_ft.ndim != 2 or combined_ft.shape != (len(timestamps), 12):
        raise ValueError(
            "combined_ft must be [N,12] ordered left[6], right[6], got "
            f"{combined_ft.shape} for {len(timestamps)} timestamps"
        )
    return causal_ft_history_from_streams(
        timestamps,
        combined_ft[:, :6],
        timestamps,
        combined_ft[:, 6:],
        anchor_timestamp=anchor_timestamp,
        num_steps=num_steps,
        stride=stride,
        frequency=frequency,
    )


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
