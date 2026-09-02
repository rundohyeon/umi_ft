"""Fail-closed startup bias calibration for native dual RG2-FT wrenches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np


@dataclass(frozen=True)
class FTStartupBiasConfig:
    sample_count: int = 200
    timeout_s: float = 5.0
    max_force_std_n: float = 0.35
    max_torque_std_nm: float = 0.03
    max_force_peak_to_peak_n: float = 1.5
    max_torque_peak_to_peak_nm: float = 0.15

    def __post_init__(self):
        if int(self.sample_count) < 2:
            raise ValueError("startup bias sample_count must be at least 2")
        values = np.asarray(
            [
                self.timeout_s,
                self.max_force_std_n,
                self.max_torque_std_nm,
                self.max_force_peak_to_peak_n,
                self.max_torque_peak_to_peak_nm,
            ],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("startup bias thresholds must be finite and positive")

    @classmethod
    def from_mapping(cls, mapping):
        return cls(**{key: mapping[key] for key in asdict(cls()) if key in mapping})


def estimate_startup_bias(timestamps, left_wrenches, right_wrenches, config):
    """Validate unloaded samples and return their 12-D native-frame mean."""

    cfg = config if isinstance(config, FTStartupBiasConfig) else FTStartupBiasConfig.from_mapping(config)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    left = np.asarray(left_wrenches, dtype=np.float64)
    right = np.asarray(right_wrenches, dtype=np.float64)
    if left.ndim != 2 or left.shape[1:] != (6,) or right.shape != left.shape:
        raise ValueError("startup left/right wrenches must have matching shape [N,6]")
    if len(ts) != len(left):
        raise ValueError("startup F/T timestamps and samples have different lengths")
    if len(ts) < cfg.sample_count:
        raise ValueError(
            f"startup bias needs {cfg.sample_count} unique samples; got {len(ts)}"
        )
    if np.any(~np.isfinite(ts)) or np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("startup F/T samples contain NaN or Inf")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("startup F/T timestamps must be unique and strictly increasing")

    samples = np.concatenate([left[-cfg.sample_count :], right[-cfg.sample_count :]], axis=1)
    std = samples.std(axis=0)
    peak_to_peak = np.ptp(samples, axis=0)
    force_idx = np.asarray([0, 1, 2, 6, 7, 8])
    torque_idx = np.asarray([3, 4, 5, 9, 10, 11])
    failures = []
    if float(std[force_idx].max()) > cfg.max_force_std_n:
        failures.append(
            f"force std {float(std[force_idx].max()):.4g} N > {cfg.max_force_std_n:.4g} N"
        )
    if float(std[torque_idx].max()) > cfg.max_torque_std_nm:
        failures.append(
            "torque std "
            f"{float(std[torque_idx].max()):.4g} Nm > {cfg.max_torque_std_nm:.4g} Nm"
        )
    if float(peak_to_peak[force_idx].max()) > cfg.max_force_peak_to_peak_n:
        failures.append(
            "force peak-to-peak "
            f"{float(peak_to_peak[force_idx].max()):.4g} N > "
            f"{cfg.max_force_peak_to_peak_n:.4g} N"
        )
    if float(peak_to_peak[torque_idx].max()) > cfg.max_torque_peak_to_peak_nm:
        failures.append(
            "torque peak-to-peak "
            f"{float(peak_to_peak[torque_idx].max()):.4g} Nm > "
            f"{cfg.max_torque_peak_to_peak_nm:.4g} Nm"
        )
    if failures:
        raise ValueError(
            "startup bias rejected: sensor/gripper was not sufficiently static and unloaded ("
            + "; ".join(failures)
            + ")"
        )
    return {
        "bias_12d": samples.mean(axis=0),
        "std_12d": std,
        "peak_to_peak_12d": peak_to_peak,
        "sample_count": int(cfg.sample_count),
        "first_timestamp": float(ts[-cfg.sample_count]),
        "last_timestamp": float(ts[-1]),
    }


def acquire_startup_bias(gripper, config):
    """Collect unique RG2-FT samples from the live controller until calibrated."""

    cfg = config if isinstance(config, FTStartupBiasConfig) else FTStartupBiasConfig.from_mapping(config)
    deadline = time.monotonic() + cfg.timeout_s
    collection_start_wall = time.time()
    collected = {}
    while time.monotonic() < deadline:
        state = gripper.get_all_state()
        ts = np.asarray(state["gripper_timestamp"], dtype=np.float64).reshape(-1)
        left = np.asarray(state["gripper_ft_left"], dtype=np.float64)
        right = np.asarray(state["gripper_ft_right"], dtype=np.float64)
        for idx, timestamp in enumerate(ts):
            if np.isfinite(timestamp) and float(timestamp) >= collection_start_wall:
                collected[float(timestamp)] = (left[idx].copy(), right[idx].copy())
        if len(collected) >= cfg.sample_count:
            ordered_ts = np.asarray(sorted(collected), dtype=np.float64)
            ordered_left = np.stack([collected[t][0] for t in ordered_ts])
            ordered_right = np.stack([collected[t][1] for t in ordered_ts])
            return estimate_startup_bias(ordered_ts, ordered_left, ordered_right, cfg)
        time.sleep(0.01)
    raise TimeoutError(
        f"startup F/T bias timed out after {cfg.timeout_s:.2f}s: "
        f"received {len(collected)}/{cfg.sample_count} unique samples"
    )


def subtract_startup_bias(left_wrenches, right_wrenches, bias_12d):
    left = np.asarray(left_wrenches)
    right = np.asarray(right_wrenches)
    bias = np.asarray(bias_12d, dtype=np.float64)
    if left.shape[-1:] != (6,) or right.shape[-1:] != (6,) or left.shape != right.shape:
        raise ValueError("native left/right wrenches must have matching trailing shape [6]")
    if bias.shape != (12,) or np.any(~np.isfinite(bias)):
        raise ValueError("startup bias must be a finite [12] vector")
    return left - bias[:6], right - bias[6:]
