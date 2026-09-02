"""Bounded grasp-force feedback for policy-predicted gripper width.

The policy's 11th action channel is a prediction of the signed native-Fz
measurement seen in the demonstrations. It is a feedback reference, never a
direct RG2 force-register command. The controller compares that reference with
the current bias-corrected measurement and applies a small bounded correction
to the policy's width command.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspForceWidthFeedbackConfig:
    kp_m_per_n: float = 1.0e-4
    force_deadband_n: float = 0.5
    max_width_correction_m: float = 1.0e-3
    target_force_min_n: float = 0.0
    target_force_max_n: float = 12.0
    width_min_m: float = 0.0
    width_max_m: float = 0.1

    def __post_init__(self):
        values = np.asarray(
            [
                self.kp_m_per_n,
                self.force_deadband_n,
                self.max_width_correction_m,
                self.target_force_min_n,
                self.target_force_max_n,
                self.width_min_m,
                self.width_max_m,
            ],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("force-feedback configuration must be finite")
        if self.kp_m_per_n <= 0:
            raise ValueError("kp_m_per_n must be positive")
        if self.force_deadband_n < 0:
            raise ValueError("force_deadband_n must be non-negative")
        if self.max_width_correction_m <= 0:
            raise ValueError("max_width_correction_m must be positive")
        if self.target_force_min_n >= self.target_force_max_n:
            raise ValueError("target force range must be strictly increasing")
        if self.width_min_m >= self.width_max_m:
            raise ValueError("width range must be strictly increasing")
        if self.max_width_correction_m > (
            self.width_max_m - self.width_min_m
        ):
            raise ValueError(
                "max width correction cannot exceed the configured width range"
            )

    @classmethod
    def from_mapping(cls, mapping):
        return cls(
            kp_m_per_n=float(mapping["kp_m_per_n"]),
            force_deadband_n=float(mapping["force_deadband_n"]),
            max_width_correction_m=float(mapping["max_width_correction_m"]),
            target_force_min_n=float(mapping["target_force_min_n"]),
            target_force_max_n=float(mapping["target_force_max_n"]),
            width_min_m=float(mapping["width_min_m"]),
            width_max_m=float(mapping["width_max_m"]),
        )


def signed_grasp_force_from_native_wrenches(
    left_wrench,
    right_wrench,
    startup_bias_12d,
) -> float:
    """Return ``0.5 * (right Fz - left Fz)`` after startup bias removal."""

    left = np.asarray(left_wrench, dtype=np.float64)
    right = np.asarray(right_wrench, dtype=np.float64)
    bias = np.asarray(startup_bias_12d, dtype=np.float64)
    if left.shape != (6,) or right.shape != (6,):
        raise ValueError(
            "current native left/right wrenches must each have shape [6]"
        )
    if bias.shape != (12,):
        raise ValueError("startup F/T bias must have shape [12]")
    if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("current native wrench contains NaN or Inf")
    if np.any(~np.isfinite(bias)):
        raise ValueError("startup F/T bias contains NaN or Inf")
    left_fz = left[2] - bias[2]
    right_fz = right[2] - bias[8]
    return float(0.5 * (right_fz - left_fz))


class GraspForceWidthFeedbackController:
    """Apply a bounded proportional correction to a width trajectory.

    Positive width means opening. Therefore a measured force above the policy's
    predicted reference produces a positive correction (open slightly), while
    force below the reference produces a negative correction (close slightly).
    """

    def __init__(self, config: GraspForceWidthFeedbackConfig):
        self.config = config

    def correct(self, policy_width_m, predicted_force_n, measured_force_n):
        width = np.asarray(policy_width_m, dtype=np.float64)
        target = np.asarray(predicted_force_n, dtype=np.float64)
        measured = float(measured_force_n)
        if width.shape != target.shape:
            raise ValueError(
                "policy width and predicted force trajectories must have the "
                f"same shape, got {width.shape} and {target.shape}"
            )
        if width.size == 0:
            raise ValueError("force-feedback trajectory cannot be empty")
        if np.any(~np.isfinite(width)) or np.any(~np.isfinite(target)):
            raise ValueError("force-feedback trajectory contains NaN or Inf")
        if not np.isfinite(measured):
            raise ValueError("measured grasp force must be finite")

        cfg = self.config
        policy_width = np.clip(width, cfg.width_min_m, cfg.width_max_m)
        target_force = np.clip(
            target, cfg.target_force_min_n, cfg.target_force_max_n
        )
        force_error = measured - target_force
        active_error = np.where(
            np.abs(force_error) <= cfg.force_deadband_n,
            0.0,
            force_error,
        )
        correction = np.clip(
            cfg.kp_m_per_n * active_error,
            -cfg.max_width_correction_m,
            cfg.max_width_correction_m,
        )
        corrected_width = np.clip(
            policy_width + correction,
            cfg.width_min_m,
            cfg.width_max_m,
        )
        return {
            "policy_width_m": policy_width,
            "predicted_force_n": target_force,
            "measured_force_n": measured,
            "force_error_n": force_error,
            "width_correction_m": correction,
            "corrected_width_m": corrected_width,
        }

    def correct_from_native_wrenches(
        self,
        policy_width_m,
        predicted_force_n,
        left_wrench,
        right_wrench,
        startup_bias_12d,
    ):
        measured = signed_grasp_force_from_native_wrenches(
            left_wrench,
            right_wrench,
            startup_bias_12d,
        )
        return self.correct(policy_width_m, predicted_force_n, measured)
