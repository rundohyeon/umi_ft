"""Fail-closed motion and force guards for dual-F/T policy deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.spatial.transform as st


class PolicySafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyMotionSafetyConfig:
    max_position_delta_m: float = 0.015
    max_rotation_delta_rad: float = 0.10
    max_gripper_delta_m: float = 0.012
    min_tcp_z_m: float = -0.024
    exclusion_sphere_radius_m: float = 0.1
    exclusion_sphere_center_m: tuple = (0.0, -0.06, -0.185)

    @classmethod
    def from_mapping(cls, mapping):
        values = {key: mapping[key] for key in asdict(cls()) if key in mapping}
        if "exclusion_sphere_center_m" in values:
            values["exclusion_sphere_center_m"] = tuple(values["exclusion_sphere_center_m"])
        return cls(**values)

    def __post_init__(self):
        positive = np.asarray(
            [
                self.max_position_delta_m,
                self.max_rotation_delta_rad,
                self.max_gripper_delta_m,
                self.exclusion_sphere_radius_m,
            ],
            dtype=np.float64,
        )
        center = np.asarray(self.exclusion_sphere_center_m, dtype=np.float64)
        if np.any(~np.isfinite(positive)) or np.any(positive <= 0):
            raise ValueError("motion safety deltas/radius must be finite and positive")
        if not np.isfinite(self.min_tcp_z_m) or center.shape != (3,) or np.any(~np.isfinite(center)):
            raise ValueError("motion safety workspace geometry is invalid")


@dataclass(frozen=True)
class FTSafetyConfig:
    max_abs_force_n: float = 40.0
    max_abs_torque_nm: float = 4.0
    max_abs_grasp_force_n: float = 20.0
    max_latest_sample_age_s: float = 0.05

    @classmethod
    def from_mapping(cls, mapping):
        return cls(**{key: mapping[key] for key in asdict(cls()) if key in mapping})

    def __post_init__(self):
        values = np.asarray(
            [
                self.max_abs_force_n,
                self.max_abs_torque_nm,
                self.max_abs_grasp_force_n,
                self.max_latest_sample_age_s,
            ],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("F/T safety limits must be finite and positive")


def validate_ft_load(
    left_wrench,
    right_wrench,
    grasp_force_n,
    config,
    *,
    latest_sample_age_s=None,
):
    cfg = config if isinstance(config, FTSafetyConfig) else FTSafetyConfig.from_mapping(config)
    left = np.asarray(left_wrench, dtype=np.float64).reshape(6)
    right = np.asarray(right_wrench, dtype=np.float64).reshape(6)
    combined = np.concatenate([left, right])
    if np.any(~np.isfinite(combined)) or not np.isfinite(grasp_force_n):
        raise PolicySafetyError("F/T overload guard received NaN or Inf")
    if latest_sample_age_s is not None:
        age = float(latest_sample_age_s)
        if not np.isfinite(age) or age < 0 or age > cfg.max_latest_sample_age_s:
            raise PolicySafetyError(
                f"latest F/T sample is stale: age={age * 1000.0:.3f} ms > "
                f"{cfg.max_latest_sample_age_s * 1000.0:.3f} ms"
            )
    max_force = float(np.max(np.abs(combined[[0, 1, 2, 6, 7, 8]])))
    max_torque = float(np.max(np.abs(combined[[3, 4, 5, 9, 10, 11]])))
    if max_force > cfg.max_abs_force_n:
        raise PolicySafetyError(
            f"F/T overload: |force|={max_force:.3f} N > {cfg.max_abs_force_n:.3f} N"
        )
    if max_torque > cfg.max_abs_torque_nm:
        raise PolicySafetyError(
            f"F/T overload: |torque|={max_torque:.3f} Nm > {cfg.max_abs_torque_nm:.3f} Nm"
        )
    if abs(float(grasp_force_n)) > cfg.max_abs_grasp_force_n:
        raise PolicySafetyError(
            f"grasp overload: |force|={abs(float(grasp_force_n)):.3f} N > "
            f"{cfg.max_abs_grasp_force_n:.3f} N"
        )


def validate_policy_waypoints(targets, current_tcp6, current_width_m, config):
    """Validate every near-term waypoint against its immediate predecessor."""

    cfg = config if isinstance(config, PolicyMotionSafetyConfig) else PolicyMotionSafetyConfig.from_mapping(config)
    poses = np.asarray(targets, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) == 0:
        raise PolicySafetyError("policy waypoints must have non-empty shape [N,7]")
    if np.any(~np.isfinite(poses)):
        raise PolicySafetyError("policy waypoint contains NaN or Inf")
    previous = np.concatenate(
        [np.asarray(current_tcp6, dtype=np.float64).reshape(6), [float(current_width_m)]]
    )
    sphere_center = np.asarray(cfg.exclusion_sphere_center_m, dtype=np.float64)
    for idx, target in enumerate(poses):
        position_delta = float(np.linalg.norm(target[:3] - previous[:3]))
        rotation_delta = float(
            (st.Rotation.from_rotvec(target[3:6]) * st.Rotation.from_rotvec(previous[3:6]).inv()).magnitude()
        )
        width_delta = abs(float(target[6] - previous[6]))
        if position_delta > cfg.max_position_delta_m:
            raise PolicySafetyError(
                f"waypoint[{idx}] position delta {position_delta:.5f} m > "
                f"{cfg.max_position_delta_m:.5f} m"
            )
        if rotation_delta > cfg.max_rotation_delta_rad:
            raise PolicySafetyError(
                f"waypoint[{idx}] rotation delta {rotation_delta:.5f} rad > "
                f"{cfg.max_rotation_delta_rad:.5f} rad"
            )
        if width_delta > cfg.max_gripper_delta_m:
            raise PolicySafetyError(
                f"waypoint[{idx}] gripper delta {width_delta:.5f} m > "
                f"{cfg.max_gripper_delta_m:.5f} m"
            )
        if float(target[2]) < cfg.min_tcp_z_m:
            raise PolicySafetyError(
                f"waypoint[{idx}] TCP z {float(target[2]):.5f} m is below "
                f"{cfg.min_tcp_z_m:.5f} m"
            )
        segment = target[:3] - previous[:3]
        segment_norm_sq = float(np.dot(segment, segment))
        if segment_norm_sq > 0:
            alpha = float(
                np.clip(
                    np.dot(sphere_center - previous[:3], segment)
                    / segment_norm_sq,
                    0.0,
                    1.0,
                )
            )
            closest_to_sphere = previous[:3] + alpha * segment
        else:
            closest_to_sphere = target[:3]
        sphere_distance = float(
            np.linalg.norm(closest_to_sphere - sphere_center)
        )
        if sphere_distance < cfg.exclusion_sphere_radius_m:
            raise PolicySafetyError(
                f"segment to waypoint[{idx}] enters exclusion sphere: "
                f"distance={sphere_distance:.5f} m"
            )
        if not 0.0 <= float(target[6]) <= 0.1:
            raise PolicySafetyError(f"waypoint[{idx}] gripper width is outside [0,0.1] m")
        previous = target
    return poses
