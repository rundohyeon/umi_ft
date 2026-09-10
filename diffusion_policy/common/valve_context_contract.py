"""Versioned strict contracts for frozen valve-state context observers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VALVE_CONTEXT_KEY = "valve_context"
VALVE_CONTEXT_V1_SCHEMA = "umi_valve_context_sidecar_v1"
VALVE_CONTEXT_V2_SCHEMA = "umi_valve_context_sidecar_v2_4state"
VALVE_CONTEXT_V1_DIM = 10
VALVE_CONTEXT_V2_DIM = 5
VALVE_CONTEXT_V1_PHASE_NAMES = (
    "approach",
    "turning",
    "endpoint_reached",
    "task_complete",
    "error",
)
VALVE_CONTEXT_V2_PHASE_NAMES = ("approach", "turning", "recovery", "error")
VALVE_CONTEXT_V1_ERROR_REASON_NAMES = (
    "none",
    "turn_no_contact",
    "post_contact_drop",
    "other",
)

# Backward-compatible aliases used by the existing v4 classifier integration.
VALVE_CONTEXT_DIM = VALVE_CONTEXT_V1_DIM
VALVE_PHASE_NAMES = VALVE_CONTEXT_V1_PHASE_NAMES
VALVE_ERROR_REASON_NAMES = VALVE_CONTEXT_V1_ERROR_REASON_NAMES


@dataclass(frozen=True)
class ValveContextRecord:
    """Validated context plus fields useful for deployment diagnostics."""

    timestamp_s: float
    values: np.ndarray
    warmed_up: bool
    phase_name: str
    error_reason_name: str
    schema: str = VALVE_CONTEXT_V1_SCHEMA


def validate_valve_context(values, *, name: str = VALVE_CONTEXT_KEY) -> np.ndarray:
    """Validate classifier context without renormalizing it.

    The policy checkpoint's normalizer intentionally stores this field as an
    identity mapping.  Re-scaling it here would violate the trained contract.
    """

    array = np.asarray(values, dtype=np.float32)
    if array.shape not in ((VALVE_CONTEXT_DIM,), (1, VALVE_CONTEXT_DIM)):
        raise ValueError(
            f"{name} must have shape [10] or [1,10], got {array.shape}"
        )
    array = array.reshape(VALVE_CONTEXT_DIM)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(array < -1e-6) or np.any(array > 1.0 + 1e-6):
        raise ValueError(f"{name} must be probabilities/flag in [0,1]")
    phase_sum = float(array[:5].sum())
    reason_sum = float(array[5:9].sum())
    if not np.isclose(phase_sum, 1.0, atol=1e-4):
        raise ValueError(f"{name} phase probabilities must sum to 1, got {phase_sum}")
    if not np.isclose(reason_sum, 1.0, atol=1e-4):
        raise ValueError(f"{name} reason probabilities must sum to 1, got {reason_sum}")
    if not np.isclose(float(array[9]), 0.0, atol=1e-6) and not np.isclose(
        float(array[9]), 1.0, atol=1e-6
    ):
        raise ValueError(f"{name}[9] warmed_up must be exactly 0 or 1")
    return array


def validate_valve_context_v2(
    values, *, name: str = VALVE_CONTEXT_KEY
) -> np.ndarray:
    """Validate four phase probabilities plus one binary validity flag."""

    array = np.asarray(values, dtype=np.float32)
    if array.shape not in ((VALVE_CONTEXT_V2_DIM,), (1, VALVE_CONTEXT_V2_DIM)):
        raise ValueError(
            f"{name} v2 must have shape [5] or [1,5], got {array.shape}"
        )
    array = array.reshape(VALVE_CONTEXT_V2_DIM)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} v2 contains NaN or Inf")
    if np.any(array < -1e-6) or np.any(array > 1.0 + 1e-6):
        raise ValueError(f"{name} v2 probabilities/valid flag must be in [0,1]")
    phase_sum = float(array[:4].sum())
    if not np.isclose(phase_sum, 1.0, atol=1e-4):
        raise ValueError(
            f"{name} v2 phase probabilities must sum to 1, got {phase_sum}"
        )
    if not np.isclose(float(array[4]), 0.0, atol=1e-6) and not np.isclose(
        float(array[4]), 1.0, atol=1e-6
    ):
        raise ValueError(f"{name} v2 context_valid must be exactly 0 or 1")
    return array


def context_from_classifier_prediction(prediction) -> ValveContextRecord:
    """Convert the frozen v4 runtime result using its documented ordering."""

    phase_probabilities = getattr(prediction, "phase_probabilities", None)
    reason_probabilities = getattr(prediction, "error_reason_probabilities", None)
    if not isinstance(phase_probabilities, dict) or not isinstance(
        reason_probabilities, dict
    ):
        raise ValueError("classifier result is missing phase/reason probabilities")
    missing_phase = [name for name in VALVE_PHASE_NAMES if name not in phase_probabilities]
    missing_reason = [
        name for name in VALVE_ERROR_REASON_NAMES if name not in reason_probabilities
    ]
    if missing_phase or missing_reason:
        raise ValueError(
            "classifier output names do not match v4 contract: "
            f"missing phase={missing_phase}, missing reason={missing_reason}"
        )
    values = validate_valve_context(
        [
            *(float(phase_probabilities[name]) for name in VALVE_PHASE_NAMES),
            *(float(reason_probabilities[name]) for name in VALVE_ERROR_REASON_NAMES),
            float(bool(getattr(prediction, "warmed_up", False))),
        ]
    )
    timestamp_s = float(getattr(prediction, "timestamp_s"))
    if not np.isfinite(timestamp_s):
        raise ValueError("classifier result timestamp must be finite")
    return ValveContextRecord(
        timestamp_s=timestamp_s,
        values=values,
        warmed_up=bool(getattr(prediction, "warmed_up")),
        phase_name=str(getattr(prediction, "phase")),
        error_reason_name=str(getattr(prediction, "error_reason")),
        schema=VALVE_CONTEXT_V1_SCHEMA,
    )


def context_from_observer_probabilities(
    probabilities,
    *,
    timestamp_s: float,
    context_valid: bool,
) -> ValveContextRecord:
    """Build the 5-D policy context from the frozen four-state observer."""

    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    values = validate_valve_context_v2(
        [*probabilities.tolist(), float(bool(context_valid))]
    )
    timestamp_s = float(timestamp_s)
    if not np.isfinite(timestamp_s):
        raise ValueError("observer result timestamp must be finite")
    phase_idx = int(np.argmax(values[:4]))
    return ValveContextRecord(
        timestamp_s=timestamp_s,
        values=values,
        warmed_up=bool(context_valid),
        phase_name=VALVE_CONTEXT_V2_PHASE_NAMES[phase_idx],
        error_reason_name="n/a",
        schema=VALVE_CONTEXT_V2_SCHEMA,
    )


def valve_context_spec(schema: str) -> dict:
    """Return serialized dimensions/names for one supported contract."""

    schema = str(schema)
    if schema == VALVE_CONTEXT_V1_SCHEMA:
        return {
            "schema": schema,
            "dim": VALVE_CONTEXT_V1_DIM,
            "phase_names": VALVE_CONTEXT_V1_PHASE_NAMES,
            "reason_names": VALVE_CONTEXT_V1_ERROR_REASON_NAMES,
            "num_phase_experts": 5,
            "value_columns": (
                "phase_approach",
                "phase_turning",
                "phase_endpoint_reached",
                "phase_task_complete",
                "phase_error",
                "reason_none",
                "reason_turn_no_contact",
                "reason_post_contact_drop",
                "reason_other",
                "warmed_up",
            ),
        }
    if schema == VALVE_CONTEXT_V2_SCHEMA:
        return {
            "schema": schema,
            "dim": VALVE_CONTEXT_V2_DIM,
            "phase_names": VALVE_CONTEXT_V2_PHASE_NAMES,
            "reason_names": (),
            "num_phase_experts": 4,
            "value_columns": (
                "phase_approach",
                "phase_turning",
                "phase_recovery",
                "phase_error",
                "context_valid",
            ),
        }
    raise ValueError(f"unsupported valve context schema {schema!r}")
