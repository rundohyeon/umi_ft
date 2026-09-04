"""Shared strict contract for the frozen valve-state classifier context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VALVE_CONTEXT_KEY = "valve_context"
VALVE_CONTEXT_DIM = 10
VALVE_PHASE_NAMES = (
    "approach",
    "turning",
    "endpoint_reached",
    "task_complete",
    "error",
)
VALVE_ERROR_REASON_NAMES = (
    "none",
    "turn_no_contact",
    "post_contact_drop",
    "other",
)


@dataclass(frozen=True)
class ValveContextRecord:
    """Validated context plus fields useful for deployment diagnostics."""

    timestamp_s: float
    values: np.ndarray
    warmed_up: bool
    phase_name: str
    error_reason_name: str


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
    )
