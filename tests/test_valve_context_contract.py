from types import SimpleNamespace

import numpy as np

from diffusion_policy.common.valve_context_contract import (
    context_from_classifier_prediction,
    validate_valve_context,
)


def test_classifier_v4_result_uses_documented_10d_order():
    result = SimpleNamespace(
        timestamp_s=12.5,
        phase="turning",
        error_reason="none",
        warmed_up=True,
        phase_probabilities={
            "approach": 0.1,
            "turning": 0.6,
            "endpoint_reached": 0.1,
            "task_complete": 0.1,
            "error": 0.1,
        },
        error_reason_probabilities={
            "none": 1.0,
            "turn_no_contact": 0.0,
            "post_contact_drop": 0.0,
            "other": 0.0,
        },
    )
    context = context_from_classifier_prediction(result)
    np.testing.assert_allclose(
        context.values,
        [0.1, 0.6, 0.1, 0.1, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    assert context.phase_name == "turning"
    assert context.warmed_up


def test_context_rejects_nonprobability_or_nonbinary_warmup():
    try:
        validate_valve_context([0.3] * 5 + [1, 0, 0, 0, 0])
    except ValueError as exc:
        assert "phase probabilities" in str(exc)
    else:
        raise AssertionError("invalid phase probabilities were accepted")
    try:
        validate_valve_context([1, 0, 0, 0, 0, 1, 0, 0, 0, 0.5])
    except ValueError as exc:
        assert "exactly 0 or 1" in str(exc)
    else:
        raise AssertionError("nonbinary warmed_up flag was accepted")
