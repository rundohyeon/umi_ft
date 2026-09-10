from types import SimpleNamespace

import numpy as np

from diffusion_policy.common.valve_context_contract import (
    VALVE_CONTEXT_V2_SCHEMA,
    context_from_classifier_prediction,
    context_from_observer_probabilities,
    validate_valve_context,
    validate_valve_context_v2,
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


def test_four_state_observer_uses_probabilities_plus_valid_flag():
    record = context_from_observer_probabilities(
        [0.1, 0.2, 0.6, 0.1],
        timestamp_s=3.5,
        context_valid=True,
    )
    np.testing.assert_allclose(record.values, [0.1, 0.2, 0.6, 0.1, 1.0])
    assert record.phase_name == "recovery"
    assert record.error_reason_name == "n/a"
    assert record.schema == VALVE_CONTEXT_V2_SCHEMA

    for invalid in (
        [0.1, 0.2, 0.3, 0.1, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.5],
    ):
        try:
            validate_valve_context_v2(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid v2 context accepted: {invalid}")
