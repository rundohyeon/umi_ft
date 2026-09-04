import numpy as np
import pytest

from umi.real_world.rg2ft_startup_bias import (
    FTStartupBiasConfig,
    correct_native_wrenches,
    estimate_startup_bias,
    startup_residual_after_software_tare,
    subtract_startup_bias,
)


def test_estimate_startup_bias_uses_unique_static_native_samples():
    rng = np.random.default_rng(3)
    bias = np.arange(12, dtype=np.float64) * 0.1
    samples = bias + rng.normal(scale=0.002, size=(20, 12))
    result = estimate_startup_bias(
        np.arange(20, dtype=np.float64) / 100.0,
        samples[:, :6],
        samples[:, 6:],
        FTStartupBiasConfig(sample_count=20),
    )
    np.testing.assert_allclose(result["bias_12d"], samples.mean(axis=0))
    left, right = subtract_startup_bias(
        samples[:, :6], samples[:, 6:], result["bias_12d"]
    )
    np.testing.assert_allclose(left.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(right.mean(axis=0), 0.0, atol=1e-12)


def test_estimate_startup_bias_rejects_duplicate_timestamps_and_contact_motion():
    cfg = FTStartupBiasConfig(sample_count=5)
    samples = np.zeros((5, 12), dtype=np.float64)
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_startup_bias([0, 1, 1, 3, 4], samples[:, :6], samples[:, 6:], cfg)

    samples[:, 0] = [0, 0, 0, 0, 5]
    with pytest.raises(ValueError, match="startup bias rejected"):
        estimate_startup_bias(np.arange(5), samples[:, :6], samples[:, 6:], cfg)


def test_software_tare_and_startup_residual_match_training_calibration_order():
    # Mimic the large fixed native offsets recorded during collection and the
    # small residual per-episode bias removed from the training sidecar.
    software_tare = np.asarray(
        [5.8, 2.6, 146.7, 0.36, 0.11, 0.07, 1.6, -5.3, -137.2, 0.43, 0.27, -0.12],
        dtype=np.float64,
    )
    residual = np.asarray(
        [0.02, -0.01, 0.03, 0.001, 0.0, -0.002, -0.03, 0.02, -0.04, 0.002, 0.0, 0.001],
        dtype=np.float64,
    )
    raw_startup_baseline = software_tare + residual
    inferred_residual = startup_residual_after_software_tare(
        raw_startup_baseline, software_tare
    )
    np.testing.assert_allclose(inferred_residual, residual)

    signal = np.asarray(
        [[0.08, 0.01, -0.04, 0.002, 0.0, 0.001, -0.05, 0.03, 0.12, -0.001, 0.0, 0.002]],
        dtype=np.float64,
    )
    raw = raw_startup_baseline[None] + signal
    left, right = correct_native_wrenches(
        raw[:, :6],
        raw[:, 6:],
        software_tare_offset_12d=software_tare,
        startup_residual_bias_12d=inferred_residual,
    )
    np.testing.assert_allclose(np.concatenate([left, right], axis=1), signal)
