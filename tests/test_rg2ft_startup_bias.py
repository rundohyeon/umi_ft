import numpy as np
import pytest

from umi.real_world.rg2ft_startup_bias import (
    FTStartupBiasConfig,
    estimate_startup_bias,
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
