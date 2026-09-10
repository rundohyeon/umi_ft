import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_valve_context_v2_sidecar.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "generate_valve_context_v2_sidecar_test", _SCRIPT_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_force_history = _MODULE._force_history


def test_sidecar_force_history_is_causal_and_episode_local():
    wrench = np.arange(8 * 12, dtype=np.float32).reshape(8, 12)
    history, mask = _force_history(
        1,
        rgb_episode_ends=np.asarray([3, 6]),
        wrench_episode_starts=np.asarray([0, 4]),
        wrench_episode_ends=np.asarray([4, 8]),
        rgb_to_wrench_end_idx=np.asarray([0, 1, 3, 4, 6, 7]),
        wrench=wrench,
    )
    assert history.shape == (50, 12)
    assert mask.shape == (50, 1)
    np.testing.assert_array_equal(history[-2:], wrench[:2])
    np.testing.assert_array_equal(history[:-2], np.repeat(wrench[:1], 48, axis=0))
    np.testing.assert_array_equal(mask[-2:], 1.0)
    np.testing.assert_array_equal(mask[:-2], 0.0)


def test_sidecar_force_history_rejects_cross_episode_mapping():
    with pytest.raises(ValueError, match="episode boundary"):
        _force_history(
            3,
            rgb_episode_ends=np.asarray([3, 6]),
            wrench_episode_starts=np.asarray([0, 4]),
            wrench_episode_ends=np.asarray([4, 8]),
            rgb_to_wrench_end_idx=np.asarray([0, 1, 3, 3, 6, 7]),
            wrench=np.zeros((8, 12), dtype=np.float32),
        )
