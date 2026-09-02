from pathlib import Path

import numpy as np
import pytest
import scipy.spatial.transform as st
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from diffusion_policy.common.nested_zarr import (
    detect_zarr_prefix,
    open_nested_zip_group,
)
from diffusion_policy.dataset.umi_dual_ft_dataset import (
    UmiDualFTDataset,
    _subtract_episode_bias,
)
from umi.common.pose_util import pose_to_mat


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATASET = REPO_ROOT / "session_260827" / "dataset.zarr.zip"
MULTIRATE_DATASET = REPO_ROOT / "session_260827" / "dataset_multirate.zarr"
FORCE_SIDECAR = REPO_ROOT / "session_260827" / "dataset_force_sidecar.zarr"
CONFIG_DIR = REPO_ROOT / "diffusion_policy" / "config"


def test_episode_bias_subtraction_preserves_native_axis_order():
    values = np.asarray(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
        ],
        dtype=np.float32,
    )
    ends = np.asarray([2, 3], dtype=np.int64)
    bias = np.asarray(
        [[0.5, 1, 1.5, 2, 2.5, 3], [3, 2.5, 2, 1.5, 1, 0.5]],
        dtype=np.float32,
    )

    actual = _subtract_episode_bias(values, ends, bias)

    np.testing.assert_array_equal(actual[:2], values[:2] - bias[0])
    np.testing.assert_array_equal(actual[2:], values[2:] - bias[1])
    np.testing.assert_array_equal(values[0], [1, 2, 3, 4, 5, 6])


def test_pose7_uses_scalar_last_xyzw_quaternion_order():
    dataset = object.__new__(UmiDualFTDataset)
    dataset.rgb_episode_ends = np.asarray([2], dtype=np.int64)
    positions = np.asarray(
        [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=np.float32
    )
    expected_rotvec = np.asarray(
        [[0.3, -0.4, 0.2], [-1.1, 0.7, 0.5]], dtype=np.float64
    )
    quat_xyzw = st.Rotation.from_rotvec(expected_rotvec).as_quat()
    pose7 = np.concatenate([positions, quat_xyzw], axis=-1).astype(np.float32)

    dataset._convert_pose7(pose7)

    np.testing.assert_array_equal(dataset.eef_pos, positions)
    actual_rotation = st.Rotation.from_rotvec(dataset.eef_rot_axis_angle)
    expected_rotation = st.Rotation.from_rotvec(expected_rotvec)
    error = (actual_rotation * expected_rotation.inv()).magnitude()
    np.testing.assert_allclose(error, 0.0, atol=2e-7)
    expected_mats = pose_to_mat(
        np.concatenate([positions, expected_rotvec], axis=-1)
    )
    np.testing.assert_allclose(dataset.pose_mats, expected_mats, atol=2e-7)


@pytest.mark.skipif(
    not ORIGINAL_DATASET.exists() or not MULTIRATE_DATASET.exists(),
    reason="260827 source and multirate datasets are not available",
)
def test_260827_xyzw_pose_matches_original_axis_angle():
    import zarr

    source_store = zarr.ZipStore(str(ORIGINAL_DATASET), mode="r")
    try:
        source = zarr.open_group(store=source_store, mode="r")
        expected_rotvec = np.asarray(
            source["data/robot0_eef_rot_axis_angle"][:], dtype=np.float64
        )
        episode_ends = np.asarray(
            source["meta/episode_ends"][:], dtype=np.int64
        )
    finally:
        source_store.close()

    multirate = zarr.open_group(str(MULTIRATE_DATASET), mode="r")
    pose7 = np.asarray(multirate["data/ts_pose_fb_0"][:], dtype=np.float32)
    assert len(pose7) == len(expected_rotvec)

    dataset = object.__new__(UmiDualFTDataset)
    dataset.rgb_episode_ends = episode_ends
    dataset._convert_pose7(pose7)

    actual_rotation = st.Rotation.from_rotvec(dataset.eef_rot_axis_angle)
    expected_rotation = st.Rotation.from_rotvec(expected_rotvec)
    error = (actual_rotation * expected_rotation.inv()).magnitude()
    assert float(np.max(error)) < 2e-7


def test_directory_zarr_store_opens_read_only(tmp_path: Path):
    import zarr

    path = tmp_path / "example.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("value", data=np.asarray([1, 2, 3], dtype=np.int64))

    info = detect_zarr_prefix(path)
    store, reopened, prefix = open_nested_zip_group(path)
    try:
        assert info.prefix == prefix == ""
        np.testing.assert_array_equal(reopened["value"][:], [1, 2, 3])
    finally:
        store.close()


def _force_contract_fixture(stored_force):
    dataset = object.__new__(UmiDualFTDataset)
    dataset.rgb_timestamps = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    dataset.rgb_episode_ends = np.asarray([3], dtype=np.int64)
    dataset.ft_left_timestamps = np.asarray([0.0, 1.0], dtype=np.float64)
    dataset.ft_right_timestamps = np.asarray([0.0, 1.0], dtype=np.float64)
    dataset.ft_left_episode_ends = np.asarray([2], dtype=np.int64)
    dataset.ft_right_episode_ends = np.asarray([2], dtype=np.int64)
    dataset.ft_left = np.zeros((2, 6), dtype=np.float32)
    dataset.ft_right = np.zeros((2, 6), dtype=np.float32)
    dataset.ft_left[:, 2] = [2.0, 4.0]
    dataset.ft_right[:, 2] = [6.0, 12.0]
    dataset.grasp_force = np.asarray(stored_force, dtype=np.float32).reshape(-1, 1)
    return dataset


def test_grasp_force_contract_is_interpolated_signed_native_fz_measurement():
    dataset = _force_contract_fixture([2.0, 3.0, 4.0])
    dataset._validate_grasp_force_contract()
    assert dataset.grasp_force_contract_max_abs_error == 0.0


def test_grasp_force_contract_rejects_command_or_wrong_sign_label():
    dataset = _force_contract_fixture([-2.0, -3.0, -4.0])
    with pytest.raises(ValueError, match="signed native-Fz measurement"):
        dataset._validate_grasp_force_contract()


@pytest.mark.skipif(
    not ORIGINAL_DATASET.exists() or not FORCE_SIDECAR.exists(),
    reason="260827 stock UMI dataset and force sidecar are not available",
)
def test_260827_config_uses_stock_zip_and_native_bias_removed_sidecar():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="train_diffusion_unet_timm_umi_dual_ft_workspace",
            overrides=["task=umi_dual_ft_260827_bias_only"],
        )

    dataset = instantiate(cfg.task.dataset)
    try:
        assert dataset.source_mode == "base_with_force_sidecar"
        assert Path(dataset.dataset_path).resolve() == ORIGINAL_DATASET.resolve()
        assert Path(dataset.force_sidecar_path).resolve() == FORCE_SIDECAR.resolve()
        assert dataset.sidecar_attrs["train_wrench_key"] == "wrench_12d"
        assert dataset.ft_bias_removed
        assert dataset.rgb_shape == (120616, 224, 224, 3)
        assert dataset.ft_left.shape == dataset.ft_right.shape == (200865, 6)
        assert dataset.n_episodes == 214
        assert int(dataset.rgb_wrench_valid.sum()) == 120402
        assert sum(row["dropped"] for row in dataset.causal_drop_report) == 214

        sidecar_store, sidecar, _ = open_nested_zip_group(FORCE_SIDECAR)
        try:
            native_wrench = np.asarray(
                sidecar["data/wrench_12d"][:64], dtype=np.float32
            )
            np.testing.assert_array_equal(dataset.ft_left[:64], native_wrench[:, :6])
            np.testing.assert_array_equal(dataset.ft_right[:64], native_wrench[:, 6:])
            np.testing.assert_array_equal(
                dataset.rgb_to_wrench_end_idx[:128],
                sidecar["data/rgb_to_wrench_end_idx"][:128],
            )
        finally:
            sidecar_store.close()

        base_store, base, _ = open_nested_zip_group(ORIGINAL_DATASET)
        try:
            np.testing.assert_array_equal(
                dataset.eef_pos[:64], base["data/robot0_eef_pos"][:64]
            )
            np.testing.assert_array_equal(
                dataset.eef_rot_axis_angle[:64],
                base["data/robot0_eef_rot_axis_angle"][:64],
            )
        finally:
            base_store.close()

        obs, action, info = dataset._sample_arrays(0, load_rgb=True)
        assert obs["camera0_rgb"].shape == (2, 3, 224, 224)
        assert obs["robot0_ft_left"].shape == (32, 6)
        assert obs["robot0_ft_right"].shape == (32, 6)
        assert action.shape == (16, 11)
        assert float(info["left_ft_age"]) >= 0.0
        assert float(info["right_ft_age"]) >= 0.0
    finally:
        dataset.close()
