"""Parity of the actual 0814 ZIP's causal F/T history and live assembler.

The test is skipped when the large robot dataset is intentionally absent (for
example in CI), but otherwise uses raw values/timestamps from the ZIP rather
than fabricated signals.
"""

from pathlib import Path
import unittest

import numpy as np

from diffusion_policy.dataset.umi_dual_ft_dataset import UmiDualFTDataset
from umi.common.pose_util import mat_to_pose
from umi.real_world.rg2ft_obs import causal_ft_history_from_streams
from umi.real_world.real_inference_util import get_real_umi_obs_dict


_ROOT = Path(__file__).resolve().parents[1]
_DATASET = Path("/home/oem/smh/indy_umi_rg_ft/demos_0814/dataset_multirate_clean.zarr.zip")


@unittest.skipUnless(_DATASET.is_file(), "0814 robot dataset is not installed")
class ActualZipDualFTParityTest(unittest.TestCase):
    def test_actual_zip_causal_ft_history_matches_training_sample(self):
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        task = OmegaConf.load(_ROOT / "diffusion_policy/config/task/umi_dual_ft.yaml")
        task = OmegaConf.create({"task": task})
        OmegaConf.resolve(task)
        task = task.task
        dataset = UmiDualFTDataset(
            shape_meta=task.shape_meta,
            dataset_path=str(_DATASET),
            data_keys=task.data_keys,
            ft=task.ft,
            pose_repr=task.pose_repr,
            action_padding=False,
            seed=42,
            val_ratio=0.05,
        )
        # Real inference is deterministic; match the evaluation path rather
        # than the training-only random episode-start rotation augmentation.
        dataset.start_pose_noise_scale = np.zeros(6, dtype=np.float64)
        offline_obs, _, info = dataset._sample_arrays(0, load_rgb=True)
        episode = int(info["episode_index"])
        # The function is module-level in the dataset implementation; calculate
        # episode bounds directly to keep this test tied to raw stream indexing.
        left_start = 0 if episode == 0 else int(dataset.ft_left_episode_ends[episode - 1])
        left_end = int(dataset.ft_left_episode_ends[episode])
        right_start = 0 if episode == 0 else int(dataset.ft_right_episode_ends[episode - 1])
        right_end = int(dataset.ft_right_episode_ends[episode])
        live_obs = causal_ft_history_from_streams(
            dataset.ft_left_timestamps[left_start:left_end],
            dataset.ft_left[left_start:left_end],
            dataset.ft_right_timestamps[right_start:right_end],
            dataset.ft_right[right_start:right_end],
            anchor_timestamp=float(info["anchor_timestamp"]),
            num_steps=int(task.ft.num_steps),
            stride=int(task.ft.stride),
            frequency=float(dataset.ft_left_hz),
        )
        episode_start = 0 if episode == 0 else int(dataset.rgb_episode_ends[episode - 1])
        image_meta = task.shape_meta.obs.camera0_rgb
        pose_meta = task.shape_meta.obs.robot0_eef_pos
        gripper_meta = task.shape_meta.obs.robot0_gripper_width
        image_idx = dataset._history_indices(
            int(dataset.indices[0][1]),
            episode_start,
            int(image_meta.horizon),
            int(image_meta.down_sample_steps),
        )
        pose_idx = dataset._history_indices(
            int(dataset.indices[0][1]),
            episode_start,
            int(pose_meta.horizon),
            int(pose_meta.down_sample_steps),
        )
        gripper_idx = dataset._history_indices(
            int(dataset.indices[0][1]),
            episode_start,
            int(gripper_meta.horizon),
            int(gripper_meta.down_sample_steps),
        )
        # Feed the same raw ZIP values through the real-evaluator assembler.
        # This is the simulated-live branch: THWC uint8 RGB, absolute pose,
        # raw width, and the timestamped causal F/T buffers above.
        raw_pose = mat_to_pose(dataset.pose_mats[pose_idx]).astype(np.float32)
        raw_rgb = dataset._get_rgb_array()[image_idx]
        simulated_env_obs = {
            "camera0_rgb": raw_rgb,
            "robot0_eef_pos": raw_pose[:, :3],
            "robot0_eef_rot_axis_angle": raw_pose[:, 3:],
            "robot0_gripper_width": dataset.gripper_width[gripper_idx],
            **live_obs,
        }
        live_policy_obs = get_real_umi_obs_dict(
            simulated_env_obs,
            task.shape_meta,
            obs_pose_repr=str(task.pose_repr.obs_pose_repr),
            tx_robot1_robot0=None,
            episode_start_pose=[
                dataset.episode_start_pose[episode].astype(np.float32)
            ],
        )
        live_policy_obs = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in live_policy_obs.items()
        }
        for key, expected in offline_obs.items():
            np.testing.assert_allclose(
                live_policy_obs[key], expected, rtol=1e-5, atol=1e-6,
                err_msg=f"offline/live mismatch for {key}",
            )
            self.assertEqual(live_policy_obs[key].shape, expected.shape)
            self.assertEqual(live_policy_obs[key].dtype, np.float32)
        for key in ("robot0_ft_left", "robot0_ft_right"):
            np.testing.assert_allclose(live_obs[key], offline_obs[key], rtol=0, atol=0)
            self.assertEqual(live_obs[key].shape, (32, 6))
            self.assertEqual(live_obs[key].dtype, np.float32)
        self.assertLessEqual(
            np.max(live_obs["robot0_ft_left_timestamps"]),
            float(info["anchor_timestamp"]),
        )
        self.assertLessEqual(
            np.max(live_obs["robot0_ft_right_timestamps"]),
            float(info["anchor_timestamp"]),
        )
        # RGB follows the dataset's deterministic decode/scaling contract.
        self.assertEqual(offline_obs["camera0_rgb"].shape, (2, 3, 224, 224))
        self.assertEqual(offline_obs["camera0_rgb"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
