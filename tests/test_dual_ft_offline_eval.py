from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from eval_dual_ft_offline import (
    ErrorAccumulator,
    _dataset_provenance,
    _rotation_error_rad,
    _select_state_name,
    _normalizer_summary,
    _update_action_metrics,
    disable_stochastic_image_transforms,
    evaluate_loader,
)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DATASET = REPO_ROOT / "session_260827" / "dataset.zarr.zip"
CURRENT_FORCE_SIDECAR = (
    REPO_ROOT / "session_260827" / "dataset_force_sidecar.zarr"
)
CONFIG_DIR = REPO_ROOT / "diffusion_policy" / "config"


def test_error_accumulator_reports_streaming_metrics():
    accumulator = ErrorAccumulator()
    accumulator.update(np.asarray([-1.0, 2.0, -3.0]))
    result = accumulator.result()
    assert result["count"] == 3
    assert np.isclose(result["mae"], 2.0)
    assert np.isclose(result["mse"], 14.0 / 3.0)
    assert np.isclose(result["rmse"], np.sqrt(14.0 / 3.0))
    assert result["max_abs"] == 3.0


def test_action_metrics_separate_grasp_force_from_width():
    names = (
        "overall",
        "position_m",
        "rotation_6d",
        "rotation_geodesic_rad",
        "rotation_geodesic_deg",
        "gripper_width_m",
        "grasp_force_N",
    )
    accumulators = {name: ErrorAccumulator() for name in names}
    target = torch.zeros(1, 16, 11)
    target[..., 3] = 1.0
    target[..., 7] = 1.0
    prediction = target.clone()
    prediction[..., 9] = 0.02
    prediction[..., 10] = 3.0

    _update_action_metrics(accumulators, prediction, target)

    assert np.isclose(accumulators["gripper_width_m"].result()["mae"], 0.02)
    assert np.isclose(accumulators["grasp_force_N"].result()["mae"], 3.0)
    assert np.isclose(
        accumulators["rotation_geodesic_rad"].result()["max_abs"], 0.0
    )
    assert np.isclose(
        accumulators["rotation_geodesic_deg"].result()["max_abs"], 0.0
    )


def test_rotation_geodesic_error_is_in_radians():
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float64)
    quarter_turn = np.asarray([0, 1, 0, -1, 0, 0], dtype=np.float64)
    error = _rotation_error_rad(quarter_turn[None], identity[None])
    np.testing.assert_allclose(error, [np.pi / 2], atol=1e-7)


def test_auto_state_selection_prefers_ema_only_when_configured():
    payload = {"state_dicts": {"model": {"x": 1}, "ema_model": {"x": 2}}}
    assert _select_state_name(payload, "auto", use_ema=True) == "ema_model"
    assert _select_state_name(payload, "auto", use_ema=False) == "model"


def test_auto_state_selection_never_falls_back_from_missing_ema():
    payload = {"state_dicts": {"model": {"x": 1}}}
    with pytest.raises(ValueError, match="no usable 'auto'"):
        _select_state_name(payload, "auto", use_ema=True)


def _complete_normalizer():
    dimensions = {
        "camera0_rgb": 1,
        "robot0_eef_pos": 3,
        "robot0_eef_rot_axis_angle": 6,
        "robot0_ft_left": 6,
        "robot0_ft_right": 6,
        "action": 11,
    }
    normalizer = LinearNormalizer()
    for key, dimension in dimensions.items():
        zeros = torch.zeros(dimension)
        ones = torch.ones(dimension)
        normalizer[key] = SingleFieldLinearNormalizer.create_manual(
            scale=ones,
            offset=zeros,
            input_stats_dict={
                "min": zeros,
                "max": ones,
                "mean": zeros,
                "std": ones,
            },
        )
    return normalizer


def test_normalizer_summary_validates_every_field_and_force_channel():
    class Policy:
        normalizer = _complete_normalizer()

    summary = _normalizer_summary(Policy())
    assert summary["grasp_force"]["max"] == 1.0
    assert summary["fields"]["robot0_ft_left"]["scale"] == [1.0] * 6


def test_normalizer_summary_rejects_zero_scale():
    class Policy:
        normalizer = _complete_normalizer()

    with torch.no_grad():
        Policy.normalizer.params_dict["robot0_ft_right"]["scale"][2] = 0
    with pytest.raises(ValueError, match="zero scale"):
        _normalizer_summary(Policy())


def test_nested_dual_encoder_image_augmentation_is_disabled():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.key_transform_map = torch.nn.ModuleDict(
                {"camera0_rgb": torch.nn.Sequential(torch.nn.Dropout(p=1.0))}
            )

    class Dual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_pose_encoder = Encoder()

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.obs_encoder = Dual()

    policy = Policy()
    disabled = disable_stochastic_image_transforms(policy)
    assert disabled == ["obs_encoder.vision_pose_encoder.camera0_rgb"]
    assert isinstance(
        policy.obs_encoder.vision_pose_encoder.key_transform_map["camera0_rgb"],
        torch.nn.Identity,
    )


def test_evaluate_loader_reports_all_11d_components_and_ft_age():
    class FakeDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            action = torch.zeros(16, 11)
            action[:, 3] = 1.0
            action[:, 7] = 1.0
            return {
                "obs": {"expected_action": action.clone()},
                "action": action,
                "sample_info": {
                    "episode_index": torch.tensor(index + 4),
                    "left_ft_age": torch.tensor(0.005 + index * 0.001),
                    "right_ft_age": torch.tensor(0.006 + index * 0.001),
                },
            }

    class FakePolicy:
        class IdentityActionNormalizer:
            def normalize(self, value):
                return value

        normalizer = {"action": IdentityActionNormalizer()}

        def compute_loss(self, batch):
            return torch.tensor(0.25, device=batch["action"].device)

        def predict_action(self, obs):
            return {"action_pred": obs["expected_action"]}

    result = evaluate_loader(
        FakePolicy(),
        DataLoader(FakeDataset(), batch_size=2),
        device=torch.device("cpu"),
        max_batches=None,
        prediction_repeats=2,
        seed=7,
        compute_diffusion_loss=True,
        ft_max_age_sec=0.012,
    )
    assert result["evaluated_samples"] == 2
    assert result["prediction_repeats"] == 2
    assert np.isclose(result["diffusion_loss"], 0.25)
    assert result["action_error"]["overall"]["max_abs"] == 0.0
    assert result["action_error"]["grasp_force_N"]["count"] == 64
    assert result["episode_indices"] == [4, 5]
    assert np.isclose(result["left_ft_latest_age_ms"]["max"], 6.0)


@pytest.mark.skipif(
    not CURRENT_DATASET.is_file() or not CURRENT_FORCE_SIDECAR.is_dir(),
    reason="260827 base dataset or force sidecar absent",
)
def test_actual_260827_validation_samples_pass_evaluator_contract():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="train_diffusion_unet_timm_umi_dual_ft_workspace"
        )
    dataset = instantiate(cfg.task.dataset)
    validation = dataset.get_validation_dataset()

    class IdentityActionNormalizer:
        def normalize(self, value):
            return value

    class ValidPredictionPolicy:
        normalizer = {"action": IdentityActionNormalizer()}

        def predict_action(self, obs):
            batch_size = obs["camera0_rgb"].shape[0]
            action = torch.zeros(batch_size, 16, 11)
            action[..., 3] = 1.0
            action[..., 7] = 1.0
            return {"action_pred": action}

    try:
        provenance = _dataset_provenance(
            dataset,
            validation,
            cfg,
            CURRENT_DATASET,
            CURRENT_FORCE_SIDECAR,
            full_content_hash=False,
        )
        loader = DataLoader(
            torch.utils.data.Subset(validation, [0, 1]),
            batch_size=2,
            shuffle=False,
        )
        result = evaluate_loader(
            ValidPredictionPolicy(),
            loader,
            device=torch.device("cpu"),
            max_batches=None,
            prediction_repeats=1,
            seed=42,
            compute_diffusion_loss=False,
            expected_obs_meta=cfg.task.shape_meta.obs,
            ft_max_age_sec=float(cfg.task.ft_max_age_sec),
        )
    finally:
        validation.close()
        dataset.close()

    assert result["evaluated_samples"] == 2
    assert provenance["effective_ft"]["source_key"] == "wrench_12d"
    assert provenance["grasp_force_target"]["key"] == (
        "derived_from_sidecar_wrench_12d"
    )
    assert provenance["force_sidecar_zarr"]["episode_bias"]["shape"] == [214, 12]
    assert result["episode_indices"] == [18]
    assert result["action_start_offset_ms"]["max"] == 0.0
    assert 750.0 < result["action_end_offset_ms"]["max"] < 752.0
