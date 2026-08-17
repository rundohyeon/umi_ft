import tempfile

import torch
from diffusers import DDIMScheduler

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.vision.dual_ft_obs_encoder import (
    CausalConv1d,
    DualFTObsEncoder,
)
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy


def _shape_meta(include_ft=True):
    obs = {
        "camera0_rgb": {"shape": [3, 64, 64], "horizon": 2, "type": "rgb"},
        "robot0_eef_pos": {"shape": [3], "horizon": 2, "type": "low_dim"},
        "robot0_eef_rot_axis_angle": {
            "shape": [6],
            "horizon": 2,
            "type": "low_dim",
        },
        "robot0_gripper_width": {
            "shape": [1],
            "horizon": 2,
            "type": "low_dim",
        },
        "robot0_eef_rot_axis_angle_wrt_start": {
            "shape": [6],
            "horizon": 2,
            "type": "low_dim",
        },
    }
    if include_ft:
        obs.update(
            {
                "robot0_ft_left": {
                    "shape": [6],
                    "horizon": 32,
                    "type": "low_dim",
                },
                "robot0_ft_right": {
                    "shape": [6],
                    "horizon": 32,
                    "type": "low_dim",
                },
            }
        )
    return {
        "obs": obs,
        "action": {"shape": [10], "horizon": 16, "down_sample_steps": 3},
    }


def _obs(batch_size=1):
    return {
        "camera0_rgb": torch.rand(batch_size, 2, 3, 64, 64),
        "robot0_eef_pos": torch.rand(batch_size, 2, 3),
        "robot0_eef_rot_axis_angle": torch.rand(batch_size, 2, 6),
        "robot0_gripper_width": torch.rand(batch_size, 2, 1),
        "robot0_eef_rot_axis_angle_wrt_start": torch.rand(batch_size, 2, 6),
        "robot0_ft_left": torch.rand(batch_size, 32, 6),
        "robot0_ft_right": torch.rand(batch_size, 32, 6),
    }


def _manual_normalizer(shape_meta):
    normalizer = LinearNormalizer()
    for key, attr in shape_meta["obs"].items():
        dim = 1 if attr["type"] == "rgb" else attr["shape"][0]
        zeros = torch.zeros(dim)
        ones = torch.ones(dim)
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
    zeros = torch.zeros(10)
    ones = torch.ones(10)
    normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
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


def _dual_encoder(shape_meta):
    return DualFTObsEncoder(
        shape_meta=shape_meta,
        model_name="resnet18",
        pretrained=False,
        frozen=False,
        global_pool="",
        transforms=None,
        feature_aggregation="avg",
        downsample_ratio=32,
        vision_feature_dim=512,
        fusion_dim=64,
        fusion_heads=8,
        fusion_feedforward_dim=128,
        ft_channel_dims=[8, 16, 32, 32],
        share_ft_encoder=False,
    )


def _policy(shape_meta):
    scheduler = DDIMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )
    policy = DiffusionUnetTimmPolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=_dual_encoder(shape_meta),
        num_inference_steps=2,
        diffusion_step_embed_dim=32,
        down_dims=[32, 64],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=True,
        input_pertub=0.0,
    )
    policy.set_normalizer(_manual_normalizer(shape_meta))
    return policy


def test_causal_conv_output_before_change_is_future_independent():
    torch.manual_seed(0)
    layer = CausalConv1d(6, 8, kernel_size=3, stride=1)
    original = torch.randn(2, 6, 16)
    modified = original.clone()
    modified[:, :, 10:] += 100.0
    y_original = layer(original)
    y_modified = layer(modified)
    torch.testing.assert_close(y_original[:, :, :10], y_modified[:, :, :10])


def test_legacy_rgb_pose_forward_contract():
    shape_meta = _shape_meta(include_ft=False)
    encoder = TimmObsEncoder(
        shape_meta=shape_meta,
        model_name="resnet18",
        pretrained=False,
        frozen=False,
        global_pool="",
        transforms=None,
        feature_aggregation="avg",
        downsample_ratio=32,
    )
    obs = {key: value for key, value in _obs().items() if key in shape_meta["obs"]}
    assert encoder(obs).shape == (1, 1056)


def test_dual_ft_forward_backward_uses_independent_encoders():
    shape_meta = _shape_meta()
    encoder = _dual_encoder(shape_meta)
    assert encoder.left_ft_encoder is not encoder.right_ft_encoder
    result = encoder(_obs(batch_size=2))
    assert result.shape == (2, 96)
    weights = torch.arange(1, 65, dtype=result.dtype)
    (result[:, :64].square() * weights).mean().backward()
    assert next(encoder.left_ft_encoder.parameters()).grad.abs().sum() > 0
    assert next(encoder.right_ft_encoder.parameters()).grad.abs().sum() > 0


def test_one_batch_train_step_and_checkpoint_reload():
    torch.manual_seed(0)
    shape_meta = _shape_meta()
    policy = _policy(shape_meta)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    batch = {
        "obs": _obs(),
        "action": torch.rand(1, 16, 10),
    }
    loss = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
        torch.save(policy.state_dict(), checkpoint.name)
        reloaded = _policy(shape_meta)
        reloaded.load_state_dict(torch.load(checkpoint.name, map_location="cpu"))
        for expected, actual in zip(policy.parameters(), reloaded.parameters()):
            torch.testing.assert_close(expected, actual)
