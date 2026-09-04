import tempfile

import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.vision.dual_ft_obs_encoder import (
    CausalConv1d,
    CausalFTEncoder,
    DualFTObsEncoder,
)
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    _build_optimizer_param_groups,
    _scheduler_resume_last_epoch,
)


def _shape_meta(include_ft=True):
    obs = {
        "camera0_rgb": {"shape": [3, 64, 64], "horizon": 2, "type": "rgb"},
        "robot0_eef_pos": {"shape": [3], "horizon": 2, "type": "low_dim"},
        "robot0_eef_rot_axis_angle": {
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
        "action": {"shape": [11], "horizon": 16, "down_sample_steps": 3},
    }


def _obs(batch_size=1):
    return {
        "camera0_rgb": torch.rand(batch_size, 2, 3, 64, 64),
        "robot0_eef_pos": torch.rand(batch_size, 2, 3),
        "robot0_eef_rot_axis_angle": torch.rand(batch_size, 2, 6),
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
    zeros = torch.zeros(11)
    ones = torch.ones(11)
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


def test_ft_encoder_final_token_uses_all_32_timesteps():
    encoder = CausalFTEncoder(
        channel_dims=(8, 8, 8, 8),
        output_dim=8,
    )
    with torch.no_grad():
        for layer in encoder.modules():
            if isinstance(layer, CausalConv1d):
                layer.conv.weight.fill_(1.0)
                layer.conv.bias.zero_()
    history = torch.ones(1, 32, 6, requires_grad=True)
    sequence = encoder.forward_sequence(history)
    assert sequence.shape == (1, 1, 8)
    sequence.sum().backward()
    influence = history.grad.abs().sum(dim=-1)[0]
    assert torch.all(influence > 0), influence


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
    assert encoder(obs).shape == (1, 1042)


def test_dual_ft_forward_backward_uses_independent_encoders():
    shape_meta = _shape_meta()
    encoder = _dual_encoder(shape_meta)
    assert encoder.left_ft_encoder is not encoder.right_ft_encoder
    assert encoder.num_fusion_tokens == 4
    assert encoder.fusion_projection.in_features == 4 * 64
    assert not hasattr(encoder, "cls_token")
    result = encoder(_obs(batch_size=2))
    assert result.shape == (2, 82)
    weights = torch.arange(1, 65, dtype=result.dtype)
    (result[:, :64].square() * weights).mean().backward()
    assert next(encoder.left_ft_encoder.parameters()).grad.abs().sum() > 0
    assert next(encoder.right_ft_encoder.parameters()).grad.abs().sum() > 0


def test_dual_ft_attention_capture_preserves_fusion_output():
    """Eval-only attention logging must not change the policy feature."""
    torch.manual_seed(0)
    encoder = _dual_encoder(_shape_meta()).eval()
    obs = _obs(batch_size=2)
    with torch.no_grad():
        expected = encoder(obs)
        encoder.set_fusion_attention_capture(True)
        actual = encoder(obs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    attention = encoder.last_fusion_attention
    assert attention is not None
    assert attention.shape == (2, 8, 4, 4)
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones((2, 8, 4), dtype=attention.dtype),
        rtol=1e-5,
        atol=1e-6,
    )


def test_one_batch_train_step_and_checkpoint_reload():
    torch.manual_seed(0)
    shape_meta = _shape_meta()
    policy = _policy(shape_meta)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    batch = {
        "obs": _obs(),
        "action": torch.rand(1, 16, 11),
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


def test_dual_ft_optimizer_groups_split_pretrained_and_new_modules():
    policy = _policy(_shape_meta())
    cfg = OmegaConf.create(
        {
            "optimizer": {"lr": 3e-4},
            "policy": {"obs_encoder": {"pretrained": True}},
            "optimizer_parameter_groups": {
                "mode": "dual_ft",
                "pretrained_vision_lr": 3e-5,
                "fusion_transformer_lr": 1e-4,
                "new_obs_lr": 3e-4,
            },
        }
    )

    groups = _build_optimizer_param_groups(policy, cfg)
    by_name = {group["name"]: group for group in groups}
    assert {name: group["lr"] for name, group in by_name.items()} == {
        "diffusion_model": 3e-4,
        "pretrained_vision": 3e-5,
        "fusion_transformer": 1e-4,
        "new_obs_modules": 3e-4,
    }

    parameter_group = {
        id(param): name
        for name, group in by_name.items()
        for param in group["params"]
    }
    assert parameter_group[id(next(policy.model.parameters()))] == "diffusion_model"
    vision_param = next(
        policy.obs_encoder.vision_pose_encoder.key_model_map.parameters()
    )
    assert parameter_group[id(vision_param)] == "pretrained_vision"
    assert parameter_group[id(next(policy.obs_encoder.fusion.parameters()))] == (
        "fusion_transformer"
    )
    for module in (
        policy.obs_encoder.left_ft_encoder,
        policy.obs_encoder.right_ft_encoder,
        policy.obs_encoder.fusion_projection,
    ):
        assert parameter_group[id(next(module.parameters()))] == "new_obs_modules"
    assert parameter_group[id(policy.obs_encoder.position_embedding)] == (
        "new_obs_modules"
    )
    assert len(parameter_group) == sum(
        1 for param in policy.parameters() if param.requires_grad
    )


def test_four_gpu_resume_scheduler_preserves_epoch90_learning_rate():
    base_lrs = [3e-4, 3e-5, 1e-4, 3e-4]
    checkpoint_lrs = [
        4.123722827092213e-05,
        4.123722827092213e-06,
        1.3745742756974046e-05,
        4.123722827092213e-05,
    ]
    params = [torch.nn.Parameter(torch.zeros(())) for _ in base_lrs]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [param],
                "lr": checkpoint_lr,
                "initial_lr": base_lr,
            }
            for param, base_lr, checkpoint_lr in zip(
                params, base_lrs, checkpoint_lrs
            )
        ]
    )
    from diffusion_policy.model.common.lr_scheduler import get_scheduler

    # epoch=90 checkpoint stores global_step=298661 before its final counter
    # increment, so resume begins at global_step=298662.
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=2000,
        num_training_steps=13125 * 120,
        last_epoch=_scheduler_resume_last_epoch(298662, 4),
    )

    assert scheduler.last_epoch == 1194648
    torch.testing.assert_close(
        torch.tensor(scheduler.get_last_lr()),
        torch.tensor(checkpoint_lrs),
        rtol=0,
        atol=0,
    )
