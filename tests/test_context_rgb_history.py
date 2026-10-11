import numpy as np
import pytest
import torch

from diffusion_policy.context.canonical_dataset import CanonicalRGBForceDataset, PHASE_NAMES
from models.rgb_force_context_encoder import RGBForceContextEncoder
from umi.real_world.rgb_force_context import RGBForceContextRuntime, ObservationUnavailable, select_context_window
from umi.real_world.valve_state_context import RGBForceValveContextRuntime
from test_context_rgb_force import canonical_files, VisionFixture


@pytest.fixture
def four_frame_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr('models.rgb_force_context_encoder.timm.create_model', lambda *a, **kw: VisionFixture())
    model = RGBForceContextEncoder(rgb_horizon=4, ft_feature_mode='raw_mean_delta', pretrained=False)
    payload = dict(schema='context_rgb_force_4frame_v3', state_dict=model.state_dict(),
        model_config=model.config, manifest=dict(phase_names=list(PHASE_NAMES)), epoch=0,
        training_config=dict(dataset=dict(rgb_horizon=4, rgb_stride=10, max_force_age_s=0.012)))
    path = tmp_path / 'four.pt'
    torch.save(payload, path)
    return path, payload, model


def test_four_image_forward_backward_checkpoint_and_shape_rejection(four_frame_checkpoint):
    path, payload, model = four_frame_checkpoint
    obs = dict(camera0_rgb=torch.rand(1, 4, 3, 224, 224),
               robot0_ft_left=torch.randn(1, 41, 6), robot0_ft_right=torch.randn(1, 41, 6))
    assert model.position.shape == (1, 7, 128)
    torch.nn.functional.cross_entropy(model(obs), torch.tensor([2])).backward()
    assert model.position.grad.shape == (1, 7, 128)
    assert model.visual_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.vision.parameters())
    model.eval()
    restored = RGBForceContextEncoder.from_checkpoint(payload).eval()
    torch.testing.assert_close(restored(obs), model(obs), rtol=0, atol=0)
    with pytest.raises(ValueError, match='RGB must'):
        model(dict(obs, camera0_rgb=obs['camera0_rgb'][:, :2]))
    runtime = RGBForceContextRuntime(path)
    assert runtime.rgb_horizon == 4 and runtime.required_rgb_history == 31
    assert runtime.metadata['rgb_history_span_s'] == 0.5
    assert runtime.required_ft_history == 41
    for schema in ('context_rgb_force_4state_v1', 'context_rgb_force_4state_ft_features_v2'):
        torch.save(dict(payload, schema=schema), path)
        with pytest.raises(ValueError, match='contracts differ'):
            RGBForceContextRuntime(path)


@pytest.mark.parametrize('canonical_files', [48], indirect=True)
def test_four_image_training_live_parity_and_episode_padding(canonical_files, four_frame_checkpoint):
    runtime = RGBForceContextRuntime(four_frame_checkpoint[0])
    ds = CanonicalRGBForceDataset(**canonical_files, rgb_horizon=4, rgb_stride=10,
                                  ft_feature_mode='raw_mean_delta')
    start, frame = 48, 48 + 40
    for index in range(start, frame + 1):
        rgb, ft = ds.history_indices(index)
        np.testing.assert_array_equal(rgb, np.maximum(index - np.array([30, 20, 10, 0]), start))
        assert ft.min() >= ds.force_starts[1]
        assert np.all(ds.force_times[ft] <= ds.rgb_times[index])
    first_rgb, _ = ds.history_indices(start)
    np.testing.assert_array_equal(first_rgb, [start] * 4)
    ids = np.arange(start, frame + 1)
    ft_start, ft_end = ds.force_starts[1], ds.force_ends[1]
    prepared, timing = runtime.prepare(ds.rgb_times[ids], [ds._get_rgb()[int(i)] for i in ids],
        ds.force_times[ft_start:ft_end], ds.wrench[ft_start:ft_end], now=ds.rgb_times[frame])
    train = ds[int(np.searchsorted(ds.indices, frame))]['obs']
    np.testing.assert_array_equal(prepared['camera0_rgb'].transpose(0, 3, 1, 2) / np.float32(255), train['camera0_rgb'])
    for key in ('robot0_ft_left', 'robot0_ft_right'):
        np.testing.assert_array_equal(prepared[key], train[key])
    assert timing['rgb_span_s'] == pytest.approx(0.5)
    with torch.inference_mode():
        expected = runtime.model({k: v[None] for k, v in train.items()})[0].numpy()
    np.testing.assert_allclose(runtime.predict(prepared)['logits'], expected, atol=1e-6)
    ds.close()


@pytest.mark.parametrize('target', [None, 0.5])
def test_half_second_selection_and_future_force_exclusion(target):
    rt = np.arange(49) / 60
    ft = np.arange(101) / 100
    ri, fi, timing = select_context_window(rt, ft, rgb_horizon=4, rgb_stride=10, rgb_target_span_s=target)
    np.testing.assert_array_equal(ri, [18, 28, 38, 48])
    assert ft[fi[-1]] <= rt[-1]
    assert len(fi) == 41
    assert timing['rgb_horizon'] == 4 and timing['rgb_span_s'] == pytest.approx(0.5)
    with pytest.raises(ObservationUnavailable, match='warming_up_rgb'):
        select_context_window(rt, ft, rgb_horizon=4, rgb_stride=10, rgb_target_span_s=target,
                              episode_start=rt[-1] - 0.3)


def test_four_frame_target_selection_at_30fps_and_missing_middle_frame():
    rt = np.arange(31) / 30
    ft = np.arange(101) / 100
    ri, _, _ = select_context_window(rt, ft, rgb_horizon=4, rgb_stride=10, rgb_target_span_s=0.5)
    np.testing.assert_array_equal(ri, [15, 20, 25, 30])
    with pytest.raises(ObservationUnavailable, match='rgb_cadence_mismatch'):
        select_context_window(rt, ft, rgb_horizon=4, rgb_stride=10)
    dropped = np.delete(rt, [19, 20, 21])
    with pytest.raises(ObservationUnavailable, match='rgb_cadence_mismatch'):
        select_context_window(dropped, ft, rgb_horizon=4, rgb_stride=10, rgb_target_span_s=0.5)


def test_policy_adapter_uses_four_images_and_excludes_future_rgb(four_frame_checkpoint):
    runtime = RGBForceValveContextRuntime(four_frame_checkpoint[0], device='cpu')
    assert runtime.rgb_target_span_s == 0.5
    runtime.reset(episode_start_timestamp_s=0)
    rt = np.arange(37) / 60
    images = np.stack([np.full((224, 224, 3), i, np.uint8) for i in range(len(rt))])
    ft = np.arange(71) / 100
    force = np.zeros((len(ft), 6), np.float32)
    record = runtime.predict_policy_anchor(timestamp_s=rt[30], rgb=images[30],
        rgb_timestamps=rt, rgb_frames=images, ft_timestamps=ft, ft_left=force, ft_right=force)
    assert record.warmed_up
    captured = runtime.get_last_classifier_model_inputs()
    np.testing.assert_array_equal(captured['camera0_rgb'], images[[0, 10, 20, 30]])
    assert captured['ft_timestamp_s'][-1] <= rt[30]
    with pytest.raises(ValueError, match='31 frames'):
        RGBForceValveContextRuntime(four_frame_checkpoint[0], device='cpu', max_rgb_buffer=16)


@pytest.mark.parametrize('canonical_files', [48], indirect=True)
def test_training_entry_saves_v3_and_rejects_changed_resume(canonical_files, four_frame_checkpoint, tmp_path, monkeypatch):
    import copy
    import json
    from omegaconf import OmegaConf
    import train_context_rgb_force as trainer
    cfg = OmegaConf.to_container(OmegaConf.load('context/qwen/config/train_context_rgb_force.yaml'), resolve=True)
    assert cfg['model']['rgb_horizon'] == cfg['dataset']['rgb_horizon'] == 4
    assert cfg['dataset']['rgb_stride'] == 10
    cfg['dataset'].update(canonical_files)
    cfg.update(output_dir=str(tmp_path / 'smoke'), device='cpu', batch_size=2, epochs=1,
               num_workers=0, amp=False, test_episode_numbers=[3, 8], val_ratio=0.2)
    # Plot generation is unrelated to the input/checkpoint contract being tested.
    monkeypatch.setattr(trainer, 'write_evaluation', lambda *a, **kw: None)
    trainer.run(cfg, smoke_batches=1)
    path = tmp_path / 'smoke/context_encoder_last.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['schema'] == 'context_rgb_force_4frame_v3'
    assert saved['state_dict']['position'].shape == (1, 7, 128)
    assert saved['training_config']['dataset']['rgb_horizon'] == 4
    assert saved['manifest']['rgb_history'] == dict(horizon=4, stride=10, span_s=0.5)
    metadata = tmp_path / 'smoke/resolved_config.yaml'
    before = metadata.read_bytes()
    changed = copy.deepcopy(cfg)
    changed['model']['rgb_horizon'] = changed['dataset']['rgb_horizon'] = 2
    changed['dataset']['rgb_stride'] = 3
    with pytest.raises(ValueError, match='do not match'):
        trainer.run(changed, resume=str(path), smoke_batches=1)
    assert metadata.read_bytes() == before
    changed = copy.deepcopy(cfg)
    changed['dataset']['rgb_horizon'] = 2
    with pytest.raises(ValueError, match='RGB horizons differ'):
        trainer.run(changed, dry_run=True)
