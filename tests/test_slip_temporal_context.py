import copy

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from diffusion_policy.context.slip_sequence_dataset import SlipSequenceDataset, split_slip_episodes
from diffusion_policy.context.slip_feature_cache import ensure_feature_cache
from models.slip_temporal_context import SlipTemporalContextEncoder, PHASE_NAMES, SCHEMA
from umi.real_world.slip_temporal_context import SlipTemporalContextRuntime, select_slip_window
from umi.real_world.rgb_force_context import ObservationUnavailable
from test_context_rgb_force import canonical_files, VisionFixture


@pytest.fixture
def tiny_model(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr('models.rgb_force_context_encoder.timm.create_model', lambda *a, **kw: VisionFixture())
    torch.manual_seed(7)
    model = SlipTemporalContextEncoder(pretrained=False, hidden_dim=32, feedforward_dim=64, dropout=0.)
    torch.nn.init.normal_(model.temporal_correction[-1].weight, std=.03)
    return model


@pytest.fixture
def slip_files(canonical_files):
    with np.load(canonical_files['label_path']) as z:
        labels = {k: z[k] for k in z.files}
    labels.update(schema='umi_manual_314_approach_error1_error2_turning_v2', phase_names=np.array(PHASE_NAMES),
                  original_episode_numbers=np.arange(1, 13))
    per_episode = np.r_[np.full(30, -100), np.full(70, 0), np.full(70, 3), np.full(70, 1),
                        np.full(30, 0), np.full(30, 3), np.full(70, 2), np.full(30, 3)]
    assert len(per_episode) == 400
    per_episode[180] = -100  # An internal ignored frame resets temporal inputs.
    labels['state_phase'] = np.tile(per_episode, 12)
    labels['state_phase_valid'] = labels['state_phase'] >= 0
    labels.pop('state_label_source'); labels.pop('label_source_names')
    np.savez(canonical_files['label_path'], **labels)
    return canonical_files


@pytest.fixture
def recording(slip_files, tiny_model, tmp_path):
    ds = SlipSequenceDataset(**slip_files)
    with torch.inference_mode():
        feature = tiny_model.encode_images(torch.zeros(1, 3, 224, 224))[0].numpy()
    cache = tmp_path / 'features.npy'
    np.save(cache, np.tile(feature, (len(ds.rgb_times), 1)))
    ds.attach_features(cache)
    yield ds
    ds.close()


def batch_at(ds, frame):
    sample = ds[int(np.searchsorted(ds.indices, frame))]
    assert sample['frame_index'] == frame
    return sample, {k: v[None] for k, v in sample['obs'].items()}


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_ignore_excluded_from_every_rgb_ft_and_history(recording):
    ds = recording
    for frame in ds.indices:
        anchors, rgb, force, valid, ages = ds.sequence_indices(frame)
        assert valid[-1] and ages[-1] == 0
        assert ds.valid[rgb[valid]].all()
        assert np.all(rgb[valid] >= ds.run_starts[frame])
        assert np.all(ds.force_times[force[valid]] <= ds.rgb_times[rgb[valid]][..., None])
        assert np.all(ds.force_times[force[valid]] >= ds.rgb_times[ds.run_starts[frame]])
        assert np.all(ds.frame_episode[rgb[valid]] == ds.frame_episode[frame])
        assert np.all(ages[:-1][valid[:-1]] > 0)
        if valid[:-1].any():
            assert np.max(ages[:-1][valid[:-1]]) <= 2.0021
    assert not ds.usable[:84].any()
    assert not ds.usable[180:235].any()
    sample, obs = batch_at(ds, 240)
    assert sample['obs']['sequence_valid'][:-1].sum() == 0
    assert set(obs) == {'rgb_features', 'robot0_ft_left', 'robot0_ft_right', 'sequence_valid', 'sequence_ages'}
    assert obs['robot0_ft_left'].shape == (1, 21, 4, 41, 6)
    splits = split_slip_episodes(ds)
    assert sorted(sum(splits.values(), [])) == list(range(12))
    assert set(splits['train']).isdisjoint(splits['validation'])
    assert set(splits['train']).isdisjoint(splits['test'])


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_force_statistics_never_use_ignore_or_held_out(recording):
    before = recording.force_statistics([0])
    initial_ignore = recording.force_times < recording.rgb_times[30]
    held_out = np.arange(len(recording.force_times)) >= recording.force_ends[0]
    recording.wrench[initial_ignore | held_out] = 1e8
    after = recording.force_statistics([0])
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_backward_history_effect_and_checkpoint_contract(recording, tiny_model):
    ds, model = recording, tiny_model
    model.set_force_statistics(*ds.force_statistics([0]))
    sample, obs = batch_at(ds, 396)
    model.train()
    result = model(obs)
    loss = (torch.nn.functional.cross_entropy(result['logits'], torch.tensor([sample['context_label']]))
            + torch.nn.functional.cross_entropy(result['observation_logits'].reshape(-1, 4), sample['sequence_labels']))
    loss.backward()
    assert model.history_gru.weight_ih.grad.abs().sum() > 0
    assert model.left_force.network[0].conv.weight.grad.abs().sum() > 0
    assert model.visual_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.vision.parameters())
    model.eval()
    with torch.no_grad():
        current = {k: v[:, -1] for k, v in obs.items() if k in ('rgb_features', 'robot0_ft_left', 'robot0_ft_right')}
        f, base = model.observe(current)
        p = torch.full((1, 20, 4), .25, requires_grad=True)
        valid, ages = obs['sequence_valid'][:, :-1], obs['sequence_ages'][:, :-1]
        without = model.fuse(f, base, p, torch.zeros_like(valid), ages)
        torch.testing.assert_close(without, base)
        with_history = model.fuse(f, base, p, valid, ages)
        changed = p.detach().clone(); changed[..., 0], changed[..., 1:] = 1, 0
        assert not torch.allclose(with_history, model.fuse(f, base, changed, valid, ages))
    invalid_ages = ages.clone(); invalid_ages[valid] = 0
    with pytest.raises(ValueError, match='strictly past'):
        model.fuse(f, base, p, valid, invalid_ages)
    payload = dict(schema=SCHEMA, phase_names=list(PHASE_NAMES), model_config=model.config, state_dict=model.state_dict())
    restored = SlipTemporalContextEncoder.from_checkpoint(payload).eval()
    torch.testing.assert_close(restored(obs)['logits'], model(obs)['logits'])
    legacy_config = dict(model.config)
    legacy_config.pop('rgb_span_s')
    legacy = SlipTemporalContextEncoder.from_checkpoint(dict(payload, model_config=legacy_config)).eval()
    torch.testing.assert_close(legacy(obs)['logits'], model(obs)['logits'])
    with pytest.raises(ValueError, match='class order'):
        SlipTemporalContextEncoder.from_checkpoint(dict(payload, phase_names=['approach', 'turning', 'recovery', 'error']))
    with pytest.raises(ValueError, match='5-sample'):
        SlipTemporalContextEncoder(pretrained=False, ft_mean_window=7, ft_delta_lag=3)


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_gru_remains_fp32_inside_bfloat16_autocast(recording, tiny_model):
    _, obs = batch_at(recording, 396)
    tiny_model.set_force_statistics(*recording.force_statistics([0]))
    dtypes = []
    handle = tiny_model.history_gru.register_forward_pre_hook(
        lambda module, inputs: dtypes.append(tuple(x.dtype for x in inputs)))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = tiny_model(obs)
    result['logits'].float().sum().backward()
    handle.remove()
    assert len(dtypes) == 20 and all(pair == (torch.float32, torch.float32) for pair in dtypes)
    assert torch.isfinite(result['logits']).all()


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_live_sequence_training_parity_and_resets(recording, tiny_model, tmp_path):
    ds, model = recording, tiny_model.eval()
    model.set_force_statistics(*ds.force_statistics([0]))
    path = tmp_path / 'model.pt'
    torch.save(dict(schema=SCHEMA, phase_names=list(PHASE_NAMES), model_config=model.config,
                    state_dict=model.state_dict()), path)
    runtime = SlipTemporalContextRuntime(path)
    runtime.reset(ds.rgb_times[181])
    for frame in ds.indices[(ds.indices >= 240) & (ds.indices < 400)]:
        sample, obs = batch_at(ds, int(frame))
        current = {k: v[-1] for k, v in sample['obs'].items() if k in ('rgb_features', 'robot0_ft_left', 'robot0_ft_right')}
        result = runtime.predict_observation(current, ds.rgb_times[frame])
        with torch.no_grad():
            expected = model(obs)['logits'].float().softmax(-1)[0].numpy()
        np.testing.assert_allclose(result['probabilities'], expected, atol=2e-6)
        assert result['history_samples'] == int(sample['obs']['sequence_valid'][:-1].sum())
    assert not np.allclose(result['probabilities'], result['observation_probabilities'])
    np.testing.assert_allclose(runtime.history[-1][1], result['observation_probabilities'])
    with pytest.raises(ValueError, match='increase'):
        runtime.predict_observation(current, ds.rgb_times[frame])
    result = runtime.predict_observation(current, ds.rgb_times[frame] + .3)
    assert result['history_samples'] == 0
    runtime.reset(ds.rgb_times[181])
    assert len(runtime.history) == 0
    rt = ds.rgb_times[:400]
    ft = ds.force_times[:ds.force_ends[0]]
    anchor = rt[300]
    ri, fi = select_slip_window(rt, ft, anchor, task_start=rt[181])
    np.testing.assert_array_equal(ri, [270, 280, 290, 300])
    assert (ft[fi] <= rt[ri, None]).all()
    expected_rgb, expected_ft = ds.sequence_indices(300)[1:3]
    np.testing.assert_array_equal(fi, expected_ft[-1])
    frames = np.zeros((400, 224, 224, 3), np.uint8)
    streamed = runtime.predict(rt, frames, ft, ds.wrench[:len(ft)], anchor_timestamp=anchor, now=anchor)
    assert streamed['history_samples'] == 0
    with pytest.raises(ObservationUnavailable, match='paired_ft'):
        select_slip_window(rt, ft, rt[210], task_start=rt[181])
    with pytest.raises(ObservationUnavailable, match='inactive'):
        runtime.predict(rt, frames, ft, ds.wrench[:len(ft)], anchor_timestamp=anchor, active=False)
    assert not runtime.history


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_cache_build_reuse_and_changed_data_rejection(recording, tiny_model, tmp_path):
    # Exercise the real writer with a few images; the remaining rows are unused.
    recording.valid[:] = False
    recording.valid[300:304] = True
    path = tmp_path / 'cache'
    manifest = ensure_feature_cache(recording, tiny_model, path, torch.device('cpu'), workers=0, batch_size=2, create=True)
    stored = np.load(path / 'features.npy', mmap_mode='r')
    with torch.no_grad():
        expected = tiny_model.encode_images(torch.zeros(4, 3, 224, 224)).numpy()
    np.testing.assert_allclose(stored[300:304], expected, atol=1e-6)
    assert ensure_feature_cache(recording, tiny_model, path, torch.device('cpu')) == manifest
    recording.fingerprint = 'different'
    with pytest.raises(ValueError, match='different recording'):
        ensure_feature_cache(recording, tiny_model, path, torch.device('cpu'))


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_full_training_entry_smoke_and_resume(recording, tiny_model, tmp_path, monkeypatch):
    import train_slip_context as trainer
    cfg = OmegaConf.to_container(OmegaConf.load('context/qwen/config/train_slip_context.yaml'), resolve=True)
    cfg['dataset'].update(dataset_path=recording.dataset_path, force_sidecar_path=recording.force_sidecar_path,
                          label_path=recording.label_path, history_steps=recording.history_steps,
                          rgb_horizon=recording.rgb_horizon, rgb_span_s=recording.rgb_span_s)
    cfg['model'] = tiny_model.config.copy()
    cfg.update(output_dir=str(tmp_path / 'run'), cache_dir=str(tmp_path / 'cache'),
               device='cpu', amp=False, num_workers=0, batch_size=2, epochs=1)
    def cached(ds, model, *a, **kw):
        model.vision.load_state_dict(tiny_model.vision.state_dict())
        ds.attach_features(recording.feature_path)
        return {'test_cache': True}
    monkeypatch.setattr(trainer, 'ensure_feature_cache', cached)
    monkeypatch.setattr(trainer, 'write_evaluation', lambda *a, **kw: None)
    report = trainer.train(cfg, smoke_batches=1)
    path = tmp_path / 'run/best.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['schema'] == SCHEMA and saved['phase_names'] == list(PHASE_NAMES)
    assert saved['smoke_test'] and 'observation_only' in saved['validation_metrics']
    assert saved['state_dict']['position'].shape == (1, 13, 32)
    assert report['ignored_frames'] == 12 * 31
    with pytest.raises(ValueError, match='Smoke-test'):
        SlipTemporalContextRuntime(path)
    cfg['epochs'] = 2
    trainer.train(cfg, resume=str(tmp_path / 'run/last.pt'), smoke_batches=1)
    changed = copy.deepcopy(cfg); changed['learning_rate'] *= 2
    with pytest.raises(ValueError, match='Resume changed'):
        trainer.train(changed, resume=str(tmp_path / 'run/last.pt'), smoke_batches=1)


@pytest.fixture
def eight_model(tiny_model):
    model = SlipTemporalContextEncoder(pretrained=False, hidden_dim=32, feedforward_dim=64, dropout=0.,
                                       rgb_horizon=8, rgb_span_s=1., history_steps=10)
    torch.nn.init.normal_(model.temporal_correction[-1].weight, std=.03)
    return model


@pytest.fixture
def eight_recording(slip_files, eight_model, tmp_path):
    with np.load(slip_files['label_path'], allow_pickle=False) as z:
        labels = {k: z[k] for k in z.files}
    targets = np.r_[np.full(30, -100), np.full(110, 0), np.full(80, 3), np.full(70, 1),
                    np.full(20, 0), np.full(20, 3), np.full(50, 2), np.full(20, 3)]
    targets[180] = -100
    labels['state_phase'] = np.tile(targets, 12)
    labels['state_phase_valid'] = labels['state_phase'] >= 0
    np.savez(slip_files['label_path'], **labels)
    ds = SlipSequenceDataset(**slip_files, rgb_horizon=8, rgb_span_s=1., history_steps=10)
    with torch.no_grad():
        feature = eight_model.encode_images(torch.zeros(1, 3, 224, 224))[0].numpy()
    cache = tmp_path / 'eight_features.npy'
    np.save(cache, np.tile(feature, (len(ds.rgb_times), 1)))
    ds.attach_features(cache)
    yield ds
    ds.close()


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_eight_rgb_alignment_ignore_and_train_statistics(eight_recording):
    ds = eight_recording
    np.testing.assert_array_equal(ds.rgb_offsets, [60, 51, 43, 34, 26, 17, 9, 0])
    for frame in ds.indices:
        anchors, rgb, ft, valid, ages = ds.sequence_indices(frame)
        assert rgb.shape == (11, 8) and ft.shape == (11, 8, 41)
        assert np.all(ds.valid[rgb[valid]])
        assert np.all(ds.frame_episode[rgb[valid]] == ds.frame_episode[frame])
        assert np.all(ds.force_times[ft[valid]] >= ds.rgb_times[ds.run_starts[frame]])
        assert np.all(ds.force_times[ft[valid]] <= ds.rgb_times[rgb[valid]][..., None])
        assert np.all((ages[:-1][valid[:-1]] > 0) & (ages[:-1][valid[:-1]] <= 1.0011))
    assert not ds.usable[180:265].any()
    sample, obs = batch_at(ds, 270)
    assert sample['obs']['sequence_valid'][:-1].sum() == 0
    assert obs['robot0_ft_left'].shape == (1, 11, 8, 41, 6)
    assert obs['rgb_features'].shape == (1, 11, 8, 768)
    rt, ft = ds.rgb_times[:400], ds.force_times[:ds.force_ends[0]]
    ri, fi = select_slip_window(rt, ft, rt[330], task_start=rt[181], rgb_horizon=8, rgb_span_s=1.)
    expected_rgb, expected_ft = ds.history_indices(330)
    np.testing.assert_array_equal(ri, expected_rgb)
    np.testing.assert_array_equal(fi, expected_ft)
    with pytest.raises(ObservationUnavailable, match='paired_ft'):
        select_slip_window(rt, ft, rt[250], task_start=rt[181], rgb_horizon=8, rgb_span_s=1.)
    before = ds.force_statistics([0])
    ds.wrench[(ds.force_times < rt[30]) | (np.arange(len(ds.force_times)) >= ds.force_ends[0])] = 1e8
    after = ds.force_statistics([0])
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_eight_rgb_backward_checkpoint_and_streaming_parity(eight_recording, eight_model, tmp_path):
    ds, model = eight_recording, eight_model
    model.set_force_statistics(*ds.force_statistics([0]))
    sample, obs = batch_at(ds, 396)
    assert model.position.shape == (1, 25, 32)
    result = model(obs)
    loss = torch.nn.functional.cross_entropy(result['logits'], torch.tensor([sample['context_label']]))
    loss = loss + torch.nn.functional.cross_entropy(result['observation_logits'].reshape(-1, 4), sample['sequence_labels'])
    loss.backward()
    assert model.history_gru.weight_ih.grad.abs().sum() > 0
    assert model.left_force.network[0].conv.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.vision.parameters())
    path = tmp_path / 'eight.pt'
    torch.save(dict(schema=SCHEMA, phase_names=list(PHASE_NAMES), model_config=model.config,
                    state_dict=model.state_dict()), path)
    runtime = SlipTemporalContextRuntime(path)
    assert runtime.metadata['rgb_horizon'] == 8
    assert runtime.metadata['startup_sensor_span_s'] == pytest.approx(1.4)
    runtime.reset(ds.rgb_times[181])
    model.eval()
    counts = []
    for frame in ds.indices[(ds.indices >= 270) & (ds.indices < 400)]:
        sample, obs = batch_at(ds, int(frame))
        current = {k: v[-1] for k, v in sample['obs'].items()
                   if k in ('rgb_features', 'robot0_ft_left', 'robot0_ft_right')}
        result = runtime.predict_observation(current, ds.rgb_times[frame])
        with torch.no_grad():
            expected = model(obs)['logits'].float().softmax(-1)[0].numpy()
        np.testing.assert_allclose(result['probabilities'], expected, atol=2e-6)
        counts.append(result['history_samples'])
    assert counts[0] == 0 and counts[1] == 1 and counts[-1] == 10
    assert len(runtime.history) <= 12
    runtime.reset(ds.rgb_times[181])
    ft = ds.force_times[:ds.force_ends[0]]
    result = runtime.predict(ds.rgb_times[:400], np.zeros((400,224,224,3), np.uint8), ft,
                             ds.wrench[:len(ft)], anchor_timestamp=ds.rgb_times[330])
    assert result['history_samples'] == 0 and len(result['rgb_timestamp_s']) == 8
    np.testing.assert_allclose(result['probabilities'], result['observation_probabilities'])


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_eight_rgb_training_config_and_smoke(eight_recording, eight_model, tmp_path, monkeypatch):
    import train_slip_context as trainer
    ds = eight_recording
    cfg = OmegaConf.to_container(OmegaConf.load('context/qwen/config/train_slip_context.yaml'), resolve=True)
    assert cfg['model']['rgb_horizon'] == 8 and cfg['model']['history_steps'] == 10
    cfg['dataset'].update(dataset_path=ds.dataset_path, force_sidecar_path=ds.force_sidecar_path,
                          label_path=ds.label_path)
    cfg['model'] = eight_model.config.copy()
    cfg.update(output_dir=str(tmp_path/'eight_run'), cache_dir=str(tmp_path/'cache'),
               device='cpu', amp=False, num_workers=0, batch_size=2, epochs=1)
    def cached(recording, model, *a, **kw):
        model.vision.load_state_dict(eight_model.vision.state_dict())
        recording.attach_features(ds.feature_path)
        return {'test_cache': True}
    monkeypatch.setattr(trainer, 'ensure_feature_cache', cached)
    monkeypatch.setattr(trainer, 'write_evaluation', lambda *a, **kw: None)
    report = trainer.train(cfg, smoke_batches=1)
    saved = torch.load(tmp_path/'eight_run/best.pt', map_location='cpu', weights_only=False)
    assert saved['state_dict']['position'].shape == (1, 25, 32)
    assert saved['model_config']['history_steps'] == 10
    assert report['rgb_offsets_frames'] == [60,51,43,34,26,17,9,0]
    changed = copy.deepcopy(cfg); changed['model']['rgb_span_s'] = .5
    with pytest.raises(ValueError, match='RGB window contracts differ'):
        trainer.train(changed, dry_run=True)
