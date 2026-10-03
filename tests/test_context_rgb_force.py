import numpy as np
import pytest
import torch
from torch import nn
import zarr

from diffusion_policy.context.canonical_dataset import CanonicalRGBForceDataset, canonical_episode_splits
from models.rgb_force_context_encoder import RGBForceContextEncoder
from diffusion_policy.context.ft_features import causal_ft_features


@pytest.fixture
def canonical_files(tmp_path):
    episodes, per_episode, force_per_episode = 12, 8, 16
    n = episodes * per_episode
    ends = np.arange(1, episodes + 1) * per_episode
    force_ends = np.arange(1, episodes + 1) * force_per_episode
    rt = np.concatenate([e * 10 + .005 + np.arange(per_episode) / 60 for e in range(episodes)])
    ft = np.concatenate([e * 10 + np.arange(force_per_episode) / 100 for e in range(episodes)])
    index = np.searchsorted(ft, rt, side='right') - 1
    names = np.array([f'episode_{e}' for e in range(episodes)])
    rgb_path, force_path, label_path = (tmp_path / n for n in ('rgb.zarr', 'force.zarr', 'labels.npz'))
    rgb = zarr.open_group(str(rgb_path), mode='w')
    rgb.create_dataset('meta/episode_ends', data=ends)
    rgb.create_dataset('data/camera0_rgb', shape=(n, 224, 224, 3), dtype='u1', chunks=(1, 224, 224, 3))
    force = zarr.open_group(str(force_path), mode='w')
    force.attrs.update(schema='umi_force_sidecar_v1', train_wrench_key='wrench_12d',
        wrench_channel_order=[f'{axis}_{side}' for side in ('l', 'r') for axis in ('fx','fy','fz','tx','ty','tz')])
    arrays = {'meta/rgb_episode_ends': ends, 'meta/source_episode': names,
        'data/rgb_timestamp_s': rt, 'data/wrench_timestamp_s': ft,
        'meta/wrench_episode_ends': force_ends, 'data/rgb_to_wrench_end_idx': index,
        'data/rgb_wrench_valid': np.ones(n, dtype=bool), 'data/rgb_wrench_age_s': rt-ft[index],
        'data/wrench_12d': np.arange(episodes*force_per_episode*12, dtype=np.float32).reshape(-1,12)}
    for key, values in arrays.items():
        force.create_dataset(key, data=values)
    targets = np.tile(np.repeat(np.arange(4), 2), episodes)
    valid = np.ones(n, dtype=np.uint8); valid[3] = 0
    np.savez(label_path, schema='valve_context_manual_v12_supervision_v2_4state',
        training_eligible=True, phase_names=['approach','turning','recovery','error'],
        state_phase=targets, state_phase_valid=valid,
        state_label_source=np.repeat(np.arange(episodes) % 3, per_episode),
        label_source_names=['v12_001_193','manual_194_214','manual_215_284'],
        rgb_timestamp_s=rt, rgb_episode_ends=ends, source_episode=names,
        episode_is_approved=np.ones(episodes, dtype=np.uint8))
    return dict(dataset_path=str(rgb_path), force_sidecar_path=str(force_path), label_path=str(label_path))


def test_canonical_no_tcp_causal_alignment_and_train_only_stats(canonical_files):
    ds = CanonicalRGBForceDataset(**canonical_files)
    assert len(ds) == 95  # Invalid labels are excluded, not converted to approach.
    sample = ds[8]
    assert set(sample['obs']) == set(RGBForceContextEncoder.input_keys)
    assert sample['obs']['camera0_rgb'].shape == (2,3,224,224)
    for frame in ds.indices:
        rgb, ft = ds.history_indices(frame)
        ep = ds.frame_episode[frame]
        assert np.all(ds.rgb_times[rgb] <= ds.rgb_times[frame])
        assert np.all(ds.force_times[ft] <= ds.rgb_times[frame])
        assert rgb.min() >= ds.rgb_starts[ep] and ft.min() >= ds.force_starts[ep]
        assert ds.targets[frame] == ds[ds.indices.searchsorted(frame)]['context_label']
    mean, std = ds.force_statistics([0])
    ds.wrench[ds.force_ends[0]:] = 1e8
    mean2, std2 = ds.force_statistics([0])
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(std, std2)
    assert ds.sample_weights[0] == pytest.approx(.3)
    assert ds.sample_weights[8] == pytest.approx(1.)
    splits = canonical_episode_splits(12, 42, .2, [3,8])
    assert splits['test'] == [2,7]
    assert sorted(sum(splits.values(), [])) == list(range(12))
    for key, episodes in splits.items():
        assert set(ds.frame_episode[ds.subset(episodes).indices]) <= set(episodes)


def test_canonical_rejects_future_force_and_wrong_label_clock(canonical_files):
    force = zarr.open_group(canonical_files['force_sidecar_path'], mode='r+')
    original = force['data/rgb_to_wrench_end_idx'][1]
    force['data/rgb_to_wrench_end_idx'][1] = original + 1
    with pytest.raises(ValueError, match='Noncausal'):
        CanonicalRGBForceDataset(**canonical_files)
    force['data/rgb_to_wrench_end_idx'][1] = original
    with np.load(canonical_files['label_path']) as saved:
        labels = {k: saved[k] for k in saved.files}
    labels['rgb_timestamp_s'][2] += .001
    np.savez(canonical_files['label_path'], **labels)
    with pytest.raises(ValueError, match='timestamps differ'):
        CanonicalRGBForceDataset(**canonical_files)


class VisionFixture(nn.Module):
    num_features = 768
    pretrained_cfg = dict(mean=(.48,.46,.4), std=(.27,.26,.28))

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 768)

    def forward(self, images):
        return self.projection(images.mean((-1,-2)))


@pytest.mark.parametrize('feature_mode', ['raw', 'raw_mean_delta'])
def test_multimodal_gradients_frozen_vision_no_pose_and_checkpoint(monkeypatch, feature_mode):
    torch.set_num_threads(2)
    monkeypatch.setattr('models.rgb_force_context_encoder.timm.create_model', lambda *a, **kw: VisionFixture())
    model = RGBForceContextEncoder(ft_feature_mode=feature_mode).train()
    obs = dict(camera0_rgb=torch.rand(2,2,3,224,224),
        robot0_ft_left=torch.randn(2,model.required_ft_history,6),
        robot0_ft_right=torch.randn(2,model.required_ft_history,6))
    logits = model(obs)
    assert logits.shape == (2,4)
    nn.functional.cross_entropy(logits, torch.tensor([0,3])).backward()
    assert not model.vision.training and all(p.grad is None for p in model.vision.parameters())
    for module in (model.visual_projection, model.left_force, model.right_force, model.transformer, model.classifier):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0
    model.eval()
    expected = model(obs)
    obs['robot0_eef_pos'] = torch.full((2,2,3), float('nan'))
    obs['robot0_eef_rot_axis_angle'] = torch.full((2,2,6), float('nan'))
    torch.testing.assert_close(model(obs), expected, rtol=0, atol=0)
    restored = RGBForceContextEncoder.from_checkpoint(dict(model_config=model.config, state_dict=model.state_dict())).eval()
    torch.testing.assert_close(restored(obs), expected, rtol=0, atol=0)
    # Every sample of the 32-point causal history must reach the CNN output.
    history = torch.randn(1,32,model.ft_channels, requires_grad=True)
    model.left_force(history).sum().backward()
    assert torch.all(history.grad.abs().sum(-1) > 0)


def test_ft_features_constant_ramp_and_causal_step():
    constant = torch.ones(2,41,6) * 7
    features = causal_ft_features(constant, 'raw_mean_delta')
    assert features.shape == (2,32,18)
    torch.testing.assert_close(features[...,:12], torch.full((2,32,12),7.))
    assert torch.count_nonzero(features[...,12:]) == 0
    ramp = torch.arange(41, dtype=torch.float32)[None,:,None].expand(1,-1,6)
    features = causal_ft_features(ramp, 'raw_mean_delta')
    torch.testing.assert_close(features[...,:6], ramp[:,9:])
    torch.testing.assert_close(features[...,6:12], ramp[:,9:] - 2)
    torch.testing.assert_close(features[...,12:], torch.full((1,32,6),5.))
    # A step in the last sample cannot affect any preceding output timestep.
    step = torch.zeros(1,41,6); step[:,-1] = 10
    features = causal_ft_features(step, 'raw_mean_delta')
    assert torch.count_nonzero(features[:,:-1]) == 0
    torch.testing.assert_close(features[:,-1,:6], torch.full((1,6),10.))
    torch.testing.assert_close(features[:,-1,6:], torch.full((1,12),2.))
    shorter = causal_ft_features(ramp[:,:30], 'raw_mean_delta')
    changed_future = ramp.clone(); changed_future[:,30:] = 999
    torch.testing.assert_close(causal_ft_features(changed_future, 'raw_mean_delta')[:,:21], shorter)


def test_feature_windows_episode_padding_and_normalization(canonical_files):
    ds = CanonicalRGBForceDataset(**canonical_files, ft_feature_mode='raw_mean_delta')
    assert ds.required_ft_history == 41
    ep = 1
    start, end = ds.force_starts[ep], ds.force_ends[ep]
    native = ds.wrench[start:end]
    padded = torch.from_numpy(np.pad(native, ((9,0),(0,0)), mode='edge')).unsqueeze(0)
    full_features = torch.cat([causal_ft_features(padded[...,:6], 'raw_mean_delta'),
                               causal_ft_features(padded[...,6:], 'raw_mean_delta')], dim=-1)[0]
    for frame in ds.indices[ds.frame_episode[ds.indices] == ep]:
        _, indices = ds.history_indices(frame)
        assert len(indices) == 41 and indices.min() >= start
        raw = torch.from_numpy(ds.wrench[indices]).unsqueeze(0)
        latest_features = torch.cat([causal_ft_features(raw[...,:6], 'raw_mean_delta'),
                                    causal_ft_features(raw[...,6:], 'raw_mean_delta')], dim=-1)[0]
        expected_indices = np.maximum(ds.force_index[frame] - np.arange(31,-1,-1), start) - start
        torch.testing.assert_close(latest_features, full_features[expected_indices])
    mean, std = ds.force_statistics([ep])
    np.testing.assert_allclose(mean, full_features.mean(0).numpy(), atol=1e-5)
    np.testing.assert_allclose(std, full_features.std(0, unbiased=False).clamp_min(1e-4).numpy(), atol=1e-5)
    assert mean.shape == std.shape == (36,)
    ds.wrench[:start] = -1e8; ds.wrench[end:] = 1e8
    changed_mean, changed_std = ds.force_statistics([ep])
    np.testing.assert_array_equal(mean, changed_mean)
    np.testing.assert_array_equal(std, changed_std)
    first_frame = int(ds.rgb_starts[ep])
    _, indices = ds.history_indices(first_frame)
    first_features = causal_ft_features(torch.from_numpy(ds.wrench[indices,:6])[None], 'raw_mean_delta')
    assert torch.count_nonzero(first_features[...,12:]) == 0


def test_features_stay_fp32_under_amp():
    history = torch.linspace(10,10.01,41)[None,:,None].expand(1,-1,6)
    expected = causal_ft_features(history, 'raw_mean_delta')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = causal_ft_features(history, 'raw_mean_delta')
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
