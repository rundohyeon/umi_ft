import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from diffusion_policy.context.slip_sequence_dataset import SlipSequenceDataset
from diffusion_policy.dataset.slip_action_dataset import (
    SlipActionDataset, expand_context_predictions, CACHE_SCHEMA, PHASE_NAMES, INPUT_KEYS,
)
from diffusion_policy.common.nested_zarr import sha256_file
from diffusion_policy.model.vision.dual_ft_obs_encoder import DualFTObsEncoder
from tools.prepare_slip_action_context import infer_context_anchors
from test_context_rgb_force import canonical_files
from test_slip_temporal_context import slip_files, recording, tiny_model, batch_at


@pytest.fixture
def cfg():
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'diffusion_policy/config')):
        return compose(config_name='train_slip_action_rgb4_workspace')


@pytest.fixture
def action_data(slip_files, tmp_path, cfg):
    root = zarr.open_group(slip_files['dataset_path'], mode='a')
    n = root['data/camera0_rgb'].shape[0]
    pos = np.zeros((n, 3)); pos[:, 0] = np.tile(np.arange(400), 12) * .001
    root.create_dataset('data/robot0_eef_pos', data=pos)
    root.create_dataset('data/robot0_eef_rot_axis_angle', data=np.zeros((n, 3)))
    root.create_dataset('data/robot0_gripper_width', data=np.full((n, 1), .02))
    ds = SlipSequenceDataset(**slip_files)
    checkpoint = tmp_path / 'classifier.pt'; checkpoint.write_bytes(b'test classifier provenance')
    metadata = dict(schema=CACHE_SCHEMA, phase_names=PHASE_NAMES, source_fingerprint=ds.fingerprint,
                    checkpoint_sha256=sha256_file(checkpoint),
                    episode_splits=dict(train=list(range(8)), validation=[8, 9], test=[10, 11]))
    cache = tmp_path / 'context.npz'
    np.savez(cache, metadata=json.dumps(metadata), frames=ds.indices,
             probabilities=np.tile(np.array([.1, .2, .3, .4], np.float32), (len(ds), 1)))
    ds.close()
    value = SlipActionDataset(**slip_files, shape_meta=cfg.shape_meta,
                              context_cache_path=cache, context_checkpoint_path=checkpoint)
    yield value
    value.close()


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_action_alignment_ignore_boundaries_and_target_only_pose(action_data):
    data, ds = action_data, action_data.recording
    assert set(data[0]['obs']) == set(INPUT_KEYS)
    for frame in data.indices:
        rgb, force = data.window_indices(frame)
        np.testing.assert_array_equal(rgb, frame - np.array([30, 20, 10, 0]))
        assert (ds.force_times[force] <= ds.rgb_times[rgb, None]).all()
        assert (ds.force_times[force] >= ds.rgb_times[ds.run_starts[frame]]).all()
        assert ds.valid[frame:frame + 46].all()
        assert ds.frame_episode[frame + 45] == ds.frame_episode[frame]
        assert 0 <= frame - data.context_source[frame] < 6
    sample = data[0]
    assert sample['obs']['robot0_ft_left'].shape == (4, 41, 6)
    assert sample['action'].shape == (16, 11)
    np.testing.assert_allclose(sample['action'][:, 0], np.arange(16) * .003, atol=1e-7)
    before = copy.deepcopy(sample['obs'])
    data.pose[:, :3, 3] += 1000
    # Even drastic modifications to the target pose cannot enter observations.
    for key, value in data[0]['obs'].items():
        torch.testing.assert_close(value, before[key])
    assert set(ds.frame_episode[data.get_validation_dataset().indices]) == {8, 9}


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_action_normalizer_excludes_heldout_and_ignore(action_data):
    data, ds = action_data, action_data.recording
    normalizer = data.get_normalizer()
    assert set(normalizer.params_dict) == set(INPUT_KEYS) | {'action'}
    frames = data.all_indices[np.isin(ds.frame_episode[data.all_indices], range(8))]
    rgb = np.unique((frames[:, None] - ds.rgb_offsets).ravel())
    force = np.unique((ds.force_index[rgb, None] - np.arange(40, -1, -1)).ravel())
    unused = np.ones(len(ds.wrench), bool); unused[force] = False
    ds.wrench[unused] = 1e12
    data.pose[ds.rgb_starts[8]:, :3, 3] = 1e12
    after = data.get_normalizer()
    for key, value in normalizer.state_dict().items():
        torch.testing.assert_close(value, after.state_dict()[key])


@pytest.mark.parametrize('canonical_files', [400], indirect=True)
def test_prediction_cache_matches_sequence_model_and_never_holds_across_gaps(recording, tiny_model):
    ds, model = recording, tiny_model.eval()
    model.set_force_statistics(*ds.force_statistics([0]))
    p = infer_context_anchors(model, ds, torch.device('cpu'), batch_size=64)
    for frame in (240, 300, 396):
        _, batch = batch_at(ds, frame)
        with torch.no_grad(): expected = model(batch)['logits'].softmax(-1)[0].numpy()
        np.testing.assert_allclose(p[np.searchsorted(ds.indices, frame)], expected, atol=2e-6)
    frames = np.array([174, 300, 396, 484])
    values, source = expand_context_predictions(ds, frames, np.tile([.1,.2,.3,.4], (4, 1)))
    assert source[179] == 174 and source[180] == -1 and source[181] == -1
    assert source[305] == 300 and source[306] == -1
    assert source[399] == 396 and source[400] == -1
    with pytest.raises(ValueError, match='10 Hz'):
        expand_context_predictions(ds, [301], [[.1,.2,.3,.4]])


def test_paired_encoder_gradients_and_tcp_independence(cfg):
    torch.set_num_threads(2)
    meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    meta['obs']['camera0_rgb']['shape'] = [3, 64, 64]
    encoder = DualFTObsEncoder(meta, model_name='resnet18', pretrained=False, frozen=False,
        global_pool='', transforms=None, feature_aggregation='avg', vision_feature_dim=512,
        fusion_dim=64, fusion_heads=8, fusion_feedforward_dim=128, ft_channel_dims=[8,16,32,32],
        ft_window_mode='per_rgb', ft_feature_mode='raw_mean_delta', valve_context_key='valve_context',
        valve_context_dim=5, valve_context_num_phase_experts=4)
    obs = dict(camera0_rgb=torch.rand(2,4,3,64,64), robot0_ft_left=torch.randn(2,4,41,6,requires_grad=True),
               robot0_ft_right=torch.randn(2,4,41,6,requires_grad=True),
               valve_context=torch.tensor([[[.1,.2,.3,.4,1.]]]).repeat(2,1,1))
    out = encoder(obs)
    assert out.shape == (2,64) and encoder.low_dim_output_dim == 0
    assert encoder.num_fusion_tokens == 12
    assert len(encoder.fusion_token_names()) == 12
    out.square().mean().backward()
    for key in ('robot0_ft_left', 'robot0_ft_right'):
        assert (obs[key].grad.abs().sum(-1) > 0).all()  # Every sample in every paired window matters.
    encoder.eval()
    with torch.no_grad():
        before = encoder(obs)
        obs['robot0_eef_pos'] = torch.full((2, 2, 3), float('nan'))
        torch.testing.assert_close(encoder(obs), before, rtol=0, atol=0)


def test_config_has_no_tcp_and_preserves_action_rate(cfg):
    assert set(cfg.shape_meta.obs) == set(INPUT_KEYS)
    assert cfg.task.action_frequency == pytest.approx(60000 / 1001 / 3)
    assert cfg.policy.obs_encoder.ft_window_mode == 'per_rgb'
    assert cfg.task.model_contract.condition_dim == 768
    assert cfg.training.full_validation and cfg.checkpoint.topk.monitor_key == 'val_loss'
