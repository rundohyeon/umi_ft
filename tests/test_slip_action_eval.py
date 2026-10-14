from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from eval_slip_action import (
    action_errors, summarize_errors, select_indices, validate_dataset_binding,
    validate_normalizer, evaluate_predictions, save_reports,
)


def identity_actions(batch=2):
    target = np.zeros((batch, 16, 11), np.float32)
    target[..., 3] = 1; target[..., 7] = 1
    return target


def test_physical_units_and_rotation_angle():
    gt = identity_actions()
    pred = gt.copy()
    pred[..., :3] = [.003, .004, 0]
    pred[..., 3:9] = [0, -1, 0, 1, 0, 0]
    pred[..., 9] = .002; pred[..., 10] = -3
    errors = action_errors(pred, gt)
    np.testing.assert_allclose(errors['position_l2_mm'], 5, atol=1e-6)
    np.testing.assert_allclose(errors['rotation_deg'], 90, atol=1e-6)
    np.testing.assert_allclose(errors['gripper_width_abs_mm'], 2, atol=1e-6)
    np.testing.assert_allclose(errors['force_abs_N'], 3, atol=1e-6)
    assert not errors['invalid_rotation'].any()


def test_invalid_rotations_and_nonfinite_actions_are_not_hidden():
    gt = identity_actions(); pred = gt.copy()
    pred[..., 3:9] = 0
    errors = action_errors(pred, gt)
    assert errors['invalid_rotation'].all() and np.isnan(errors['rotation_deg']).all()
    with pytest.raises(ValueError, match='demonstrated rotation'):
        action_errors(gt, pred)
    pred[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        action_errors(pred, gt)


def test_future_summary_excludes_current_anchor_and_preserves_repeats():
    gt = identity_actions()[None].repeat(3, axis=0)
    pred = gt.copy(); pred[..., 0, 0] = 10
    errors = action_errors(pred, gt)
    report = summarize_errors(errors, np.array([True, False]))
    assert report['anchors'] == 1 and report['prediction_repeats'] == 3
    assert report['position_l2_mm']['count'] == 45
    assert report['position_l2_mm']['mean'] == 0
    current = summarize_errors(errors, steps=slice(0, 1))
    assert current['position_l2_mm']['mean'] == 10000
    empty = summarize_errors(errors, np.array([False, False]))
    assert empty['position_l2_mm']['mean'] is None


def test_grid_uses_episode_local_frame_numbers_and_original_episode_ids():
    ds = SimpleNamespace(indices=np.array([6, 7, 12, 26, 27, 32]),
        recording=SimpleNamespace(rgb_starts=np.array([0, 20]),
            frame_episode=np.r_[np.zeros(20, int), np.ones(20, int)],
            original_episode_numbers=np.array([291, 295])))
    selected, total = select_indices(ds)
    np.testing.assert_array_equal(selected, [0, 2, 3, 5]); assert total == 4
    selected, total = select_indices(ds, episode=295)
    np.testing.assert_array_equal(ds.indices[selected], [26, 32]); assert total == 2
    selected, total = select_indices(ds, max_samples=2)
    np.testing.assert_array_equal(selected, [0, 5]); assert total == 4
    with pytest.raises(ValueError, match='No eligible'):
        select_indices(ds, episode=1)


def test_checkpoint_binds_class_order_data_and_splits():
    expected = {k: 'same' for k in ('schema', 'phase_names', 'source_fingerprint', 'checkpoint_sha256',
        'episode_splits', 'model_config', 'observation_contract', 'history_source',
        'prediction_source', 'context_cadence_frames')}
    ds = SimpleNamespace(action_training_metadata=dict(expected))
    validate_dataset_binding(expected, ds)
    for key in ('episode_splits', 'source_fingerprint', 'checkpoint_sha256', 'phase_names'):
        ds.action_training_metadata = dict(expected, **{key: 'changed'})
        with pytest.raises(ValueError, match=key): validate_dataset_binding(expected, ds)


def test_normalizer_needs_pose_free_fields_and_identity_context():
    params = {k: dict(scale=torch.ones(n), offset=torch.zeros(n)) for k, n in
              dict(camera0_rgb=1, robot0_ft_left=6, robot0_ft_right=6, valve_context=1, action=11).items()}
    policy = SimpleNamespace(normalizer=SimpleNamespace(params_dict=params))
    validate_normalizer(policy)
    params['valve_context']['offset'] += .1
    with pytest.raises(ValueError, match='identity'): validate_normalizer(policy)
    params['valve_context']['offset'].zero_(); params['action']['scale'][0] = 0
    with pytest.raises(ValueError, match='Nonpositive'): validate_normalizer(policy)
    params['action']['scale'][0] = 1; params['robot0_eef_pos'] = {}
    with pytest.raises(ValueError, match='pose-free'): validate_normalizer(policy)


def test_evaluation_reproducibility_repeats_and_no_input_label():
    class FakePolicy:
        def compute_loss(self, batch): return torch.tensor(.25)
        def predict_action(self, obs):
            assert set(obs) == {'camera0_rgb', 'robot0_ft_left', 'robot0_ft_right', 'valve_context'}
            out = torch.from_numpy(identity_actions(len(obs['camera0_rgb'])))
            out[..., :3] += torch.randn_like(out[..., :3]) * .01
            return dict(action_pred=out)
    samples = [dict(obs=dict(camera0_rgb=torch.zeros(4,3,8,8),
                            robot0_ft_left=torch.zeros(4,41,6), robot0_ft_right=torch.zeros(4,41,6),
                            valve_context=torch.tensor([[.1,.2,.3,.4,1.]])),
                    action=torch.from_numpy(identity_actions(1)[0])) for _ in range(3)]
    loader = DataLoader(samples, batch_size=2)
    a, target, metrics = evaluate_predictions(FakePolicy(), loader, torch.device('cpu'), repeats=2, seed=7)
    b, _, _ = evaluate_predictions(FakePolicy(), loader, torch.device('cpu'), repeats=2, seed=7)
    assert a.shape == (2,3,16,11) and target.shape == (3,16,11)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[0], a[1])
    assert metrics['diffusion_loss_no_augmentation'] == .25


def test_saved_report_keeps_predictions_and_empty_phase_counts(tmp_path):
    gt = identity_actions()
    ds = SimpleNamespace(recording=SimpleNamespace(frame_episode=np.array([0,1]),
        original_episode_numbers=np.array([291,295]), targets=np.array([0,2]), rgb_times=np.array([1.,2.])),
        context=np.tile([.1,.2,.3,.4,1.],(2,1)), context_source=np.array([0,1]))
    report = save_reports(tmp_path, dict(epoch_1based=9, weights='ema_model', split='test', anchor_stride=6),
                          gt[None], gt, np.array([0,1]), ds)
    assert report['evaluated_episodes'] == 2 and report['evaluated_samples'] == 2
    assert report['per_phase']['error1']['anchors'] == 0
    assert report['future_steps_1_to_15']['position_l2_mm']['mean'] == 0
    saved = np.load(tmp_path / 'predictions.npz')
    np.testing.assert_array_equal(saved['predicted_actions'], gt[None])
    assert saved['horizon_seconds'][0] == 0
    assert (tmp_path / 'errors_by_horizon.png').stat().st_size > 0
