import json
import numpy as np
import pytest

from diffusion_policy.context.raw_recording import estimate_width_sync
from diffusion_policy.context.data import load_labeling_base, causal_summary, causal_rgb
from diffusion_policy.context.force_visual import evidence_at, settings


def test_estimate_sync_from_width_not_force():
    ct = np.arange(0, 10, .01)
    width = .1-.08*np.exp(-((ct-3)/.35)**2)-.04/(1+np.exp(-(ct-3.5)*10))
    vt = np.arange(0, 8, 1/60)
    true_shift = -.864
    visual = 1.05*np.interp(vt-true_shift, ct, width)+.06
    result = estimate_width_sync(vt, visual, ct, width)
    assert abs(result['csv_relative_to_video_shift_s']-true_shift) <= .002
    assert result['correlation'] > .999
    assert 'not_hardware_verified' in result['status']


def test_reject_sync_without_motion():
    t = np.arange(0, 8, .01)
    with pytest.raises(ValueError, match='Insufficient'):
        estimate_width_sync(t, np.full_like(t,.1), t, np.full_like(t,.1))


def test_reject_periodic_ambiguous_sync():
    ct = np.arange(0, 12, .01); vt = np.arange(0,8,1/60)
    cw = .1+.05*np.sin(ct*2*np.pi)
    vw = .1+.05*np.sin((vt+.864)*2*np.pi)
    with pytest.raises(ValueError, match='ambiguous'):
        estimate_width_sync(vt, vw, ct, cw)


@pytest.fixture
def raw_config(tmp_path):
    path = tmp_path/'clip.npz'
    rt = np.arange(24)/60; ft = np.arange(-.5,.6,.01)
    force = np.zeros((len(ft),6)); force[ft>.25,0] = 5
    np.savez(path, rgb=np.zeros((24,224,224,3),dtype=np.uint8), rgb_timestamp_s=rt,
             wrench_timestamp_s=ft, wrench_left=force, wrench_right=force,
             gripper_width_m=np.full((24,1),.05), metadata_json=json.dumps({'force_bias_removed':True}))
    return dict(recording_path=str(path), dataset_overrides=[],
                inputs=dict(tcp_position=False,tcp_rotation=False,gripper_width=True,force_torque=True))


def test_causal_clip_adapter_with_no_fabricated_pose(raw_config):
    base, _ = load_labeling_base(raw_config)
    assert not hasattr(base,'pose_mats')
    summary, start, end = causal_summary(base,0,6,121,raw_config['inputs'])
    assert not any('rotation' in key or 'position' in key for key in summary['signals'])
    assert summary['signals']['right_force_norm_N']['current'] == [0.]
    frames, timestamps = causal_rgb(base,0,start,end,8)
    assert len(frames)==7
    assert evidence_at(base,0,6,settings({}))['contact'] is False
    assert evidence_at(base,0,23,settings({}))['contact'] is True
    base.close()


@pytest.mark.parametrize('key', ['tcp_position','tcp_rotation'])
def test_pose_inputs_are_rejected_for_raw_clip(raw_config,key):
    raw_config['inputs'][key] = True
    with pytest.raises(ValueError, match='no TCP pose'):
        load_labeling_base(raw_config)


def test_override_rejected_for_raw_clip(raw_config):
    with pytest.raises(ValueError, match='overrides'):
        load_labeling_base(raw_config,['task=umi_dual_ft'])
