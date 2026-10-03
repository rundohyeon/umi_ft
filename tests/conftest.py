"""Shared fixtures for the five-class context tests."""
import pytest
import torch

@pytest.fixture
def context_meta():
    return {'obs':{'camera0_rgb':{'shape':[3,16,16],'horizon':2,'type':'rgb'},
        'robot0_eef_pos':{'shape':[3],'horizon':2,'type':'low_dim'},
        'robot0_eef_rot_axis_angle':{'shape':[6],'horizon':2,'type':'low_dim'},
        'robot0_ft_left':{'shape':[6],'horizon':32,'type':'low_dim'},
        'robot0_ft_right':{'shape':[6],'horizon':32,'type':'low_dim'}},'action':{'shape':[11],'horizon':16}}

@pytest.fixture
def context_obs(context_meta):
    return {k:torch.randn(2,v['horizon'],*v['shape']) for k,v in context_meta['obs'].items()}

@pytest.fixture
def context_base(tmp_path):
    from scripts.smoke_context_pipeline import make_fixture
    from diffusion_policy.context.data import load_base
    overrides=make_fixture(tmp_path)
    base,cfg=load_base(overrides=overrides)
    yield base,cfg
    base.close()
