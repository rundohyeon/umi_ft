import copy
import numpy as np
import pytest
from diffusion_policy.context.data import causal_summary,assert_causal,frame_view


def test_future_state_ft_and_action_targets_do_not_enter_context(context_base):
    base,_=context_base
    view=frame_view(base,[0])
    i=next(i for i,(_,current) in enumerate(view.indices) if current==10)
    before,_,info=view._sample_arrays(i,load_rgb=True)
    summary,_,_=causal_summary(view,0,10,8)
    assert_causal(info)
    anchor=base.rgb_timestamps[10]
    view.pose_mats=view.pose_mats.copy();view.pose_mats[11:,:3,3]+=100
    view.ft_left=view.ft_left.copy();view.ft_left[view.ft_left_timestamps>anchor]+=100
    view.grasp_force=view.grasp_force.copy();view.grasp_force[:]+=1000
    after,_,_=view._sample_arrays(i,load_rgb=True)
    for key in before:np.testing.assert_array_equal(before[key],after[key])
    assert summary==causal_summary(view,0,10,8)[0]


def test_future_timestamps_and_cross_episode_windows(context_base):
    base,_=context_base
    view=frame_view(base,[1]);obs,_,info=view._sample_arrays(0,load_rgb=False)
    assert_causal(info)
    assert info['left_ft_timestamps'].min()>=base.rgb_timestamps[64]
    info['left_ft_timestamps'][-1]=info['anchor_timestamp']+1
    with pytest.raises(ValueError,match='Future'):assert_causal(info)
    _,start,end=causal_summary(base,1,0,32);assert start==end==0

@pytest.mark.parametrize('key',['rgb_timestamps','pose_timestamps','left_ft_timestamps','right_ft_timestamps'])
def test_each_modality_clock_rejects_future(context_base,key):
    base,_=context_base
    _,_,info=base._sample_arrays(0,load_rgb=False)
    info[key]=info[key].copy()
    info[key][-1]=info['anchor_timestamp']+.001
    with pytest.raises(ValueError,match='Future'):assert_causal(info)
