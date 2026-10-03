import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from diffusion_policy.context.data import causal_summary,causal_rgb,force_history,episode_bounds
from diffusion_policy.context.labels import definitions,parse_response
from scripts.generate_context_labels import generate,make_prompt,label_with_retries


def test_greedy_generation_ignores_checkpoint_sampling_defaults_without_warning():
    import warnings
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from scripts.generate_context_labels import LocalLabeler
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=8,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
        bos_token_id=1,eos_token_id=7,pad_token_id=7,attn_implementation='eager')).eval()
    model.generation_config.do_sample=True
    model.generation_config.temperature=1e-6
    model.generation_config.top_p=.8
    model.generation_config.top_k=20
    original=model.generation_config.to_dict()
    ids=torch.tensor([[1,2,3]])
    class Tokenizer:
        eos_token_id=7
        def apply_chat_template(self,*args,**kwargs):return ids
        def decode(self,tokens,**kwargs):return tokens.tolist()
    labeler=LocalLabeler.__new__(LocalLabeler)
    labeler.vision=False;labeler.device='cpu';labeler.max_new_tokens=2
    labeler.model=model;labeler.tokenizer=Tokenizer()
    with warnings.catch_warnings(record=True) as baseline_warnings,torch.inference_mode():
        warnings.simplefilter('always')
        expected=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
            do_sample=False,num_beams=1,max_new_tokens=2,pad_token_id=7)
    assert any('temperature' in str(w.message) for w in baseline_warnings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual=labeler('fixture')
    assert not [w for w in caught if 'do_sample' in str(w.message)]
    assert actual==expected[0,ids.shape[1]:].tolist()
    assert model.generation_config.to_dict()==original


def test_rotation_is_sent_but_position_is_excluded(context_base):
    base,_=context_base
    base.pose_mats=base.pose_mats.copy()
    base.pose_mats[:11,:3,:3]=Rotation.from_rotvec(np.arange(11)[:,None]*np.array([[0.,0.,.1]])).as_matrix()
    summary,_,_=causal_summary(base,0,10,8)
    assert 'eef_position_m' not in summary['signals']
    np.testing.assert_allclose(summary['signals']['tcp_rotation']['relative_rotvec_rad'][-1],[0,0,.7])
    assert 'tcp_angular_velocity_rad_s' in summary['signals']
    base.pose_mats[:,:3,3]+=12345
    base.pose_mats[11:,:3,:3]=Rotation.from_rotvec([1,2,3]).as_matrix()
    assert causal_summary(base,0,10,8)[0]==summary


def test_rgb_sampling_is_causal_and_episode_local(context_base,monkeypatch):
    base,_=context_base
    rgb=np.zeros((int(base.rgb_episode_ends[-1]),3,3,3),dtype=np.uint8)
    rgb[:,0,0,0]=np.arange(len(rgb))%256
    monkeypatch.setattr(base,'_get_rgb_array',lambda:rgb)
    images,metadata=causal_rgb(base,1,0,5,count=4)
    assert [entry['anchor_index'] for entry in metadata]==[0,1,3,5]
    assert [int(image[0,0,0]) for image in images]==[64,65,67,69]
    assert all(entry['offset_s']<=0 for entry in metadata)
    rgb[70:]=255
    after,_=causal_rgb(base,1,0,5,count=4)
    for a,b in zip(images,after):np.testing.assert_array_equal(a,b)
    first,_=causal_rgb(base,1,0,0,count=4)
    assert len(first)==1


def test_force_history_keeps_brief_peaks_in_their_original_order():
    times=np.arange(100)/100
    values=np.zeros((100,6))
    values[17,:3]=[0,0,-8]
    values[64,:3]=[3,4,0]
    history=force_history(values,times,1.,4)
    samples=np.array(history['samples'])
    assert len(samples)<=10 and history['source_samples']==100
    assert np.all(np.diff(samples[:,0])>0)
    np.testing.assert_allclose(samples[samples[:,1]==8][0],[-.83,8])
    np.testing.assert_allclose(samples[samples[:,1]==5][0],[-.36,5])
    assert samples[0,0]==-1. and samples[-1,0]==-.01
    assert history['last_sample_age_s']==.01
    # The same aggregate statistics at a different time must give a different trace.
    shifted=np.zeros_like(values);shifted[42,:3]=values[17,:3];shifted[89,:3]=values[64,:3]
    assert force_history(shifted,times,1.,4)['samples']!=history['samples']


def test_missing_force_is_not_reported_as_measured_zero():
    missing=force_history(np.empty((0,6)),np.array([]),1.,4)
    zero=force_history(np.zeros((1,6)),np.array([.9]),1.,4)
    assert missing['available'] is False and missing['samples']==[]
    assert missing['last_sample_age_s'] is None
    assert zero['available'] is True and zero['samples']==[[-.1,0.]]
    assert zero['last_sample_age_s']==.1
    with pytest.raises(ValueError,match='future'):
        force_history(np.zeros((1,6)),np.array([1.1]),1.,4)


@pytest.mark.parametrize('episode',[0,1])
def test_right_force_history_excludes_future_and_other_episodes(context_base,episode):
    base,_=context_base
    lo,_=episode_bounds(base,episode)
    start,anchor=base.rgb_timestamps[lo+11],base.rgb_timestamps[lo+30]
    times=base.ft_right_timestamps
    included=(times>=start)&(times<=anchor)
    base.ft_right=base.ft_right.copy();base.ft_right[:]=0.
    peak=np.flatnonzero(included)[len(np.flatnonzero(included))//2]
    base.ft_right[peak,2]=-7.
    summary,ws,we=causal_summary(base,episode,30,20,{'force_history_bins':4})
    assert (ws,we)==(11,30)
    trace=summary['signals']['right_force_history']
    samples=np.array(trace['samples'])
    assert trace['source_samples']==int(included.sum())
    assert np.all(samples[:,0]<=0)
    assert np.any(samples[:,1]==7.)
    base.ft_right[~included,:]=9999.
    assert causal_summary(base,episode,30,20,{'force_history_bins':4})[0]==summary
    assert 'right_force_history' not in causal_summary(base,episode,30,20,{'force_torque':False,'force_history_bins':4})[0]['signals']


def test_force_magnitudes_are_independent_of_axis_direction_and_torque(context_base):
    base,_=context_base
    base.ft_left=base.ft_left.copy();base.ft_right=base.ft_right.copy()
    base.ft_left[:]=[-2.,3.,-6.,100.,-200.,300.]
    base.ft_right[:]=[3.,-4.,0.,-400.,500.,-600.]
    inputs={'force_history_bins':4}
    summary,_,_=causal_summary(base,0,30,20,inputs)
    signals=summary['signals']
    assert signals['left_force_norm_N']['current']==[7.]
    assert signals['right_force_norm_N']['current']==[5.]
    assert signals['right_force_history']['columns']==['offset_s','force_norm_N']
    assert all(len(row)==2 and row[1]==5. for row in signals['right_force_history']['samples'])
    assert {key for key in signals if key.startswith(('left_','right_'))}=={
        'left_force_norm_N','right_force_norm_N','right_force_history'}
    # Changing force direction and all torques must leave Qwen's JSON unchanged.
    base.ft_left[:]=[0.,7.,0.,-900.,800.,-700.]
    base.ft_right[:]=[0.,0.,-5.,600.,-500.,400.]
    assert causal_summary(base,0,30,20,inputs)[0]==summary


def test_images_and_previous_predictions_survive_resume(context_base,tmp_path,monkeypatch):
    import scripts.generate_context_labels as script
    base,_=context_base;cfg,digest=definitions('context/qwen/config/context_labels.yaml')
    cfg.update(output_dir=str(tmp_path/'labels'),model_path='vision-fixture')
    monkeypatch.setattr(script,'make_prompt',lambda contexts,summary:json.dumps(summary))
    captured=[]
    def labeler(prompt,images=None):
        summary=json.loads(prompt);captured.append(summary)
        assert len(images)==len(summary['rgb_frames']) and len(images)>0
        assert all(frame.shape==(224,224,3) for frame in images)
        assert 'right_force_history' in summary['signals']
        assert summary['signals']['right_force_history']['columns']==['offset_s','force_norm_N']
        assert all(len(row)==2 and row[1]>=0 for row in summary['signals']['right_force_history']['samples'])
        assert all(row[0]<=0 for row in summary['signals']['right_force_history']['samples'])
        return json.dumps(dict(class_id=len(summary['previous_predictions'])%4,confidence=.6,reason='fixture'))
    generate(base,cfg,digest,labeler,episode=0,max_windows=2)
    assert captured[0]['previous_predictions']==[]
    assert captured[0]['progress']['last_predicted_class'] is None
    assert captured[1]['previous_predictions'][0]['anchor_index']==0
    generate(base,cfg,digest,labeler,episode=0,resume=True,max_windows=1)
    assert [row['anchor_index'] for row in captured[-1]['previous_predictions']]==[0,6]
    assert captured[-1]['progress']['last_predicted_class']==1
    assert 'completion_percentage' not in captured[-1]['progress']
    # A new episode must not inherit earlier episodes' predictions.
    generate(base,cfg,digest,labeler,episode=1,resume=True,max_windows=1)
    assert captured[-1]['previous_predictions']==[]


def test_prompt_has_no_fixed_answer_and_accepts_semantic_unknown():
    cfg,_=definitions('context/qwen/config/context_labels.yaml')
    prompt=make_prompt(cfg['contexts'],{})
    assert '"confidence": 0.84' not in prompt and '"class_id": 0' not in prompt
    assert 'does NOT grasp or pinch' in prompt
    answer='{"class_id":-1,"confidence":0.2,"reason":"Force measurements are unavailable"}'
    row,attempts,failures=label_with_retries(lambda prompt:answer,prompt,2,num_classes=4)
    assert row['class_id']==-1 and len(attempts)==1 and failures==0


def test_resume_rejects_history_holes(context_base,tmp_path):
    base,_=context_base;cfg,digest=definitions('context/qwen/config/context_labels.yaml')
    cfg.update(output_dir=str(tmp_path/'labels'),model_path='fixture')
    def labeler(prompt,images=None):return '{"class_id":0,"confidence":0.5,"reason":"fixture"}'
    generate(base,cfg,digest,labeler,max_windows=3)
    path=tmp_path/'labels/auto_labels.jsonl'
    lines=path.read_text().splitlines(True);path.write_text(lines[0]+lines[2])
    with pytest.raises(ValueError,match='chronological prefix'):
        generate(base,cfg,digest,labeler,resume=True)
