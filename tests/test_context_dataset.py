import numpy as np
import pytest
from diffusion_policy.context.data import split_episodes,validate_label_alignment,ContextFrames
from diffusion_policy.context.labels import LabelIndex


def test_splits_disjoint_and_reproducible():
    split=split_episodes(214)
    assert split==split_episodes(214)
    assert sorted(sum(split.values(),[]))==list(range(214))
    assert not set(split['train'])&set(split['validation'])
    assert not set(split['train'])&set(split['test'])


def test_label_hold_is_past_only_and_reviewed_default(context_base):
    base,_=context_base
    rows=[dict(episode_id=0,anchor_index=i,timestamp=float(base.rgb_timestamps[i]),class_id=c,confidence=.95) for i,c in [(2,1),(6,4)]]
    index=LabelIndex(rows,source='all_auto')
    assert index.resolve(0,1)[0]==-1
    assert index.resolve(0,5)[0]==1
    assert index.resolve(0,6)[0]==4
    assert index.resolve(1,5)[0]==-1
    assert len(ContextFrames(base,[0],LabelIndex(rows)))==0
    dataset=ContextFrames(base,[0],index)
    sample=dataset[0]
    assert sample['anchor_index']==2 and sample['context_label'].item()==1
    assert 'action' not in sample and 'action' not in sample['obs']
    rows[0]['timestamp']=float(base.rgb_timestamps[3])
    with pytest.raises(ValueError,match='timestamp'):validate_label_alignment(base,rows)


def test_weak_labels_keep_lower_weights_and_unknown_review_blocks_fallback():
    auto=[dict(episode_id=0,anchor_index=0,timestamp=0.,class_id=1,confidence=.95)]
    review=[dict(episode_id=0,anchor_index=1,timestamp=1.,reviewed_class_id=-1,is_reviewed=True,is_modified=True)]
    index=LabelIndex(auto,review,mix_auto=True)
    assert index.resolve(0,0)==(1,.3,'auto')
    assert index.resolve(0,1)[0]==-1


def test_stage_b_uses_stage_a_splits_and_oracle_filters_unknown(context_base,tmp_path):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from diffusion_policy.context.labels import review_segment,save_review,atomic_json
    from diffusion_policy.context.data import source_fingerprint,episode_bounds
    base,cfg=context_base
    rows=[]
    for ep in range(base.n_episodes):
        lo,hi=episode_bounds(base,ep)
        rows=review_segment(rows,LabelIndex(),ep,1,3,base.rgb_timestamps[lo:hi],2)
    path=tmp_path/'reviewed_labels.parquet';save_review(path,rows)
    splits=split_episodes(base.n_episodes)
    split_path=tmp_path/'splits.json';atomic_json(split_path,dict(splits=splits,source_fingerprint=source_fingerprint(base)))
    config=OmegaConf.to_container(cfg.task.dataset,resolve=True)
    config.update(_target_='diffusion_policy.context.data.ContextUmiDataset',context_split_path=str(split_path),
        require_known_context=True,context_labels=dict(auto_path=str(tmp_path/'auto_labels.jsonl'),reviewed_path=str(path)))
    dataset=instantiate(config)
    validation=dataset.get_validation_dataset()
    assert {ep for ep,_ in dataset.indices}==set(splits['train'])
    assert {ep for ep,_ in validation.indices}==set(splits['validation'])
    assert all(dataset[i]['context_label']==2 for i in range(len(dataset)))
    splits['validation'][0]=splits['train'][0]
    atomic_json(split_path,dict(splits=splits,source_fingerprint=source_fingerprint(base)))
    with pytest.raises(Exception,match='overlap'):instantiate(config)
