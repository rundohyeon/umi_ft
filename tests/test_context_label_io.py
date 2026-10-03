import json
import numpy as np
import pytest
from diffusion_policy.context.labels import *
from scripts.generate_context_labels import label_with_retries,generate


def test_four_class_labels_reject_removed_class(tmp_path):
    response=json.dumps(dict(class_id=4,confidence=.9,reason='removed class'))
    row,attempts,failures=label_with_retries(lambda prompt:response,'test',1,num_classes=4)
    assert row['class_id']==-1 and len(attempts)==failures==2
    assert parse_response(response,num_classes=5)['class_id']==4
    path=tmp_path/'auto_labels.jsonl'
    old=dict(episode_id=0,anchor_index=0,timestamp=0.,class_id=4,confidence=.9)
    path.write_text(json.dumps(old)+'\n')
    with pytest.raises(ValueError,match='for 4 classes'):read_rows(path,num_classes=4)
    with pytest.raises(ValueError,match='for 4 classes'):LabelIndex([old],num_classes=4)
    with pytest.raises(ValueError,match='review class'):
        review_segment([],LabelIndex(num_classes=4),0,0,0,np.arange(2),4)


def test_definitions_determine_hydra_class_count(tmp_path):
    import yaml
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    cfg={'contexts':{i:dict(name=f'class {i}',description='fixture') for i in range(4)},'unknown_label':-1}
    path=tmp_path/'classes.yaml';path.write_text(yaml.safe_dump(cfg))
    register_context_resolvers()
    with initialize_config_dir(version_base=None,config_dir=str(Path('diffusion_policy/config').resolve())):
        policy_cfg=compose(config_name='train_context_aware_policy',overrides=[f'context_training.definitions_path={path}'])
    assert policy_cfg.policy.context.num_classes==4
    assert policy_cfg.task.dataset.context_labels.num_classes==4
    cfg['contexts'][4]=cfg['contexts'].pop(3)
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='consecutive'):definitions(path)


def test_four_class_evaluation(tmp_path):
    from diffusion_policy.context.evaluation import classification_metrics,write_evaluation
    probabilities=np.eye(4)*.8+.05
    labels=[0,1,2,3]
    metrics=write_evaluation(tmp_path,labels,probabilities,[0]*4,np.arange(4),names=['approach','turning','finish','error'])
    assert metrics['confusion_matrix']==np.eye(4,dtype=int).tolist()
    assert metrics['macro_f1']==1. and len(metrics['per_class'])==4
    with pytest.raises(ValueError,match='class range'):
        classification_metrics([0,1,2,4],probabilities)

@pytest.mark.parametrize('response',['{}','{"class_id":true,"confidence":0.9,"reason":"x"}',
    '{"class_id":5,"confidence":0.9,"reason":"x"}','{"class_id":1,"confidence":NaN,"reason":"x"}',
    'prefix {"class_id":1,"confidence":0.9,"reason":"x"}',
    '```','``````','```json```','```json\n','```json\n{',
    '```json\n{}\n``` trailing','```json {"class_id":',None])
def test_invalid_json_never_assigns_class(response):
    row,attempts,failures=label_with_retries(lambda prompt:response,'test',2)
    assert row['class_id']==-1 and len(attempts)==failures==3


@pytest.mark.parametrize('prefix,suffix',[
    ('',''),('```json\n','\n```'),('```\n','\n```'),
    ('```json ','```'),('```','```'),('```JSON\r\n','\r\n```')])
def test_complete_json_fences_remain_valid(prefix,suffix):
    value=dict(class_id=2,confidence=.7,reason='Visible valve rotation')
    assert parse_response(prefix+json.dumps(value)+suffix,num_classes=4)==value


def test_truncated_fence_can_recover_on_retry():
    responses=iter(['```','```json {"class_id":2,"confidence":0.7,"reason":"rotation"}```'])
    row,attempts,failures=label_with_retries(lambda prompt:next(responses),'test',2,num_classes=4)
    assert row['class_id']==2 and failures==1 and len(attempts)==2


def test_broken_fences_do_not_interrupt_parallel_generation(context_base,tmp_path):
    base,_=context_base;cfg,digest=definitions('context/qwen/config/context_labels.yaml')
    cfg.update(output_dir=str(tmp_path/'labels'),model_path='fixture')
    def broken(prompt,images=None):return '```'
    rows=generate(base,cfg,digest,broken,parallel_labelers=[broken],max_windows=2)
    assert len(rows)==2 and all(row['class_id']==-1 for row in rows)
    assert all(row['parsing_failures']==cfg['max_retries']+1 for row in rows)
    path=tmp_path/'labels/auto_labels.jsonl';before=path.read_bytes()
    def valid(prompt,images=None):return '{"class_id":1,"confidence":0.7,"reason":"rotation"}'
    resumed=generate(base,cfg,digest,valid,parallel_labelers=[valid],resume=True,max_windows=2)
    assert len(resumed)==4 and all(row['class_id']==1 for row in resumed[2:])
    assert path.read_bytes().startswith(before)


def test_review_cannot_overwrite_auto_or_approve_neighbors(tmp_path):
    row=dict(episode_id=0,anchor_index=0,timestamp=0.,class_id=2,confidence=.9)
    path=tmp_path/'auto_labels.jsonl';path.write_text(json.dumps(row)+'\n');before=path.read_bytes()
    reviews=review_segment([],LabelIndex([row]),0,1,2,np.arange(5),3)
    save_review(tmp_path/'reviewed_labels.parquet',reviews)
    loaded=read_rows(tmp_path/'reviewed_labels.parquet')
    index=LabelIndex([row],loaded)
    assert index.resolve(0,0)==(-1,0.,'unknown')
    assert index.resolve(0,1)[0]==3 and index.resolve(0,3)[0]==-1
    with pytest.raises(ValueError):save_review(path,reviews)
    assert path.read_bytes()==before


def test_generate_resume_and_manifest_guard(context_base,tmp_path):
    base,_=context_base;cfg,digest=definitions('context/qwen/config/context_labels.yaml')
    cfg['output_dir']=str(tmp_path/'labels');cfg['model_path']='test-fixture'
    called=[]
    def labeler(prompt,images=None):called.append(prompt);return '{"class_id":1,"confidence":0.9,"reason":"test"}'
    generate(base,cfg,digest,labeler,max_windows=2)
    assert len(called)==2
    generate(base,cfg,digest,labeler,max_windows=1,resume=True)
    assert len(called)==3
    cfg['history_length']+=1
    with pytest.raises(ValueError,match='Resume'):generate(base,cfg,digest,labeler,resume=True,max_windows=1)


def test_parallel_labelers_overlap_keep_order_and_resume_serially(context_base,tmp_path):
    from threading import Barrier,Event
    base,_=context_base;cfg,digest=definitions('context/qwen/config/context_labels.yaml')
    cfg.update(output_dir=str(tmp_path/'parallel'),model_path='test-fixture',label_stride=2)
    barrier=Barrier(2);second_finished=Event();calls=[0,0]
    def worker(index):
        def label(prompt,images=None):
            calls[index]+=1
            barrier.wait(timeout=10)
            if index==0:
                assert second_finished.wait(timeout=10)
            else:
                second_finished.set()
            return json.dumps(dict(class_id=index,confidence=.9,reason='parallel fixture'))
        return label
    rows=generate(base,cfg,digest,worker(0),parallel_labelers=[worker(1)],end_index=10,max_windows=4)
    assert calls==[2,2]
    assert [(r['episode_id'],r['anchor_index'],r['class_id']) for r in rows]==[(0,0,0),(1,0,1),(0,2,0),(1,2,1)]
    path=tmp_path/'parallel/auto_labels.jsonl';before=path.read_bytes()
    rows=generate(base,cfg,digest,lambda prompt,images=None:'{"class_id":3,"confidence":0.9,"reason":"resume"}',
                  resume=True,episode=1,end_index=10,max_windows=1)
    assert path.read_bytes().startswith(before)
    assert len(rows)==5 and rows[-1]['anchor_index']==4 and rows[-1]['class_id']==3
    assert read_rows(tmp_path/'parallel/auto_labels.parquet',num_classes=4)==rows


def test_parallel_failure_keeps_completed_rows_for_resume(context_base,tmp_path):
    base,_=context_base;cfg,digest=definitions('context/qwen/config/context_labels.yaml')
    cfg.update(output_dir=str(tmp_path/'failed'),model_path='test-fixture',label_stride=2)
    def good(prompt,images=None):return '{"class_id":1,"confidence":0.9,"reason":"test"}'
    def bad(prompt,images=None):raise RuntimeError('worker failed')
    with pytest.raises(RuntimeError,match='worker failed'):
        generate(base,cfg,digest,good,parallel_labelers=[bad],max_windows=2)
    path=tmp_path/'failed/auto_labels.jsonl'
    assert len(read_rows(path,num_classes=4))==1
    before=path.read_bytes()
    rows=generate(base,cfg,digest,good,parallel_labelers=[good],resume=True,max_windows=2)
    assert path.read_bytes().startswith(before)
    assert [(r['episode_id'],r['anchor_index']) for r in rows]==[(0,0),(0,2),(1,0)]


@pytest.mark.parametrize('devices',[
    ['cpu'],['cuda','cuda:0'],['cuda:0','cuda:0'],['cuda:0','cuda:2']])
def test_parallel_cli_rejects_invalid_visible_devices(monkeypatch,devices):
    import sys
    import torch
    from scripts.generate_context_labels import main
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(sys,'argv',['generate_context_labels.py','--devices',*devices])
    with pytest.raises(SystemExit) as exc:main()
    assert exc.value.code==2
