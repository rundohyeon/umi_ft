import json
import numpy as np
import pytest

from diffusion_policy.context.force_visual import (
    settings, measure_force, evidence_at, prepare_summary, parse_visual_response, decide)
from diffusion_policy.context.labels import definitions
from scripts.generate_context_labels import generate, label_with_retries


CFG = settings({})
TIMES = np.linspace(-.3, 0., 31)


def constant(force, previous_contact=None):
    return measure_force(np.full(31, force), TIMES, 0., CFG, previous_contact)


def vision(motion='engaged_rotation', visibility='visible'):
    return dict(motion=motion, visibility=visibility, confidence=.98)


def test_noise_cannot_establish_turning_or_finish_and_is_not_automatically_approach():
    evidence = measure_force(np.resize([0., .1, .2], 31), TIMES, 0., CFG)
    assert evidence['available'] and evidence['contact'] is False
    assert evidence['trend'] == 'stable' and not evidence['sharp_drop']
    assert decide(vision(), evidence)['class_id'] == -1
    assert decide(vision('stationary'), evidence)['class_id'] == -1
    assert decide(vision('approaching'), evidence)['class_id'] == 0
    assert decide(vision('independent_rotation'), evidence)['class_id'] == 3


@pytest.mark.parametrize('before,current,expected',[(None,.4,None),(True,.4,True),
    (False,.4,False),(True,.3,False),(False,.5,True)])
def test_contact_hysteresis(before, current, expected):
    assert constant(current, before)['contact'] is expected


def test_contact_and_visual_rotation_are_both_needed():
    evidence = constant(3.)
    assert decide(vision(), evidence)['class_id'] == 1
    assert decide(vision('stationary'), evidence)['class_id'] == -1
    assert decide(vision(visibility='unclear'), evidence)['class_id'] == -1


def test_sharp_drop_overrides_rotation_and_current_force_is_rendered_from_data():
    norms = np.where(TIMES < -.07, 14., .14)
    evidence = measure_force(norms, TIMES, 0., CFG)
    assert evidence['sharp_drop'] and evidence['trend'] == 'decreasing'
    result = decide(vision(), evidence)
    assert result['class_id'] == 2 and result['decision_source'] == 'force_rule'
    assert result['confidence'] == 0.  # Do not invent an LLM probability for a rule.
    assert result['reason'].startswith('Right |F| now 0.14 N;')
    # A drop needs no preceding increase, retreat or rotation stop.
    assert decide(vision('stationary'), evidence)['class_id'] == 2
    # The explicit disappearance error condition still has precedence.
    assert decide(vision(visibility='lost'), evidence)['class_id'] == 3
    # A drop in an old window cannot keep declaring finish indefinitely.
    assert not constant(.14)['sharp_drop']


def test_isolated_peak_and_small_force_jitter_do_not_trigger_finish():
    norms = np.full(31, .1); norms[10] = 14.
    assert not measure_force(norms, TIMES, 0., CFG)['sharp_drop']
    norms = np.where(TIMES < -.07, 14., 13.)
    assert not measure_force(norms, TIMES, 0., CFG)['sharp_drop']


def test_missing_stale_sparse_and_gapped_force_are_not_zero_contact():
    for norms, times in [(np.array([]), np.array([])),
                         (np.full(10,3.), np.linspace(-.5,-.2,10)),
                         (np.array([3.]), np.array([0.]))]:
        evidence = measure_force(norms, times, 0., CFG, True)
        assert not evidence['available'] and evidence['contact'] is None
        assert not evidence['sharp_drop'] and decide(vision(),evidence)['class_id'] == -1
    # Enough recent samples for contact, but a gap prevents trend/drop inference.
    times = np.r_[TIMES[TIMES < -.16], TIMES[TIMES > -.04]]
    evidence = measure_force(np.where(times < -.1,14.,.14),times,0.,CFG)
    assert evidence['available'] and evidence['trend'] == 'unknown'
    assert not evidence['sharp_drop']


def test_force_extraction_excludes_future_and_other_episodes(context_base):
    base, _ = context_base
    base.ft_right = base.ft_right.copy()
    expected = evidence_at(base,1,25,CFG)
    lo = int(base.rgb_episode_ends[0]); anchor = base.rgb_timestamps[lo+25]
    first = int(base.ft_right_episode_ends[0]); end = int(base.ft_right_episode_ends[1])
    base.ft_right[:first] = 999.
    base.ft_right[end:] = 999.
    base.ft_right[base.ft_right_timestamps > anchor] = 999.
    assert evidence_at(base,1,25,CFG) == expected
    with pytest.raises(ValueError,match='future'):
        measure_force([1.],[.1],0.,CFG)


def test_llm_cannot_supply_force_numbers_or_force_descriptions():
    valid = dict(visibility='visible', motion='approaching', confidence=.8)
    assert parse_visual_response(json.dumps(valid)) == valid
    response = json.dumps(dict(valid,reason='Force is 14N',force=14))
    result, attempts, failures = label_with_retries(lambda prompt:response,'fixture',1,
        response_parser=parse_visual_response,
        failure_result=dict(visibility='unclear',motion='unclear',confidence=0.))
    assert failures == 2 and result['motion'] == 'unclear'
    summary = prepare_summary({'signals':{'right_force_norm_N':{'max':[14.]},
        'right_force_history':{}, 'left_force_norm_N':{}, 'tcp_rotation':{}}},constant(.14))
    assert set(summary['signals']) == {'tcp_rotation'}
    assert summary['right_force_evidence']['current_N'] == .14


def test_force_rules_and_visual_evidence_survive_resume_and_threshold_changes_reject(context_base,tmp_path):
    base, _ = context_base
    cfg,digest = definitions('context/qwen/config/context_labels_qwen35.yaml')
    cfg.update(output_dir=str(tmp_path/'labels'),model_path='fixture')
    class Labeler:
        runtime = {'fixture': True}
        def __call__(self,prompt,images=None):
            assert 'right_force_evidence' in prompt
            assert 'right_force_norm_N' not in prompt
            return json.dumps(vision('approaching'))
    worker = Labeler()
    rows=generate(base,cfg,digest,worker,episode=0,max_windows=3)
    prefix=(tmp_path/'labels/auto_labels.jsonl').read_bytes()
    resumed=generate(base,cfg,digest,worker,episode=0,max_windows=1,resume=True)
    assert (tmp_path/'labels/auto_labels.jsonl').read_bytes().startswith(prefix)
    assert len(resumed)==4
    for row in resumed:
        assert json.loads(row['visual_evidence_json'])['motion']=='approaching'
        assert 'contact' in json.loads(row['force_evidence_json'])
    cfg['force_decision']['contact_on_N']=.6
    with pytest.raises(ValueError,match='Resume'):
        generate(base,cfg,digest,worker,episode=0,max_windows=1,resume=True)
