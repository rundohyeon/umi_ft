"""Causal numeric force evidence plus constrained Qwen visual observations.

These configurable thresholds are experimental labeling rules, not calibrated
probabilities. Raw model observations and numeric evidence are saved separately.
"""
from __future__ import annotations

import json
import math
import numpy as np


DEFAULTS = dict(contact_on_N=.5, contact_off_N=.3, recent_s=.05,
                reference_start_s=.25, reference_end_s=.1, max_age_s=.05,
                max_gap_s=.05, min_samples=3, trend_deadband_N=.2,
                drop_absolute_N=1., drop_fraction=.35)
VISIBILITY = {'visible', 'lost', 'unclear'}
MOTION = {'approaching', 'engaged_rotation', 'independent_rotation',
          'retreating', 'stationary', 'unclear'}


def settings(overrides):
    if set(overrides) - set(DEFAULTS):
        raise ValueError('Unknown force decision settings')
    cfg = dict(DEFAULTS, **overrides)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v <= 0 for v in cfg.values()):
        raise ValueError('Force decision settings must be positive finite numbers')
    if (cfg['contact_off_N'] >= cfg['contact_on_N'] or cfg['drop_fraction'] >= 1
            or not cfg['recent_s'] < cfg['reference_end_s'] < cfg['reference_start_s']
            or type(cfg['min_samples']) is not int or cfg['min_samples'] < 2):
        raise ValueError('Invalid force thresholds or time windows')
    return cfg


def measure_force(norms, times, anchor, cfg, previous_contact=None):
    """Use native samples, never the window maximum or the LLM's arithmetic."""
    norms, times = np.asarray(norms, dtype=float), np.asarray(times, dtype=float)
    if (norms.shape != times.shape or norms.ndim != 1 or not np.isfinite(norms).all()
            or not np.isfinite(times).all() or np.any(norms < 0)
            or np.any(times > anchor) or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid or future force samples')
    out = dict(available=False, current_N=None, recent_median_N=None,
               reference_median_N=None, delta_N=None, contact=None,
               trend='unknown', sharp_drop=False, sample_age_s=None,
               recent_samples=0, reference_samples=0,
               contact_on_N=cfg['contact_on_N'], contact_off_N=cfg['contact_off_N'])
    if not len(times):
        return out
    out.update(current_N=round(float(norms[-1]), 5),
               sample_age_s=round(float(anchor-times[-1]), 6))
    recent = (times >= anchor-cfg['recent_s'])
    reference = ((times >= anchor-cfg['reference_start_s'])
                 & (times < anchor-cfg['reference_end_s']))
    out.update(recent_samples=int(recent.sum()), reference_samples=int(reference.sum()))
    if (anchor-times[-1] > cfg['max_age_s'] or recent.sum() < cfg['min_samples']
            or np.any(np.diff(times[recent]) > cfg['max_gap_s'])):
        return out
    current = float(np.median(norms[recent]))
    contact = (True if current >= cfg['contact_on_N'] else
               False if current <= cfg['contact_off_N'] else previous_contact)
    out.update(available=True, recent_median_N=round(current, 5), contact=contact)
    whole = times >= anchor-cfg['reference_start_s']
    if (reference.sum() < cfg['min_samples']
            or times[reference][0] > anchor-cfg['reference_start_s']+cfg['max_gap_s']
            or times[reference][-1] < anchor-cfg['reference_end_s']-cfg['max_gap_s']
            or np.any(np.diff(times[whole]) > cfg['max_gap_s'])):
        return out
    before = float(np.median(norms[reference]))
    delta = current-before
    trend = ('increasing' if delta > cfg['trend_deadband_N'] else
             'decreasing' if delta < -cfg['trend_deadband_N'] else 'stable')
    out.update(reference_median_N=round(before, 5), delta_N=round(delta, 5), trend=trend,
               sharp_drop=bool(before >= cfg['contact_on_N']
                   and -delta >= cfg['drop_absolute_N']
                   and -delta >= before*cfg['drop_fraction']))
    return out


def evidence_at(base, episode, anchor_index, cfg, previous_contact=None):
    from diffusion_policy.context.data import episode_bounds
    lo, _ = episode_bounds(base, episode)
    anchor = float(base.rgb_timestamps[lo+anchor_index])
    ends = base.ft_right_episode_ends
    start, end = (0 if episode == 0 else int(ends[episode-1])), int(ends[episode])
    times = base.ft_right_timestamps[start:end]
    left = np.searchsorted(times, anchor-cfg['reference_start_s'], side='left')
    right = np.searchsorted(times, anchor, side='right')
    values = base.ft_right[start+left:start+right, :3]
    return measure_force(np.linalg.norm(values, axis=1), times[left:right], anchor, cfg, previous_contact)


def prepare_summary(summary, force):
    # Do not place old peaks, a 2-second net delta, or noisy derivatives beside
    # the current reading. The original signed axes never reach the model.
    result = dict(summary)
    result['signals'] = {k:v for k,v in summary['signals'].items()
                         if not k.startswith(('left_force', 'right_force'))}
    result['right_force_evidence'] = force
    return result


def make_visual_prompt(contexts, summary):
    return (
        'Inspect the RGB frames from oldest to newest and describe the CURRENT visible motion. '
        'The robot operates the orange lever with one RIGHT finger, without grasping. '
        'The camera is attached to the gripper: background movement occurs during approach, '
        'repositioning, retreat, AND lever turning. Background rotation alone does not prove '
        'lever rotation or finger-lever engagement. Inspect the finger tip relative to the lever. '
        'During genuine engaged rotation, the lever and finger can stay fixed in the image '
        'while the fixed rig rotates. This is only a possibility, not the default answer.\n'
        'Numeric right_force_evidence was computed from synchronized native |F| samples. '
        'contact=false means the measurements are below the contact threshold, including sensor noise. '
        'Do not infer pushing from a small nonzero magnitude. Do not recompute force, substitute '
        'past force or previous labels, or output force values. Previous predictions are fallible.\n'
        'Return one JSON object with exactly visibility, motion, confidence. '
        'visibility: "visible" when the lever is visible now; "lost" only when a previously visible '
        'lever actually disappears by the last frame; otherwise "unclear". '
        'motion: "approaching" when the finger closes its gap to the lever; '
        '"engaged_rotation" only for visible finger-lever engagement and coherent rotation with '
        'contact=true; "independent_rotation" only for visible lever rotation without finger '
        'engagement (camera/background movement alone is insufficient); "retreating" when the '
        'finger moves away; "stationary" for no meaningful relative motion; otherwise "unclear". '
        'A still frame cannot establish motion. confidence is a number from 0 to 1. '
        'Do not output class_id, reason, numbers describing force, or any additional keys.\n'
        'The final class is assigned separately: a measured sharp force drop takes precedence '
        'over turning, and turning requires measured contact.\n'
        'Observation window (causal, SI units):\n' + json.dumps(summary, sort_keys=True))


def parse_visual_response(text):
    if not isinstance(text, str):
        raise ValueError('Expected visual JSON text')
    text = text.strip()
    if text.startswith('```'):
        if len(text) < 6 or not text.endswith('```'):
            raise ValueError('Incomplete visual JSON fence')
        text = text[3:-3].strip()
        if text[:4].lower() == 'json':
            text = text[4:].strip()
    value = json.loads(text)
    if (not isinstance(value, dict) or set(value) != {'visibility','motion','confidence'}
            or value['visibility'] not in VISIBILITY or value['motion'] not in MOTION):
        raise ValueError('Invalid visual evidence schema')
    confidence = value['confidence']
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError('Invalid visual confidence')
    return value


def decide(visual, force):
    visibility, motion = visual['visibility'], visual['motion']
    class_id, source, explanation = -1, 'insufficient_evidence', 'Insufficient task-phase evidence.'
    if visibility == 'lost':
        class_id, source, explanation = 3, 'vision', 'Previously visible lever disappeared.'
    elif force['available'] and force['sharp_drop']:
        class_id, source, explanation = 2, 'force_rule', 'Sharp force drop: finish overrides continued rotation.'
    elif visibility == 'visible' and motion == 'approaching':
        class_id, source, explanation = 0, 'vision', 'Finger visibly approaches the lever.'
    elif (visibility == 'visible' and force['available'] and force['contact'] is True
          and motion == 'engaged_rotation'):
        class_id, source, explanation = 1, 'vision_and_force', 'Engaged rotation with measured contact.'
    elif (visibility == 'visible' and force['available'] and force['contact'] is False
          and motion == 'independent_rotation'):
        class_id, source, explanation = 3, 'vision_and_force', 'Lever rotates independently with no measured contact.'
    elif motion == 'engaged_rotation' and force['contact'] is not True:
        explanation = 'Turning rejected: force contact is not established.'
    current = 'missing' if force['current_N'] is None else f"{force['current_N']:.2f} N"
    contact = {True:'yes', False:'no', None:'unknown'}[force['contact']]
    recent = ''
    if force['reference_median_N'] is not None:
        recent = f" ({force['reference_median_N']:.2f} -> {force['recent_median_N']:.2f} N)"
    reason = (f"Right |F| now {current}; contact={contact}; {force['trend']}{recent}. "
              f"Visual={motion}, lever={visibility}. {explanation}")
    return dict(class_id=class_id, reason=reason,
                confidence=visual['confidence'] if source in ('vision','vision_and_force') else 0.,
                decision_source=source, visual_evidence_json=json.dumps(visual),
                force_evidence_json=json.dumps(force))
