"""Offline episode-order corrections to unknown automatic labels.

The approach rule uses a future turning prediction. These are derived weak
labels, not new causal model predictions or human-reviewed ground truth.
"""
from collections import defaultdict

VERSION = 'unknown_phase_fill_v1'
RAW_FIELDS = ('class_id', 'confidence', 'reason', 'decision_source', 'provenance')


def fill_unknown_phases(rows):
    """Fill only unknowns outside the first turning/finish boundaries per episode."""
    episodes = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row['episode_id'], row['anchor_index'])
        if key in seen:
            raise ValueError('Duplicate episode/anchor')
        if 'postprocessing_version' in row or any('raw_'+k in row for k in RAW_FIELDS):
            raise ValueError('Use original predictions, not already postprocessed labels')
        if 'reviewed_class_id' in row:
            raise ValueError('Phase filling applies only to automatic predictions')
        if row['class_id'] not in (-1, 0, 1, 2, 3):
            raise ValueError('Expected approach/turning/finish/error class IDs')
        seen.add(key)
        episodes[row['episode_id']].append(row)
    boundaries = {}
    for ep, group in episodes.items():
        boundaries[ep] = tuple(min((r['anchor_index'] for r in group if r['class_id'] == cid),
                                   default=None) for cid in (1, 2))
    output = []
    for row in rows:
        new = dict(row)
        new.update({'raw_'+k: row.get(k) for k in RAW_FIELDS})
        new.update(postprocessing_version=VERSION, postprocessing_rule='',
                   postprocessing_reference_anchor=-1)
        turning, finish = boundaries[row['episode_id']]
        rule, reference, target = '', -1, None
        if row['class_id'] == -1:
            # Finish wins if anomalous phase order makes both rules applicable.
            if finish is not None and row['anchor_index'] > finish:
                rule, reference, target = 'after_first_finish', finish, 2
            elif turning is not None and row['anchor_index'] < turning:
                rule, reference, target = 'before_first_turning', turning, 0
        if rule:
            name = 'approach' if target == 0 else 'finish'
            boundary = 'turning' if target == 0 else 'finish'
            relation = 'before' if target == 0 else 'after'
            new.update(class_id=target, confidence=0., decision_source='phase_rule',
                       provenance='phase_rule', postprocessing_rule=rule,
                       postprocessing_reference_anchor=reference,
                       reason=f'Offline episode-order rule: unknown -> {name}; '
                              f'{relation} first {boundary} at frame {reference}. '
                              'Original prediction and evidence are preserved.')
        output.append(new)
    return output
