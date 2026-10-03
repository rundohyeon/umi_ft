import copy
import hashlib
import json

import pytest

from diffusion_policy.context.phase_fill import fill_unknown_phases


def rows(classes, episode=0):
    return [dict(episode_id=episode, anchor_index=6*i, timestamp=i/10,
                 class_id=cid, confidence=.7, reason='original observation',
                 provenance='unknown' if cid == -1 else 'auto',
                 decision_source='insufficient_evidence' if cid == -1 else 'vision',
                 force_evidence_json='{}', visual_evidence_json='{}')
            for i, cid in enumerate(classes)]


@pytest.mark.parametrize('classes,expected', [
    ([-1, -1, 0, 1, -1, 1, 2, -1, -1], [0, 0, 0, 1, -1, 1, 2, 2, 2]),
    ([-1, 3, 1, 2, 3, -1, 1, -1], [0, 3, 1, 2, 3, 2, 1, 2]),
    ([-1, -1], [-1, -1]),
    ([-1, 1, -1], [0, 1, -1]),
    ([-1, 2, -1], [-1, 2, 2]),
    ([-1, 2, -1, 1, -1], [0, 2, 2, 1, 2]),
])
def test_phase_boundaries_preserve_explicit_classes_and_middle_unknowns(classes, expected):
    original = rows(classes)
    before = copy.deepcopy(original)
    result = fill_unknown_phases(original)
    assert [r['class_id'] for r in result] == expected
    assert original == before
    for old, new in zip(original, result):
        assert new['raw_class_id'] == old['class_id']
        assert new['raw_reason'] == old['reason']
        assert new['raw_confidence'] == old['confidence']
        assert new['raw_decision_source'] == old['decision_source']
        assert new['raw_provenance'] == old['provenance']
        assert new['force_evidence_json'] == old['force_evidence_json']
        if old['class_id'] != new['class_id']:
            assert old['class_id'] == -1
            assert new['decision_source'] == new['provenance'] == 'phase_rule'
            assert new['confidence'] == 0
        else:
            assert all(new[k] == v for k, v in old.items())


def test_episode_boundaries_and_unsorted_input():
    source = rows([-1, 1, 2, -1], 0) + rows([-1, -1, -1], 1)
    source.reverse()
    result = fill_unknown_phases(source)
    assert all(r['class_id'] == -1 for r in result if r['episode_id'] == 1)
    assert [(r['episode_id'], r['anchor_index']) for r in source] == [
        (r['episode_id'], r['anchor_index']) for r in result]


def test_rejects_reprocessing_duplicate_or_reviewed_rows():
    source = rows([-1, 1, 2])
    with pytest.raises(ValueError, match='already postprocessed'):
        fill_unknown_phases(fill_unknown_phases(source))
    with pytest.raises(ValueError, match='Duplicate'):
        fill_unknown_phases(source + source[:1])
    source[0]['reviewed_class_id'] = 0
    with pytest.raises(ValueError, match='automatic'):
        fill_unknown_phases(source)


def test_postprocess_complete_episode_and_preserve_source(context_base, tmp_path):
    from diffusion_policy.context.data import source_fingerprint
    from diffusion_policy.context.labels import definitions, read_rows
    from tools.postprocess_context_labels import postprocess
    base, _ = context_base
    cfg, digest = definitions('context/qwen/config/context_labels_qwen35.yaml')
    directory = tmp_path/'original'; directory.mkdir()
    cfg['output_dir'] = str(directory)
    source = directory/'auto_labels.jsonl'
    manifest = dict(definition_hash=digest, source_fingerprint=source_fingerprint(base), label_stride=6)
    (directory/'labeling_manifest.json').write_text(json.dumps(manifest))
    saved = rows([-1, -1, 0, 1, -1, 1, 2, -1, 3, -1, -1])
    for r in saved:
        r.update(timestamp=float(base.rgb_timestamps[r['anchor_index']]), definition_hash=digest)
    source.write_text(''.join(json.dumps(r)+'\n' for r in saved))
    original = source.read_bytes()
    out = tmp_path/'derived'
    report = postprocess(base, cfg, digest, [0], out)
    assert source.read_bytes() == original
    result = read_rows(out/'auto_labels.jsonl', 4)
    assert result == read_rows(out/'auto_labels.parquet', 4)
    assert report['after']['corrected_count'] == 5
    assert report['after']['unknown_count'] == 1
    derived = json.loads((out/'labeling_manifest.json').read_text())['postprocessing']
    assert derived['source_sha256'] == hashlib.sha256(original).hexdigest()
    assert derived['output_sha256'] == hashlib.sha256((out/'auto_labels.jsonl').read_bytes()).hexdigest()
    assert derived['uses_future_phase_boundary']
    with pytest.raises(FileExistsError):
        postprocess(base, cfg, digest, [0], out)
    # A hole and a missing tail must both be rejected, never filled as predictions.
    for incomplete in (saved[:-1], saved[:3]+saved[4:]):
        source.write_text(''.join(json.dumps(r)+'\n' for r in incomplete))
        with pytest.raises(ValueError, match='incomplete'):
            postprocess(base, cfg, digest, [0], tmp_path/'incomplete')
        assert not (tmp_path/'incomplete').exists()
