#!/usr/bin/env python
"""Fill unknown phases in complete episodes, preserving original predictions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_policy.context.data import (
    load_labeling_base, episode_bounds, source_fingerprint, protect_raw_data, validate_label_alignment)
from diffusion_policy.context.labels import (
    definitions, read_rows, validate_definition_version, write_parquet, atomic_json)
from diffusion_policy.context.phase_fill import VERSION, fill_unknown_phases


def statistics(rows):
    counts = Counter(r['class_id'] for r in rows)
    return dict(sample_count=len(rows), count_per_class={str(i): counts[i] for i in range(4)},
                unknown_count=counts[-1],
                corrected_count=sum(bool(r.get('postprocessing_rule')) for r in rows))


def postprocess(base, cfg, digest, episodes, output):
    if [cfg['contexts'][i]['name'] for i in range(len(cfg['contexts']))] != [
            'approach', 'turning', 'finish', 'error']:
        raise ValueError('Phase filling requires approach/turning/finish/error in order')
    source = Path(cfg['output_dir'])/'auto_labels.jsonl'
    original = source.read_bytes()
    manifest_path = source.parent/'labeling_manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('postprocessing'):
        raise ValueError('Use the original prediction directory')
    if manifest['definition_hash'] != digest or manifest['source_fingerprint'] != source_fingerprint(base):
        raise ValueError('Saved predictions do not match the definitions/recording')
    rows = read_rows(source, 4)
    validate_definition_version(rows, digest)
    validate_label_alignment(base, rows)
    selected = sorted(set(episodes))
    if not selected:
        raise ValueError('Select at least one episode')
    frame_counts = {}
    stride = manifest['label_stride']
    if type(stride) is not int or stride < 1:
        raise ValueError('Invalid saved stride')
    for ep in selected:
        lo, hi = episode_bounds(base, ep)
        anchors = sorted(r['anchor_index'] for r in rows if r['episode_id'] == ep)
        if anchors != list(range(0, hi-lo, stride)):
            raise ValueError(f'Episode {ep} is incomplete; finish labeling first')
        frame_counts[str(ep)] = hi-lo
    rows = sorted((r for r in rows if r['episode_id'] in selected),
                  key=lambda r: (r['episode_id'], r['anchor_index']))
    corrected = fill_unknown_phases(rows)
    output = Path(output).resolve()
    protect_raw_data(base, output)
    if output.exists():
        raise FileExistsError(output)
    # Refuse to produce a mixed snapshot if labeling is still writing.
    if source.read_bytes() != original or manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError('Source predictions changed while postprocessing')
    encoded = ''.join(json.dumps(r, allow_nan=False)+'\n' for r in corrected).encode()
    manifest['postprocessing'] = dict(
        version=VERSION, source_labels=str(source.resolve()),
        source_sha256=hashlib.sha256(original).hexdigest(),
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        output_sha256=hashlib.sha256(encoded).hexdigest(),
        episodes=selected, frame_counts=frame_counts, requires_complete_episode=True,
        uses_future_phase_boundary=True, only_unknown=True,
        rules=['unknown after first finish -> finish', 'unknown before first turning -> approach'],
        precedence='finish before approach; explicit classes and middle unknowns preserved',
        corrected_confidence='0 (uncalibrated rule, not model confidence)')
    report = dict(before=statistics(rows), after=statistics(corrected), episodes={
        str(ep): dict(before=statistics([r for r in rows if r['episode_id'] == ep]),
                      after=statistics([r for r in corrected if r['episode_id'] == ep]))
        for ep in selected})
    output.mkdir(parents=True)
    (output/'auto_labels.jsonl').write_bytes(encoded)
    write_parquet(output/'auto_labels.parquet', corrected)
    atomic_json(output/'labeling_manifest.json', manifest)
    atomic_json(output/'labeling_statistics.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='context/qwen/config/context_labels_qwen35.yaml')
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    cfg, digest = definitions(args.config)
    base, _ = load_labeling_base(cfg)
    try:
        print(json.dumps(postprocess(base, cfg, digest, args.episodes, args.output_dir), indent=2))
    finally:
        base.close()


if __name__ == '__main__':
    main()
