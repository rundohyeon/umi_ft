"""Validated, immutable automatic labels and explicit human-review provenance."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import yaml


def definitions(path):
    cfg = yaml.safe_load(Path(path).read_text())
    contexts = cfg.get('contexts')
    if (not isinstance(contexts, dict) or len(contexts) < 2
            or any(type(key) is not int for key in contexts)
            or set(contexts) != set(range(len(contexts)))
            or cfg.get('unknown_label') != -1):
        raise ValueError('Context IDs must be consecutive integers starting at 0, with at least two classes and unknown_label=-1')
    for value in cfg['contexts'].values():
        if not value.get('name') or not value.get('description'):
            raise ValueError('Every context needs a name and description')
    digest = hashlib.sha256(json.dumps(cfg['contexts'], sort_keys=True).encode()).hexdigest()
    return cfg, digest


def parse_response(text, num_classes=5):
    if not isinstance(text, str):
        raise ValueError('Expected a JSON text response')
    text = text.strip()
    if text.startswith('```'):
        if len(text)<6 or not text.endswith('```'):
            raise ValueError('Incomplete JSON code fence')
        text = text[3:-3].strip()
        if text[:4].lower()=='json':
            text = text[4:].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'class_id', 'confidence', 'reason'}:
        raise ValueError('Expected exactly class_id, confidence, reason')
    if type(value['class_id']) is not int or value['class_id'] not in range(-1,num_classes):
        raise ValueError('Invalid class ID')
    c = value['confidence']
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1:
        raise ValueError('Invalid confidence')
    if not isinstance(value['reason'], str) or not 0 < len(value['reason'].strip()) <= 1000:
        raise ValueError('Invalid reason')
    return value


def read_rows(path, num_classes=5):
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix == '.parquet':
        import pandas as pd
        rows = pd.read_parquet(path).to_dict('records')
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        key = (int(row['episode_id']), int(row['anchor_index']))
        if key in seen:
            raise ValueError(f'Duplicate label {key}')
        seen.add(key)
        validate_class_ids([row], num_classes)
        if 'reviewed_class_id' in row and (not isinstance(row.get('is_reviewed'), (bool, np.bool_)) or not isinstance(row.get('is_modified'), (bool, np.bool_))):
            raise ValueError('Review provenance flags must be booleans')
        if 'confidence' in row and (not math.isfinite(float(row['confidence'])) or not 0 <= row['confidence'] <= 1):
            raise ValueError('Invalid stored confidence')
        if not math.isfinite(float(row['timestamp'])):
            raise ValueError('Invalid label timestamp')
    return rows


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def write_parquet(path, rows):
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, path)


class LabelIndex:
    """Auto uses causal hold; human decisions apply only to explicitly reviewed frames."""
    def __init__(self, auto=(), reviewed=(), source='reviewed', threshold=.9,
                 reviewed_weight=1., auto_weight=.3, mix_auto=False, num_classes=5):
        if source not in ('reviewed', 'high_confidence_auto', 'all_auto'):
            raise ValueError('Invalid label source')
        self.num_classes = num_classes
        auto, reviewed = list(auto), list(reviewed)
        validate_class_ids(auto + reviewed, num_classes)
        self.source, self.threshold = source, threshold
        self.reviewed_weight, self.auto_weight, self.mix_auto = reviewed_weight, auto_weight, mix_auto
        self.reviewed = {(int(r['episode_id']), int(r['anchor_index'])): r for r in reviewed}
        self.auto = {}
        for r in auto:
            self.auto.setdefault(int(r['episode_id']), []).append(r)
        for ep in self.auto:
            self.auto[ep].sort(key=lambda r: r['anchor_index'])
        self.anchors = {ep: np.array([r['anchor_index'] for r in rows]) for ep, rows in self.auto.items()}

    def automatic(self, episode, index):
        indices = self.anchors.get(episode, [])
        pos = np.searchsorted(indices, index, side='right') - 1
        return self.auto[episode][pos] if pos >= 0 else None

    def resolve(self, episode, index):
        reviewed = self.reviewed.get((episode, index))
        # An explicit reviewed unknown also blocks weak-label fallback.
        if reviewed and reviewed.get('is_reviewed'):
            label = int(reviewed['reviewed_class_id'])
            return label, self.reviewed_weight if label >= 0 else 0., 'modified' if reviewed['is_modified'] else 'reviewed'
        if self.source == 'reviewed' and not self.mix_auto:
            return -1, 0., 'unknown'
        row = self.automatic(episode, index)
        if row is None or row['class_id'] == -1:
            return -1, 0., 'unknown'
        if (self.source == 'high_confidence_auto' or self.mix_auto) and row['confidence'] < self.threshold:
            return -1, 0., 'unknown'
        return int(row['class_id']), self.auto_weight, 'auto'


def review_segment(existing, automatic, episode, start, end, timestamps, label=None, definition_hash=None):
    """Inclusive local frame bounds; label=None approves the held automatic class."""
    if not 0 <= start <= end < len(timestamps):
        raise ValueError('Invalid review segment')
    if label is not None and (isinstance(label, bool) or not isinstance(label, (int, np.integer))
                              or not -1 <= label < automatic.num_classes):
        raise ValueError('Invalid review class')
    rows = {(int(r['episode_id']), int(r['anchor_index'])): dict(r) for r in existing}
    now = datetime.now(timezone.utc).isoformat()
    for i in range(start, end + 1):
        auto = automatic.automatic(episode, i)
        auto_id = -1 if auto is None else int(auto['class_id'])
        corrected = auto_id if label is None else label
        rows[(episode, i)] = dict(episode_id=episode, anchor_index=i, timestamp=float(timestamps[i]),
            auto_class_id=auto_id, auto_confidence=0. if auto is None else float(auto['confidence']),
            reviewed_class_id=corrected, is_modified=corrected != auto_id, is_reviewed=True,
            review_timestamp=now, definition_hash=definition_hash, provenance='unknown' if corrected == -1 else ('modified' if corrected != auto_id else 'reviewed'))
    return [rows[key] for key in sorted(rows)]


def save_review(path, rows):
    path = Path(path)
    if path.name != 'reviewed_labels.parquet':
        raise ValueError('Review writes must target reviewed_labels.parquet, never automatic labels')
    write_parquet(path, rows)


def validate_definition_version(rows, digest):
    for row in rows:
        actual = row.get('definition_hash')
        if actual and actual != digest:
            raise ValueError('Labels use different context definitions; create a new version/output directory')


def validate_class_ids(rows, num_classes):
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError('num_classes must be an integer >= 2')
    for row in rows:
        for field in ('class_id', 'auto_class_id', 'reviewed_class_id'):
            if field not in row:
                continue
            value = row[field]
            if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                    or not -1 <= value < num_classes):
                raise ValueError(f'Invalid stored {field}={value!r} for {num_classes} classes')


def register_context_resolvers():
    # Resolved once by train.py, then stored as an integer in policy checkpoints.
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver(
        'context_num_classes', lambda path: len(definitions(path)[0]['contexts']), replace=True)
