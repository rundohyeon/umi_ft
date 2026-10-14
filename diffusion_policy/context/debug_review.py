"""Frame review of canonical RGB/F/T data; edits are separate, portable JSON patches."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from diffusion_policy.context.canonical_dataset import CanonicalRGBForceDataset, PHASE_NAMES
from diffusion_policy.context.ft_features import causal_ft_features


def file_digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class DebugRecording:
    def __init__(self, dataset_path, force_sidecar_path, label_path):
        self.dataset = CanonicalRGBForceDataset(dataset_path, force_sidecar_path, label_path,
                                               ft_feature_mode='raw_mean_delta')
        with np.load(label_path, allow_pickle=False) as labels:
            self.episode_names = labels['source_episode'].astype(str).tolist()
        self.label_digest = file_digest(label_path)

    def bounds(self, episode):
        ds = self.dataset
        if not 0 <= episode < len(ds.rgb_ends):
            raise ValueError('Episode is out of range')
        return int(ds.rgb_starts[episode]), int(ds.rgb_ends[episode])

    def original_class(self, frame):
        ds = self.dataset
        return int(ds.targets[frame]) if ds.valid[frame] else -1

    @lru_cache(maxsize=128)
    def image(self, frame):
        return np.asarray(self.dataset._get_rgb()[int(frame)])

    def causal_force(self, frame):
        ds = self.dataset
        episode = ds.frame_episode[frame]
        start, end = ds.force_starts[episode], ds.force_ends[episode]
        index = int(np.searchsorted(ds.force_times[start:end], ds.rgb_times[frame], side='right')) - 1
        if index < 0:
            return None, None
        index += int(start)
        return ds.wrench[index], float(ds.rgb_times[frame] - ds.force_times[index])

    @lru_cache(maxsize=4)
    def force_episode(self, episode):
        ds = self.dataset
        start, end = ds.force_starts[episode], ds.force_ends[episode]
        raw = ds.wrench[start:end]
        # Causal, native five-sample windows, repeating the episode's first
        # sample at startup just like the training loader. Never center the mean.
        features = [causal_ft_features(torch.from_numpy(
            np.pad(raw[:, side], ((9, 0), (0, 0)), mode='edge'))[None],
            mode='raw_mean_delta')[0].numpy() for side in (slice(0, 6), slice(6, 12))]
        mean = np.concatenate([f[:, 6:12] for f in features], axis=-1)
        delta = np.concatenate([f[:, 12:] for f in features], axis=-1)
        return ds.force_times[start:end], raw, mean, delta

    def apply(self, edits, episode, start_frame, end_frame, class_id):
        start, end = self.bounds(episode)
        if not 0 <= start_frame <= end_frame < end - start:
            raise ValueError('범위는 0 ≤ 시작 프레임 ≤ 끝 프레임 < 에피소드 길이여야 합니다.')
        if class_id not in (-1, 0, 1, 2, 3):
            raise ValueError('Invalid class ID')
        result = dict(edits)
        for frame in range(start + start_frame, start + end_frame + 1):
            if self.original_class(frame) == class_id:
                result.pop(frame, None)
            else:
                result[frame] = class_id
        return result

    def document(self, edits):
        ds = self.dataset
        rows = []
        for frame, class_id in sorted(edits.items()):
            if not 0 <= frame < len(ds.rgb_times) or class_id not in (-1, 0, 1, 2, 3):
                raise ValueError('Invalid review edit')
            episode = int(ds.frame_episode[frame])
            rows.append(dict(global_frame=int(frame), episode_number=episode + 1,
                             frame=int(frame - ds.rgb_starts[episode]),
                             timestamp=float(ds.rgb_times[frame]),
                             original_class_id=int(ds.targets[frame]), original_valid=bool(ds.valid[frame]),
                             class_id=int(class_id), valid=class_id >= 0))
        return dict(schema='canonical_context_debug_review_v1',
                    source_label_sha256=self.label_digest, phase_names=list(PHASE_NAMES),
                    source_labels=ds.label_path, saved_at=datetime.now(timezone.utc).isoformat(), edits=rows)

    def load(self, path):
        path = Path(path)
        if not path.exists():
            return {}, None
        data = path.read_bytes()
        document = json.loads(data)
        if (document.get('schema') != 'canonical_context_debug_review_v1'
                or document.get('source_label_sha256') != self.label_digest
                or document.get('phase_names') != list(PHASE_NAMES)):
            raise ValueError('Review JSON belongs to a different label source or schema')
        edits = {}
        for row in document['edits']:
            frame, class_id = row['global_frame'], row['class_id']
            if type(frame) is not int or type(class_id) is not int or frame in edits:
                raise ValueError('Invalid or duplicate review frame')
            # Check the complete frame mapping before displaying a restored edit.
            expected = self.document({frame: class_id})['edits'][0]
            if row != expected:
                raise ValueError('Review frame metadata does not match the recording')
            edits[frame] = class_id
        return edits, hashlib.sha256(data).hexdigest()

    def save(self, path, edits, expected_digest):
        path = Path(path).expanduser().resolve()
        if path.suffix != '.json':
            raise ValueError('Review output must be a separate .json file')
        ds = self.dataset
        for source in (ds.dataset_path, ds.force_sidecar_path, ds.label_path):
            source = Path(source).resolve()
            if path == source or source in path.parents:
                raise ValueError('Cannot write review output into a source dataset')
        data = (json.dumps(self.document(edits), indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.json.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if file_digest(path) != expected_digest:
                raise ValueError('다른 창에서 저장 파일을 변경했습니다. 수정 내용을 확인하고 페이지를 새로 여세요.')
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
                    temp_name = temp.name
                    temp.write(data)
                    temp.flush()
                    os.fsync(temp.fileno())
                os.replace(temp_name, path)
            finally:
                if temp_name and Path(temp_name).exists():
                    Path(temp_name).unlink()
        return hashlib.sha256(data).hexdigest()
