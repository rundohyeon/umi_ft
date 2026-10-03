"""Timestamped diagnostics for the existing real-robot history-buffer path."""
import json
from pathlib import Path
import numpy as np


def log_context(result, timestamp, cycle, directory, log_every=10):
    context=result.get('context')
    if context is None: return
    raw=context['raw_probabilities'][0].detach().cpu().tolist()
    smoothed=context['smoothed_probabilities'][0].detach().cpu().tolist()
    used=context['used_probabilities'][0].detach().cpu().tolist()
    if not np.isfinite(raw+smoothed+used).all():
        raise ValueError('Nonfinite online context probabilities')
    record=dict(timestamp=float(timestamp),cycle=int(cycle),predicted=int(np.argmax(raw)),
                raw_probabilities=raw,smoothed_probabilities=smoothed,used_probabilities=used)
    if directory is not None:
        path=Path(directory)/'context_probabilities.jsonl'
        with path.open('a') as stream: stream.write(json.dumps(record)+'\n')
    if log_every>0 and cycle%log_every==0:
        print(f'context: predicted={record["predicted"]} probabilities={np.round(raw,4).tolist()}')
