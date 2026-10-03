"""Exercise the real interpreter boundary without loading model weights."""
import json
import subprocess
import sys

import numpy as np
import pytest

from diffusion_policy.context import qwen35


@pytest.fixture
def worker_script(tmp_path, monkeypatch):
    script = tmp_path / 'worker.py'
    script.write_text('''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('worker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class FakeInference:
    def __init__(self, **options):
        print('loading noise')
        self.runtime = {'device': options['device'], 'enable_thinking': False}
    def __call__(self, prompt, images):
        print('generation noise')
        if prompt == 'crash':
            raise RuntimeError('fixture inference failure')
        return json.dumps({'prompt': prompt, 'pixels': [list(im.getdata()) for im in images]})
module.Inference = FakeInference
module.serve(json.loads(sys.argv[2]))
''')
    original = subprocess.Popen
    def launch(args, **kwargs):
        return original([sys.executable, '-u', str(script), args[2], args[3]], **kwargs)
    monkeypatch.setattr(qwen35.subprocess, 'Popen', launch)


def test_worker_preserves_rgb_order_prompt_and_sequential_requests(worker_script):
    labeler = qwen35.Qwen35Labeler(sys.executable, '.', device='cpu')
    frames = [np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
              np.full((2, 3, 3), 255, dtype=np.uint8)]
    try:
        assert labeler.runtime == {'device': 'cpu', 'enable_thinking': False}
        for prompt in ['force |F| 증가\n{"last": 4.2}', 'next prediction']:
            response = json.loads(labeler(prompt, frames))
            assert response['prompt'] == prompt
            assert response['pixels'] == [im.reshape(-1, 3).tolist() for im in frames]
        with pytest.raises(ValueError, match='RGB'):
            labeler('empty')
    finally:
        labeler.close()
    assert labeler.process.returncode == 0


def test_worker_failure_is_reported_and_process_reaped(worker_script):
    labeler = qwen35.Qwen35Labeler(sys.executable, '.', device='cpu')
    try:
        with pytest.raises(RuntimeError, match='worker exited'):
            labeler('crash', [np.zeros((2, 2, 3), dtype=np.uint8)])
    finally:
        labeler.close()
    assert labeler.process.returncode != 0


def test_resume_requires_same_qwen35_runtime(context_base, tmp_path):
    from diffusion_policy.context.labels import definitions
    from scripts.generate_context_labels import generate
    base, _ = context_base
    cfg, digest = definitions('context/qwen/config/context_labels_qwen35.yaml')
    cfg.update(output_dir=str(tmp_path / 'labels'), model_path='fixture')
    class Labeler:
        runtime = {'transformers': 'fixture-version', 'enable_thinking': False}
        def __call__(self, prompt, images=None):
            assert prompt.endswith(cfg['prompt_suffix'])
            return '{"visibility":"unclear","motion":"unclear","confidence":0.2}'
    labeler = Labeler()
    generate(base, cfg, digest, labeler, episode=0, max_windows=1)
    manifest = json.loads((tmp_path / 'labels/labeling_manifest.json').read_text())
    assert manifest['decoding']['enable_thinking'] is False
    assert manifest['inference_runtime'] == labeler.runtime
    assert manifest.get('prompt_suffix','') == cfg['prompt_suffix']
    generate(base, cfg, digest, labeler, episode=0, max_windows=1, resume=True)
    labeler.runtime = dict(labeler.runtime, transformers='different-version')
    with pytest.raises(ValueError, match='Resume'):
        generate(base, cfg, digest, labeler, episode=0, max_windows=1, resume=True)
    labeler.runtime = manifest['inference_runtime']
    cfg['prompt_suffix'] += ' Changed formatting.'
    with pytest.raises(ValueError, match='Resume'):
        generate(base, cfg, digest, labeler, episode=0, max_windows=1, resume=True)
