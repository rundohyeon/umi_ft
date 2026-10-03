"""Qwen3.5 inference in an isolated interpreter, with lossless local RGB transport.

The parent uses the existing UMI dataset environment. Only the worker imports
modern Transformers/PyTorch; stdout is reserved for the JSON-lines protocol.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys


class Qwen35Labeler:
    def __init__(self, python, model_path, *, offline=True, seed=42,
                 max_new_tokens=160, device='cuda:0', max_pixels=224*224):
        options = dict(model_path=str(Path(model_path).resolve()), offline=offline,
                       seed=seed, max_new_tokens=max_new_tokens, device=device,
                       max_pixels=max_pixels)
        self.process = subprocess.Popen(
            [str(python), '-u', str(Path(__file__).resolve()), json.dumps(options)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            ready = self._receive()
            if ready.get('status') != 'ready':
                raise RuntimeError(f'Unexpected Qwen3.5 startup response: {ready}')
            self.runtime = ready['runtime']
        except BaseException:
            self.close()
            raise

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Qwen3.5 worker exited; see its traceback above')
        value = json.loads(line)
        if 'error' in value:
            raise RuntimeError(f"Qwen3.5 worker: {value['error']}")
        return value

    def __call__(self, prompt, images=None):
        from PIL import Image
        if not images:
            raise ValueError('Qwen3.5 vision labeling requires causal RGB frames')
        encoded = []
        for frame in images:
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format='PNG')
            encoded.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
        request = dict(prompt=prompt, images_png=encoded)
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        return self._receive()['response']

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()


class Inference:
    def __init__(self, model_path, offline, seed, max_new_tokens, device, max_pixels):
        import os
        from importlib.metadata import version, PackageNotFoundError
        os.environ.setdefault('TRITON_CACHE_DIR', str(
            Path(__file__).resolve().parents[2] / 'context/qwen/.triton_cache'))
        if offline:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        torch.manual_seed(seed)
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=offline, trust_remote_code=False,
            min_pixels=64*64, max_pixels=max_pixels)
        self.device, self.max_new_tokens = device, max_new_tokens
        dtype = torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=offline, trust_remote_code=False,
            dtype=dtype, attn_implementation='sdpa', device_map={'': device}).eval()
        self.runtime = dict(python=sys.version.split()[0], torch=torch.__version__,
                            transformers=transformers.__version__, dtype=str(dtype),
                            attention='sdpa', enable_thinking=False)
        for package in ('flash-linear-attention', 'causal-conv1d', 'triton'):
            try:
                self.runtime[package] = version(package)
            except PackageNotFoundError:
                self.runtime[package] = None

    def __call__(self, prompt, images):
        import torch
        content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': prompt}]
        text = self.processor.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        tokens = self.processor(text=[text], images=images, padding=True,
                                return_tensors='pt').to(self.device)
        with torch.inference_mode():
            result = self.model.generate(
                **tokens, max_new_tokens=self.max_new_tokens, do_sample=False,
                num_beams=1, temperature=1., top_p=1., top_k=50,
                pad_token_id=self.processor.tokenizer.eos_token_id)
        return self.processor.tokenizer.decode(
            result[0, tokens['input_ids'].shape[1]:], skip_special_tokens=True)


def serve(options):
    from PIL import Image
    # Library progress/log messages must never corrupt the response stream.
    with contextlib.redirect_stdout(sys.stderr):
        inference = Inference(**options)
    print(json.dumps(dict(status='ready', runtime=inference.runtime)), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        images = [Image.open(io.BytesIO(base64.b64decode(value))).convert('RGB')
                  for value in request['images_png']]
        with contextlib.redirect_stdout(sys.stderr):
            response = inference(request['prompt'], images)
        print(json.dumps(dict(response=response)), flush=True)


if __name__ == '__main__':
    serve(json.loads(sys.argv[1]))
