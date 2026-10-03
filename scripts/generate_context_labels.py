#!/usr/bin/env python
"""Offline greedy local Qwen labeling. Never imports into policy/runtime code."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from diffusion_policy.context.data import load_base, load_labeling_base, causal_summary, causal_rgb, episode_bounds, source_fingerprint, validate_label_alignment
from diffusion_policy.context.labels import definitions, parse_response, read_rows, write_parquet, atomic_json


class LocalLabeler:
    def __init__(self, model_path, offline=True, seed=42, max_new_tokens=160, device='cpu',
                 backend='text', max_pixels=224*224):
        if offline:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        torch.manual_seed(seed)
        self.vision=backend=='qwen2_5_vl'
        if self.vision:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.processor=AutoProcessor.from_pretrained(model_path,local_files_only=offline,
                trust_remote_code=False,use_fast=False,min_pixels=56*56,max_pixels=max_pixels)
            self.tokenizer=self.processor.tokenizer
            self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
                local_files_only=offline,trust_remote_code=False,attn_implementation='eager',
                torch_dtype=torch.float16 if torch.device(device).type=='cuda' else torch.float32).to(device).eval()
        elif backend=='text':
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=offline, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=offline,
                trust_remote_code=False).to(device).eval()
        else:
            raise ValueError(f'Unknown labeling backend: {backend}')
        self.device, self.max_new_tokens = device, max_new_tokens

    def __call__(self, prompt, images=None):
        import torch
        if self.vision:
            from PIL import Image
            if not images: raise ValueError('Vision labeling requires causal RGB frames')
            content=[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]
            text=self.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            tokens=self.processor(text=[text],images=[Image.fromarray(frame) for frame in images],
                padding=True,return_tensors='pt').to(self.device)
        else:
            if images: raise ValueError('Text-only Qwen cannot receive RGB; use the vision backend/environment')
            ids = self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                tokenize=True, add_generation_prompt=True, return_tensors='pt').to(self.device)
            tokens={'input_ids':ids,'attention_mask':torch.ones_like(ids)}
        with torch.inference_mode():
            result = self.model.generate(**tokens,
                max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                # Neutralize sampling-only defaults shipped with the checkpoint.
                # Greedy decoding and checkpoint files stay unchanged.
                temperature=1.0, top_p=1.0, top_k=50,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(result[0,tokens['input_ids'].shape[1]:], skip_special_tokens=True)


def make_prompt(contexts, summary):
    classes = '\n'.join(f'{i}: {v["name"]}: {v["description"]}' for i,v in sorted(contexts.items()))
    return ('Label the robot situation at the LAST observation, using only evidence available up to that time.\n'
        'The robot pushes the orange valve lever with its RIGHT finger. It does NOT grasp or pinch the valve.\n'
        'CAMERA GEOMETRY: This is a moving camera attached to the robot gripper, not a fixed external camera. '
        'During turning the camera, finger and lever rotate together. The lever can therefore stay nearly '
        'stationary in image coordinates while the fixed gray rig, signs and background rotate across frames. '
        'Compare those fixed background features relative to the finger/lever, especially near the LAST frame. '
        'Right-finger force plus coherent background rotation while finger-lever interaction is maintained '
        'is evidence of turning, even if the lever stays in the same image position. '
        'An image-stationary lever is not sufficient evidence that turning stopped or that the phase is unknown. '
        'Distinguish background rotation from the translation or scale change caused by approaching/receding. '
        'Camera repositioning alone does not prove lever turning. TCP rotation can corroborate camera rotation '
        'but does not replace visual interaction and right-finger force evidence.\n'
        'Task phase definitions:\n' + classes
        + '\nUse synchronized visual and force evidence. right_force_norm_N and right_force_history belong '
        'to the RIGHT finger; left-finger force cannot substitute for right-finger force in turning. '
        'Force measurements contain only the nonnegative magnitude |F| = sqrt(Fx^2 + Fy^2 + Fz^2) '
        'in newtons; signed force components and torque values are not supplied. '
        'Force increase/decrease refers to changes in this magnitude over time. '
        'right_force_history rows contain offset_s and force_norm_N, ordered from oldest to newest; offsets '
        'are seconds relative to the last RGB frame. They retain short force peaks as well as low values. '
        'Compare force at the time of the inferred turning motion, rather than a maximum from another time. '
        'Missing or stale force measurements are unknown, not measured zero force. '
        'No numerical contact or sharp-drop threshold has been calibrated; compare recent force magnitudes '
        'over time and choose unknown when force evidence is ambiguous. '
        'A previously visible lever disappearing is an error under the task definition, including during retreat. '
        'For finish, use a sharp decrease in the RIGHT finger force magnitude near the LAST observation. '
        'Compare consecutive recent right_force_history measurements, rather than only the net change '
        'across the whole window. A preceding force rise, rotation stop, and retreat are not prerequisites. '
        'Steady low force or a small fluctuation alone does not establish finish. '
        'Do not label an earlier frame finish using a force drop that occurs in future frames.\n'
        'Observation window (causal, SI units):\n' + json.dumps(summary,sort_keys=True)
        + '\nRGB images follow rgb_frames order from oldest to newest. '
        'Previous predictions are fallible context, NOT ground truth: do not copy them when observations disagree. '
        'They are not proof that a force drop or a visual event occurred. Elapsed time is not completion percentage. '
        'Choose unknown (-1) if the supplied observations do not establish any task phase; approach is not a fallback. '
        'Return one JSON object only, with exactly three keys: reason (a short description of background '
        'rotation relative to the finger/lever, lever visibility, synchronized right-finger force, '
        'and the recent force-magnitude change when relevant), '
        'class_id (integer), confidence (number from 0 to 1). '
        f'Allowed class_id values: -1 and 0 through {len(contexts)-1}. Do not include any other keys.')


def label_with_retries(labeler, prompt, retries, num_classes=5, images=None,
                       response_parser=None, failure_result=None):
    attempts, failures = [], 0
    for attempt in range(retries+1):
        actual_prompt = prompt if attempt == 0 else prompt + '\nYour previous response was invalid. Return one strict JSON object only.'
        response = labeler(actual_prompt,images=images) if images else labeler(actual_prompt)
        attempts.append({'prompt':actual_prompt, 'response':response})
        try:
            result = response_parser(response) if response_parser else parse_response(response,num_classes)
            return result, attempts, failures
        except (ValueError, TypeError, KeyError):
            failures += 1
    return (failure_result if failure_result is not None else
            {'class_id':-1, 'confidence':0., 'reason':'Invalid structured output after limited retries'}), attempts, failures


def prior_context(rows, episode, anchor, timestamp, class_names, count=8):
    """Only earlier predictions from this episode; never end time or future labels."""
    past=sorted((row for row in rows if row['episode_id']==episode and row['anchor_index']<anchor),
                key=lambda row:row['anchor_index'])
    recent=[dict(anchor_index=row['anchor_index'],seconds_ago=float(timestamp-row['timestamp']),
                 class_id=row['class_id'],name=class_names.get(row['class_id'],'unknown'),confidence=row['confidence'])
            for row in past[-count:]] if count else []
    progress={'source':'previous_automatic_predictions','last_predicted_class':None,'seconds_since_predicted_transition':None}
    if past:
        last=past[-1];start=last['timestamp']
        for row in reversed(past[:-1]):
            if row['class_id']!=last['class_id']:break
            start=row['timestamp']
        progress.update(last_predicted_class=last['class_id'],seconds_since_predicted_transition=float(timestamp-start))
    return recent,progress


def model_identity(path):
    p = Path(path)
    if not p.is_dir():
        return str(path)
    # Content hashes bind local model/config/tokenizer bytes, not just the path name.
    digest = hashlib.sha256()
    for f in sorted(p.rglob('*')):
        if f.is_file():
            digest.update(str(f.relative_to(p)).encode())
            with f.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def generate(base, cfg, definition_hash, labeler, *, resume=False, episode=None,
             start_index=0, end_index=None, max_windows=None, identity=None,
             parallel_labelers=()):
    """Distribute independent causal windows; only this thread reads/writes data."""
    from diffusion_policy.context.data import protect_raw_data
    num_classes=len(cfg['contexts'])
    output = Path(cfg['output_dir'])
    protect_raw_data(base, output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'auto_labels.jsonl'
    manifest_path = output/'labeling_manifest.json'
    inputs=dict(rgb=False,rgb_frames=4,tcp_position=False,tcp_rotation=True,gripper_width=True,
                force_torque=True,force_history_bins=0,previous_labels=True,previous_label_count=8)
    inputs.update(cfg.get('inputs',{}))
    if inputs['previous_label_count']<0 or inputs['rgb_frames']<1 or inputs['force_history_bins']<0:
        raise ValueError('Invalid RGB/prior-label history length')
    decision_mode = cfg.get('decision_mode','llm_class')
    if decision_mode not in ('llm_class','force_visual_v1'):
        raise ValueError('Unknown labeling decision mode')
    force_cfg = None
    prompt_builder = make_prompt
    if decision_mode == 'force_visual_v1':
        from diffusion_policy.context.force_visual import (
            settings, evidence_at, prepare_summary, make_visual_prompt, parse_visual_response, decide)
        if [cfg['contexts'][i]['name'] for i in range(num_classes)] != ['approach','turning','finish','error']:
            raise ValueError('Force/visual decisions require approach/turning/finish/error classes in order')
        if not inputs['rgb'] or not inputs['force_torque']:
            raise ValueError('Force/visual decisions require RGB and native force inputs')
        force_cfg = settings(cfg.get('force_decision',{}))
        prompt_builder = make_visual_prompt
    manifest = {'definition_hash':definition_hash, 'contexts':cfg['contexts'], 'label_version':cfg['label_version'],
        'model_name':cfg['model_path'], 'model_identity':identity or cfg['model_path'],
        'source_fingerprint':source_fingerprint(base), 'history_length':cfg['history_length'],
        'label_stride':cfg['label_stride'], 'seed':cfg['seed'], 'max_retries':cfg['max_retries'],
        'max_new_tokens':cfg['max_new_tokens'], 'offline':cfg['offline'], 'prompt_version':7, 'num_classes':num_classes,
        'backend':cfg.get('backend','text'),'inputs':inputs,'start_index':start_index,
        'prompt_template_hash':hashlib.sha256(prompt_builder(cfg['contexts'],{}).encode()).hexdigest(),
        'max_pixels':cfg.get('max_pixels',224*224),
        'decoding':{'do_sample':False,'num_beams':1}}
    if cfg.get('backend') == 'qwen3_5':
        manifest['decoding']['enable_thinking'] = False
        manifest['inference_runtime'] = labeler.runtime
    if cfg.get('prompt_suffix'):
        manifest['prompt_suffix'] = cfg['prompt_suffix']
        manifest['prompt_template_hash'] = hashlib.sha256(
            (prompt_builder(cfg['contexts'],{}) + cfg['prompt_suffix']).encode()).hexdigest()
    if force_cfg is not None:
        manifest.update(decision_mode=decision_mode, force_decision=force_cfg,
                        prompt_version=8, force_decision_version=1)
    # Compare canonical JSON to avoid YAML integer-key conversion differences.
    manifest = json.loads(json.dumps(manifest))
    if path.exists() and not resume:
        raise FileExistsError('Automatic labels exist; use --resume or a different output directory')
    if resume and path.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume configuration/model/data/definitions differ from cached labeling run')
    else:
        atomic_json(manifest_path, manifest)
    rows = read_rows(path,num_classes)
    validate_label_alignment(base,rows)
    seen = {(r['episode_id'], r['anchor_index']) for r in rows}
    by_episode=defaultdict(list)
    for row in rows:by_episode[row['episode_id']].append(row)
    # With prediction history, an existing later label cannot predate a missing
    # earlier prediction. A prefix is recoverable after interruption; a hole is not.
    for ep,saved in by_episode.items():
        anchors=sorted(row['anchor_index'] for row in saved)
        if anchors!=list(range(start_index,start_index+len(anchors)*cfg['label_stride'],cfg['label_stride'])):
            raise ValueError('Cached labels must form a chronological prefix per episode; use a new output directory')
    if cfg['label_stride'] < 1 or start_index < 0 or (max_windows is not None and max_windows < 1):
        raise ValueError('Invalid stride/index/window limit')
    episodes = iter(range(len(base.rgb_episode_ends)) if episode is None else [episode])
    def episode_windows(ep):
        lo,hi=episode_bounds(base,ep)
        stop=min(hi-lo,end_index if end_index is not None else hi-lo)
        for anchor in range(start_index,stop,cfg['label_stride']):
            if (ep,anchor) not in seen:yield ep,anchor,lo
    labelers = (labeler, *parallel_labelers)
    active=[None]*len(labelers)
    class_names={i:entry['name'] for i,entry in cfg['contexts'].items()}
    submitted=0
    with ThreadPoolExecutor(max_workers=len(labelers)) as executor, path.open('a') as stream:
        while True:
            pending = []
            for slot,worker in enumerate(labelers):
                if max_windows is not None and submitted>=max_windows:break
                entry=None
                while entry is None:
                    if active[slot] is None:
                        ep=next(episodes,None)
                        if ep is None:break
                        active[slot]=episode_windows(ep)
                    entry=next(active[slot],None)
                    if entry is None:active[slot]=None
                if entry is None:continue
                ep,anchor,lo=entry
                timestamp=float(base.rgb_timestamps[lo+anchor])
                summary, ws, we = causal_summary(base,ep,anchor,cfg['history_length'],inputs)
                summary['episode_elapsed_s']=float(timestamp-base.rgb_timestamps[lo])
                images=None
                if inputs['rgb']:
                    images,summary['rgb_frames']=causal_rgb(base,ep,ws,we,inputs['rgb_frames'])
                if inputs['previous_labels']:
                    summary['previous_predictions'],summary['progress']=prior_context(by_episode[ep],ep,anchor,
                        timestamp,class_names,inputs['previous_label_count'])
                force, response_options = None, {}
                if force_cfg is not None:
                    past = by_episode[ep]
                    previous_contact = (json.loads(past[-1]['force_evidence_json'])['contact']
                                        if past else None)
                    force = evidence_at(base,ep,anchor,force_cfg,previous_contact)
                    summary = prepare_summary(summary,force)
                    response_options = dict(response_parser=parse_visual_response,
                        failure_result=dict(visibility='unclear',motion='unclear',confidence=0.))
                prompt = prompt_builder(cfg['contexts'],summary) + cfg.get('prompt_suffix','')
                future = executor.submit(label_with_retries,worker,prompt,cfg['max_retries'],num_classes,images,
                                         **response_options)
                pending.append((ep,anchor,lo,ws,we,prompt,future,force))
                submitted+=1
            if not pending:break
            # Different episodes run concurrently; each episode is strictly sequential
            # so its previous prediction is available before the next prompt is built.
            for ep,anchor,lo,ws,we,prompt,future,force in pending:
                result, attempts, failures = future.result()
                if force is not None:
                    result = decide(result,force)
                row = dict(episode_id=ep, anchor_index=anchor, timestamp=float(base.rgb_timestamps[lo+anchor]),
                    **result, model_name=cfg['model_path'],label_version=cfg['label_version'],window_start=ws,window_end=we,
                    definition_hash=definition_hash, provenance='unknown' if result['class_id']==-1 else 'auto',
                    prompt=prompt, attempts_json=json.dumps(attempts), parsing_failures=failures)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
                by_episode[ep].append(row)
    if rows: write_parquet(output/'auto_labels.parquet', rows)
    counts = {str(i):sum(r['class_id']==i for r in rows) for i in range(num_classes)}
    stats = dict(sample_count=len(rows),count_per_class=counts,
        class_distribution={k:v/max(1,len(rows)) for k,v in counts.items()},
        unknown_count=sum(r['class_id']==-1 for r in rows),
        average_confidence=float(np.mean([r['confidence'] for r in rows])) if rows else 0.,
        parsing_failures=sum(r['parsing_failures'] for r in rows))
    atomic_json(output/'labeling_statistics.json', stats)
    print(json.dumps(stats,indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='context/qwen/config/context_labels.yaml')
    parser.add_argument('--model-path')
    device_args=parser.add_mutually_exclusive_group()
    device_args.add_argument('--device',default='cpu',help='One device, e.g. cpu or cuda:0')
    device_args.add_argument('--devices',nargs='+',help='One model per visible CUDA device, e.g. cuda:0 cuda:1')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--episode',type=int)
    parser.add_argument('--start-index',type=int,default=0)
    parser.add_argument('--end-index',type=int)
    parser.add_argument('--max-windows',type=int)
    parser.add_argument('--label-stride',type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--dataset-override',action='append',default=[])
    parser.add_argument('--allow-placeholder-classes',action='store_true',help='Only for testing the mechanics; labels will not have task semantics')
    args=parser.parse_args()
    devices=args.devices or [args.device]
    if args.devices:
        import torch
        try:
            devices=[str(torch.device(device)) for device in devices]
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        if (len(set(devices)) != len(devices)
                or any(torch.device(device).type != 'cuda' or torch.device(device).index is None for device in devices)):
            parser.error('--devices requires distinct explicit CUDA devices, e.g. cuda:0 cuda:1')
        count=torch.cuda.device_count()
        if any(torch.device(device).index >= count for device in devices):
            parser.error(f'Requested CUDA device is unavailable; visible device count is {count}')
    cfg,digest=definitions(args.config)
    if not args.allow_placeholder_classes and any('TODO' in v['description'] for v in cfg['contexts'].values()):
        parser.error('Define the context descriptions in the YAML before labeling')
    for name in ('model_path','label_stride','output_dir'):
        if getattr(args,name) is not None: cfg[name]=getattr(args,name)
    base,_=load_labeling_base(cfg,args.dataset_override)
    labelers=[]
    try:
        print(f'Labeling devices: {devices}; CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}',flush=True)
        if cfg.get('inputs',{}).get('rgb') and cfg.get('backend','text')=='text':
            raise ValueError('RGB labeling requires a vision backend and its separate VLM environment')
        for device in devices:
            if cfg.get('backend') == 'qwen3_5':
                from diffusion_policy.context.qwen35 import Qwen35Labeler
                worker=Qwen35Labeler(cfg['inference_python'],cfg['model_path'],
                    offline=cfg['offline'],seed=cfg['seed'],max_new_tokens=cfg['max_new_tokens'],
                    device=device,max_pixels=cfg.get('max_pixels',224*224))
            else:
                worker=LocalLabeler(cfg['model_path'],cfg['offline'],cfg['seed'],cfg['max_new_tokens'],device,
                    backend=cfg.get('backend','text'),max_pixels=cfg.get('max_pixels',224*224))
            labelers.append(worker)
        generate(base,cfg,digest,labelers[0],resume=args.resume,episode=args.episode,start_index=args.start_index,
            end_index=args.end_index,max_windows=args.max_windows,identity=model_identity(cfg['model_path']),
            parallel_labelers=labelers[1:])
    finally:
        for worker in labelers:
            if hasattr(worker,'close'): worker.close()
        base.close()

if __name__=='__main__': main()
