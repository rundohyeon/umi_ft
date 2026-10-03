#!/usr/bin/env python3
"""Passive real-robot RGB/F/T classifier evaluation; see docs/context_rgb_force_real_eval.md."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import yaml

from umi.real_world.rgb_force_context import (
    PHASE_NAMES, ObservationUnavailable, RGBForceContextRuntime,
    TimingLimits, TrainingImageTransform,
)


ROOT = Path(__file__).resolve().parent


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value)}')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + '\n')


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).expanduser().open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def self_test(runtime):
    """Synthetic forward pass, not a real-data accuracy or latency benchmark."""
    n = runtime.required_ft_history
    # Extra future samples verify that history selection stops at the RGB anchor.
    ft_times = np.arange(n + 20, dtype=np.float64) / 100
    anchor = ft_times[n - 1] + 0.005
    rgb_times = anchor - np.arange(3, -1, -1) / 60
    frames = np.full((4, 224, 224, 3), 128, dtype=np.uint8)
    wrench = np.zeros((len(ft_times), 12), dtype=np.float32)
    wrench[:n, 8] = np.linspace(0, 2, n)
    wrench[n:] = 999
    obs, timing = runtime.prepare(rgb_times, frames, ft_times, wrench, now=anchor)
    assert timing['ft_timestamps'][-1] <= anchor
    result = runtime.predict(obs)
    assert np.isclose(sum(result['probabilities']), 1, atol=1e-6)
    return dict(mode='self-test', synthetic_data=True, checkpoint=runtime.metadata,
                timing=timing, prediction=result)


def render_panel(rgb, record):
    import cv2
    panel = np.zeros((620, 800, 3), dtype=np.uint8)
    if rgb is not None:
        panel[:448, :448] = cv2.resize(rgb[..., ::-1], (448, 448))
    valid = record['valid']
    title = (f"{record['phase']}  {record['confidence']:.3f}" if valid
             else f"NO PREDICTION: {record['status']}")
    cv2.putText(panel, title[:85], (12, 477), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (80, 230, 80) if valid else (0, 180, 255), 1, cv2.LINE_AA)
    if valid:
        for i, (phase, probability) in enumerate(zip(PHASE_NAMES, record['probabilities'])):
            y = 50 + i * 90
            cv2.putText(panel, f'{phase}: {probability:.3f}', (465, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (465, y + 10), (465 + int(310 * probability), y + 28), (190, 130, 40), -1)
        text = (f"|F| left/right: {record['force_norm_left_n']:.2f} / "
                f"{record['force_norm_right_n']:.2f} N    "
                f"inference: {record['inference_ms']:.1f} ms")
        cv2.putText(panel, text, (12, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(panel, f"RGB age: {record['result_age_s']*1000:.0f} ms; "
                    f"F/T age at RGB: {record['ft_age_s']*1000:.1f} ms", (12, 555),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, 'q: quit   r: new episode (keep startup bias)', (12, 598),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return panel


def run_live(args, runtime):
    # Lazy imports: inspect/self-test/replay never open camera, Modbus, or robot controllers.
    import cv2
    from umi.real_world.context_sensor_streams import CameraStream, NativeFTStream

    with args.config.expanduser().open() as f:
        cfg = yaml.safe_load(f)
    if args.camera is not None:
        cfg['camera']['device'] = args.camera
    if args.gripper_ip is not None:
        cfg['ft']['hostname'] = args.gripper_ip
    if not cfg['camera']['device']:
        raise ValueError('Set --camera /dev/v4l/by-id/<camera>-video-index0 or camera.device in --config')
    limits = TimingLimits(**cfg['timing'])
    if limits.ft_sample_hz != 100:
        raise ValueError('Checkpoint was trained on native 100 Hz F/T')
    aruco_path = Path(cfg['aruco_config']).expanduser()
    if not aruco_path.is_absolute():
        aruco_path = ROOT / aruco_path
    transform = TrainingImageTransform(aruco_path)
    # Warm up model allocation before collecting live observations.
    self_test(runtime)
    output = args.output or ROOT / 'outputs' / 'context_real_eval' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.save_inputs:
        (output / 'inputs').mkdir()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    metadata = dict(schema='rgb_force_context_live_eval_v1', checkpoint=runtime.metadata,
                    checkpoint_path=str(args.checkpoint.expanduser().resolve()),
                    checkpoint_sha256=checkpoint_sha256(args.checkpoint),
                    git_commit=revision.stdout.strip(), config=cfg, timing=asdict(limits),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    timestamp_convention='Unix receive time minus configured latency',
                    wrench_convention='native sensor frame, software bias subtracted once, N/Nm',
                    requested_prediction_hz=args.rate, save_inputs=args.save_inputs)
    write_json(output / 'metadata.json', metadata)
    print(f'Output: {output}', flush=True)
    print('Keep both fingers unloaded and the gripper still for startup software bias calibration.', flush=True)
    started = time.monotonic()
    counts = dict(records=0, valid_predictions=0)
    exit_status = 'completed'
    try:
        with NativeFTStream(**cfg['ft']) as ft, CameraStream(**cfg['camera']) as camera, \
                (output / 'predictions.jsonl').open('w', buffering=1) as log:
            calibration = ft.calibrate(cfg['startup_bias'])
            bias = calibration['bias_12d']
            metadata['startup_bias'] = calibration
            write_json(output / 'metadata.json', metadata)
            episode_start, episode = time.time(), 0
            last_anchor, last_console = None, -np.inf
            print('Calibrated. Waiting for a full native history; q quits, r starts a new episode.', flush=True)
            deadline = time.monotonic()
            while args.max_seconds is None or time.monotonic() - started < args.max_seconds:
                rgb_times, frames = camera.snapshot()
                ft_times, raw_wrenches = ft.snapshot()
                now = time.time()
                obs, timing = None, {}
                try:
                    # prepare() rejects warm-up buffers before interpreting their values.
                    corrected = (np.stack(raw_wrenches) - bias if raw_wrenches else np.empty((0, 12)))
                    obs, timing = runtime.prepare(
                        rgb_times, frames, ft_times, corrected,
                        image_transform=transform, episode_start=episode_start, now=now, limits=limits)
                    if timing['anchor_timestamp'] == last_anchor:
                        raise ObservationUnavailable('waiting_for_new_rgb')
                    last_anchor = timing['anchor_timestamp']
                    result = runtime.predict(obs)
                    result_age = time.time() - last_anchor
                    if not 0 <= result_age <= limits.max_rgb_age_s:
                        raise ObservationUnavailable('inference_result_too_old')
                    result.update(status='ok', result_age_s=result_age,
                                  force_norm_left_n=float(np.linalg.norm(obs['robot0_ft_left'][-1, :3])),
                                  force_norm_right_n=float(np.linalg.norm(obs['robot0_ft_right'][-1, :3])))
                except ObservationUnavailable as exc:
                    result = dict(valid=False, status=str(exc), class_id=None, phase=None,
                                  confidence=None, probabilities=None, logits=None)
                record = dict(index=counts['records'], episode=episode,
                              recorded_at=time.time(), **timing, **result)
                if args.save_inputs and obs is not None:
                    relative_path = f"inputs/{counts['records']:08d}.npz"
                    np.savez_compressed(output / relative_path, **obs,
                                        rgb_timestamps=timing['rgb_timestamps'],
                                        ft_timestamps=timing['ft_timestamps'])
                    record['input_file'] = relative_path
                log.write(json.dumps(record, allow_nan=False) + '\n')
                counts['records'] += 1
                counts['valid_predictions'] += int(result['valid'])
                if time.monotonic() - last_console >= 1:
                    label = f"{record['phase']} {record['confidence']:.3f}" if record['valid'] else record['status']
                    print(f"episode={episode} {label}", flush=True)
                    last_console = time.monotonic()
                if not args.headless:
                    # Show the actual last input image, not an unrelated latest camera frame.
                    panel = render_panel(None if obs is None else obs['camera0_rgb'][-1], record)
                    cv2.imshow('RGB + F/T context observer', panel)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                    if key == ord('r'):
                        episode += 1
                        episode_start, last_anchor = time.time(), None
                deadline = max(deadline + 1 / args.rate, time.monotonic())
                time.sleep(max(0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        exit_status = 'interrupted'
    except Exception as exc:
        exit_status = f'failed: {exc}'
        raise
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        write_json(output / 'summary.json', dict(**counts, exit_status=exit_status,
                                                elapsed_s=time.monotonic() - started))
    print(f"Saved {counts['valid_predictions']} valid predictions to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trusted context_encoder_best.pt (may be renamed best.pt)')
    parser.add_argument('--mode', choices=['live', 'inspect', 'self-test', 'replay'], default='live')
    parser.add_argument('--device', default='cuda:0', help='Logical device after CUDA_VISIBLE_DEVICES; cpu is also supported')
    parser.add_argument('--config', type=Path, default=ROOT / 'example/eval_context_rgb_force.yaml')
    parser.add_argument('--camera', help='V4L2 device path; overrides camera.device')
    parser.add_argument('--gripper-ip', help='RG2-FT sensor IP; overrides ft.hostname')
    parser.add_argument('--output', type=Path, help='New output directory (must not already exist)')
    parser.add_argument('--rate', type=float, default=10, help='Maximum predictions per second; sensors always capture at 60/100 Hz')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--headless', action='store_true', help='No GUI; stop with Ctrl-C or --max-seconds')
    parser.add_argument('--max-seconds', type=float, help='Bound live run duration including calibration')
    parser.add_argument('--save-inputs', action='store_true', help='Save exact preprocessed input windows as NPZ for replay')
    parser.add_argument('--replay-inputs', type=Path, help='One NPZ window saved by --save-inputs')
    args = parser.parse_args()
    if not np.isfinite(args.rate) or args.rate <= 0 or args.cpu_threads < 1:
        parser.error('--rate and --cpu-threads must be positive')
    if args.max_seconds is not None and (not np.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error('--max-seconds must be finite and positive')
    if args.mode == 'replay' and args.replay_inputs is None:
        parser.error('--mode replay requires --replay-inputs')
    torch.set_num_threads(args.cpu_threads)
    runtime = RGBForceContextRuntime(args.checkpoint, device='cpu' if args.mode == 'inspect' else args.device)
    if args.mode == 'inspect':
        print(json.dumps(runtime.metadata, indent=2))
    elif args.mode == 'self-test':
        print(json.dumps(self_test(runtime), indent=2))
    elif args.mode == 'replay':
        with np.load(args.replay_inputs, allow_pickle=False) as data:
            obs = {key: data[key] for key in runtime.model.input_keys}
        print(json.dumps(runtime.predict(obs), indent=2))
    else:
        run_live(args, runtime)


if __name__ == '__main__':
    main()
