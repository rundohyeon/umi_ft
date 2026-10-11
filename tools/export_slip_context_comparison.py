#!/usr/bin/env python
"""Render held-out slip RGB/F/T with ground truth and both branches of best.pt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader, Subset

from diffusion_policy.context.slip_sequence_dataset import SlipSequenceDataset
from diffusion_policy.context.slip_feature_cache import ensure_feature_cache
from models.slip_temporal_context import SlipTemporalContextEncoder, PHASE_NAMES
from train_context_rgb_force import autocast_context, move_obs

W, H = 1600, 1080
BG, PANEL, TEXT, MUTED = '#101720', '#1b2735', '#eef4fb', '#a6b6ca'
COLORS = {0: '#60b3ff', 1: '#ff9866', 2: '#d49bff', 3: '#64dfad', -1: '#475569', -100: '#475569'}
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS = {n: ImageFont.truetype(FONT_PATH, n, index=1) for n in (14, 16, 18, 20, 22, 26, 30)}
AXIS_COLORS = ('#71b8ff', '#ffb56b', '#cc95ff', '#62dfc1')


def text(draw, xy, value, size=18, fill=TEXT):
    draw.text(xy, str(value), font=FONTS[size], fill=fill)


def name(cid):
    return PHASE_NAMES[cid] if 0 <= cid < 4 else 'ignore'


@torch.inference_mode()
def predict_episode(ds, payload, episode, model, device):
    test = ds.subset(payload['manifest']['episode_splits']['test'])
    test.attach_features(ds.feature_path)
    positions = np.flatnonzero(test.frame_episode[test.indices] == episode)
    if not len(positions):
        raise ValueError('Episode has no eligible test observations')
    bs = payload['training_config']['batch_size']
    # Preserve the exact batch boundaries of the saved full-test evaluation.
    start = int(positions[0]) // bs * bs
    end = min(len(test), (int(positions[-1]) // bs + 1) * bs)
    loader = DataLoader(Subset(test, range(start, end)), batch_size=bs, shuffle=False, num_workers=0)
    frames, final, base, history = [], [], [], []
    try:
        for batch in loader:
            with autocast_context(device, bool(payload['training_config']['amp'])):
                out = model(move_obs(batch, device))
            selected = batch['episode_id'].numpy() == episode
            frames.extend(batch['frame_index'].numpy()[selected].tolist())
            final.extend(out['logits'].float().softmax(-1).cpu().numpy()[selected])
            base.extend(out['observation_logits'][:, -1].float().softmax(-1).cpu().numpy()[selected])
            history.extend(batch['obs']['sequence_valid'][:, :-1].sum(-1).numpy()[selected].tolist())
    finally:
        test.close()
    return np.asarray(frames), np.asarray(base), np.asarray(final), np.asarray(history)


def runs(values):
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return zip(boundaries[:-1], boundaries[1:])


def make_background(times, wrench_times, wrench, labels, base_display, final_display, original, source, epoch, rgb_horizon=4):
    duration = float(times[-1] + np.median(np.diff(times)))
    canvas = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(canvas)
    text(d, (24, 15), f'Context 비교 · 원본 EP{original:03d} · {source}', 30)
    text(d, (24, 54), f'checkpoint epoch {epoch} · test episode · 예측 약 10 Hz / RGB 59.94 fps', 16, MUTED)
    for left in (600, 1092):
        d.rounded_rectangle((left, 82, left + 484, 343), radius=12, fill=PANEL)
    text(d, (620, 92), 'RGB/F/T 관측만 · history 보정 전', 20)
    text(d, (1112, 92), 'RGB/F/T + history · 최종 예측', 20)
    text(d, (24, 618 if rgb_horizon == 8 else 684), f'모델에 들어간 RGB {rgb_horizon}장 · 최근 추론 시점 기준', 18)
    text(d, (24, 890), '양손 F/T: bias 제거 원본 값 · 모델은 각 RGB 이전 41개 측정 사용', 16, MUTED)
    charts = []
    for side, row_y, offset in (('오른손', 400, 6), ('왼손', 640, 0)):
        for torque, x0, x1 in ((False, 653, 1062), (True, 1155, 1564)):
            y0, y1 = row_y + 30, row_y + 155
            values = wrench[:, offset + (3 if torque else 0):offset + (6 if torque else 3)]
            values = np.c_[values, np.linalg.norm(values, axis=-1)]
            lo, hi = min(0., float(values.min())), max(.01, float(values.max()))
            margin = max(.01, .08 * (hi - lo))
            lo, hi = lo - margin, hi + margin
            def px(t): return x0 + t / duration * (x1 - x0)
            def py(v): return y1 - (v - lo) / (hi - lo) * (y1 - y0)
            unit, prefix = ('N·m', 'T') if torque else ('N', 'F')
            text(d, (x0 - 40, row_y - 24), f'{side} {prefix} ({unit})', 20)
            for a, b in runs(labels):
                if labels[a] == 2:
                    right = times[b] if b < len(times) else duration
                    d.rectangle((px(times[a]), y0, px(right), y1), fill='#32213e')
            for v in np.linspace(lo, hi, 4):
                yy = py(v)
                d.line((x0, yy, x1, yy), fill='#344150')
                tick = f'{v:.3f}' if torque else f'{v:.1f}'
                text(d, (x0 - 47, yy - 10), tick, 14, MUTED)
            for j, color in enumerate(AXIS_COLORS):
                points = [(px(t), py(v)) for t, v in zip(wrench_times, values[:, j]) if 0 <= t <= duration]
                if len(points) > 1:
                    d.line(points, fill=color, width=2)
                text(d, (x0 + j * 95, row_y + 2), (prefix + 'xyz'[j]) if j < 3 else f'|{prefix}|', 14, color)
            for t in np.arange(0, duration, 5):
                text(d, (px(t) - 4, y1 + 4), f'{t:.0f}s', 14, MUTED)
            charts.append((x0, y0, x1, y1, row_y, offset, torque))
    text(d, (600, 873), '보라색 배경 = 정답 error2 구간 · 전체 그래프는 검토용', 16, MUTED)
    text(d, (24, 924), '전체 타임라인', 20)
    for title, vals, y in (('정답', labels, 962), ('관측만', base_display, 990), ('+ history', final_display, 1018)):
        text(d, (24, y - 2), title, 16)
        for a, b in runs(vals):
            d.rectangle((150 + a / len(vals) * 1424, y, 150 + b / len(vals) * 1424, y + 19),
                        fill=COLORS.get(int(vals[a]), COLORS[-1]))
    x = 220
    for i, nm in enumerate(PHASE_NAMES):
        d.rectangle((x, 929, x + 16, 945), fill=COLORS[i])
        text(d, (x + 23, 922), nm, 16)
        x += 170
    text(d, (965, 924), '회색: ignore / 워밍업 / 유효 예측 없음', 16, MUTED)
    text(d, (24, 1051), '확률은 모델 softmax 출력 · 같은 체크포인트 두 분기의 비교 · 정답 라벨을 입력으로 사용하지 않음', 16, MUTED)
    return canvas, charts, duration


def render(background, charts, duration, local, frames, times, labels, pred_index, anchors,
           probabilities, final_probabilities, history, force_times, wrench, rgb_start,
           rgb_offsets=(30, 20, 10, 0), history_steps=20):
    canvas = background.copy()
    d = ImageDraw.Draw(canvas)
    now = float(times[local])
    main_size = 480 if len(rgb_offsets) == 8 else 544
    canvas.paste(Image.fromarray(frames[local]).resize((main_size, main_size), Image.Resampling.BILINEAR),
                 (24 + (544 - main_size)//2, 122))
    text(d, (24, 88), f'정답: {name(int(labels[local]))}', 22, COLORS.get(int(labels[local]), MUTED))
    text(d, (1100, 23), f'{now:5.2f} / {duration:.2f} s    frame {local} / {len(times)-1}', 20)
    j = int(pred_index[local])
    if j >= 0:
        anchor = int(anchors[j]) - rgb_start
        for x, p in ((620, probabilities[j]), (1112, final_probabilities[j])):
            cid = int(p.argmax())
            text(d, (x, 125), f'{name(cid)}  {100*p[cid]:.1f}%', 26, COLORS[cid])
            for c in range(4):
                yy = 176 + c * 32
                text(d, (x, yy - 5), PHASE_NAMES[c], 16, COLORS[c])
                d.rounded_rectangle((x + 115, yy, x + 360, yy + 15), radius=3, fill='#344150')
                if p[c] > 0:
                    d.rectangle((x + 115, yy, x + 115 + 245 * float(p[c]), yy + 15), fill=COLORS[c])
                text(d, (x + 370, yy - 6), f'{100*p[c]:5.1f}%', 16)
            text(d, (x, 311), '일치' if cid == labels[local] else '정답과 다름', 16,
                 '#77dfb4' if cid == labels[local] else '#ff8787')
        text(d, (600, 356), f'추론 frame {anchor} · {times[anchor]:.2f}s · 직전 예측 유지 {(now-times[anchor])*1000:.0f} ms', 16, MUTED)
        text(d, (1210, 356), f'history {int(history[j])}/{history_steps}', 16, MUTED)
        for k, offset in enumerate(rgb_offsets):
            ri = anchor - offset
            size = 100 if len(rgb_offsets) == 8 else 128
            x = 24 + (k % 4)*139
            y = 650 + (k//4)*122 if len(rgb_offsets) == 8 else 722
            canvas.paste(Image.fromarray(frames[ri]).resize((size, size), Image.Resampling.BILINEAR), (x, y))
            text(d, (x, y + size + 4), f'{times[ri]:.2f}s', 14, MUTED)
    else:
        status = 'ignore · 평가 제외' if labels[local] < 0 else '센서 입력 워밍업 / 유효 관측 없음'
        text(d, (620, 158), status, 20, MUTED)
        text(d, (1112, 158), status, 20, MUTED)
        text(d, (24, 738), '추론에 필요한 RGB/F/T를 모으는 중', 18, MUTED)
    force_i = int(np.searchsorted(force_times, now, side='right') - 1)
    for x0, y0, x1, y1, row_y, offset, torque in charts:
        xx = x0 + now / duration * (x1 - x0)
        d.line((xx, y0, xx, y1), fill='white', width=2)
        if force_i >= 0:
            a = wrench[force_i, offset + (3 if torque else 0):offset + (6 if torque else 3)]
            text(d, (x0 - 40, row_y + 188), f'x {a[0]:+.2f}   y {a[1]:+.2f}   z {a[2]:+.2f}', 16)
            if not torque:
                text(d, (x0 + 280, row_y + 188), f'|F| {np.linalg.norm(a):.2f} N', 14, MUTED)
    xx = 150 + local / len(times) * 1424
    d.line((xx, 958, xx, 1040), fill='white', width=2)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='outputs/slip_context_rgb8_1s_history10/best.pt')
    parser.add_argument('--original-episode', type=int, default=291)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload.get('smoke_test'):
        raise ValueError('Use a fully trained checkpoint')
    ds = SlipSequenceDataset(**payload['training_config']['dataset'])
    try:
        if ds.fingerprint != payload['manifest']['source_fingerprint']:
            raise ValueError('Dataset differs from the checkpoint')
        matches = np.flatnonzero(ds.original_episode_numbers == args.original_episode)
        if len(matches) != 1:
            raise ValueError('Unknown original episode')
        ep = int(matches[0])
        if ep not in payload['manifest']['episode_splits']['test']:
            raise ValueError('Choose a held-out test episode')
        model = SlipTemporalContextEncoder.from_checkpoint(payload).eval().requires_grad_(False).to(device)
        cache = ensure_feature_cache(ds, model, payload['training_config']['cache_dir'], device, create=False)
        if cache != payload['cache_manifest']:
            raise ValueError('Feature cache differs from the trained run')
        anchors, base, final, history = predict_episode(ds, payload, ep, model, device)
        prior = np.load(Path(args.checkpoint).parent/'test_eval/predictions.npz', allow_pickle=False)
        selected = prior['episode_ids'] == ep
        if not np.array_equal(prior['labels'][selected], ds.targets[anchors]):
            raise ValueError('Saved evaluation label alignment differs')
        if not np.allclose(prior['timestamps'][selected], ds.rgb_times[anchors], atol=1e-6, rtol=0):
            raise ValueError('Saved evaluation timestamp alignment differs')
        agreement = float(np.mean(final.argmax(-1) == prior['probabilities'][selected].argmax(-1)))
        max_diff = float(np.max(np.abs(final - prior['probabilities'][selected])))
        if agreement != 1. or max_diff > 1e-4:
            raise ValueError(f'Replay differs from saved evaluation: {agreement=}, {max_diff=}')
        lo, hi = int(ds.rgb_starts[ep]), int(ds.rgb_ends[ep])
        fs, fe = int(ds.force_starts[ep]), int(ds.force_ends[ep])
        times = ds.rgb_times[lo:hi] - ds.rgb_times[lo]
        dt = float(np.median(np.diff(times)))
        if not np.allclose(np.diff(times), 1001/60000, atol=1e-6, rtol=0):
            raise ValueError('Expected 59.94 fps RGB')
        dt = 1001/60000
        labels = ds.targets[lo:hi]
        ft = ds.force_times[fs:fe] - ds.rgb_times[lo]
        wrench = ds.wrench[fs:fe]
        with np.load(ds.label_path, allow_pickle=False) as z:
            source = str(z['source_episode'][ep])
        # Predictions are only held causally for their scheduled six-frame period.
        ids = np.arange(lo, hi)
        pred_index = np.searchsorted(anchors, ids, side='right') - 1
        safe = np.maximum(pred_index, 0)
        available = (pred_index >= 0) & (ids - anchors[safe] < 6) & ds.valid[lo:hi]
        pred_index[~available] = -1
        display = []
        for p in (base, final):
            arr = np.full(hi-lo, -1, np.int64)
            arr[available] = p[pred_index[available]].argmax(-1)
            display.append(arr)
        rgb = np.asarray(ds._get_rgb()[lo:hi])
        background, charts, duration = make_background(times, ft, wrench, labels, *display, args.original_episode, source,
                                                       payload['epoch']+1, model.rgb_horizon)
        render_config = dict(rgb_offsets=ds.rgb_offsets, history_steps=model.history_steps)
        y = ds.targets[anchors]
        mistake = (y == 2) & (final.argmax(-1) == 0)
        harmed = (base.argmax(-1) == y) & (final.argmax(-1) != y)
        candidates = np.flatnonzero(harmed & (y == 2))
        if not len(candidates):
            candidates = np.flatnonzero(mistake)
        preview_frame = int(anchors[candidates[0]] - lo) if len(candidates) else int(anchors[len(anchors)//2]-lo)
        metadata = dict(original_episode=args.original_episode, episode_index=ep, source=source,
            checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_epoch=payload['epoch']+1,
            rgb_horizon=model.rgb_horizon, rgb_span_s=model.rgb_span_s, history_steps=model.history_steps,
            rgb_offsets_frames=ds.rgb_offsets.tolist(),
            phase_names=list(PHASE_NAMES), fps=60000/1001, frames=hi-lo, duration_s=(hi-lo)*dt,
            prediction_count=len(anchors), error2_count=int((y==2).sum()), error2_as_approach=int(mistake.sum()),
            baseline_accuracy=float((base.argmax(-1)==y).mean()), history_accuracy=float((final.argmax(-1)==y).mean()),
            history_helped=int(((base.argmax(-1)!=y)&(final.argmax(-1)==y)).sum()),
            history_harmed=int(harmed.sum()), preview_frame=preview_frame,
            saved_test_prediction_agreement=agreement, max_probability_difference=max_diff,
            note='Both branches from the same checkpoint; offline causal predictions at eligible 10 Hz anchors. RGB and F/T inputs exclude ignore. Full force plots are for human review.',
            display='Causal hold up to 6 RGB frames, ignore and warmup grey',
            error2_intervals_s=[[float(times[a]),float(times[b-1]+dt)] for a,b in runs(labels) if labels[a]==2])
        output.mkdir(parents=True)
        (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
        np.savez_compressed(output/'comparison_predictions.npz',global_frames=anchors,local_frames=anchors-lo,
            timestamps=ds.rgb_times[anchors],labels=y,observation_probabilities=base,history_probabilities=final,history_count=history)
        render(background, charts, duration, preview_frame, rgb, times, labels, pred_index, anchors,
               base, final, history, ft, wrench, lo, **render_config).save(output/'preview.jpg',quality=92)
        print(json.dumps(dict(event='replayed',**metadata),ensure_ascii=False),flush=True)
        video = output/f'EP{args.original_episode:03d}_comparison_1x.mp4'
        command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                 '-s',f'{W}x{H}','-r','60000/1001','-i','-','-an','-c:v','libx264',
                 '-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for local in range(hi-lo):
                canvas=render(background,charts,duration,local,rgb,times,labels,pred_index,anchors,base,final,history,ft,wrench,lo,**render_config)
                process.stdin.write(canvas.tobytes())
                if local%300==0:
                    print(json.dumps(dict(event='render',frame=local,total=hi-lo)),flush=True)
            process.stdin.close()
            err=process.stderr.read().decode()
            if process.wait():
                raise RuntimeError(err)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stderr.close()
        ranges=metadata['error2_intervals_s']
        if ranges:
            start=max(0.,ranges[0][0]-1)
            end=min(duration,ranges[-1][1]+1)
            slow=output/f'EP{args.original_episode:03d}_error2_0.5x.mp4'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-ss',str(start),'-t',str(end-start),
                '-i',str(video),'-vf','setpts=2*PTS','-r','30000/1001','-an','-c:v','libx264',
                '-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(slow)],check=True)
            metadata['slow_motion_original_range_s']=[start,end]
        metadata['complete']=True
        (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(event='complete',output=str(output))),flush=True)
    finally:
        ds.close()


if __name__=='__main__':
    main()
