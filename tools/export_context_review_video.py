#!/usr/bin/env python
"""Export recorded RGB, saved predictions, and synchronized native right-finger force."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from diffusion_policy.context.data import load_labeling_base, episode_bounds, source_fingerprint, protect_raw_data, validate_label_alignment
from diffusion_policy.context.labels import definitions, read_rows, validate_definition_version


WIDTH, HEIGHT = 1280, 800
BG, PANEL, FG, MUTED = '#101720', '#192330', '#f1f5fa', '#a9b6c6'
COLORS = {0: '#65b7fa', 1: '#5cdda5', 2: '#b59bff', 3: '#ff7979', -1: '#ffce70', None: '#637388'}
NAMES = {0: '접근 · approach', 1: '회전 · turning', 2: '완료 · finish', 3: '에러 · error', -1: '판단 불가 · unknown', None: '라벨 없음 · NOT LABELED'}
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS = {size: ImageFont.truetype(FONT, size, index=1) for size in (14, 16, 18, 20, 24, 30)}


def label_at(rows, anchors, frame, stride, frame_count=None):
    """Hold between anchors; cover the final short stride only for a finished episode."""
    if not anchors or frame < 0 or (frame_count is not None and frame >= frame_count):
        return None
    if frame > anchors[-1]:
        final_anchor = None if frame_count is None else ((frame_count-1)//stride)*stride
        if anchors[-1] != final_anchor:
            return None
    i = bisect.bisect_right(anchors, frame)-1
    if i < 0 or frame-anchors[i] >= stride:
        return None
    return rows[i]


def wrapped(draw, text, xy, width, size=16, lines=4):
    x, y = xy
    words = text.split()
    line = ''
    used = 0
    for word in words:
        candidate = f'{line} {word}'.strip()
        if draw.textlength(candidate, font=FONTS[size]) <= width:
            line = candidate
        else:
            draw.text((x, y), line, font=FONTS[size], fill=FG)
            y += size+7
            used += 1
            line = word
            if used == lines-1:
                break
    if line:
        if used == lines-1:
            # Keep the video card bounded; full saved reasons remain in labels.json.
            line = line[:max(1, int(width/(size*.6)))-3]+'...'
        draw.text((x, y), line, font=FONTS[size], fill=FG)


def episode_data(base, episode, rows, stride):
    lo, hi = episode_bounds(base, episode)
    ts = np.asarray(base.rgb_timestamps[lo:hi])
    dt = float(np.median(np.diff(ts)))
    if not np.allclose(np.diff(ts), dt, atol=1e-4, rtol=.01):
        raise ValueError('This exporter requires a regular RGB clock')
    start = 0 if episode == 0 else int(base.ft_right_episode_ends[episode-1])
    end = int(base.ft_right_episode_ends[episode])
    ft = np.asarray(base.ft_right_timestamps[start:end])-ts[0]
    force = np.asarray(base.ft_right[start:end, :3])
    ordered = sorted((r for r in rows if r['episode_id'] == episode), key=lambda r:r['anchor_index'])
    anchors = [r['anchor_index'] for r in ordered]
    return dict(episode=episode, lo=lo, count=hi-lo, times=ts-ts[0], dt=dt,
        duration=float(ts[-1]-ts[0]+dt), force_times=ft, force=force,
        norms=np.linalg.norm(force, axis=1), rows=ordered, anchors=anchors, stride=stride)


def backdrop(ep, limits, cfg):
    canvas = Image.new('RGB', (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((716, 80, 1258, 768), radius=16, fill=PANEL)
    recording_name = cfg.get('recording_name', f"에피소드 {ep['episode']:02d}")
    d.text((24, 15), f"Qwen {cfg['label_version']} 검토  /  {recording_name}", font=FONTS[30], fill=FG)
    subtitle = ('에피소드 순서 보정 포함 · 정답 라벨 아님' if cfg.get('postprocessing')
                else '기록 영상 전체 구간 · 정답 라벨 아님')
    d.text((24, 57), subtitle, font=FONTS[16], fill=MUTED)
    reason_title = ('힘 규칙 + Qwen 영상 판단' if cfg.get('decision_mode') == 'force_visual_v1'
                    else 'Qwen 판단 이유')
    if cfg.get('postprocessing'):
        reason_title = '판단 및 구간 보정 근거'
    d.text((738, 195), reason_title, font=FONTS[16], fill=MUTED)
    d.text((738, 337), '오른쪽 손가락 힘 · RGB 시점 이전의 최근 측정값', font=FONTS[16], fill=MUTED)
    # Entire-episode force plot is for human inspection, not a Qwen input.
    left, top, right, bottom = 770, 462, 1234, 620
    ymin, ymax = limits
    def px(t): return left + t/ep['duration']*(right-left)
    def py(v): return bottom - (v-ymin)/(ymax-ymin)*(bottom-top)
    for value in np.linspace(ymin, ymax, 5):
        y = py(value)
        d.line((left, y, right, y), fill='#334050')
        d.text((727, y-9), f'{value:.1f}', font=FONTS[14], fill=MUTED)
    d.text((727, 438), 'N', font=FONTS[14], fill=MUTED)
    plot_colors = ['#79bcff', '#ffb875', '#cd9aff', '#63dfc2']
    for j, name in enumerate(('Fx', 'Fy', 'Fz', '|F|')):
        x = 775+j*105
        d.line((x, 443, x+18, 443), fill=plot_colors[j], width=3)
        d.text((x+24, 431), name, font=FONTS[16], fill=FG)
        values = ep['force'][:, j] if j < 3 else ep['norms']
        mask = (ep['force_times'] >= 0) & (ep['force_times'] <= ep['duration'])
        points = [(px(t), py(v)) for t,v in zip(ep['force_times'][mask], values[mask])]
        if len(points) > 1:
            d.line(points, fill=plot_colors[j], width=2)
    for t in np.arange(0, ep['duration'], 2.):
        d.text((px(t)-8, bottom+7), f'{t:.0f}s', font=FONTS[14], fill=MUTED)
    d.text((738, 656), '저장된 예측 범위 · 회색은 미추론 구간', font=FONTS[16], fill=MUTED)
    for frame in range(ep['count']):
        row = label_at(ep['rows'], ep['anchors'], frame, ep['stride'], ep['count'])
        color = COLORS[None if row is None else row['class_id']]
        d.rectangle((px(ep['times'][frame]), 692, px(ep['times'][frame]+ep['dt']), 711), fill=color)
    d.text((738, 732), '전체 힘 그래프는 사람의 검토용입니다.', font=FONTS[14], fill=MUTED)
    d.text((24, 773), '원본 RGB 224×224 확대 표시 · 표시된 힘은 라벨링에 사용한 데이터', font=FONTS[14], fill=MUTED)
    return canvas, px


def render(background, px, frame, rgb, ep):
    canvas = background.copy()
    canvas.paste(Image.fromarray(cv2.resize(rgb, (672, 672), interpolation=cv2.INTER_LINEAR)), (24, 88))
    d = ImageDraw.Draw(canvas)
    now = float(ep['times'][frame])
    d.text((735, 22), f"시간 {now:5.2f}s / {ep['times'][-1]:.2f}s    프레임 {frame:03d} / {ep['count']-1}", font=FONTS[20], fill=FG)
    row = label_at(ep['rows'], ep['anchors'], frame, ep['stride'], ep['count'])
    cid = None if row is None else row['class_id']
    d.text((738, 96), NAMES[cid], font=FONTS[24], fill=COLORS[cid])
    if row is None:
        note = '이 프레임에는 저장된 Qwen 예측이 없습니다.'
        reason = 'No saved prediction. Review the recorded motion and force directly.'
    else:
        direct = '추론한 프레임' if row['anchor_index'] == frame else '직전 예측 표시'
        note = f"신뢰도 {row['confidence']:.2f}  ·  {direct} (frame {row['anchor_index']})"
        if row.get('decision_source') == 'phase_rule':
            note = f"순서 규칙 보정 · 원본 unknown · frame {row['anchor_index']}"
        elif row.get('decision_source') == 'force_rule':
            note = f"힘 규칙 판정 · 신뢰도 미보정 · {direct} (frame {row['anchor_index']})"
        elif row.get('decision_source') in ('vision','vision_and_force'):
            note = f"영상 신뢰도 {row['confidence']:.2f} · {direct} (frame {row['anchor_index']})"
        elif row.get('decision_source') == 'insufficient_evidence':
            note = f"근거 부족 · {direct} (frame {row['anchor_index']})"
        reason = row['reason']
    d.text((738, 144), note, font=FONTS[16], fill=MUTED)
    wrapped(d, reason, (738, 224), 496, lines=4)
    index = int(np.searchsorted(ep['force_times'], now, side='right')-1)
    if index < 0:
        d.text((738, 370), '힘 측정 없음', font=FONTS[20], fill=MUTED)
    else:
        fx, fy, fz = ep['force'][index]
        age = (now-ep['force_times'][index])*1000
        d.text((738, 367), f'Fx {fx:+.2f}   Fy {fy:+.2f}   Fz {fz:+.2f} N', font=FONTS[20], fill=FG)
        d.text((738, 400), f"|F| {ep['norms'][index]:.2f} N   ·   측정 시차 {age:.1f} ms", font=FONTS[18], fill='#63dfc2')
    x = px(now)
    d.line((x, 458, x, 620), fill='white', width=2)
    d.polygon([(x-5, 685), (x+5, 685), (x, 692)], fill='white')
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='context/qwen/config/context_labels.yaml')
    parser.add_argument('--episodes', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--labels-dir', help='Saved original or postprocessed label directory')
    parser.add_argument('--require-complete', action='store_true',
        help='Refuse to export if any scheduled prediction is missing')
    args = parser.parse_args()
    cfg, digest = definitions(args.config)
    labels_path = Path(args.labels_dir or cfg['output_dir'])/'auto_labels.jsonl'
    original = labels_path.read_bytes()
    rows = read_rows(labels_path, len(cfg['contexts']))
    manifest = json.loads((labels_path.parent/'labeling_manifest.json').read_text())
    if manifest['definition_hash'] != digest:
        raise ValueError('Configuration and saved label definitions differ')
    if manifest.get('postprocessing'):
        if manifest['postprocessing']['output_sha256'] != hashlib.sha256(original).hexdigest():
            raise ValueError('Postprocessed labels differ from their manifest')
        cfg = dict(cfg, postprocessing=manifest['postprocessing'])
    validate_definition_version(rows, digest)
    base, _ = load_labeling_base(cfg)
    out = Path(args.output_dir).resolve()
    try:
        protect_raw_data(base, out)
        if out.exists():
            raise FileExistsError(out)
        if source_fingerprint(base) != manifest['source_fingerprint']:
            raise ValueError('Recording differs from the labeling manifest')
        validate_label_alignment(base, rows)
        episodes = [episode_data(base, ep, rows, manifest['label_stride']) for ep in args.episodes]
        for ep in episodes:
            ep['complete'] = ep['anchors'] == list(range(0, ep['count'], ep['stride']))
            if args.require_complete and not ep['complete']:
                raise ValueError(f"Episode {ep['episode']} has missing predictions; finish labeling first")
        dt = episodes[0]['dt']
        if not all(abs(ep['dt']-dt) < 1e-6 for ep in episodes):
            raise ValueError('Selected episodes have different RGB rates')
        values = np.concatenate([np.concatenate((ep['force'].ravel(), ep['norms'])) for ep in episodes])
        low, high = min(0., float(values.min())), max(1., float(values.max()))
        margin = max(.2, (high-low)*.08)
        limits = low-margin, high+margin
        out.mkdir(parents=True)
        normal = out/'review_1x.mp4'
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(1/dt), '-i', '-',
            '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '19',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(normal)]
        metadata = dict(config=str(Path(args.config).resolve()), labels=str(labels_path.resolve()),
            label_sha256=hashlib.sha256(original).hexdigest(), fps=1/dt,
            force_bias_removed=bool(base.ft_bias_removed), force_display='latest native sample at or before RGB time',
            label_display='held between anchors within stride; final short stride held only when final scheduled anchor is saved',
            episodes=[])
        if manifest.get('postprocessing'):
            metadata['postprocessing'] = manifest['postprocessing']
        if cfg.get('recording_path'):
            metadata['recording'] = base.metadata
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            rgb_array = base._get_rgb_array()
            cursor = 0
            for ep in episodes:
                background, px = backdrop(ep, limits, cfg)
                for frame in range(ep['count']):
                    rgb = np.array(rgb_array[ep['lo']+frame])
                    process.stdin.write(render(background, px, frame, rgb, ep).tobytes())
                    if frame in (0, ep['anchors'][-1] if ep['anchors'] else 0, ep['count']-1):
                        render(background, px, frame, rgb, ep).save(out/f"ep{ep['episode']:02d}_frame{frame:04d}.jpg")
                metadata['episodes'].append(dict(episode_id=ep['episode'], frame_count=ep['count'],
                    video_start_s=cursor*dt, duration_s=ep['duration'],
                    all_scheduled_predictions_present=ep['complete'],
                    labeled_anchors=ep['anchors'], last_labeled_time_s=float(ep['times'][ep['anchors'][-1]]) if ep['anchors'] else None))
                cursor += ep['count']
                print(f"Rendered episode {ep['episode']}: {ep['count']} frames", flush=True)
            process.stdin.close()
            errors = process.stderr.read().decode()
            if process.wait():
                raise RuntimeError(errors)
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        finally:
            process.stderr.close()
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(normal),
            '-vf', 'setpts=2*PTS', '-r', str(.5/dt), '-an', '-c:v', 'libx264', '-threads', '2',
            '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(out/'review_0.5x.mp4')], check=True)
        (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
        (out/'labels.json').write_text(json.dumps([r for r in rows if r['episode_id'] in args.episodes], ensure_ascii=False, indent=2))
        if labels_path.read_bytes() != original:
            raise RuntimeError('Label source changed while exporting')
        print(json.dumps(dict(output_dir=str(out), frames=cursor, duration_s=cursor*dt), indent=2))
    finally:
        base.close()


if __name__ == '__main__':
    main()
