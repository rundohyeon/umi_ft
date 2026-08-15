"""Render a demo episode as: left = GoPro video, right = RG2-FT plots.

Takes a demo folder produced by record_gopro_rg2ft.py (raw_video.mp4 +
rg2ft.csv + video_timestamps.csv), time-aligns the F/T log to each video frame
via the wall-clock timestamps, and writes a side-by-side mp4 with a moving
time cursor on the force / torque / width plots.

Usage:
  conda activate umi-rec
  python scripts_real/render_gopro_rg2ft.py <demo_dir> [-o out.mp4]
  # e.g. scripts_real/render_gopro_rg2ft.py ~/umi_data/session_.../demos/demo_0000
"""
import os
import csv
import argparse
import pathlib

import numpy as np
import av
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FT_COLS = ['fx_l', 'fy_l', 'fz_l', 'tx_l', 'ty_l', 'tz_l',
           'fx_r', 'fy_r', 'fz_r', 'tx_r', 'ty_r', 'tz_r']


def load_rg2ft(path):
    ts, ft, width = [], [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            ts.append(float(row[0]))
            ft.append([float(v) for v in row[1:13]])
            width.append(float(row[13]))
    return (np.array(ts), np.array(ft, dtype=np.float64),
            np.array(width) * 1000.0)  # width -> mm


def load_frame_timestamps(path):
    """Expand video_timestamps.csv (captured ts + encode repeats) to one
    absolute timestamp per ENCODED frame."""
    out = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t = float(row[0])
            repeats = int(row[1]) if len(row) > 1 else 1
            out.extend([t] * max(1, repeats))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('demo_dir')
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--right_width', type=int, default=900,
                    help='pixel width of the plot panel')
    ap.add_argument('--crf', type=int, default=20)
    args = ap.parse_args()

    demo = pathlib.Path(os.path.expanduser(args.demo_dir)).absolute()
    video_path = demo / 'raw_video.mp4'
    out_path = (pathlib.Path(os.path.expanduser(args.output)).absolute()
                if args.output else demo / 'render.mp4')

    ts_ft, ft, width = load_rg2ft(demo / 'rg2ft.csv')
    frame_ts = load_frame_timestamps(demo / 'video_timestamps.csv')

    # decode all video frames
    container = av.open(str(video_path))
    vstream = container.streams.video[0]
    fps = float(vstream.average_rate)
    frames = [f.to_ndarray(format='bgr24') for f in container.decode(video=0)]
    container.close()
    n = min(len(frames), len(frame_ts))
    frames, frame_ts = frames[:n], frame_ts[:n]
    H, W = frames[0].shape[:2]
    print(f"video: {W}x{H} @ {fps:.1f}fps, {n} frames | rg2ft: {len(ts_ft)} samples")

    # normalize time to start of episode for display
    t0 = min(frame_ts[0], ts_ft[0])
    tt = ts_ft - t0
    ftt = frame_ts - t0

    # ---- build the plot figure (persistent; only the cursor moves) ----
    RW = args.right_width
    dpi = 100
    fig, axes = plt.subplots(3, 1, figsize=(RW / dpi, H / dpi), dpi=dpi,
                             sharex=True)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.95, bottom=0.07, hspace=0.25)
    colors_l = ['#e6194B', '#3cb44b', '#4363d8']   # x,y,z left
    colors_r = ['#f58231', '#42d4f4', '#911eb4']   # x,y,z right

    # forces
    axf = axes[0]
    for i, c in zip([0, 1, 2], colors_l):
        axf.plot(tt, ft[:, i], color=c, lw=1.0, label=FT_COLS[i])
    for i, c in zip([6, 7, 8], colors_r):
        axf.plot(tt, ft[:, i], color=c, lw=1.0, ls='--', label=FT_COLS[i])
    axf.set_ylabel('Force [N]')
    axf.legend(ncol=6, fontsize=6, loc='upper right')
    axf.grid(alpha=0.3)

    # torques
    axt = axes[1]
    for i, c in zip([3, 4, 5], colors_l):
        axt.plot(tt, ft[:, i], color=c, lw=1.0, label=FT_COLS[i])
    for i, c in zip([9, 10, 11], colors_r):
        axt.plot(tt, ft[:, i], color=c, lw=1.0, ls='--', label=FT_COLS[i])
    axt.set_ylabel('Torque [Nm]')
    axt.legend(ncol=6, fontsize=6, loc='upper right')
    axt.grid(alpha=0.3)

    # width
    axw = axes[2]
    axw.plot(tt, width, color='#000075', lw=1.2)
    axw.set_ylabel('Width [mm]')
    axw.set_xlabel('time [s]')
    axw.grid(alpha=0.3)
    axw.set_xlim(tt[0], tt[-1])

    cursors = [ax.axvline(ftt[0], color='red', lw=1.2) for ax in axes]
    title = fig.suptitle('', fontsize=9)

    fig.canvas.draw()

    def render_panel(t, vals, w):
        for cur in cursors:
            cur.set_xdata([t, t])
        title.set_text(
            f"t={t:5.2f}s   fz_l={vals[2]:6.1f}N  fz_r={vals[8]:6.1f}N   "
            f"width={w:5.1f}mm")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        rgb = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        return cv2.resize(rgb, (RW, H))

    # ---- write side-by-side video ----
    out = av.open(str(out_path), mode='w')
    ostream = out.add_stream('h264', rate=int(round(fps)))
    ostream.width = W + RW
    ostream.height = H
    ostream.pix_fmt = 'yuv420p'
    ostream.codec_context.options = {'crf': str(args.crf)}

    # interpolate F/T + width onto each frame's timestamp
    ft_i = np.stack([np.interp(frame_ts, ts_ft, ft[:, k]) for k in range(12)], axis=1)
    w_i = np.interp(frame_ts, ts_ft, width)

    for idx in range(n):
        panel = render_panel(ftt[idx], ft_i[idx], w_i[idx])
        combo = np.hstack([frames[idx], panel])
        vframe = av.VideoFrame.from_ndarray(combo, format='bgr24')
        for pkt in ostream.encode(vframe):
            out.mux(pkt)
        if idx % 30 == 0:
            print(f"  rendered {idx+1}/{n}")
    for pkt in ostream.encode():
        out.mux(pkt)
    out.close()
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == '__main__':
    main()
