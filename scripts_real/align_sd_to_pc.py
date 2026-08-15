"""Estimate the time offset between a GoPro SD-card recording and the
HDMI-captured demo recorded by record_gopro_rg2ft.py.

The HDMI capture carries PC wall-clock (epoch) timestamps in
video_timestamps.csv; the SD recording carries the GoPro IMU (GPMF) but only a
relative timeline. Both films the same scene, so we align them by
cross-correlating a per-frame motion-energy signal. This deliberately ignores
the GoPro RTC (wrong timezone / free-running drift) and recovers the mapping

    pc_epoch = sd_relative_time + offset

Usage:
  conda activate umi-rec
  python scripts_real/align_sd_to_pc.py --sd /path/GX010123.MP4 \
      --demo ~/umi_data/session_.../demos/demo_0001

  # one SD file usually spans a whole session: check several demos agree
  python scripts_real/align_sd_to_pc.py --sd GX010123.MP4 \
      --demo .../demo_0001 --demo .../demo_0002 --demo .../demo_0003
"""
import csv
import json
import argparse
import pathlib

import numpy as np
import av
import cv2

RATE = 30.0        # Hz, common grid the two signals are resampled onto
THUMB = (64, 36)   # motion energy is computed on this tiny grayscale image


def motion_energy(video_path, keep_mask=None):
    """Per-frame-pair motion energy of a video.

    Returns (idx, energy) where idx[k] is the index of the *second* frame of
    pair k. Frames whose keep_mask entry is False are skipped entirely, which
    is how HDMI duplicate ("repeats") frames are removed.
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    idx, energy = [], []
    prev = None
    for i, frame in enumerate(container.decode(video=0)):
        if keep_mask is not None:
            if i >= len(keep_mask):
                break
            if not keep_mask[i]:
                continue
        small = cv2.resize(
            frame.to_ndarray(format="gray"), THUMB, interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        if prev is not None:
            energy.append(float(np.abs(small - prev).mean()))
            idx.append(i)
        prev = small
    container.close()
    return np.asarray(idx), np.asarray(energy, dtype=np.float64)


def load_frame_timestamps(path):
    """video_timestamps.csv -> (per-encoded-frame epoch, first-of-group mask).

    The recorder duplicates a captured frame `repeats` times to fill the
    constant-fps encoder timeline; only the first copy is a genuinely new
    observation.
    """
    ts, first = [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t = float(row[0])
            n = int(row[1]) if len(row) > 1 else 1
            for k in range(max(1, n)):
                ts.append(t)
                first.append(k == 0)
    return np.asarray(ts), np.asarray(first, dtype=bool)


def resample(t, v, grid):
    """Zero-mean / unit-std resampling of an irregular signal onto `grid`."""
    out = np.interp(grid, t, v)
    out -= out.mean()
    s = out.std()
    return out / s if s > 1e-9 else out


def parabolic_peak(y, k):
    """Sub-sample peak refinement around integer argmax k."""
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    a, b, c = y[k - 1], y[k], y[k + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return float(k)
    return k + 0.5 * (a - c) / denom


def align_one(sd_sig, demo_dir):
    """Cross-correlate one demo against the (already computed) SD signal.

    Returns (offset, score) where offset is the PC epoch of SD relative time 0.
    """
    demo_dir = pathlib.Path(demo_dir).expanduser().absolute()
    frame_ts, first_mask = load_frame_timestamps(demo_dir / "video_timestamps.csv")

    idx, energy = motion_energy(demo_dir / "raw_video.mp4", keep_mask=first_mask)
    if len(energy) < 8:
        raise RuntimeError(f"{demo_dir.name}: too few usable frames")

    # energy[k] spans the gap between two kept frames -> stamp it at the midpoint
    kept = np.flatnonzero(first_mask)
    pos = np.searchsorted(kept, idx)
    t_hdmi = 0.5 * (frame_ts[kept[pos]] + frame_ts[kept[np.maximum(pos - 1, 0)]])

    grid_h = np.arange(t_hdmi[0], t_hdmi[-1], 1.0 / RATE)
    h = resample(t_hdmi, energy, grid_h)

    sd_t, sd_v = sd_sig
    grid_s = np.arange(sd_t[0], sd_t[-1], 1.0 / RATE)
    s = resample(sd_t, sd_v, grid_s)

    # Pad so the demo may hang off either end of the SD recording. Both signals
    # are zero-mean, so the padding contributes nothing to the correlation.
    pad = len(h)
    s_pad = np.concatenate([np.zeros(pad), s, np.zeros(pad)])

    corr = np.correlate(s_pad, h, mode="full") / len(h)
    k = int(np.argmax(corr))
    k_ref = parabolic_peak(corr, k)
    m = k_ref - (len(h) - 1) - pad    # sd[m + j] matches hdmi[j]
    offset = grid_h[0] - m / RATE + grid_s[0]

    peak = float(corr[k])
    # peak sharpness: compare against the correlation outside a +-1s guard band
    guard = int(RATE)
    mask = np.ones(len(corr), dtype=bool)
    mask[max(0, k - guard):k + guard + 1] = False
    runner_up = float(corr[mask].max()) if mask.any() else 0.0
    return offset, peak, runner_up, len(h) / RATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd", required=True, help="GoPro SD recording (GX*.MP4)")
    ap.add_argument("--demo", action="append", required=True,
                    help="demo dir with raw_video.mp4 + video_timestamps.csv "
                         "(repeat for several demos of the same session)")
    ap.add_argument("-o", "--output", default=None,
                    help="write the result as JSON (default: <sd>.pc_offset.json)")
    args = ap.parse_args()

    sd_path = pathlib.Path(args.sd).expanduser().absolute()
    out_path = (pathlib.Path(args.output).expanduser().absolute()
                if args.output else sd_path.with_suffix(".pc_offset.json"))

    print(f"[sd] decoding {sd_path.name} ...")
    with av.open(str(sd_path)) as c:
        sd_fps = float(c.streams.video[0].average_rate)
    sd_idx, sd_energy = motion_energy(sd_path)
    sd_sig = ((sd_idx - 0.5) / sd_fps, sd_energy)
    print(f"[sd] {len(sd_energy)} frame pairs @ {sd_fps:.3f}fps "
          f"({sd_idx[-1] / sd_fps:.1f}s)")

    results = []
    for demo in args.demo:
        offset, peak, runner_up, dur = align_one(sd_sig, demo)
        name = pathlib.Path(demo).name
        ratio = peak / runner_up if runner_up > 1e-9 else float("inf")
        flag = "ok" if (peak > 0.3 and ratio > 1.5) else "LOW CONFIDENCE"
        print(f"[{name}] offset={offset:.6f}  peak={peak:.3f} "
              f"(runner-up {runner_up:.3f}, ratio {ratio:.2f})  "
              f"dur={dur:.1f}s  -> {flag}")
        results.append({
            "demo": str(pathlib.Path(demo).expanduser().absolute()),
            "offset": offset, "peak": peak, "runner_up": runner_up,
            "duration_s": dur,
        })

    offsets = np.array([r["offset"] for r in results])
    print(f"\nmedian offset = {np.median(offsets):.6f}")
    if len(offsets) > 1:
        spread = offsets.max() - offsets.min()
        print(f"spread across demos = {spread * 1000:.1f} ms "
              f"({'consistent' if spread < 0.1 else 'INCONSISTENT - check inputs'})")

    payload = {
        "sd_video": str(sd_path),
        "sd_fps": sd_fps,
        "offset": float(np.median(offsets)),
        "note": "pc_epoch = sd_relative_time + offset",
        "per_demo": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
