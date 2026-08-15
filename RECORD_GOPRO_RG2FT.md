# GoPro (capture card) + RG2-FT synchronized recorder

A single console tool for UMI-style data collection on a laptop: it records the
GoPro video (via an HDMI capture card) together with the OnRobot RG2-FT
force/torque log, one episode at a time, into per-episode folders that mirror
the UMI SLAM output layout.

## Hardware (verified on this laptop, 2026-07-22)

```
GoPro Hero13 --HDMI--> Elgato HD60 X --USB--> /dev/video2   (live 1080p video)
OnRobot RG2-FT --Modbus/TCP--> 192.168.2.1:502 (slave 65)   (12-axis F/T + width)
```

- Capture card shows up as **/dev/video2** (`Elgato HD60 X`). The Linux UVC path
  only exposes **YUYV** (uncompressed), so bandwidth-limited: ~20 fps at 1080p,
  ~30 fps at 720p. This is the *live HDMI feed*, lower quality than the GoPro's
  own on-SD recording.
- RG2-FT answers Modbus/TCP at **192.168.2.1:502**, slave **65**.

## Environment

Runs in the dedicated `umi-rec` conda env (created on this laptop):

```
conda activate umi-rec   # python 3.11 + numpy<2, zarr 2.16, av, opencv, (pymodbus present but unused)
```

Modbus is spoken over a **raw socket** (no pymodbus dependency), so this tool is
independent of the control env's pymodbus version and never blocks on it.

## ⚠️ One gripper master at a time

The RG2-FT must be commanded by **exactly one** program. This tool opens its own
Modbus/TCP connection and continuously commands the gripper, so the OnRobot ROS
driver (`ros2 launch onrobot_rg_control bringup.launch.py`, incl. any duplicates
inside the `indy_umi` docker container) **must be stopped first** — otherwise the
two masters fight and the gripper oscillates (open/close never settle). Stop them
with e.g. `docker exec indy_umi pkill -9 -f onrobot_rg` and confirm
`ss -tn | grep :502` shows no connection before launching this tool. Conversely,
stop this tool before using the ROS driver again.

## Run

```
scripts_real/run_record_gopro_rg2ft.sh ~/umi_data/session_20260722
# or directly:
conda activate umi-rec
python scripts_real/record_gopro_rg2ft.py -o ~/umi_data/session_20260722
```

### Keys
| key            | action                                   |
|----------------|------------------------------------------|
| `space` or `r` | start / stop recording an episode        |
| `o`            | open gripper                             |
| `c`            | close gripper                            |
| `z`            | tare the F/T sensor (software zero)      |
| `u`            | clear the tare: back to raw F/T readings  |
| `q` / Ctrl-C   | quit                                     |

### F/T tare (`z`) — why it is done in software

The gripper's own `out_zero` (Modbus register 0) is **not** a bias snapshot:
while it is 1 the RG2-FT reports every force and torque as literally 0, so a
grasped object also reads 0 N — it stops measuring. (The OnRobot ROS driver
documents it as *"if the value is 1, all force and torque values will be set to
0"* and writes 0 at startup to enable live readings.) It is therefore useless
as a tare, and this tool always keeps it at 0, clearing a leftover latch on
connect so an earlier `out_zero=1` cannot silently blank a log.

`z` instead averages the last ~25 raw samples (~0.25 s) and subtracts that
baseline from then on, so **real forces keep coming through** — only the
constant bias (~100 N on this sensor) is removed. Press it with the gripper
unloaded. The status line shows `tared` instead of `raw`, and the tared values
are what gets written to `rg2ft.csv`; the offset is saved next to it as
`ft_offset.json` so the raw signal stays recoverable
(`raw = rg2ft.csv value + ft_offset`). `u` clears the tare.

The status line shows: recording state, current episode folder (or the number
of episodes recorded so far), gripper width, fz (left/right), grip-detect —
plus capture fps / encoded frame count when `--save_video` is on.

## Live F/T monitor (no recording)

When you only want to *see* the forces/torques — no episodes, no files, capture
card never opened:

```
scripts_real/run_monitor_rg2ft.sh
# or directly:
conda activate umi-rec
python scripts_real/monitor_rg2ft.py
```

It draws all 12 axes (both fingers) as bipolar bars with a peak-hold column,
plus gripper width, grip-detect, |F| per finger and the actual Modbus sample
rate, refreshed at 20 Hz. Same gripper keys as the recorder:

| key          | action                                                     |
|--------------|------------------------------------------------------------|
| `o`          | open gripper                                                |
| `c`          | close gripper                                               |
| `z`          | tare the F/T sensor (software zero), same as the recorder    |
| `x`          | clear the tare: back to raw sensor values                    |
| `p`          | reset the peak-hold column                                  |
| `q` / Ctrl-C | quit                                                        |

The tare is the same software zero as the recorder's (see above) and lives in
this process only — it writes no data. Options: `--fmax` (force bar full scale, default 20 N),
`--tmax` (torque full scale, default 1 Nm), `--smooth` (display EMA weight per
sample, default 0.4; pass `1.0` for raw unsmoothed values), `--refresh_hz`,
`--zero_on_start`, `--no_color`, plus the same `--host/--port/--slave_id/
--rg2ft_hz/--gripper_force` as the recorder.

The "one gripper master at a time" rule above applies here too: stop the ROS
driver (and the recorder) before running this.

## Output layout

By default only the RG2-FT log is written (the GoPro records to its own SD
card); pass `--save_video` to also encode the capture-card feed.

```
<session_dir>/demos/
  demo_153042/             # HHMMSS at the moment recording started
    rg2ft.csv              # timestamp, 12-axis F/T (N/Nm), width_m
    ft_offset.json         # tare offset, only if `z` was pressed
    raw_video.mp4          # h264 capture-card feed   (--save_video only)
    video_timestamps.csv   # per-frame ts + repeats   (--save_video only)
  demo_153418/
    ...
```

- Folder naming is `demos/demo_<HHMMSS>`, taken from the recording start time
  (local clock — the same clock the csv timestamps use); a name collision gets
  `_2`, `_3`, ... appended. The `demos/demo_*` prefix still matches UMI's SLAM
  output layout.
- `rg2ft.csv` columns:
  `timestamp, fx_l,fy_l,fz_l,tx_l,ty_l,tz_l, fx_r,fy_r,fz_r,tx_r,ty_r,tz_r, width_m`
  Forces in N, torques in Nm, width in meters, timestamp is unix wall-clock.

## Sync

Video and F/T are driven by one process and share the wall clock
(`time.time()`). On every tested episode the F/T log **fully brackets** the
video (starts at/just before the first frame, ends at/just after the last), so
F/T can be interpolated onto each video frame's timestamp with no
extrapolation. Measured start/end skew is a few ms (the first episode after
launch may have a larger, harmless leading F/T margin from V4L2 warm-up).

## Options

`--save_video` (off by default; when off the capture device is never opened),
`--video_dev` (default /dev/video2), `--width/--height/--fps`
(default 1920x1080@30), `--host/--port/--slave_id`
(default 192.168.2.1:502/65), `--rg2ft_hz` (default 100),
`--crf` (default 21), `--gripper_force` (N, default 20).

## Status / TODO

- [x] Capture-card video recording (verified: h264 1080p decodes cleanly)
- [x] RG2-FT logging to CSV at 100 Hz (verified against live sensor)
- [x] Per-episode `demos/demo_<HHMMSS>` folders (csv only by default,
      `--save_video` adds the mp4)
- [x] Video/F/T start-stop sync (verified: F/T brackets video, few-ms skew)
- [x] Gripper open/close actuation (verified: both move and HOLD, no oscillation
      — once the ROS driver is stopped so this is the only Modbus master).
      `open` commands max width then rests; `close` commands width 0 and keeps
      commanding to hold against the gripper's spring-open rest state.
- [x] `z` (F/T zero/tare) — fixed 2026-08-07 as a **software** tare. The
      gripper's `out_zero` register cannot be used for this: it does not bias
      the signal, it forces every F/T reading to 0 while set (verified on the
      real gripper — a grasped object read 0 N), so it is now always held at 0.
      `z` subtracts an averaged baseline instead, keeping live forces, and the
      offset is written per episode to `ft_offset.json`. (Logic verified
      against a fake device: bias removed, a +12.5 N load still read 12.5 N.)
- [ ] Optional: also emit `rg2ft.zarr` (timestamp + gripper_ft) so the existing
      `scripts_slam_pipeline/08_merge_rg2ft.py` can consume episodes directly
      (it currently reads zarr, not csv).
```
