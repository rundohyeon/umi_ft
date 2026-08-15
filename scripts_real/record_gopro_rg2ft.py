"""GoPro (via HDMI capture card) + RG2-FT synchronized demonstration recorder.

One console tool for UMI-style data collection on a laptop:

  GoPro --HDMI--> Elgato HD60 X --USB--> /dev/video2  (live 1080p video)
  OnRobot RG2-FT --Modbus/TCP--> 192.168.2.1:502      (12-axis F/T + width)

Press one key to start/stop an episode. By default only the RG2-FT
force/torque log is written (the GoPro records to its own SD card); each
episode gets a folder named after the time the recording started:

  <output>/demos/demo_HHMMSS/rg2ft.csv

With --save_video the capture-card video is encoded as well, mirroring the
UMI SLAM output layout:

  <output>/demos/demo_HHMMSS/raw_video.mp4
  <output>/demos/demo_HHMMSS/video_timestamps.csv

The gripper can also be opened/closed from this tool while collecting.

Keys:
  [space] or r : start / stop recording an episode
  o            : open gripper
  c            : close gripper
  z            : tare the F/T sensor (software zero; live forces still shown)
  u            : clear the tare
  q or Ctrl-C  : quit

Run inside the `umi-rec` conda env:
  python scripts_real/record_gopro_rg2ft.py -o ~/umi_data/session_2026xxxx
"""
import os
import sys
import time
import select
import termios
import tty
import pathlib
import threading
import argparse

import numpy as np
import cv2
import av

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from umi.real_world.rg2ft_recorder import RG2FTRecorder, FT_COLUMNS
from umi.common.timestamp_accumulator import get_accumulate_timestamp_idxs


class GoProRG2FTRecorder:
    def __init__(self, output_dir, video_dev='/dev/video2',
                 width=1920, height=1080, fps=30,
                 host='192.168.2.1', port=502, slave_id=65,
                 rg2ft_hz=100, crf=21, codec='h264',
                 gripper_force=20.0, save_video=False):
        self.output_dir = pathlib.Path(os.path.expanduser(output_dir)).absolute()
        self.demos_dir = self.output_dir.joinpath('demos')
        self.demos_dir.mkdir(parents=True, exist_ok=True)

        self.video_dev = video_dev
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.codec = codec
        self.save_video = save_video

        self.rg2ft = RG2FTRecorder(
            hostname=host, port=port, slave_id=slave_id,
            frequency=rg2ft_hz, default_force=gripper_force, verbose=False)

        # capture / encode state (guarded by _rec_lock)
        self._rec_lock = threading.Lock()
        self._recording = False
        self._container = None
        self._stream = None
        self._start_time = None
        self._arm_time = None
        self._next_global_idx = 0
        self._n_encoded = 0
        self._ts_file = None
        self._ts_writer = None
        self._cur_demo_dir = None

        # live status
        self._last_frame_time = 0.0
        self._cap_fps = 0.0
        self._frame_shape = None

        self._n_episodes = 0
        self._cur_demo_name = '-'
        self._stop_event = threading.Event()
        self._cap_thread = None
        self._cap = None

    def _new_demo_dir(self, start_time):
        """demos/demo_HHMMSS, named after the moment recording started."""
        stamp = time.strftime('%H%M%S', time.localtime(start_time))
        demo_dir = self.demos_dir.joinpath(f'demo_{stamp}')
        # same clock time on a different day of the same session: disambiguate
        n = 2
        while demo_dir.exists():
            demo_dir = self.demos_dir.joinpath(f'demo_{stamp}_{n}')
            n += 1
        return demo_dir

    # ---------- lifecycle ----------
    def start(self):
        # rg2ft first (fail fast if the sensor is unreachable)
        self.rg2ft.start()
        if not self.save_video:
            # F/T only: never touch the capture card
            return
        cap = cv2.VideoCapture(self.video_dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.rg2ft.stop()
            raise RuntimeError(f"could not open capture device {self.video_dev}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()

    def stop(self):
        if self._recording:
            self.stop_recording()
        self._stop_event.set()
        if self._cap_thread is not None:
            self._cap_thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self.rg2ft.stop()

    # ---------- capture / encode ----------
    def _capture_loop(self):
        n = 0
        t_win = time.monotonic()
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            t = time.time()
            self._last_frame_time = t
            self._frame_shape = frame.shape

            # rolling fps estimate
            n += 1
            now = time.monotonic()
            if now - t_win >= 1.0:
                self._cap_fps = n / (now - t_win)
                n = 0
                t_win = now

            with self._rec_lock:
                if self._recording and self._container is not None:
                    self._encode_frame(frame, t)

        # ensure clean close if still recording
        with self._rec_lock:
            if self._recording:
                self._finalize_container()

    def _encode_frame(self, frame_bgr, frame_time):
        # drop frames captured before the recording was armed: V4L2 can hand
        # back one buffered (stale) frame right after start, which would make
        # the video appear to start earlier than the F/T log.
        if self._arm_time is not None and frame_time < self._arm_time:
            return
        if self._start_time is None:
            self._start_time = frame_time
        local_idxs, _, self._next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=[frame_time],
            start_time=self._start_time,
            dt=1.0 / self.fps,
            next_global_idx=self._next_global_idx)
        n_repeats = len(local_idxs)
        if n_repeats == 0:
            return
        av_frame = av.VideoFrame.from_ndarray(frame_bgr, format='bgr24')
        for _ in range(n_repeats):
            for packet in self._stream.encode(av_frame):
                self._container.mux(packet)
            self._n_encoded += 1
        if self._ts_writer is not None:
            self._ts_writer.write(f"{frame_time:.6f},{n_repeats}\n")

    def _finalize_container(self):
        try:
            if self._stream is not None:
                for packet in self._stream.encode():
                    self._container.mux(packet)
        except Exception:
            pass
        if self._container is not None:
            self._container.close()
        if self._ts_file is not None:
            self._ts_file.close()
        self._container = None
        self._stream = None
        self._ts_file = None
        self._ts_writer = None

    # ---------- episode control ----------
    def start_recording(self):
        if self._recording:
            return None
        demo_dir = self._new_demo_dir(time.time())
        demo_dir.mkdir(parents=True, exist_ok=True)
        if not self.save_video:
            with self._rec_lock:
                self._n_encoded = 0
                self._cur_demo_dir = demo_dir
                self._cur_demo_name = demo_dir.name
                self._recording = True
            self.rg2ft.start_episode()
            return demo_dir
        video_path = demo_dir.joinpath('raw_video.mp4')
        # open + initialize the h264 encoder OUTSIDE the lock: it can take
        # ~0.3s, during which the capture loop keeps grabbing fresh frames but
        # does not encode (recording still False). Flipping the flag under the
        # lock afterwards keeps video/F/T start aligned to within ~one frame.
        container = av.open(str(video_path), mode='w')
        stream = container.add_stream(self.codec, rate=self.fps)
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = 'yuv420p'
        stream.codec_context.options = {'crf': str(self.crf)}
        ts_file = open(demo_dir.joinpath('video_timestamps.csv'), 'w')
        ts_file.write("timestamp,repeats\n")
        with self._rec_lock:
            self._container = container
            self._stream = stream
            self._start_time = None
            self._arm_time = time.time()
            self._next_global_idx = 0
            self._n_encoded = 0
            self._cur_demo_dir = demo_dir
            self._cur_demo_name = demo_dir.name
            self._ts_file = ts_file
            self._ts_writer = ts_file
            self._recording = True
        # start F/T logging (its own thread/lock), right after the flag flips
        self.rg2ft.start_episode()
        return demo_dir

    def stop_recording(self):
        with self._rec_lock:
            if not self._recording:
                return None
            # stop encoding immediately so video and F/T end at the same instant
            # (h264 container flush below can take ~0.3s; do NOT keep logging
            # F/T during it)
            self._recording = False
            demo_dir = self._cur_demo_dir
        data = self.rg2ft.end_episode()
        with self._rec_lock:
            self._finalize_container()
        csv_path = demo_dir.joinpath('rg2ft.csv')
        RG2FTRecorder.save_episode_csv(csv_path, data)
        # keep the tare recoverable: raw = rg2ft.csv value + this offset
        if np.any(data['ft_offset']):
            import json
            demo_dir.joinpath('ft_offset.json').write_text(json.dumps(
                {'note': 'software tare; raw_ft = rg2ft.csv ft + ft_offset',
                 'columns': list(FT_COLUMNS),
                 'ft_offset': [float(v) for v in data['ft_offset']]}, indent=2))
        self._n_episodes += 1
        return demo_dir, len(data['timestamp'])

    # ---------- gripper ----------
    def open_gripper(self):
        self.rg2ft.open_gripper()

    def close_gripper(self):
        self.rg2ft.close_gripper()

    def zero_ft(self):
        return self.rg2ft.zero_ft()

    def unzero_ft(self):
        self.rg2ft.unzero_ft()


HELP = ("[space]/r record  |  o open  |  c close  |  "
        "z tare-FT  |  u un-tare  |  q quit")


def main():
    parser = argparse.ArgumentParser(
        description="GoPro(capture card) + RG2-FT synchronized recorder")
    parser.add_argument('-o', '--output', required=True,
        help="session directory; episodes saved under <output>/demos/demo_HHMMSS")
    parser.add_argument('--save_video', action='store_true',
        help="also record the capture-card video (default: RG2-FT csv only)")
    parser.add_argument('--video_dev', default='/dev/video2')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--host', default='192.168.2.1')
    parser.add_argument('--port', type=int, default=502)
    parser.add_argument('--slave_id', type=int, default=65)
    parser.add_argument('--rg2ft_hz', type=int, default=100)
    parser.add_argument('--crf', type=int, default=21)
    parser.add_argument('--gripper_force', type=float, default=20.0)
    args = parser.parse_args()

    rec = GoProRG2FTRecorder(
        output_dir=args.output, video_dev=args.video_dev,
        width=args.width, height=args.height, fps=args.fps,
        host=args.host, port=args.port, slave_id=args.slave_id,
        rg2ft_hz=args.rg2ft_hz, crf=args.crf, gripper_force=args.gripper_force,
        save_video=args.save_video)

    if args.save_video:
        print(f"Connecting RG2-FT @ {args.host}:{args.port} and capture {args.video_dev} ...")
    else:
        print(f"Connecting RG2-FT @ {args.host}:{args.port} (video disabled, csv only) ...")
    rec.start()
    print(f"Ready. Output: {rec.demos_dir}  (episode folder: demo_<HHMMSS> at record start)")
    print(HELP)

    is_tty = sys.stdin.isatty()
    old_attrs = None
    if is_tty:
        old_attrs = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def read_key():
        if not is_tty:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    last_status = 0.0
    try:
        while True:
            key = read_key()
            if key is not None:
                if key in (' ', 'r'):
                    if rec._recording:
                        demo_dir, n = rec.stop_recording()
                        frames = f"{rec._n_encoded} frames, " if rec.save_video else ""
                        print(f"\n[STOP] {demo_dir.name}: {frames}"
                              f"{n} F/T samples -> {demo_dir}")
                    else:
                        demo_dir = rec.start_recording()
                        print(f"\n[REC ] {demo_dir.name} recording...")
                elif key == 'o':
                    rec.open_gripper(); print("\n[grip] open")
                elif key == 'c':
                    rec.close_gripper(); print("\n[grip] close")
                elif key == 'z':
                    off = rec.zero_ft()
                    if off is None:
                        print("\n[ft  ] no sample yet, tare skipped")
                    else:
                        print(f"\n[ft  ] tared (press unloaded) offset fz="
                              f"({off[2]:.1f},{off[8]:.1f})N - live forces "
                              f"still measured")
                elif key == 'u':
                    rec.unzero_ft(); print("\n[ft  ] tare cleared (raw)")
                elif key == 'q':
                    break

            now = time.time()
            if now - last_status >= 0.25:
                last_status = now
                with rec.rg2ft._lock:
                    ft = rec.rg2ft.last_ft.copy()
                    w = rec.rg2ft.last_width
                    det = rec.rg2ft.last_grip_det
                    tared = bool(np.any(rec.rg2ft.ft_offset))
                state = 'REC ' if rec._recording else 'idle'
                fzl, fzr = ft[2], ft[8]
                vid = (f"cap={rec._cap_fps:4.1f}fps enc={rec._n_encoded:5d} | "
                       if rec.save_video else "")
                name = rec._cur_demo_name if rec._recording else f"n={rec._n_episodes}"
                zer = 'tared' if tared else 'raw  '
                line = (f"[{state}] {name} {vid}"
                        f"w={w*1000:5.1f}mm fz=({fzl:6.1f},{fzr:6.1f})N det={det} "
                        f"{zer} ")
                sys.stdout.write('\r' + line)
                sys.stdout.flush()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if old_attrs is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)
        print("\nShutting down...")
        rec.stop()
        print("Done.")


if __name__ == '__main__':
    main()
