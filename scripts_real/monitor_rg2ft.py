"""RG2-FT live force/torque monitor.

Recording-free companion to record_gopro_rg2ft.py: it connects to the OnRobot
RG2-FT over Modbus/TCP and just draws the 12-axis force/torque (both fingers)
plus the gripper width in the terminal, refreshed several times a second. No
files are written and the capture card is never opened.

  OnRobot RG2-FT --Modbus/TCP--> 192.168.2.1:502   (12-axis F/T + width)

Keys (same as the recorder):
  o            : open gripper
  c            : close gripper
  z            : tare the F/T sensor (software zero; live forces still shown)
  x            : clear the tare (back to raw sensor values)
  p            : reset the peak-hold column
  q or Ctrl-C  : quit

Run inside the `umi-rec` conda env:
  python scripts_real/monitor_rg2ft.py
"""
import os
import sys
import time
import shutil
import select
import termios
import tty
import argparse

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from umi.real_world.rg2ft_recorder import RG2FTRecorder

AXES = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
IS_FORCE = [True, True, True, False, False, False]

RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
CYAN = '\033[36m'


def color_for(frac):
    """green / yellow / red by how close the reading is to the bar limit."""
    if frac >= 0.9:
        return RED
    if frac >= 0.5:
        return YELLOW
    return GREEN


def bar(value, limit, width, use_color=True):
    """Centered bipolar bar: zero at the middle, negative left, positive right."""
    half = max(1, (width - 1) // 2)
    frac = min(1.0, abs(value) / limit) if limit > 0 else 0.0
    n = int(round(frac * half))
    fill = '#' * n
    pad = ' ' * (half - n)
    if value < 0:
        left, right = pad + fill, ' ' * half
    else:
        left, right = ' ' * half, fill + pad
    body = f"{left}|{right}"
    if use_color:
        return f"{color_for(frac)}{body}{RESET}"
    return body


class FTMonitor:
    def __init__(self, host, port, slave_id, hz, gripper_force,
                 fmax, tmax, smooth, use_color):
        self.rg2ft = RG2FTRecorder(
            hostname=host, port=port, slave_id=slave_id,
            frequency=hz, default_force=gripper_force, verbose=False)
        self.host = host
        self.port = port
        self.fmax = fmax
        self.tmax = tmax
        self.smooth = smooth
        self.use_color = use_color

        self.disp = np.zeros(12, dtype=np.float64)   # smoothed display values
        self.peak = np.zeros(12, dtype=np.float64)   # |value| peak hold
        self.width_m = 0.0
        self.grip_det = 0
        self.last_sample_time = 0.0
        self.sample_hz = 0.0
        self._n_prev = 0
        self._t_prev = 0.0
        self._prev_sample_time = 0.0
        self._n_reads = 0

    # ---------- data ----------
    def poll(self):
        with self.rg2ft._lock:
            ft = self.rg2ft.last_ft.copy()
            self.width_m = self.rg2ft.last_width
            self.grip_det = self.rg2ft.last_grip_det
            t = self.rg2ft.last_sample_time
        # sensor-side update rate (counts distinct sample timestamps)
        if t != self._prev_sample_time:
            self._prev_sample_time = t
            self._n_reads += 1
        now = time.monotonic()
        if self._t_prev == 0.0:
            self._t_prev = now
        elif now - self._t_prev >= 1.0:
            self.sample_hz = (self._n_reads - self._n_prev) / (now - self._t_prev)
            self._n_prev = self._n_reads
            self._t_prev = now
        self.last_sample_time = t

        val = ft
        a = self.smooth
        self.disp = val if a >= 1.0 else (a * val + (1.0 - a) * self.disp)
        self.peak = np.maximum(self.peak, np.abs(self.disp))

    def zero(self):
        """Software tare: subtract the current baseline, keep live forces."""
        self.rg2ft.zero_ft()
        self.disp = np.zeros(12, dtype=np.float64)
        self.peak = np.zeros(12, dtype=np.float64)

    def clear_tare(self):
        """Back to raw sensor readings."""
        self.rg2ft.unzero_ft()
        self.peak = np.zeros(12, dtype=np.float64)

    @property
    def tared(self):
        return bool(np.any(self.rg2ft.ft_offset))

    def reset_peak(self):
        self.peak = np.zeros(12, dtype=np.float64)

    # ---------- rendering ----------
    def render(self):
        cols = shutil.get_terminal_size((100, 30)).columns
        # 2 fingers side by side when wide enough, else stacked
        side_by_side = cols >= 104
        bar_w = 21 if side_by_side else min(41, max(11, cols - 40))

        stale = (time.time() - self.last_sample_time) > 1.0
        lines = []
        b, r, d, c = ((BOLD, RESET, DIM, CYAN) if self.use_color
                      else ('', '', '', ''))
        lines.append(f"{b}RG2-FT live force / torque{r}   "
                     f"{d}{self.host}:{self.port}{r}")
        fl = np.linalg.norm(self.disp[0:3])
        fr = np.linalg.norm(self.disp[6:9])
        status = []
        status.append(f"width={self.width_m*1000:6.1f}mm")
        status.append(f"grip_det={self.grip_det}")
        status.append(f"|F|L={fl:5.1f}N |F|R={fr:5.1f}N")
        status.append(f"{self.sample_hz:5.1f}Hz")
        if self.tared:
            status.append("TARED")
        if stale:
            status.append("!! NO DATA !!")
        lines.append('  '.join(status))
        lines.append('')

        half = max(1, (bar_w - 1) // 2)
        scale_hdr = '-' + ' ' * (half - 1) + '0' + ' ' * (half - 1) + '+'

        def finger_block(offset, title):
            out = [f"{c}{title}{r}",
                   f"{d}  {'axis':<4}{'value':>7}    [{scale_hdr}] "
                   f"{'peak':>7}{r}"]
            for i in range(6):
                v = self.disp[offset + i]
                pk = self.peak[offset + i]
                if IS_FORCE[i]:
                    lim, unit, fmt = self.fmax, 'N ', '{:7.2f}'
                else:
                    lim, unit, fmt = self.tmax, 'Nm', '{:7.3f}'
                out.append(f"  {AXES[i]:<4}{fmt.format(v)} {unit} "
                           f"[{bar(v, lim, bar_w, self.use_color)}] "
                           f"{fmt.format(pk)}")
            return out

        left = finger_block(0, 'LEFT finger')
        right = finger_block(6, 'RIGHT finger')
        if side_by_side:
            pad = max(len(_strip(x)) for x in left) + 4
            for a_line, b_line in zip(left, right):
                gap = ' ' * max(1, pad - len(_strip(a_line)))
                lines.append(a_line + gap + b_line)
        else:
            lines.extend(left)
            lines.append('')
            lines.extend(right)
        lines.append('')
        lines.append(f"{d}scale: force +-{self.fmax:g}N  torque +-{self.tmax:g}Nm{r}")
        lines.append(f"{b}o{r} open  {b}c{r} close  {b}z{r} zero-FT  "
                     f"{b}x{r} clear-tare  {b}p{r} reset-peak  {b}q{r} quit")

        buf = ['\033[H']
        for line in lines:
            buf.append(line + '\033[K\n')
        buf.append('\033[J')
        sys.stdout.write(''.join(buf))
        sys.stdout.flush()


def _strip(s):
    """Visible length helper: drop ANSI escapes."""
    out = []
    i = 0
    while i < len(s):
        if s[i] == '\033':
            while i < len(s) and s[i] not in 'm':
                i += 1
            i += 1
            continue
        out.append(s[i])
        i += 1
    return ''.join(out)


def main():
    parser = argparse.ArgumentParser(
        description="RG2-FT live force/torque monitor (no recording)")
    parser.add_argument('--host', default='192.168.2.1')
    parser.add_argument('--port', type=int, default=502)
    parser.add_argument('--slave_id', type=int, default=65)
    parser.add_argument('--rg2ft_hz', type=int, default=100,
        help="Modbus polling rate")
    parser.add_argument('--refresh_hz', type=float, default=20.0,
        help="screen refresh rate")
    parser.add_argument('--gripper_force', type=float, default=20.0)
    parser.add_argument('--fmax', type=float, default=20.0,
        help="force bar full scale, N")
    parser.add_argument('--tmax', type=float, default=1.0,
        help="torque bar full scale, Nm")
    parser.add_argument('--smooth', type=float, default=0.4,
        help="display EMA weight of each new sample (1.0 = raw, no smoothing)")
    parser.add_argument('--zero_on_start', action='store_true',
        help="tare the F/T right after connecting")
    parser.add_argument('--no_color', action='store_true')
    args = parser.parse_args()

    mon = FTMonitor(host=args.host, port=args.port, slave_id=args.slave_id,
                    hz=args.rg2ft_hz, gripper_force=args.gripper_force,
                    fmax=args.fmax, tmax=args.tmax,
                    smooth=min(1.0, max(0.01, args.smooth)),
                    use_color=(not args.no_color) and sys.stdout.isatty())

    print(f"Connecting RG2-FT @ {args.host}:{args.port} ...")
    mon.rg2ft.start()
    if args.zero_on_start:
        time.sleep(0.3)
        mon.zero()

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

    sys.stdout.write('\033[2J\033[?25l')  # clear screen, hide cursor
    dt = 1.0 / max(1.0, args.refresh_hz)
    last_draw = 0.0
    try:
        while True:
            key = read_key()
            if key is not None:
                if key == 'o':
                    mon.rg2ft.open_gripper()
                elif key == 'c':
                    mon.rg2ft.close_gripper()
                elif key == 'z':
                    mon.zero()
                elif key == 'x':
                    mon.clear_tare()
                elif key == 'p':
                    mon.reset_peak()
                elif key == 'q':
                    break

            mon.poll()
            now = time.monotonic()
            if now - last_draw >= dt:
                last_draw = now
                mon.render()
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write('\033[?25h')  # show cursor
        if old_attrs is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)
        print("\nShutting down...")
        mon.rg2ft.stop()
        print("Done.")


if __name__ == '__main__':
    main()
