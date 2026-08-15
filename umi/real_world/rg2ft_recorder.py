"""RG2-FT logger + simple open/close control for demonstration collection.

Companion to rg2ft_controller.py (which does the real-time control loop for
policy eval). This module is meant to run *during teleoperated data collection*:
it continuously polls the OnRobot RG2-FT F/T + width registers over Modbus/TCP,
buffers timestamped samples while an episode is active, and (on request) sends
simple open / close / zero commands to the gripper.

Everything talks Modbus/TCP over a raw socket so this module is independent of
the pymodbus version used by the control env. A single background thread owns
the socket: every loop it first drains a small command queue (open/close/zero
writes) and then performs one F/T read, so reads and writes never race.

Register map (see rg2ft_protocol.py):
  read  addr=257 count=26  -> F/T (r2[2..7],[11..16]), width r2[23], busy, det
  write addr=0 out_zero, addr=2 r_gfr (force 1/10N), addr=3 r_gwd (width 1/10mm),
        addr=4 r_ctr (1 = execute move)

Per-episode data is exposed as numpy arrays (timestamp, gripper_ft, width) and
can be written to CSV via save_episode_csv(); the CSV columns are:
  timestamp, fx_l,fy_l,fz_l,tx_l,ty_l,tz_l,
             fx_r,fy_r,fz_r,tx_r,ty_r,tz_r, width_m
"""
import os
import csv
import time
import pathlib
import threading
from collections import deque

import numpy as np

from umi.real_world.rg2ft_protocol import (
    GRIPPER_SPECS,
    FT_STATUS_ADDRESS, FT_STATUS_COUNT,
    FT_REGISTER_INDICES, FT_FORCE_MASK,
    to_int16, width_to_meters, meters_to_width_reg)

FT_COLUMNS = ['fx_l', 'fy_l', 'fz_l', 'tx_l', 'ty_l', 'tz_l',
              'fx_r', 'fy_r', 'fz_r', 'tx_r', 'ty_r', 'tz_r']


class _ModbusTcpClient:
    """Read/write Modbus/TCP client for the RG2-FT, backed by pymodbus.

    Uses the same library and transaction handling as the working OnRobot ROS
    driver (Ros2-OnRobot-RG2-FT/comModbusTcp.py). A hand-rolled raw-socket
    client de-synced under the rapid read+write mix this recorder does, which
    corrupted the status reads and made the gripper oscillate; pymodbus manages
    transaction IDs and framing correctly.
    """

    def __init__(self, hostname, port=502, slave_id=65, timeout=1.0):
        self.hostname = hostname
        self.port = port
        self.slave_id = slave_id
        self.timeout = timeout
        self._client = None
        # pymodbus renamed the unit kwarg to device_id in v3
        try:
            from pymodbus.client.sync import ModbusTcpClient  # noqa: F401
            self._v2 = True
        except ModuleNotFoundError:
            self._v2 = False
        self._slave_kwargs = ({'unit': slave_id} if self._v2
                              else {'device_id': slave_id})

    def connect(self):
        self.close()
        if self._v2:
            from pymodbus.client.sync import ModbusTcpClient
        else:
            from pymodbus.client import ModbusTcpClient
        self._client = ModbusTcpClient(
            self.hostname, port=self.port, timeout=self.timeout)
        if not self._client.connect():
            raise ConnectionError(
                f"could not connect to {self.hostname}:{self.port}")

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def read_holding_registers(self, address, count):
        resp = self._client.read_holding_registers(
            address=address, count=count, **self._slave_kwargs)
        if resp is None or resp.isError():
            raise IOError(f"Modbus read error @{address}: {resp}")
        return list(resp.registers)

    def write_register(self, address, value):
        resp = self._client.write_register(
            address=address, value=int(value) & 0xFFFF, **self._slave_kwargs)
        if resp is None or resp.isError():
            raise IOError(f"Modbus write error @{address}: {resp}")


def read_ft_status_raw(client):
    """Read F/T + width + status.

    Returns (ft(12,), width_reg, busy, grip_det, sta) where `sta` is the
    finger/proximity status flag (g_sta) that the OnRobot ROS driver uses to
    gate command sending: 0 when the fingers/prox sensors are idle, 1 otherwise.
    """
    r2 = client.read_holding_registers(
        address=FT_STATUS_ADDRESS, count=FT_STATUS_COUNT)
    raw = np.array([to_int16(r2[i]) for i in FT_REGISTER_INDICES], dtype=np.float64)
    ft = np.where(FT_FORCE_MASK, raw / 10.0, raw / 100.0)
    # g_sta: sta_fing_l=r2[0], sta_fing_r=r2[9], sta_prox_l=r2[17], sta_prox_r=r2[20]
    sta = 0 if (r2[0] == 0 and r2[9] == 0 and r2[17] == 0 and r2[20] == 0) else 1
    # width is signed: reads slightly negative (~-0.6mm) when fully closed
    return ft, to_int16(r2[23]), r2[24], r2[25], sta


class RG2FTRecorder:
    def __init__(self,
            hostname='192.168.2.1',
            port=502,
            slave_id=65,
            gripper_type='rg2ft',
            frequency=100,
            default_force=20.0,   # N, used for open/close commands
            verbose=False):
        assert gripper_type in GRIPPER_SPECS
        self.hostname = hostname
        self.port = port
        self.slave_id = slave_id
        self.gripper_type = gripper_type
        self.frequency = frequency
        self.max_width_reg = GRIPPER_SPECS[gripper_type]['max_width']
        self.max_force = GRIPPER_SPECS[gripper_type]['max_force']
        self.default_force = default_force
        self.verbose = verbose

        # shared state (protected by _lock)
        self._lock = threading.Lock()
        self._recording = False
        self._ts_buf = []
        self._ft_buf = []
        self._w_buf = []

        # one-shot register writes: list of (address, value)
        self._cmd_lock = threading.Lock()
        self._cmd_q = deque()
        # F/T zero (tare) is done in SOFTWARE, not with the gripper's out_zero
        # register. out_zero (register 0) is not a bias snapshot: while it is 1
        # the RG2-FT reports all forces and torques as literally 0 -- it stops
        # measuring, so a grasped object reads 0 N too. (The OnRobot ROS driver
        # documents it as "if the value is 1, all force and torque values will
        # be set to 0" and forces it to 0 at startup to enable live readings;
        # this class does the same on connect.) The tare below instead
        # subtracts a baseline captured from the live signal, so real forces
        # keep coming through.
        self._raw_buf = deque(maxlen=100)      # recent raw F/T, for tare avg
        self.ft_offset = np.zeros(12, dtype=np.float64)
        self.last_ft_raw = np.zeros(12, dtype=np.float64)
        # Grip target commanded by the poll thread (Modbus registers 0,2,3,4,
        # same as the OnRobot ROS driver). The RG2-FT springs back to fully
        # open when it stops being commanded, so:
        #   mode='hold' (close/set_width): keep commanding every loop to hold.
        #   mode='rest' (open): command until the target width is reached, then
        #     stop -- the rest state IS fully open, so it settles there instead
        #     of the re-grip oscillation continuous open commands cause.
        self._target_lock = threading.Lock()
        self._target_reg = None      # None = no active command
        self._target_force_reg = 0
        self._target_mode = 'hold'   # 'hold' | 'rest'

        # latest reading for live display (always updated)
        self.last_ft = np.zeros(12, dtype=np.float64)
        self.last_width = 0.0
        self.last_busy = 0
        self.last_grip_det = 0
        self.last_sample_time = 0.0
        self.n_samples_current = 0

        self._thread = None
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._connect_error = None

    # ---------- lifecycle ----------
    def start(self, wait=True, timeout=5.0):
        assert self._thread is None
        self._stop_event.clear()
        self._connected_event.clear()
        self._connect_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if wait:
            if not self._connected_event.wait(timeout=timeout):
                self.stop()
                raise TimeoutError(
                    f"RG2-FT did not connect at {self.hostname}:{self.port} "
                    f"within {timeout}s")
            if self._connect_error is not None:
                self.stop()
                raise self._connect_error

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def is_recording(self):
        with self._lock:
            return self._recording

    # ---------- episode control ----------
    def start_episode(self):
        with self._lock:
            if self._recording:
                return
            self._ts_buf = []
            self._ft_buf = []
            self._w_buf = []
            self._recording = True
            self.n_samples_current = 0

    def end_episode(self):
        """Stop accumulating; return dict of numpy arrays for the episode."""
        with self._lock:
            self._recording = False
            ts = np.asarray(self._ts_buf, dtype=np.float64)
            ft = (np.stack(self._ft_buf, axis=0).astype(np.float64)
                  if self._ft_buf else np.zeros((0, 12), dtype=np.float64))
            w = np.asarray(self._w_buf, dtype=np.float64)
            self._ts_buf = []
            self._ft_buf = []
            self._w_buf = []
            offset = self.ft_offset.copy()
        # gripper_ft is tared; raw = gripper_ft + ft_offset
        return {'timestamp': ts, 'gripper_ft': ft, 'width': w,
                'ft_offset': offset}

    @staticmethod
    def save_episode_csv(path, data):
        """Write episode dict (timestamp, gripper_ft, width) to a CSV file."""
        path = pathlib.Path(os.path.expanduser(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = data['timestamp']
        ft = data['gripper_ft']
        w = data['width']
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp'] + FT_COLUMNS + ['width_m'])
            for i in range(len(ts)):
                row = [f"{ts[i]:.6f}"] + [f"{v:.4f}" for v in ft[i]] + \
                      [f"{w[i]:.5f}"]
                writer.writerow(row)
        return path

    # ---------- gripper control (executed by the poll thread) ----------
    def _queue_move(self, width_m, force_n, mode):
        """mode='hold': command continuously (needed to stay closed against the
        gripper's spring-open rest state). mode='rest': command only until the
        width reaches the target, then stop -- the gripper's uncommanded rest
        state is fully open, so this settles cleanly at open without the
        re-grip oscillation that continuous open commands cause."""
        r_gwd = meters_to_width_reg(width_m, self.max_width_reg)
        r_gfr = int(max(0, min(self.max_force, round(force_n * 10))))
        with self._target_lock:
            self._target_reg = r_gwd
            self._target_force_reg = r_gfr
            self._target_mode = mode

    def open_gripper(self, force_n=None):
        self._queue_move(self.max_width_reg / 10000.0,
                         self.default_force if force_n is None else force_n,
                         mode='rest')

    def close_gripper(self, force_n=None):
        self._queue_move(0.0,
                         self.default_force if force_n is None else force_n,
                         mode='hold')

    def set_width(self, width_m, force_n=None):
        self._queue_move(width_m,
                         self.default_force if force_n is None else force_n,
                         mode='hold')

    def zero_ft(self, n_avg=25):
        """Tare in software: subtract the current baseline from here on.

        Averages the most recent `n_avg` raw samples (~0.25 s at 100 Hz) so a
        noisy sample cannot skew the offset. Live forces keep coming through --
        only the constant bias is removed -- so a grasped object still reads a
        real force. Press with the gripper unloaded. Returns the offset, or
        None if no sample has arrived yet.
        """
        with self._lock:
            if not self._raw_buf:
                return None
            samples = list(self._raw_buf)[-max(1, int(n_avg)):]
            self.ft_offset = np.mean(np.stack(samples, axis=0), axis=0)
            return self.ft_offset.copy()

    def unzero_ft(self):
        """Clear the tare: back to raw sensor readings."""
        with self._lock:
            self.ft_offset = np.zeros(12, dtype=np.float64)

    @property
    def is_tared(self):
        with self._lock:
            return bool(np.any(self.ft_offset))

    # ---------- internals ----------
    def _run(self):
        client = _ModbusTcpClient(
            self.hostname, port=self.port, slave_id=self.slave_id, timeout=1.0)
        try:
            client.connect()
            read_ft_status_raw(client)  # verify readable
            # Enable live F/T readings: out_zero latches inside the gripper and
            # survives restarts, and while it is 1 every force/torque reads 0.
            # Clear it here so a leftover latch cannot silently blank the log.
            try:
                client.write_register(0, 0)
            except Exception:
                pass
            self._connected_event.set()
        except Exception as e:
            self._connect_error = e
            self._connected_event.set()
            return

        dt = 1.0 / self.frequency
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            # 1) drain + execute one-shot writes (e.g. F/T zero)
            pending = []
            with self._cmd_lock:
                while self._cmd_q:
                    pending.append(self._cmd_q.popleft())
            # 2) resolve grip target
            with self._target_lock:
                tgt = self._target_reg
                tgt_force = self._target_force_reg
                tgt_mode = self._target_mode
            try:
                for addr, val in pending:
                    client.write_register(addr, val)
                # 3) read F/T + status
                ft_raw, grip_width_reg, busy, grip_det, sta = read_ft_status_raw(client)
                t_measure = time.time()
                # 4) command the target while fingers are idle (g_sta gate),
                #    using the same registers 0,2,3,4 as the OnRobot ROS driver.
                #    mode='rest' (open): once the width has reached the target,
                #    stop commanding so the gripper settles open instead of
                #    re-gripping. mode='hold' (close): keep commanding so it
                #    stays closed against the spring-open rest state.
                if tgt is not None and not sta:
                    client.write_register(0, 0)          # out_zero (keep live)
                    client.write_register(2, tgt_force)  # r_gfr (force)
                    client.write_register(3, tgt)        # r_gwd (target width)
                    client.write_register(4, 1)          # r_ctr (execute)
                if tgt is not None and tgt_mode == 'rest' \
                        and abs(grip_width_reg - tgt) <= 50:  # within ~5mm
                    with self._target_lock:
                        if self._target_reg == tgt and self._target_mode == 'rest':
                            self._target_reg = None
            except Exception as e:
                if self.verbose:
                    print(f"[rg2ft_recorder] modbus error: {e}, reconnecting...")
                client.close()
                time.sleep(0.2)
                try:
                    client.connect()
                except Exception:
                    pass
                continue

            width_m = width_to_meters(grip_width_reg)
            with self._lock:
                # tare in software; ft_offset is all zeros until zero_ft()
                self._raw_buf.append(ft_raw)
                ft = ft_raw - self.ft_offset
                self.last_ft_raw = ft_raw
                self.last_ft = ft
                self.last_width = width_m
                self.last_busy = busy
                self.last_grip_det = grip_det
                self.last_sample_time = t_measure
                if self._recording:
                    self._ts_buf.append(t_measure)
                    self._ft_buf.append(ft)
                    self._w_buf.append(width_m)
                    self.n_samples_current = len(self._ts_buf)

            next_t += dt
            sleep_t = next_t - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.monotonic()

        client.close()


# ---------- standalone test CLI ----------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="RG2-FT logger/control test.")
    parser.add_argument('--hostname', default='192.168.2.1')
    parser.add_argument('--port', type=int, default=502)
    parser.add_argument('--slave_id', type=int, default=65)
    parser.add_argument('--frequency', type=int, default=100)
    parser.add_argument('-o', '--output_csv', default=None,
        help="if set, record --duration s and save a CSV")
    parser.add_argument('--duration', type=float, default=3.0)
    parser.add_argument('--monitor', action='store_true',
        help="print live F/T at 5Hz, no recording")
    parser.add_argument('--open', action='store_true', help="open then exit")
    parser.add_argument('--close', action='store_true', help="close then exit")
    args = parser.parse_args()

    rec = RG2FTRecorder(hostname=args.hostname, port=args.port,
        slave_id=args.slave_id, frequency=args.frequency, verbose=True)
    rec.start()
    print(f"connected to RG2-FT at {args.hostname}:{args.port}")
    try:
        if args.open:
            rec.open_gripper(); time.sleep(1.5)
        elif args.close:
            rec.close_gripper(); time.sleep(1.5)
        elif args.monitor:
            while True:
                with rec._lock:
                    ft = rec.last_ft.copy(); w = rec.last_width
                print(f"width={w*1000:6.1f}mm  "
                      f"fL=[{ft[0]:6.1f},{ft[1]:6.1f},{ft[2]:6.1f}]N  "
                      f"fR=[{ft[6]:6.1f},{ft[7]:6.1f},{ft[8]:6.1f}]N",
                      end='\r', flush=True)
                time.sleep(0.2)
        else:
            print(f"recording {args.duration}s at {args.frequency}Hz...")
            rec.start_episode(); time.sleep(args.duration)
            data = rec.end_episode()
            print(f"got {len(data['timestamp'])} samples")
            if args.output_csv:
                p = rec.save_episode_csv(args.output_csv, data)
                print("saved", p)
    except KeyboardInterrupt:
        pass
    finally:
        rec.stop()
