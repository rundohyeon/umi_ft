"""Read-only camera and RG2-FT streams for the standalone context observer.

This module does not import robot controllers or send any Modbus writes.
Capture continues independently of inference so native histories are retained.
"""
from __future__ import annotations

from collections import deque
import threading
import time

import numpy as np

from umi.real_world.rg2ft_protocol import RG2FTModbusClient, read_ft_status_full
from umi.real_world.rg2ft_startup_bias import FTStartupBiasConfig, estimate_startup_bias


class SensorStream:
    def __init__(self, capacity):
        self._samples = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    def append(self, timestamp, value):
        with self._lock:
            if self._samples and timestamp <= self._samples[-1][0]:
                raise RuntimeError('Sensor clock moved backwards; restart the observer')
            self._samples.append((timestamp, value))

    def snapshot(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f'{type(self).__name__} failed: {self._error}') from self._error
            samples = list(self._samples)
        return np.asarray([s[0] for s in samples], dtype=np.float64), [s[1] for s in samples]

    def _worker(self):
        try:
            self.run()
        except Exception as exc:
            with self._lock:
                self._error = exc

    def __enter__(self):
        self._thread = threading.Thread(target=self._worker, daemon=True, name=type(self).__name__)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=3)


class NativeFTStream(SensorStream):
    def __init__(self, hostname, *, port=502, slave_id=65, frequency=100,
                 receive_latency=0.01, timeout=0.5, client_factory=RG2FTModbusClient):
        super().__init__(capacity=4096)
        if frequency != 100 or not np.isfinite(receive_latency) or receive_latency < 0:
            raise ValueError('This classifier requires native 100 Hz F/T and nonnegative latency')
        self.client = client_factory(hostname, port=port, slave_id=slave_id, timeout=timeout)
        self.frequency, self.receive_latency = frequency, receive_latency

    def run(self):
        try:
            self.client.connect()
            deadline = time.monotonic()
            while not self._stop.is_set():
                wrench, _, _, _, status = read_ft_status_full(self.client)
                received = time.time()
                if status != 0 or not np.isfinite(wrench).all():
                    raise RuntimeError(f'RG2-FT sensor status={status}; no valid F/T measurement')
                self.append(received - self.receive_latency, wrench)
                # Never fabricate samples to catch up after a slow Modbus read.
                deadline = max(deadline + 1 / self.frequency, time.monotonic())
                self._stop.wait(max(0, deadline - time.monotonic()))
        finally:
            self.client.close()

    def calibrate(self, config):
        cfg = FTStartupBiasConfig.from_mapping(config)
        start = time.time()
        deadline = time.monotonic() + cfg.timeout_s
        while time.monotonic() < deadline:
            ts, values = self.snapshot()
            keep = ts >= start
            if keep.sum() >= cfg.sample_count:
                wrench = np.stack(values)[keep]
                times = ts[keep]
                if np.max(np.diff(times[-cfg.sample_count:])) > 0.025:
                    raise ValueError('Startup F/T polling has gaps; check connection and 100 Hz rate')
                return estimate_startup_bias(times, wrench[:, :6], wrench[:, 6:], cfg)
            self._stop.wait(0.01)
        raise TimeoutError(f'Could not acquire {cfg.sample_count} fresh F/T samples for software bias')


class CameraStream(SensorStream):
    def __init__(self, device, *, resolution=(1920, 1080), fps=60,
                 receive_latency=0.125, capture_factory=None):
        super().__init__(capacity=16)
        if not 59 <= fps <= 61 or not np.isfinite(receive_latency) or receive_latency < 0:
            raise ValueError('Camera must capture at about 60 FPS; latency must be nonnegative')
        self.device, self.resolution, self.fps = device, tuple(resolution), fps
        self.receive_latency, self.capture_factory = receive_latency, capture_factory

    def run(self):
        import cv2
        factory = self.capture_factory or cv2.VideoCapture
        cap = factory(self.device, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open V4L2 camera {self.device}')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self._stop.is_set():
                ok, frame = cap.read()
                received = time.time()
                if not ok:
                    raise RuntimeError('Camera read failed')
                if frame.shape != (self.resolution[1], self.resolution[0], 3):
                    raise RuntimeError(f'Camera returned {frame.shape}, requested {self.resolution}')
                self.append(received - self.receive_latency, frame)
        finally:
            cap.release()
