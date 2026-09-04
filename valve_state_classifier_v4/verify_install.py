#!/usr/bin/env python3
"""Verify checkpoint integrity, imports, strict state loading, and one forward pass."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from valve_state_classifier import ValveStateRuntime


HERE = Path(__file__).resolve().parent
EXPECTED_SHA256 = "74763bcf640b05c8e9ea25f35021b1c42743ed348ba9c2829338720bc53558b1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    args = parser.parse_args()
    checkpoint = HERE / "model/final.pt"
    actual = sha256_file(checkpoint)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {actual}")

    runtime = ValveStateRuntime(checkpoint, device=args.device)
    for index in range(50):
        runtime.append_wrench(index / 100.0, np.zeros(12, dtype=np.float32))
    result = runtime.predict(
        timestamp_s=0.50,
        rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        position_m=np.zeros(3, dtype=np.float32),
        rotation_axis_angle_rad=np.zeros(3, dtype=np.float32),
        gripper_width_m=0.05,
    )
    print("checkpoint SHA-256: OK")
    print("strict model load: OK")
    print("forward pass: OK")
    print("device:", runtime.device)
    print("dummy output:", result.phase, f"confidence={result.confidence:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
