#!/usr/bin/env python3
"""Minimal streaming integration example using synthetic sensor values."""

from pathlib import Path

import numpy as np

from valve_state_classifier import ValveStateRuntime


HERE = Path(__file__).resolve().parent


def main() -> None:
    runtime = ValveStateRuntime(HERE / "model/final.pt", device="auto")

    # In the real control loop, enqueue every F/T sample received up to the
    # current RGB timestamp. wrench_12d order is:
    # [L_Fx,L_Fy,L_Fz,L_Tx,L_Ty,L_Tz,R_Fx,R_Fy,R_Fz,R_Tx,R_Ty,R_Tz].
    for index in range(50):
        runtime.append_wrench(
            timestamp_s=index / 100.0,
            wrench_12d=np.zeros(12, dtype=np.float32),
        )

    # Replace these values with the current camera and robot observations.
    prediction = runtime.predict(
        timestamp_s=0.50,
        rgb=np.zeros((224, 224, 3), dtype=np.uint8),  # RGB, not BGR
        position_m=np.zeros(3, dtype=np.float32),
        rotation_axis_angle_rad=np.zeros(3, dtype=np.float32),
        gripper_width_m=0.05,
    )
    print(prediction.to_dict())

    # Call runtime.reset() at every episode boundary. Do not carry temporal or
    # force histories from one episode into the next.


if __name__ == "__main__":
    main()
