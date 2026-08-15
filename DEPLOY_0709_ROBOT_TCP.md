# 0709 robot-TCP deployment

This folder is intended to be copied and run as one self-contained directory.
The scripts resolve every project file relative to their own location; do not
replace those paths with paths from another checkout.

## Coordinate contract

- The model action is a relative motion in the current physical TCP frame.
- TCP `+X` is camera-image down, TCP `+Y` is camera left, and TCP `+Z` is
  camera forward.
- Dataset-to-robot axes and signs are identity.
- Indy reports flange/tool0 because no Tool/TCP is registered in CONTY. Runtime
  feedback therefore applies `T_base_tcp = T_base_flange @ Trans_z(0.235)`, and
  commands apply the inverse transform before sending an absolute flange pose.
- Indy UVW is extrinsic XYZ Euler in degrees:
  `R = Rz(W) @ Ry(V) @ Rx(U)`.

## Local artifacts

- `artifacts/0709_robot_tcp/checkpoints/latest.ckpt`
- `artifacts/0709_robot_tcp/dataset_robot_tcp.zarr.zip`
- `example/eval_robots_config_indy.yaml`

The checkpoint still contains the original training-machine dataset path as
provenance metadata. `eval_real_indy.py` prints that value, then replaces it in
memory with the local `artifacts/0709_robot_tcp/dataset_robot_tcp.zarr.zip`
before constructing the workspace. It does not access the `indy_umi_dkim`
checkout at runtime.

Verify them after copying the folder:

```bash
cd indy_umi_114
(cd artifacts/0709_robot_tcp && sha256sum -c SHA256SUMS)
```

## Bring-up order

Activate the UMI environment first. The target computer must also have the
Neuromeka and Dynamixel SDK dependencies, access to Indy at `192.168.1.10`, the
HD60 camera, and the Dynamixel at `/dev/ttyUSB0` (ID 1, 57600 baud).

First run the one-cycle no-motion audit:

```bash
./run_eval_0709_plan_only.sh
```

`plan_only` suppresses robot action commands, and this wrapper also disables the
gripper connection. It still needs the robot feedback connection and live camera
because it performs one real policy inference.

After inspecting the printed TCP/model input and proposed motion, run:

```bash
./run_eval_0709_robot.sh --max_policy_iters 1 --action_scale 0.2 --freeze_rotation
```

The command above is the first physical-motion smoke test: one inference cycle,
20% translation scale, and frozen rotation. After its TCP-axis direction is
confirmed, remove those three safety overrides to execute the trained policy's
full translation and rotation:

```bash
./run_eval_0709_robot.sh
```

The real wrapper uses integrated Dynamixel feedback, keeps model rotation enabled,
does not load the old match/replay dataset, and does not apply FOV95 rectification.
Press `c` only after checking the live image and current TCP. Keep the emergency
stop available during the first low-risk test.

Initial coordinate validation intentionally keeps action-latency compensation
off. Do not enable the configured 0.1 s compensation until timestamps and the
measured camera/inference/robot latency are redesigned and verified together.

Environment overrides are available without editing the scripts:

```bash
PYTHON_BIN=/path/to/umi/bin/python \
EVAL_OUTPUT_DIR="$PWD/data/my_eval" \
./run_eval_0709_robot.sh
```

Do not add `-m/--match_dataset` for normal policy execution. Absolute SLAM/tag
poses are not robot-base poses, even though the learned action itself is relative.
