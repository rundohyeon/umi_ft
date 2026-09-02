# Evaluating the dual-finger F/T policy

Run commands from the repository root in the `umi` environment. Only load a
trusted local checkpoint: PyTorch/dill checkpoint loading can execute Python.

## Quick offline validation

The evaluator never imports or opens robot, camera, RG2/Modbus, or controller
interfaces.

```bash
conda run -n umi python eval_dual_ft_offline.py \
  --checkpoint /path/to/checkpoints/latest.ckpt \
  --dataset session_260827/dataset.zarr.zip \
  --force-sidecar session_260827/dataset_force_sidecar.zarr \
  --weights auto \
  --device cuda \
  --batch-size 8 \
  --max-samples 256 \
  --seed 42 \
  --output data/eval_dual_ft/offline_metrics.json
```

`--weights auto` evaluates `ema_model` when the checkpoint says
`training.use_ema=true`; a missing EMA is treated as a corrupt checkpoint and
does not silently fall back to `model`.

For the complete 5,778-sample validation split:

```bash
conda run -n umi python eval_dual_ft_offline.py \
  --checkpoint /path/to/checkpoints/latest.ckpt \
  --dataset session_260827/dataset.zarr.zip \
  --force-sidecar session_260827/dataset_force_sidecar.zarr \
  --device cuda \
  --max-samples 0 \
  --full-dataset-hash \
  --output data/eval_dual_ft/validation_full.json
```

The full-content hash reads all roughly 3 GB, including RGB chunks. Without
that flag, the report still content-hashes every effective non-RGB input and
target array and records a path/size manifest.

## What is checked and reported

The evaluator fails closed unless the checkpoint contains:

- condition `[B,786]`, action `[B,16,11]`, and timestep embedding 32;
- stock offline position/axis-angle pose with no quaternion round trip, while
  live/legacy 7-D pose remains fixed to `xyzw` (`qx,qy,qz,qw`);
- official-form fusion over four local tokens, pose-only 18-D proprioception,
  architecture marker 2, and left/right temporal marker 1;
- native left/right `[32,6]` F/T sliced only from sidecar `wrench_12d`, which
  already contains the per-episode bias removal, with identity axis/sign;
- a complete, finite checkpoint normalizer for every observation and all 11
  action channels.

Metrics include diffusion loss, raw overall error, position metres,
rotation-6D error, rotation geodesic degrees, width metres, signed grasp-force
newtons, checkpoint-normalized action error, F/T causal age, action timestamp
span, and inference latency.

The JSON also records the train/validation episode split, sync verdict/offset
provenance, bias hash/statistics, full-split causal ages, array schema, action
rate/horizon, checkpoint SHA-256, and exact selected sample indices.

The force target is a measured label and width-feedback reference, not a direct
RG2 force-register command:

```text
0.5 * ((right native Fz - right episode-bias Fz)
     - (left native Fz  - left episode-bias Fz))
```

It is linearly aligned to RGB timestamps. The dataset loader numerically
verifies this formula before evaluation.

The checkpoint also fixes the bounded width correction to `0.0001 m/N`, with a
`0.5 N` deadband and `1 mm` maximum correction. Offline inspection reports
these values. Live startup acquires 200 unique unloaded samples, rejects a
moving/contact-loaded calibration, applies the new native 12-D bias only to
this process, and writes calibration provenance beside the evaluation logs.

## Metadata-only checkpoint inspection

```bash
conda run -n umi python eval_real_indy_rg2_dual_ft.py \
  --checkpoint /path/to/checkpoints/latest.ckpt \
  --inspect-checkpoint
```

## Live deployment status

Start with a single controller-connected planning cycle:

```bash
conda run -n umi python eval_real_indy_rg2_dual_ft.py \
  --checkpoint /path/to/checkpoints/latest.ckpt \
  --robot-config example/eval_robots_config_indy_rg2.yaml \
  --log-dir data/eval_dual_ft/run1 \
  --match-dataset session_260827/dataset.zarr.zip \
  --n-action-steps 1 \
  --dry-run \
  --max-cycles 1 \
  --reference-arg=--print_motion_debug
```

`--dry-run` suppresses waypoint submission from the evaluation script but
still connects to the robot/RG2 controllers; it is not a passive protocol-free
mode. After checking the bias record, exact policy image/first-frame overlap,
coordinates, force sign, and predicted deltas, remove `--dry-run` for live
actuation. Teleop remains in control until `c` is pressed.

The live path is fail-closed on stale F/T history, failed startup calibration,
non-identity dataset coordinate transforms, preprocessing mismatch, F/T
overload, per-waypoint position/rotation/width deltas, workspace limits, and
NaN/Inf. An Indy watchdog disarms streaming after the final queued waypoint;
stopping or a safety exception clears future waypoints and holds the measured
TCP. Hardware emergency stop and low-speed commissioning are still required.
