# Indy + RG2-FT evaluation path

This is the RG2-FT-specific deployment path.  The preserved Dynamixel entry
point is `eval_real_indy_dynamixel.py`; do not use its YAML for RG2-FT.

## Control path

```text
eval_real_indy_rg2.py
  -> umi.real_world.umi_env.UmiEnv
  -> umi.real_world.rg2ft_controller.RG2FTController
  -> umi.real_world.rg2ft_protocol.RG2FTModbusClient
  -> OnRobot Compute Box (Modbus/TCP)
```

Policy and teleoperation actions use gripper width in metres.  RG2-FT register
units are 0.1 mm, so 0.0 / 0.05 / 0.1 m map to 0 / 500 / 1000.

The default hardware configuration is
`example/eval_robots_config_indy_rg2.yaml`:

- Indy: `192.168.1.10`
- OnRobot Compute Box: `192.168.2.1:502`
- Modbus device/slave ID: `65`
- RG2-FT width: `0..0.1 m`
- Indy flange-to-training-TCP offset: `0.252 m`
- grip force: `20 N`
- startup auto-open: disabled

Confirm these addresses on the deployment machine before connecting.

## First-scene camera overlay

The launcher loads `data/dataset_ft.zarr.zip` by default and overlays episode 0's
first `camera0_rgb` image at 50% opacity on the live camera pop-up.  Select a
different initial scene with `MATCH_EPISODE=<index>`, or provide another Zarr
dataset with `MATCH_DATASET=/path/to/dataset.zarr.zip`.

## F/T zero at startup

Evaluation software-tares both RG2-FT finger sensors automatically at startup.
It averages the latest 25 raw samples (about 0.25 seconds at 100 Hz) and
subtracts that 12-channel baseline first. The following unloaded startup
calibration is converted into the small residual *after* that tare, matching
the training sidecar's `capture software tare -> episode standing bias`
order. Policy history, force feedback, and F/T safety use this same corrected
signal. Keep the unloaded gripper still while both stages run. Use
`--no_zero_ft_on_start` only when raw sensor values are intentionally required;
change the averaging window with `--ft_zero_samples N`.

The current live RG2-FT transport permits a latest causal F/T age of **20 ms**
(`RG2_FT_MAX_AGE_SEC=0.020`). This is a runtime freshness guard, not force
filtering: every sample remains causal and the independent 50 ms F/T overload
guard remains active. The checkpoint metadata records 12 ms from the training
rig's alignment audit, but that margin caused false stops from a sub-1 ms
transport jitter on this deployment computer. Override only with a positive
seconds value, for example `RG2_FT_MAX_AGE_SEC=0.015`.

## Per-evaluation diagnostics

Each policy run creates `data/eval_indy_rg2/eval_logs/ep*/` containing:

- `comparison.mp4`: training-match frame, exact live policy image, TCP comparison, current horizon-0 output, fusion-attention heatmap, frozen-observer context, and physical startup-bias-corrected left/right F/T input traces (captured before safety rejection as well);
- `input_ft_history.csv`: all 32 causal left/right F/T samples used at every policy call, in physical and normalized units;
- `input_ft_timeline.png`: latest causal F/T sample over the evaluation;
- `policy_outputs.csv`: every raw 11-D output and decoded TCP/gripper target across the full 16-step horizon, including safety-rejected outputs;
- `scheduled_actions.csv`: final scaled/F/T-corrected rows and timestamps that would be sent (`will_send_to_robot=0` in plan-only);
- `fusion_attention.csv`, `fusion_attention_summary.json`, and `fusion_attention_mean.png`: the 4-token RGB/F/T fusion self-attention.

The launcher enables fusion-attention capture by default. Disable only that
capture with `RG2_SAVE_FUSION_ATTENTION=0`. Attention indicates query-to-key
mixing in the fusion layer; it is not causal proof that an input caused the
robot action. With `--show_policy_image`, the same output/attention diagnostic
frame is also shown live in an OpenCV window.

## Four-state valve-context checkpoint

The current contract is `umi_valve_context_sidecar_v2_4state`. The frozen
observer is `answer/best_context.pt` with SHA-256
`b759155d33bd0c00fb5a673f072d38737bedc95e455540beaf74dce0b700c281`.
It outputs `[P(approach), P(turning), P(recovery), P(error), context_valid]`.
The observer is a separate required artifact, not a submodule embedded in the
action-policy checkpoint.

The evaluator reconstructs the observer's two-frame, stride-three causal
window online. Each frame uses the latest 50 native 12-D F/T samples at or
before that RGB timestamp. TCP pose is expressed relative to the latest of the
two frames. Context is routed through the trainable four-expert policy
conditioner and therefore directly influences the predicted action.

Start with command submission disabled:

```bash
cd /ros2_ws/src/indy_umi_rg_ft
RG2_ENABLE_MOTION=0 \
RG2_CHECKPOINT="$PWD/data/context_v2_latest.ckpt" \
VALVE_CLASSIFIER_CHECKPOINT="$PWD/answer/best_context.pt" \
MATCH_DATASET="$PWD/data/dataset_ft.zarr.zip" \
./deploy_real_indy_rg2.sh --steps_per_inference 1 --max_policy_iters 1
```

The evaluator validates the context schema, 5-D input, four-state order, and
observer SHA before policy execution. Enable `RG2_ENABLE_MOTION=1` only after
checking the planned output and all startup/safety diagnostics.

## Legacy 5-phase valve-context checkpoint (2026-09-04)

`data/latest_rg_tf_context.ckpt` is not interchangeable with the ordinary
Dual-F/T checkpoint. It requires the frozen v4 classifier at
`valve_state_classifier_v4/model/final.pt`; the evaluator verifies its SHA-256
against the value embedded in the context checkpoint before enabling policy
control.

Run a conservative, command-suppressed check first:

```bash
cd /ros2_ws/src/indy_umi_rg_ft
RG2_CHECKPOINT="$PWD/data/latest_rg_tf_context.ckpt" \
VALVE_CLASSIFIER_CHECKPOINT="$PWD/valve_state_classifier_v4/model/final.pt" \
MATCH_DATASET="$PWD/data/dataset_ft.zarr.zip" \
RG2_ENABLE_MOTION=0 \
./deploy_real_indy_rg2.sh --steps_per_inference 4 --action_scale 1.0
```

For real motion, change only `RG2_ENABLE_MOTION=1` after validating the
camera, startup F/T bias prompt, current TCP, classifier SHA, and planned
outputs. A dedicated 60 Hz context worker now reads the UVC stream separately
from the roughly 20 Hz diffusion-policy loop. It replays every unseen RGB
frame through the classifier, uses the same startup-bias-corrected native F/T
stream as the policy, and never supplies F/T newer than each RGB timestamp.
The policy waits for the context record whose timestamp **and RGB bytes** match
its own final camera image, so it cannot accidentally receive a future-frame
context value. It primes a 0.5-second live-camera preroll because the first
latency-compensated policy RGB may slightly predate `eval_t_start`. If a worker
poll misses an actual timestamp gap, it recovers up to the latest 120 retained
frames; if a requested exact policy anchor is absent, it performs one bounded
32-frame anchor recovery rather than substituting a newer context value. Each
policy episode resets classifier history; its first 61 RGB frames have
`warmed_up=0` until the full temporal context exists.

Context runs add `valve_context.csv` to the evaluation log and show current
phase/reason/probabilities in `comparison.mp4`. The 10 values are phase
probabilities (5), error-reason probabilities (4), then the warm-up flag (1);
they are passed to the context policy without external normalization.
`context_worker_summary.json` records the processed-frame count, mean/max
camera-frame period, worker polls, timestamp-gap recovery polls,
exact-anchor recovery polls, and any worker error. Check
this file after each run: software can recover buffered frames, but cannot
invent frames the GoPro/Elgato/UVC source never delivered. The camera should
therefore be configured and verified at 60 fps where the training setup used
60 fps.

### Context input capture for retraining/debugging

Context runs now save the complete frozen-classifier input contract in
`eval_logs/ep*/context_inputs/` by default (`RG2_SAVE_CONTEXT_INPUTS=1`):

- `images/frame_*.png`: lossless 224×224 RGB frames, exactly the image passed
  to the frozen classifier (not a display screenshot);
- `context_frames.csv`: image filename/timestamp, TCP xyz, raw TCP axis-angle
  rotation, gripper width, and the classifier phase/reason output for every
  classifier frame;
- `context_wrenches.csv`: each unique causal startup-bias-corrected native
  12-D F/T sample at its source timestamp, in N/Nm. `context_frames.csv` links
  each RGB frame to its latest preceding wrench timestamp;
- `classifier_windows/window_*.npz`: one archive for every classifier
  prediction. This is authoritative for the 16-step temporal input actually
  selected at that instant: RGB timestamps and source image-file references,
  TCP-derived `lowdim` `[x,y,z,rot6d,width]`, physical F/T histories, the
  exact model-scaled F/T histories, and the causal F/T masks;
- `classifier_windows/index.csv`: classifier timestamp, archive name, temporal
  step count, and F/T history length.

The classifier directly consumes no physical IMU feature, so neither a
`context_imu.csv` file nor a fabricated zero IMU vector is emitted. Its direct
low-dimensional input is TCP position, rotation-6D, and gripper width; the raw
axis-angle needed to reproduce rotation-6D is retained in `context_frames.csv`.
The RGB PNG is lossless, so the exact pre-normalizer classifier tensor is
reconstructed without approximation as `moveaxis(rgb, -1, 1).astype(float32) /
255.0`; its window ordering comes from `rgb_image_file`. The ImageNet
mean/std constants are in `capture_manifest.json`. This avoids copying the
same 16 RGB tensors into every camera-rate archive and prevents logging from
stalling real-time context inference.

Use `RG2_SAVE_CONTEXT_INPUTS=0` only to avoid the additional lossless-image
disk usage. The capture is intentionally camera-rate/force-rate rather than
only one row per slow diffusion-policy replan.

### Exact diffusion-policy input capture

For a second algorithm or offline reproduction, the evaluator also writes
`eval_logs/ep*/policy_inputs/` by default (`RG2_SAVE_POLICY_INPUTS=1`). This is
separate from `context_inputs/`: it is one sample per diffusion-policy call and
captures the exact NumPy observation dictionary after live preprocessing and
immediately before `policy.predict_action` (before its internal normalizer).

- `samples/sample_*.npz`: authoritative arrays passed to the policy: two
  `camera0_rgb` frames in `TCHW` float32, relative TCP position, rotation-6D,
  causal left/right F/T histories, and `valve_context` for the context
  checkpoint;
- `images/`: lossless PNG views of those two exact policy RGB frames;
- `ft_history.csv`: the same 32-step-per-finger policy F/T tensors with their
  causal source timestamps;
- `index.csv`: policy iteration, RGB anchor timestamp, archive, and image
  filenames.

Use `RG2_SAVE_POLICY_INPUTS=0` only when this per-inference capture is not
needed. The archive contains every and only `shape_meta.obs` input key; this
checkpoint defines no IMU key, so no IMU placeholder file is emitted.

### Optional TCP momentum experiment

The default is disabled (`RG2_MOTION_MOMENTUM_PREVIOUS_WEIGHT=0`). To blend
the last actually submitted TCP increment with the new policy increment at the
requested 1:2 previous:current ratio, set:

```bash
RG2_MOTION_MOMENTUM_PREVIOUS_WEIGHT=0.3333333333
```

Only TCP position and orientation are blended; F/T width feedback remains
unchanged. The state is updated only after waypoint submission, persists
through F/T contact and watchdog holds, and the normal position/rotation/
gripper safety gate still validates the blended waypoint afterward.

## Python dependency

Install the additional pinned dependency in the full UMI eval environment:

```bash
python -m pip install -r requirements_rg2ft.txt
```

`rg2ft_protocol.py` supports pymodbus `unit`, `slave`, and current
`device_id` APIs.  The pinned/tested API is pymodbus 3.14.0.

## Safe first run

Set the checkpoint and Python environment, then run the launcher.  Its default
is plan-only: it connects to RG2-FT and reads width/F/T, but the controller does
not write a motion command until an explicit width waypoint is scheduled.

```bash
PYTHON_BIN=/path/to/umi/bin/python \
RG2_CHECKPOINT=/path/to/rg2ft_latest.ckpt \
./deploy_real_indy_rg2.sh
```

After validating printed actions, addresses, current width and emergency-stop
access, explicitly enable motion:

```bash
PYTHON_BIN=/path/to/umi/bin/python \
RG2_CHECKPOINT=/path/to/rg2ft_latest.ckpt \
RG2_ENABLE_MOTION=1 \
./deploy_real_indy_rg2.sh
```

## Safety behaviour

- Connecting alone does not open or close the gripper.
- Commands are clamped to the RG2-FT physical range.
- Intermediate/closed widths are refreshed to hold against spring opening.
- A fully-open target is released after arrival to avoid re-grip oscillation.
- Modbus response errors are detected; the controller reconnects after a
  runtime communication failure.
- Signed slightly-negative fully-closed readings are clipped to 0 m before
  entering policy observations.

Hardware-free tests:

```bash
python -m unittest tests.test_rg2ft_protocol tests.test_rg2ft_controller -v
```
