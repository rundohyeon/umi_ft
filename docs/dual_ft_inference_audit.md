# Dual-F/T evaluation and deployment audit

## Status

Training, checkpoint inspection, hardware-free offline evaluation, and the
guarded Indy/RG2 live loop are implemented. Controller-connected `--dry-run`
suppresses waypoint submission but is not passive: it still opens the robot,
camera, and RG2 interfaces.

## Checkpoint contract

`diffusion_policy/common/dual_ft_contract.py` validates a checkpoint without
importing AV, OpenCV, UmiEnv, or robot adapters. It requires:

| Item | Required contract |
|---|---|
| Observation | RGB `[2,3,224,224]`, pose `[2,3]+[2,6]`, left/right F/T `[32,6]` |
| Condition | fusion 768 + pose-only proprioception 18 = 786 |
| Fusion | four tokens, 8 heads, FF 2048, one layer, dropout 0, learnable positions, flatten/project `3072→768` |
| Diffusion time | embedding 32 |
| Action | `[16,11]`: relative pose9, absolute width1, signed measured force1 |
| Offline pose source | stock ZIP position + rotation axis-angle, no quaternion conversion |
| Temporal marker | architecture 2; left/right full-window marker 1 |
| F/T semantics | native frames, identity axes/signs, six independent channels, episode-bias only |
| Normalizer | checkpoint-owned entries for RGB, pose, both F/T streams, and action |

Both `model` and `ema_model` states, when present, are checked. If
`training.use_ema=true`, missing EMA is corruption rather than a reason to
fall back to model weights.

## Dataset contract

The effective 260827 training input is the stock UMI ZIP plus its force
sidecar. The only F/T value key used for training is:

```text
left  = wrench_12d[:, :6]
right = wrench_12d[:, 6:]
```

`wrench_12d` already has the recorded per-episode standing bias removed in the
native sensor frames. `wrench_force_tcp_6d` is never loaded.

No coordinate transform, axis permutation, sign flip, TCP rotation, torque
translation, or left/right averaging is applied. Channel order is
`Fx,Fy,Fz,Tx,Ty,Tz`, with N and N·m units.

Offline robot pose is read directly from the stock ZIP's position and
axis-angle arrays, so there is no quaternion round trip. The `xyzw` checkpoint
marker remains the required convention for live/legacy 7-D pose sources.

The output force label is not a direct gripper-force command:

```text
derived grasp-force label =
    0.5 * ((right native Fz - right episode-bias Fz)
         - (left native Fz  - left episode-bias Fz))
```

It is a signed measurement linearly aligned from the 100 Hz stream to each RGB
timestamp. The dataset constructor derives the label from the same native
sidecar stream and verifies the relation over all 214 episodes.

At deployment it is interpreted as a reference for bounded proportional width
feedback. The current bias-corrected measurement is compared with the predicted
force; the resulting width correction uses `0.0001 m/N`, a `0.5 N` deadband,
and a maximum magnitude of `1 mm`. Target force is clipped to `[0,12] N` and
width to `[0,0.1] m`. No value is written to the RG2 force register.

F/T history uses causal zero-order hold and repeat-first dataset padding. This
is distinct from the encoder's no-padding convolution contract:

```text
dataset: 32 samples over about 0.31 s, repeat-first before stream start
encoder: 32 → 16 → 8 → 4 → 2 → 1, kernel 2 / stride 2, no conv padding
```

## Offline evaluator

`eval_dual_ft_offline.py` directly instantiates one policy and loads only the
selected model state. It does not create a workspace, optimizer, robot, camera,
or Modbus connection. Image augmentation is disabled before policy
construction and the checkpoint's training normalizer is restored strictly.

It reports:

- diffusion loss with deterministic per-batch seeds;
- raw MAE/MSE/RMSE/max error overall and separately for position, rotation-6D,
  gripper width, and measured grasp force;
- rotation geodesic error in radians and degrees;
- checkpoint-normalized action error;
- inference latency, F/T causal age, action start/end timestamp offsets;
- checkpoint SHA-256 and immutable-file identity;
- selected validation indices, episodes, split size, sync verdict/offset
  provenance, bias hash/statistics, schema, rates, causal drops, and action
  semantics.

By default it evaluates a seeded 256-sample subset. `--max-samples 0` uses
the complete validation split (5,778 samples across 11 episodes).
`--full-dataset-hash` additionally reads and hashes every RGB chunk; otherwise
all effective non-RGB arrays are still content-hashed.

The evaluator rejects NaN/Inf, bad shapes/keys, future observations, stale F/T,
timestamp inconsistencies, a changed live `latest.ckpt`, output paths that
would overwrite checkpoint/dataset data, and incomplete or zero-scale
normalizers.

## Current time-alignment evidence

The transferred dataset contains 214 episodes and 200,865 F/T samples. Stored
sync verdicts are `pass=65`, `check=149`, `fail=0`. After applying the
stored offsets, latest-causal F/T age is approximately median 5.0 ms, p99
9.9 ms, maximum 11.0 ms; the first RGB anchor of each episode has no prior F/T
and is dropped.

These ages prove the implemented causal lookup on the already-aligned clocks.
They do not independently prove that all 149 `check` offset estimates are
physically exact. The evaluator records this caveat and the original sync
metadata.

## Live implementation and remaining commissioning boundary

The live loop now restores model/EMA weights and the checkpoint normalizer,
assembles independent causal `[32,6]` left/right histories, estimates a fresh
native-frame startup bias from unique stationary samples, and connects the
11th action channel to the bounded width controller. It never sends that
channel to the RG2 force register.

Training images were generated as: detected-tag inpaint, predefined gripper
mask, `finger=False`, resize/crop ratio 1.0, and no fisheye rectification. The
live dual-F/T path rejects incompatible preprocessing flags and provides an
exact policy-input/first-training-frame overlap view.

This was also checked against episode 0 numerically: replaying the generator on
raw video frame 2 gives mean absolute pixel error 0.627 after JPEG-XL storage,
versus 19.70 with no mask and 17.08 with the additional finger mask. The stored
black-pixel fraction is 0.1208 (generator replay 0.1262); adding the finger mask
raises it to 0.2516. The completed pipeline log and its default `out_fov=None`
path confirm that distortion correction was not used.

The model-side dataset transform is required to be identity. Neuromeka's
millimetre/Euler protocol conversion and the configured flange-to-fingertip TCP
remain necessary hardware-interface conversions; they are not transformations
of the learned dataset or native F/T axes.

Deadline-aware action scheduling keeps force references paired with the exact
surviving pose/width rows. Guards reject F/T overload, per-waypoint translation,
rotation and width jumps, workspace violations, and non-finite values. Safety
stops clear pending waypoints and hold the measured TCP, while a controller
watchdog disarms after the final scheduled target.

The numeric limits in `example/eval_robots_config_indy_rg2.yaml` are conservative
software commissioning limits, not a certified safety system. Validate sensor
signs, exclusion geometry, TCP offset, speed/delta limits, physical sync, and
the robot emergency stop with one planning-only cycle and one low-speed live
cycle before an unrestricted run.
