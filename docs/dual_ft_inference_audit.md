# Dual-F/T Indy inference audit

## Status: PARTIALLY_IMPLEMENTED → deployment entry point completed

`eval_real_indy_rg2.py` already contained the working Indy/RG2 camera, pose,
10-D coordinate-action, frame-conversion, keyboard, and timestamped waypoint
path.  It also restored the dual-F/T checkpoint and used causal history in
`UmiEnv.get_obs()`.  It did not expose a checkpoint-only inspection command or
a dedicated dry-run CLI, and its F/T transport represented the atomic RG2-FT
reply as one combined buffer.  This change preserves the motion path and adds:

- `eval_real_indy_rg2_dual_ft.py`: FT-only checkpoint inspection and a safe
  dry-run wrapper around the reference evaluator;
- distinct logical left/right F/T arrays in `RG2FTController` and independent
  causal lookup in `causal_ft_history_from_streams()`;
- fail-closed finite/empty/non-monotonic/stale F/T validation;
- loading-time validation of both FT encoders and both checkpoint normalizers;
- actual-ZIP causal-history parity and unit tests.

The wrapper never begins real policy control automatically: normal operation
still uses the reference UI's explicit start action. `--dry-run` maps to the
reference `--plan_only` route and never calls `UmiEnv.exec_actions`.

## Source of truth and training call graph

| Contract | Source | Implementation |
|---|---|---|
| Zarr / RGB decode | `diffusion_policy/dataset/umi_dual_ft_dataset.py:_sample_arrays` | `rgb_0` is decoded HWC `uint8`, moved to `[T,3,224,224]`, converted to `float32 / 255`. |
| RGB live preprocessing | `umi/real_world/umi_env.py:get_obs` → `umi.real_world.real_inference_util:get_real_umi_obs_dict` | Existing single-GoPro resize/mask/inpainting path is retained; eval turns stochastic image transforms into `Identity`. |
| Pose / relative coordinates | dataset `_sample_arrays` and `get_real_umi_obs_dict` | Existing relative pose representation, episode-start rotation feature, and action decode are retained. |
| F/T sampling | dataset `_ft_history` | target grid is `anchor - arange(31..0) / actual_ft_hz`; `searchsorted(..., side="right") - 1`; pre-stream slots repeat the first sample. |
| Live F/T sampling | `umi.real_world.rg2ft_obs:causal_ft_history_from_streams` | Same grid and causal zero-order hold separately for left/right. |
| Normalization | `DiffusionUnetTimmPolicy.predict_action` | **B**: `policy.normalizer.normalize(obs_dict)` exactly once internally; the eval passes raw RGB/pose/F/T tensors and never normalizes them itself. Action is unnormalized internally before return. |
| Condition | `DualFTObsEncoder.forward` | encoder fuses CLS + two RGB tokens + left FT token + right FT token, then appends 32-D low-dimensional features: `[B,800]`. |
| Action | `DiffusionUnetTimmPolicy.predict_action` | full unnormalized `[B,16,10]` is returned as `action_pred`. |

The current checkpoint stores both independently parameterized
`left_ft_encoder` / `right_ft_encoder` plus a distinct `LinearNormalizer`
entry for each key. `inspect_dual_ft_checkpoint_payload()` rejects a checkpoint
without any of those states; no identity normalizer fallback exists.

## Live sensor contract

| Item | Contract |
|---|---|
| RGB | one GoPro UVC stream from `MultiUvcCamera`; camera timestamp is the policy anchor. |
| Pose | `IndyInterpolationController` `ActualTCPPose`, interpolated to camera-anchor timestamps. |
| Left/right F/T | one atomic `RG2FTController` Modbus/TCP status read. It publishes distinct `gripper_ft_left` and `gripper_ft_right` 6-D arrays; both share one timestamp because the device returns them in one status block. |
| Register parser | `umi/real_world/rg2ft_protocol.py:parse_ft_status`. Register indices are `[2..7,11..16]`, left then right. Forces divide by 10 to N; torques divide by 100 to N·m. |
| Channel order / frame | `Fx,Fy,Fz,Tx,Ty,Tz`, raw native left/right sensor frames. No software permutation, sign flip, filtering, tare/bias subtraction, wrist transform, or controller is added. |
| Clock | camera, robot, RG2-FT receive and command scheduling use the existing `time.time()` wall-clock domain. The controller stores `gripper_timestamp = time.time() - receive_latency`; `UmiEnv` uses camera timestamps as anchors. |

The 0814 audit measured left/right latest causal age p95 `9.504 ms`, max
`11.182 ms` at 100.000095 Hz. `task.ft_max_age_sec=0.012` is therefore a
configurable fail-closed bound derived from that report. A different transport
or a non-identical sensor mounting requires a fresh alignment/mounting audit
before real actuation.

## Model and execution contract

```text
camera0_rgb                       [1, 2, 3, 224, 224] float32, raw [0,1]
robot pose/rotation/gripper       [1, 2, D] float32, raw representation
robot0_ft_left / robot0_ft_right  [1, 32, 6] float32, raw N/Nm
encoder condition                 [1, 800]
policy action_pred                [1, 16, 10], unnormalized coordinate action
```

Prediction horizon is never shortened. `eval_real_indy_rg2.py` selects rows
starting at index 0, then removes only rows whose *scheduled timestamps* have
already gone stale. Thus the normal two-action execution slice is indices
`[0, 1]`; it is not an arbitrary post-model truncation and is preserved for
the 1/2/4/8-step setting. The command period is `1 / 19.980011909 =
50.050 ms`; two actions imply a nominal 100.100 ms replanning interval.

On missing stream, invalid shape, NaN/Inf, no causal sample, stale F/T, bad
checkpoint normalizer/encoder, or non-finite model output, the code raises
before `exec_actions`. Stale action rows are discarded by the pre-existing
timestamp filter; if none remain it schedules only the next safe row.

## Validation scope

`tests/test_dual_ft_inference.py` uses synthetic streams to verify independent
left/right lookup, causal timestamps, repeat-first padding, stale rejection,
checkpoint normalizer ownership, and RGB-only rejection.

`tests/test_dual_ft_online_offline_parity.py` opens the real 0814 ZIP when it
is installed. It reconstructs a simulated-live THWC `uint8` RGB stream,
absolute pose, gripper width and independent timestamped F/T buffers, then
passes them through `get_real_umi_obs_dict`. Every final model observation key
(RGB, relative pose, gripper, start-relative rotation, left F/T and right F/T)
is compared to the dataset sample with `assert_allclose`; shapes/dtypes and
causal timestamp indexes are also checked. Physical camera/robot parity and a
100-cycle dry-run still need live hardware and were not run by this code task.
