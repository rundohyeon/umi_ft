# eval_real_indy Policy Diagnosis

Date: 2026-06-26  
Main script: `scripts/indy_umi/eval_real_indy.py`  
Policy checkpoint: `scripts/indy_umi/scripts/waypoints/0624_latest.ckpt`  
Training dataset: `scripts/indy_umi/dataset.zarr.zip`  
Recent live log used heavily: `scripts/indy_umi/data/eval_run/eval_logs/ep37_20260625_141629/log.csv`

## 1. Goal

The real task is to move the Indy RP2 robot toward the film end protruding from the roll, grasp it, and continue the learned rolling/manipulation behavior.

The training demonstrations were collected around this film-end grasping task. The live evaluation used `eval_real_indy.py` with the `0624_latest.ckpt` checkpoint.

## 2. Main Symptom

During live policy execution, the robot did not approach the film target. The observed failures were:

- The robot moved away from the expected film-end direction.
- The motion was not explained by only one wrong axis.
- `x` sign testing did not solve the issue.
- `y` motion became excessive even from physically similar starting poses.
- `z` motion and rotation also looked wrong.
- When the film was already grasped and the policy was run, the behavior was still not consistent with the training demonstrations.

This made the policy look random or broken at first, but later tests showed the policy itself is not the primary failure.

## 3. Important Files And Current Assumptions

### Evaluation script

`scripts/indy_umi/eval_real_indy.py`

Important behavior:

- Loads the checkpoint.
- Gets live camera and robot observation from `UmiEnv`.
- Converts live observation with `get_real_umi_obs_dict`.
- Runs `policy.predict_action`.
- Converts predicted relative action to absolute TCP waypoints with `get_real_umi_action`.
- Sends waypoints to Indy.

### Robot config

`scripts/indy_umi/example/eval_robots_config_indy.yaml`

Current Indy rotation config:

```yaml
indy_task_rot_is_euler: true
indy_task_rot_euler_seq: "zxz"
indy_task_rot_euler_in_degrees: true
indy_task_rot_euler_extrinsic: false
indy_task_frame_xyz_signs: [1, 1, 1]
```

Meaning:

```python
Rotation.from_euler("zxz", UVW_deg, degrees=True).as_rotvec()
```

So the current code assumes Indy `XYZUVW` orientation is intrinsic `zxz` Euler angles in degrees.

### Camera calibration

Training SLAM used:

`scripts/indy_umi/assets/gopro13_ultrawide_2000x1500.yaml`

We added support so `eval_real_indy.py` and `eval_real_indy_debug.py` can use this ORB-SLAM style fisheye calibration by default.

This improved camera consistency, but it did not solve the live robot motion failure.

## 4. Checkpoint And Dataset Contract

The checkpoint config says:

```yaml
obs_pose_repr: relative
action_pose_repr: relative
```

Action shape:

```yaml
action:
  shape: [10]
  horizon: 16
  down_sample_steps: 3
  rotation_rep: rotation_6d
```

Observation includes:

```yaml
camera0_rgb
robot0_eef_pos
robot0_eef_rot_axis_angle
robot0_gripper_width
robot0_eef_rot_axis_angle_wrt_start
```

This is critical:

The policy does not output absolute world poses directly. It outputs a relative pose representation. During eval, that relative action is decoded using the current live TCP pose as the base.

Code path:

- Training relative transform:

```python
out = np.linalg.inv(base_pose_mat) @ pose_mat
```

- Eval relative decode:

```python
out = base_pose_mat @ pose_mat
```

This means if the live TCP rotation is wrong, the same policy output will be decoded into the wrong world direction.

Relevant files:

- `scripts/indy_umi/diffusion_policy/common/pose_repr_util.py`
- `scripts/indy_umi/umi/real_world/real_inference_util.py`

## 5. Hypotheses Tested

| Hypothesis | Verification Method | Result |
|---|---|---|
| Single `x` sign issue | Changed/tested `x -> +x` behavior | Not sufficient. `y`, `z`, and rotation were still wrong |
| Robot xyz axes are globally wrong | Manual axis tests and live displacement logs | Not explained by simple axis sign flip |
| Camera feature problem | Applied GoPro13 fisheye YAML used during train SLAM | Camera input became more consistent, but motion failure remained |
| Policy/checkpoint is corrupted | Loaded `0624_latest.ckpt` and `dataset.zarr.zip`; ran offline zarr inference | Policy works very well on zarr observations |
| zarr and ckpt mismatch | Compared checkpoint normalizer/action contract with zarr | No evidence of mismatch |
| Bad visual recognition of film | Fed zarr observations directly into policy | Policy predicts demo-like actions, so visual feature is not the main issue |
| `wrt_start` start-pose channel is the main issue | Compared offline demo-start vs current-start modes | In tested zarr samples, difference was tiny. Could be secondary live OOD issue, not primary |
| Live robot rotation convention/TCP frame mismatch | Compared live rotvec with zarr rotvec and decoded same relative action through both frames | Strongly confirmed |

## 6. Offline zarr Policy Test

We tested the policy without the robot by feeding observations from:

`scripts/indy_umi/dataset.zarr.zip`

into:

`scripts/indy_umi/scripts/waypoints/0624_latest.ckpt`

The policy was evaluated on zarr samples around the grasp/roll phase.

Result:

- Position error against zarr future demo pose: about `0.09 cm` to `0.23 cm`.
- Rotation error: about `0.3 deg` to `2 deg`.

Interpretation:

The policy is not random. The checkpoint and zarr dataset are internally consistent. On training-style input, the policy predicts the learned trajectory very accurately.

## 7. Live ep37 Analysis

Log:

`scripts/indy_umi/data/eval_run/eval_logs/ep37_20260625_141629/log.csv`

Live start TCP rotation:

```text
[-0.905, 1.081, -0.548]
```

Typical zarr rotations at visually/task-similar phases:

```text
[2.491, 0.061, 0.179]
[2.529, 0.150, 0.082]
[2.626, 0.011, 0.035]
```

Rotation difference between live ep37 start and similar zarr phases:

```text
about 155 to 162 degrees
```

Even searching the entire zarr dataset, the closest zarr rotation to live ep37 was still about:

```text
112.9 degrees away
```

This means the live `robot0_eef_rot_axis_angle` is far outside the training rotation distribution.

## 8. Why y Motion Becomes Wrong

The strongest confirmation came from decoding the same relative action under two different TCP bases.

Example zarr relative action decoded with correct zarr TCP base:

```text
[-0.25, -0.30, -0.16] cm
```

The same relative action decoded with live ep37 TCP rotation:

```text
[+0.13, +0.40, +0.08] cm
```

So the policy may output a correct local action, but because live TCP rotation is wrong, the world-space result gets rotated into the wrong direction. This explains the observed excessive or wrong-sign `y` movement.

## 9. Current Root Cause

The most likely root cause is:

```text
Indy live TCP orientation is not represented in the same frame/convention as the zarr training TCP orientation.
```

More specifically:

- Indy RP2 returns task pose as `XYZUVW`.
- `UVW` is not a SciPy rotvec.
- Current code interprets `UVW` as intrinsic `zxz` Euler degrees.
- The resulting live rotvec does not match the zarr/SLAM training TCP frame.
- Because the checkpoint uses `action_pose_repr: relative`, wrong live rotation corrupts action decoding.

This is not primarily:

- Film feature shortage.
- Policy collapse.
- zarr corruption.
- Simple x/y/z sign mistake.
- Camera calibration alone.

## 10. Why Simple Axis Sign Flip Is Not Enough

The live-to-zarr rotation difference is about `155 to 162 degrees`, and the relative rotation is a full 3D orientation mismatch, not a simple one-axis sign flip.

Therefore changing only:

```text
x -> -x
y -> -y
z -> -z
```

is unlikely to solve the problem by itself.

The rotation convention or fixed TCP/tool-frame orientation offset must be resolved.

## 11. Next Required Verification

We need one live run that prints Indy raw task pose.

The controller already has debug prints:

```text
[Indy] raw get_control_data()['p']=...
[Indy] interpreted TCP (m, rotvec rad)=...
[Indy] cmd=... euler_scipy_seq=... rot_is_euler=...
```

These are printed by:

`scripts/indy_umi/umi/real_world/indy_interpolation_controller.py`

The important value is raw `p[3:6]`.

Once raw `UVW` is known, test all candidates:

```text
xyz, xzy, yxz, yzx, zxy, zyx,
xyx, xzx, yxy, yzy, zxz, zyz
```

with both:

```text
intrinsic and extrinsic
```

The correct interpretation should produce a live rotvec close to zarr task-phase rotations, roughly:

```text
[2.5 ~ 2.8, 0.0 ~ 0.3, 0.0 ~ 0.2]
```

## 12. Suggested Safe Debug Command

Run from:

```bash
cd /home/idim/dkim/umi/scripts/indy_umi
```

Use plan-only first:

```bash
/home/idim/miniforge3/envs/umi/bin/python eval_real_indy.py \
  -i scripts/waypoints/0624_latest.ckpt \
  -o data/eval_run \
  -rc example/eval_robots_config_indy.yaml \
  --plan_only \
  --print_motion_debug \
  --print_policy_output \
  --print_model_input \
  --pose_eval_audit \
  --dataset_zarr dataset.zarr.zip
```

Goal of this run:

1. Do not move the robot.
2. Capture the raw Indy `p` print.
3. Capture interpreted TCP rotvec.
4. Compare the interpreted live rotvec to zarr rotation distribution.

## 13. If Raw `p` Is Captured

After capturing:

```text
[Indy] raw get_control_data()['p']=[X, Y, Z, U, V, W]
```

the next step is to compute all possible Euler interpretations and pick the one closest to zarr rotations at the same physical pose.

Candidate fix areas:

1. `indy_task_rot_euler_seq`
2. `indy_task_rot_euler_extrinsic`
3. A fixed TCP/tool-frame rotation offset
4. Possibly separate observation-frame and command-frame conversion if Indy command convention differs from feedback convention

Do not run full policy execution until this rotation match is solved.

## 14. Current Practical Conclusion

The current policy failure is best explained by this chain:

```text
Indy raw XYZUVW
 -> currently interpreted as intrinsic zxz Euler deg
 -> live rotvec becomes far from zarr training rotvec
 -> policy receives out-of-distribution proprio rotation
 -> relative action is decoded using wrong current TCP rotation
 -> world-space y/z/rotation movement becomes wrong
 -> robot does not approach film target
```

So the next meaningful work is not more policy training yet. The next work is to align the live Indy TCP rotation convention/frame with the zarr/SLAM training TCP frame.

