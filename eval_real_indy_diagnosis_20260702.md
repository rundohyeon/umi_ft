# eval_real_indy Session Log — 2026-07-02

Continues: `eval_real_indy_diagnosis_20260626.md` (root-cause hypothesis: live Indy TCP
rotation frame does not match the UMI/SLAM training frame).

Main script: `eval_real_indy.py`
Robot config: `example/eval_robots_config_indy.yaml`
Controller: `umi/real_world/indy_interpolation_controller.py`
Checkpoint: `scripts/waypoints/epoch100_axis.ckpt`
Training data (the real one — see correction #1 below): `data/axix_data_zarrfile/dataset_axis_newP.zarr.zip`

## 0. Correction: which zarr is the real training data

The zarr at the repo root (`/home/idim/dkim/data/`, `meta/`) is **not** the film
task. It's an unrelated UMI example dataset (living-room / cups scene). The
real film/roller training data is `dataset.zarr.zip` and `dataset_axis_newP.zarr.zip`
under `data/axix_data_zarrfile/`. Both encode the same episodes; `newP`
mirrors the y/z sign of `robot0_eef_rot_axis_angle` relative to the plain
`dataset.zarr.zip`. **The deployed checkpoint (`epoch100_axis.ckpt`) is trained
on the `newP` convention** — verified by matching its action normalizer stats
against each dataset's relative-action stats.

## 1. Confirmed the relative-action decode mechanics

Before touching anything, verified from `pose_repr_util.py` +
`real_inference_util.get_real_umi_action` + live log residuals (ep10, 58
steps) that the executed absolute target is:

```
p_target = R_current_TCP_rotation @ t_rel_from_policy + p_current
```

not a world-frame delta. Residual of this hypothesis against logged
`converted - obs` was 0.000000 m (exact); the alternative (`p_target =
t_rel + p_current`) had a mean residual of 0.37 mm — refuted. This means a
wrong *live* TCP rotation reading corrupts the direction of every executed
action even when the policy's local output is correct.

## 2. First (incomplete) rotation fix — gravity-only, 2 of 3 DOF

Compared live `robot0_eef_rot_axis_angle` against the newP training
distribution:

- Live start rotvecs were 70–90° geodesic from every training frame.
- Gravity check (tool-frame "up" component, invariant to any world-yaw
  difference between the SLAM map and the robot base): training demos ≈
  [-0.01, -0.50, -0.86] (gripper tilted down), live ≈ [-0.15, -0.81, +0.57]
  (reads as tilted up) — a ~95° physical inconsistency confirmed against
  video (gripper is visibly looking down at the rollers).
- Solved the minimal rotation explaining the gravity mismatch: **Rx(+90°)**
  (residual 9.7°).

Implemented a `tool_rot_offset_deg` mechanism in
`indy_interpolation_controller.py`:
- `_robot_to_umi_pose` (feedback): strips the offset, `R_umi = R_read · R_offset⁻¹`
- `_umi_to_robot_pose` (command): re-adds it, `R_robot_cmd = R_umi_cmd · R_offset`
- Exact inverses (roundtrip error 1e-16), constructed via
  `scipy.Rotation.from_euler("xyz", deg, degrees=True)`.
- Wired through `UmiEnv.__init__` (`indy_tool_rot_offset_deg` kwarg) and
  `eval_real_indy.py` (`rc.get("indy_tool_rot_offset_deg", ...)`).
- Config set to `[90, 0, 0]`.

**Live result:** gravity tilt error dropped from 94–96° to 6–20° (correct),
and executed-direction-vs-training-direction improved from ~80° to ~18–30°.
But the robot still did not approach the film — it drifted at roughly demo
speed but in a direction skewed too far off, or crept at near-zero speed on
an overlay-aligned start (see §4). **Gravity alone only constrains 2 of 3
rotational DOF** — the third (rotation about the gravity axis, i.e. yaw) was
silently wrong and had been mis-classified as "harmless world-yaw offset"
(cancels under `action_pose_repr: relative`) in the 6/26 diagnosis.

## 3. Unrelated bugs fixed along the way

These surfaced as separate crashes during live testing, independent of the
rotation issue:

- **ArUco API removed in OpenCV ≥ 4.7**: `cv2.aruco.detectMarkers()`
  (functional API) no longer exists; must use
  `cv2.aruco.ArucoDetector(dict, params).detectMarkers(img)`. Added a
  `_aruco_detect_markers()` compat helper in `umi/common/cv_util.py` (and the
  duplicate top-level `cv_util.py`), used by `umi_env.py`'s
  `apply_training_image_preprocess`. Verified functionally on this host's
  OpenCV 4.7.0 (generate + detect roundtrip).
- **Elgato HD60X camera capture hangs forever** (`UmiEnv.start_wait` timeout,
  process alive but never signals ready): the capture was never given an
  explicit FOURCC, so V4L2 defaults to uncompressed YUYV, which exceeds
  practical USB throughput at 1080p60 — `cap.grab()` succeeds but
  `cap.retrieve()` never returns real frames, and the retrieve-failure branch
  has no sleep, so the subprocess spins forever without crashing. Added
  `cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))` before
  resolution/fps in `uvc_camera.py`. (A commented-out line hinting at this
  exact fix already existed in the unused debug script `scripts/uvc_camera.py`.)
- **`--match_dataset` overlay silently loaded 0 frames** when pointed at a
  bare `dataset*.zarr.zip` (the code only looked for a session `videos/`
  folder). Added a fallback in `eval_real_indy.py` that reads each episode's
  first frame directly from the zarr's `camera0_rgb` array
  (imagecodecs-registered for JPEG-XL) when no `videos/` folder exists.
  Verified: loads all 79 episode first-frames from `dataset_axis_newP.zarr.zip`.
- **`e`/`w` match-episode keys were no-ops** unless `--match_episode` (`-me`)
  was explicitly passed, and even then clamped against the *live* recording
  buffer's episode count instead of the match dataset's. Fixed to
  auto-initialize from the loaded frame map and clamp against it.

## 4. Second calibration — full 3-DOF tool offset

User's earlier same-day test (`replay_axix_zarr_episode.py`, position-only
replay with `xyz_map=identity` and rotation frozen) **physically reached the
film** by executing zarr *world* positions directly as robot-base positions
with no rotation remapping at all. This is a strong physical constraint:
**the zarr/SLAM world frame is identical to the robot base frame — there is
no world-frame yaw offset.** That invalidates the earlier assumption that
residual disagreement beyond the gravity-constrained 2 DOF was "harmless
world yaw" — it was real, uncompensated tool-frame error.

With world-side rotation pinned to identity, the tool offset became fully
solvable from a single physically-aligned pose. Used the `--match_dataset`
overlay (§3) to align a live start pose to a demo start frame by eye
(ep15, `data/eval_run/eval_logs/ep15_20260702_074037/`), then solved:

```
E_total = R_demo_start⁻¹ · R_raw_zxz_reading
```

Result: `|E_total| ≈ 112°`, `euler_xyz_deg ≈ [111.93, 29.00, 71.39]`.
Cross-checked against two earlier (non-aligned, closest-start-guess) logs
ep13/ep14 — independent estimates agreed within 15–28°, consistent with a
single fixed offset rather than noise.

Config updated:
```yaml
"indy_tool_rot_offset_deg": [111.93, 29.00, 71.39],
```

Offline verification (re-simulating old logs' raw readings through the new
offset): corrected rotvecs land at [-2.46 to -2.55, ~0, ±0.15], vs. training
demo starts clustered at [-2.8 to -3.0, 0, ±0.1] — 0–17° residual across the
three logs (vs. 55–90° before this calibration, 70–96° before any fix at all).

**Status: not yet validated live with this second calibration** (as of
writing). This was calibrated from a single aligned pose plus two
lower-confidence cross-checks — if the resulting motion is toward the film
but skewed by ~10–15°, that's the expected residual from single-pose
calibration and can be refined with another aligned-overlay log.

## 5. New debug tool: zarr episode visualizer

`render_pose_traj_video.py` (new file). Renders an mp4 for one training
episode: camera0_rgb on the left, three panels (XY/YZ/XZ) on the right, each
showing the position trajectory plus the live orientation drawn as a local
xyz axis frame (red/green/blue) at the current pose. Useful for building
physical intuition about what a given rotvec "looks like" without staring at
numbers.

```bash
python render_pose_traj_video.py -d data/axix_data_zarrfile/dataset_axis_newP.zarr.zip -e 0
```

## 6. Files touched today

- `umi/real_world/indy_interpolation_controller.py` — `tool_rot_offset_deg` mechanism
- `umi/real_world/umi_env.py` — offset kwarg plumbing + ArUco compat fix
- `eval_real_indy.py` — offset plumbing + match_dataset overlay/e-w fixes
- `example/eval_robots_config_indy.yaml` — `indy_tool_rot_offset_deg: [111.93, 29.00, 71.39]`
- `umi/common/cv_util.py`, `cv_util.py` — ArUco `detectMarkers` compat helper
- `umi/real_world/uvc_camera.py` — MJPG FOURCC for Elgato capture
- `render_pose_traj_video.py` — new visualization tool

## 7. Next steps

1. Re-test with the `[111.93, 29.00, 71.39]` offset using the overlay
   (`-m data/axix_data_zarrfile/dataset_axis_newP.zarr.zip --allow_rotation`).
   Confirm motion is toward the film. If skewed, capture one more
   overlay-aligned start and re-solve `E_total`.
2. The gripper section of `eval_robots_config_indy.yaml` is still fully
   commented out — no grasp will execute even once approach is correct.
   Needs to be re-enabled once approach direction is confirmed good.
3. Consider averaging `E_total` over several aligned poses (rather than one)
   once approach direction is roughly right, to tighten the residual below
   the current single-pose calibration noise floor.
