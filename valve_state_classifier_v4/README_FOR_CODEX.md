# Codex handoff — integrate the valve-state classifier

## Objective

Integrate this pretrained, causal valve-state classifier into the robot/model codebase on the receiving computer. Use the exported checkpoint and runtime in this directory. Do not retrain the model, alter its architecture, or recreate preprocessing unless the user explicitly requests that work.

The portable bundle is self-contained for inference and does not require the original training repository, zarr replay, force sidecar, diffusion-policy package, or label-generation scripts.

## Start here

Relevant files, in priority order:

1. `valve_state_classifier.py` — canonical model architecture, checkpoint loader, preprocessing, and stateful streaming runtime.
2. `model/final.pt` — epoch-23 checkpoint. Do not rename state-dict keys or partially load it.
3. `example_streaming.py` — minimal call sequence.
4. `MODEL_CARD.md` — architecture, training snapshot, and limitations.
5. `README_KO.md` — environment setup and offline NPZ interface.
6. `verify_install.py` — checksum, strict load, and forward-pass test.

The intended public integration API is:

```python
from valve_state_classifier import ValveStateRuntime

runtime = ValveStateRuntime("model/final.pt", device="cuda")
runtime.append_wrench(ft_timestamp_s, wrench_12d)
result = runtime.predict(
    timestamp_s=rgb_timestamp_s,
    rgb=rgb_224x224_uint8,
    position_m=robot_position_xyz,
    rotation_axis_angle_rad=robot_rotation_vector,
    gripper_width_m=gripper_width,
)
```

Use `result.phase`, `result.phase_id`, `result.confidence`, and `result.phase_probabilities` as the primary output. Error details are in `result.error_reason` and `result.error_reason_probabilities`.

## Exact input contract

| Input | Shape | Unit / format | Expected rate |
|---|---:|---|---:|
| RGB | `(224,224,3)` | uint8, **RGB order**, not OpenCV BGR | ~59.94/60 Hz |
| TCP position | `(3,)` | metres | aligned to RGB |
| TCP orientation | `(3,)` | axis-angle / rotation vector, radians | aligned to RGB |
| Gripper width | scalar | metres | aligned to RGB |
| Wrench | `(12,)` | `[L_Fx,L_Fy,L_Fz,L_Tx,L_Ty,L_Tz,R_Fx,R_Fy,R_Fz,R_Tx,R_Ty,R_Tz]`; N and Nm | ~100 Hz |
| Timestamp | scalar | seconds on one monotonic clock | strictly increasing |

Before changing adapter code, inspect the receiving codebase and explicitly resolve:

- whether camera frames are RGB or BGR;
- whether image crop/resize already produces the same 224×224 view used during training;
- the exact ordering of the two six-axis wrench sensors;
- whether forces are already normalized (the runtime expects raw N/Nm and performs training-time scaling internally);
- whether robot orientation is an axis-angle vector rather than Euler angles or quaternion;
- whether all sensor timestamps share the same clock;
- where episode/task boundaries are emitted.

Do not silently guess any unresolved coordinate convention. Surface it to the user if it cannot be established from the receiving repository.

## Causal timing invariants

- Enqueue every F/T sample using `append_wrench()` before predicting the RGB observation at that time.
- Never enqueue or select an F/T sample newer than the RGB timestamp being predicted.
- Call `predict()` for every approximately 60 Hz RGB/robot observation, even if downstream control consumes predictions more slowly. The model samples 16 observations at stride 4.
- The RGB/robot temporal span is approximately 1.001 seconds.
- Each sampled RGB observation contains its preceding 50 F/T samples, approximately 0.49 seconds at 100 Hz.
- Call `runtime.reset()` at every episode boundary. Histories must never cross episodes.
- The full RGB context needs 61 raw frames. Before then, prediction is allowed and matches training-time left clamping, but `result.warmed_up` is `False`.
- `ValveStateRuntime` is stateful and is not designed for concurrent calls. Protect it with the receiving system's single-threaded callback or a lock.

## Output definitions

Phase IDs and order are fixed:

```text
0 approach
1 turning
2 endpoint_reached
3 task_complete
4 error
```

Error-reason IDs and order are fixed:

```text
0 none
1 turn_no_contact
2 post_contact_drop
3 other
```

Do not change these indices when connecting to an enum, ROS message, policy observation, logger, or state machine. Add an explicit mapping at the integration boundary if the receiving system uses a different order.

## Recommended integration sequence

1. Run `python verify_install.py --device cpu` or `--device cuda` unchanged.
2. Locate the receiving system's synchronized RGB, TCP pose, gripper, and F/T paths.
3. Add a small adapter around `ValveStateRuntime`; keep the supplied runtime file unchanged initially.
4. Reset the runtime at the existing episode/task-start boundary.
5. Log timestamp, `warmed_up`, phase probabilities, selected phase, and error reason for initial trials.
6. First run in observation-only mode. Do not immediately connect the classifier to motion-stop or other safety-critical actuation.
7. Compare recorded predictions with video and sensor logs, especially approach rotations and no-contact retries.
8. Only after validation, expose the result to the policy/controller in the form requested by the user.

## Acceptance checks for Codex

The integration is complete only when all applicable checks pass:

- `verify_install.py` reports checkpoint hash, strict load, and forward pass as `OK`.
- The adapter receives 224×224 RGB frames in RGB order.
- Raw 12-D N/Nm wrench values are passed without duplicate scaling.
- RGB and wrench timestamps are monotonic and causal.
- `runtime.reset()` is exercised at a real episode boundary.
- `warmed_up` changes from false to true after 61 RGB frames at normal rate.
- All five probabilities are finite and sum to approximately one.
- Inference runs on the intended device without unbounded queues or cross-episode state.
- At least one recorded episode is reviewed against video before control behavior is changed.

Add tests in the receiving repository for its adapter and message mapping. Do not edit the checkpoint or weaken strict state loading to make an incompatible implementation appear to work.

## Known model facts

- Checkpoint schema: `valve_state_classifier_v1`.
- Label provenance: automatic `rule_generated_unreviewed_v4` pseudo-labels.
- Best checkpoint epoch: 23.
- Original environment: Python 3.9.18, PyTorch 2.1.0, torchvision 0.16.0, NumPy 1.24.4.
- Checkpoint SHA-256: `74763bcf640b05c8e9ea25f35021b1c42743ed348ba9c2829338720bc53558b1`.
- Held-out pseudo-label test: accuracy 0.9603, macro-F1 0.9495, error precision/recall 0.9141/0.9141.

These test values measure agreement with episode-disjoint v4 pseudo-labels, not independently human-annotated safety performance.

## Suggested prompt on the receiving computer

After placing this folder inside the destination repository, the user can give Codex this prompt:

```text
Read valve_state_classifier_v4/README_FOR_CODEX.md completely. Inspect this repository's
camera, robot-state, force/torque, timestamp, and episode-boundary interfaces. Integrate
ValveStateRuntime as an observation-only module first, preserving every input contract and
causal timing invariant in the handoff. Add adapter tests and run the supplied verification.
Report any unresolved channel ordering or coordinate convention instead of guessing it.
```
