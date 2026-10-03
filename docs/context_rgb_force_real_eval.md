# RGB + F/T context classifier: real-robot evaluation and handoff

This guide is the handoff for the destination computer's operator and Codex.
The entry point is `eval_real_context_rgb_force.py`. It runs the trained Stage A
classifier on a live camera and RG2-FT. The screen shows the actual model image,
four probabilities, force norms, and timing; JSONL records every prediction or
unavailable-input status. This is an observer, with no robot motion or gripper
commands. Run an existing teleoperation/controller program separately if motion
is needed. The classifier checkpoint alone cannot produce actions.

## 1. Move the code and checkpoint

On the destination computer, from the existing repository checkout:

```bash
git fetch origin
git switch umi_ft_context-3
git pull --ff-only origin umi_ft_context-3
mkdir -p checkpoints/context_rgb_force
```

Copy this file from the training computer using your preferred transfer method:

```text
/home/metafarmers/dkim/umi_ft/outputs/context_rgb_force_4state_ft_features/context_encoder_best.pt
```

Place it at `checkpoints/context_rgb_force/best.pt` in the destination checkout.
The original filename also works; pass its path with `--checkpoint`. Git does
not transfer `.pt` files. Compare `sha256sum` on both computers after copying.
The current best file is from epoch 123 with validation macro F1 approximately
0.84484, schema `context_rgb_force_4state_ft_features_v2`. It includes the whole
visual encoder, trained heads, architecture settings, class order, and F/T
normalization. The training YAML's default epoch count does not describe the
completed run, and changing that YAML does not change this checkpoint.

No training images, force sidecar, label NPZ, Hugging Face cache, Qwen virtualenv,
optimizer configuration, or external pretrained model is needed for inference.
For a new environment, follow the repository's installation instructions and
`conda_environment.yaml`; the existing `umi` environment is preferred. The code
was checked with Python 3.9, PyTorch 2.1.0, timm 0.9.7, and OpenCV 4.7 with ArUco.
For live RG2-FT access also install:

```bash
conda activate umi
python -m pip install -r requirements_rg2ft.txt
python -c 'import torch, timm, cv2; assert hasattr(cv2, "aruco"); print(torch.__version__, timm.__version__, cv2.__version__)'
```

No Indy SDK is imported by this observer. Only load a trusted checkpoint from
your training run: these training artifacts contain Python metadata as well as
weight tensors.

## 2. Check the checkpoint without hardware

From the repository root:

```bash
HF_HUB_OFFLINE=1 python eval_real_context_rgb_force.py \
  --checkpoint checkpoints/context_rgb_force/best.pt --mode inspect

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 python eval_real_context_rgb_force.py \
  --checkpoint checkpoints/context_rgb_force/best.pt --mode self-test --device cpu
```

`inspect` restores the complete model on CPU and prints its contract. `self-test`
also selects a causal synthetic input window and runs a forward pass. Neither
opens a camera nor a Modbus connection. Synthetic class predictions are not an
accuracy test. Checkpoint loading uses `pretrained=False`, so offline restoration
does not download CLIP. Smoke-test training checkpoints, incompatible observers,
wrong class order, and invalid normalization buffers are rejected.

IDs are fixed: **0 approach, 1 turning, 2 recovery, 3 error**. Do not replace
`recovery` with `finish` or use the Qwen labeling definitions/thresholds. This
classifier uses both fingers' full signed force and torque, not just `|F|`.

## 3. Configure the destination sensors

Find the camera device with `ls -l /dev/v4l/by-id/`. Edit a copy of
`example/eval_context_rgb_force.yaml` for the actual machine and pass `--config`.
The shipped configuration uses:

| Setting | Default | Meaning |
|---|---|---|
| camera.device | required | V4L2 path, also settable with `--camera` |
| camera.resolution | 1920 × 1080 | Match collection camera/capture-card mode and framing |
| camera.fps | 60 | Must capture approximately 60 native frames/s |
| camera.receive_latency | 0.125 s | Receive time minus this value estimates the image time |
| ft.hostname | 192.168.2.1 | RG2-FT sensor IP, also settable with `--gripper-ip` |
| ft.port / slave_id | 502 / 65 | Modbus TCP endpoint |
| ft.frequency | 100 Hz | Continuous native reads, independent of model rate |
| ft.receive_latency | 0.010 s | Receive time minus this value estimates the F/T time |
| startup_bias.sample_count | 200 | Stationary, unloaded samples for software bias |

The latency defaults come from the existing setup; **measure/verify them on the
new computer**. They are receive-time corrections, not hardware synchronization.
The observer checks the resulting clocks, sample intervals, and RGB-to-F/T age;
those checks cannot detect a consistently wrong latency offset. Camera geometry,
sensor mounting/axis signs, and exposure should match collection. A 30 FPS camera
mode changes the trained `[t-3,t]` interval and is rejected by the cadence check.

Keep both fingers unloaded and still before starting. The program collects a
fresh software bias, checks sample stability, and subtracts it once from native
readings. A stable applied load cannot be distinguished from an unloaded offset,
so do not calibrate while pushing the lever. There are **no Modbus writes**, no
device tare commands, and no reuse of a bias saved on the training computer.
Restart unloaded to recalibrate after a mounting or offset change.

Only one application should own this camera device. If teleoperation or a policy
already captures it, integrate the runtime into that program's existing streams
as described below instead of opening the same camera twice. Existing programs
can retain their normal robot control; the observer does not supply robot actions.

## 4. Run the live observer

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
python eval_real_context_rgb_force.py \
  --checkpoint checkpoints/context_rgb_force/best.pt \
  --config example/eval_context_rgb_force.yaml \
  --camera /dev/v4l/by-id/YOUR_CAMERA-video-index0 \
  --gripper-ip 192.168.2.1 \
  --device cuda:0 --rate 10 --save-inputs
```

This uses physical GPU 2 as logical `cuda:0`; set `CUDA_VISIBLE_DEVICES=3` to use
physical GPU 3. One GPU is enough for this single-stream classifier; DDP is not
used for evaluation. On a computer with different GPU numbering, select an
available physical GPU. `--device cpu` also works but may be too slow for live
input-age limits. The configured rate is a maximum, not a guaranteed rate.

Press **q/Esc** to quit, **r** for a new episode. Reset discards pre-reset sensor
history from inference and keeps the startup bias. It waits for a full new
window (about 0.4 s of F/T plus camera latency), rather than carrying old history
into a new episode. For SSH without a display, append
`--headless --max-seconds 60`; Ctrl-C also stops the process.

## 5. Input contract — preserve this when integrating

| Input | Contract |
|---|---|
| RGB | Two native frames `[t-3,t]`, about 50 ms apart at 60 Hz |
| Image preprocessing | Inpaint ArUco `DICT_4X4_50`, mask gripper on raw frame, resize/center crop to 224 × 224, BGR→RGB |
| Image options | Mirror retained, no mirror swap, no lens warp, no augmentation; `finger=False`, `use_aa=False` |
| F/T | Last 41 consecutive native samples **at or before** the latest selected RGB time |
| Channels | Left `[Fx,Fy,Fz,Tx,Ty,Tz]`, right same; sensor frame, forces N, torques Nm |
| Timing | Latest causal F/T must be at most 12 ms older than RGB; no future samples or resampling |
| Features inside model | Per finger: raw six axes + trailing mean over five samples + difference from mean five samples earlier |
| Model normalization | RGB CLIP mean/std and all 36 learned F/T feature mean/std buffers are in `best.pt` |
| Outputs | Four logits, four softmax probabilities, argmax class; no smoothing or heuristic relabeling |

Raw F/T is required: do not pass precomputed 18-channel features, divide by
training standard deviation yourself, rotate to TCP, take absolute values,
or smooth twice. TCP position/orientation and gripper width are never model
inputs. Live startup waits for a complete window; the offline training loader
instead repeats the first available sample at episode boundaries. Once enough
history exists, sample selection is the same.

The reusable API is `umi.real_world.rgb_force_context.RGBForceContextRuntime`:

```python
from umi.real_world.rgb_force_context import RGBForceContextRuntime

runtime = RGBForceContextRuntime("checkpoints/context_rgb_force/best.pt", device="cuda:0")
# Supply rolling buffers of EVERY native sample, not only 10 Hz policy observations.
# rgb_frames: already preprocessed uint8 RGB [N,224,224,3]
# wrench_12d: bias-corrected native N/Nm [M,12], left first then right.
# rgb_times and ft_times: strictly increasing Unix seconds on the same corrected clock.
obs, timing = runtime.prepare(rgb_times, rgb_frames, ft_times, wrench_12d,
                              episode_start=episode_start, now=now)
prediction = runtime.predict(obs)
```

For raw BGR buffers pass `image_transform=TrainingImageTransform(aruco_config_path)`
to `prepare`. For already processed replay-buffer/UmiEnv images, omit that
argument. Apply bias correction exactly once upstream. Catch
`ObservationUnavailable` to show an unavailable status; do not convert it into
class 3, zero-valued F/T, or an old held prediction. Also verify the result's age
after inference, as the standalone script does. Integrating these probabilities
into a Stage B action policy requires a separate compatible policy loader.

## 6. Logs and reproducing a prediction

Each run creates a new `outputs/context_real_eval/<timestamp>/` directory:

- `metadata.json`: checkpoint SHA-256, epoch, class order, Git commit, sensor
  settings, timing conventions, and measured startup bias/stability.
- `predictions.jsonl`: timestamps, probabilities, class/confidence, inference
  latency, F/T age and history span, or `valid=false` with the reason.
- `summary.json`: valid/total records, elapsed time, and exit/failure status.
- `inputs/*.npz` with `--save-inputs`: exact two processed RGB images and both
  41×6 F/T arrays, plus selected timestamps, before internal normalization.

```bash
HF_HUB_OFFLINE=1 python eval_real_context_rgb_force.py \
  --checkpoint checkpoints/context_rgb_force/best.pt --device cpu \
  --mode replay \
  --replay-inputs outputs/context_real_eval/RUN/inputs/00000020.npz
```

Replay opens no hardware and does not reapply bias/image preprocessing. CPU/GPU
rounding may cause small differences. Input recording is optional and consumes
disk; without it, metadata and prediction logs are still saved. Ground-truth
labels are not available live, so the script does not report accuracy/F1.

`warming_up_*` is expected at startup/reset. Persistent `rgb_cadence_mismatch`,
`ft_cadence_mismatch`, or `gap_in_ft_history` means the capture/polling rate does
not match training. `stale_ft_at_rgb_anchor` indicates alignment/availability
problems; inspect latency settings and native sample timestamps. A stale camera
or slow inference also produces an explicit unavailable record. Sensor read or
status errors stop the run and are recorded in the summary. Correct acquisition
before treating these runs as classifier evaluation.

## Verification and limits

The test suite covers native causal selection, future-sample exclusion, reset
boundaries, stale/gapped inputs, checkpoint compatibility/normalization, exact
input replay, image preprocessing, sensor units, read-only Modbus, and independent
camera capture. Run it with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_context_real_eval.py tests/test_context_rgb_force.py \
  tests/test_rg2ft_protocol.py tests/test_rg2ft_startup_bias.py -q
```

Implementation validation passed 33 tests. With the actual epoch-123 checkpoint,
four manually labelled dataset examples (one per class) produced exactly the
same input tensors and CPU logits through this runtime and the training dataset
loader. CPU and physical GPU 2 replay both completed. These are implementation
checks, not a new estimate of model accuracy. The checkpoint can also be checked
offline on the destination with `--mode self-test`.

Physical camera/Modbus acquisition, latency calibration, and classification
quality on the destination robot have not been tested here and must still be
validated there.
