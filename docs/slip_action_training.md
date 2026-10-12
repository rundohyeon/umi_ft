# Slip action policy: RGB 4 / paired F/T / no TCP input

This is the action-policy path for the slip classifiers with class order
`approach, error1, error2, turning`. It uses
`train_slip_action_rgb4_workspace`, not the older A/T/R/E context task.

For a different computer, start with the [handoff guide](slip_action_handoff.md).
It lists the artifacts outside Git, environment, required recording paths, and
offline evaluation commands. Real-robot integration for this action contract
is still pending.

The policy observation dictionary contains exactly:

| Key | Unbatched shape | Meaning |
| --- | --- | --- |
| `camera0_rgb` | `[4,3,224,224]` | `[t-30,t-20,t-10,t]` at 59.94 FPS |
| `robot0_ft_left` | `[4,41,6]` | Each RGB's own preceding native left F/T window |
| `robot0_ft_right` | `[4,41,6]` | Each RGB's own preceding native right F/T window |
| `valve_context` | `[1,5]` | Four predicted probabilities in the above order, then validity |

TCP position, orientation, gripper feedback, manual labels and episode outcomes
are absent from the model inputs. Demonstrated TCP remains necessary to construct
the training action targets; a live controller also needs its current pose to
convert predicted relative motions into robot targets. This does not feed TCP
into the neural network.

The RGB anchors span 0.5005 seconds. At each anchor, the latest F/T sample is
strictly at or before that RGB timestamp, at most 12 ms old. Each 41-sample window
spans approximately 0.4 seconds. Its trailing five-sample mean and the difference
from the mean five samples earlier produce `[32,18]` features: six raw, six mean,
six delta channels. There is no future interpolation in an observation.
Since the oldest image has its own preceding F/T window, the required total
sensor buffer is approximately **0.9 seconds**, not 0.5 seconds. Missing history
is excluded; no repetition across ignore or episode boundaries is used.

The pretrained CLIP ViT-B/16 backbone is frozen. Separate trainable left/right
F/T CNNs encode each window. The fusion Transformer receives 12 ordered tokens:
`RGB0,left0,right0,...,RGB3,left3,right3`. The context probabilities modulate that
feature using four residual experts. The resulting **768-D** condition feeds
the diffusion U-Net. It no longer includes the former 18 TCP features.

Action targets stay `[16,11]`: anchor-relative position 3, rotation-6D 6,
absolute gripper width 1, and signed force reference 1. Targets use every third
RGB frame, about 19.98 Hz; changing the RGB observation spacing does not change
the action frequency. The force target is the existing bias-removed
`0.5*(right_Fz-left_Fz)` interpolated onto target timestamps. It is a width-feedback
reference, not a direct gripper force-register command.

## Data and context preparation

Data: `/home/metafarmers/dkim/data/slip_processed_context/`:
`dataset.zarr.zip`, `dataset_force_sidecar.zarr`, and `state_labels_v2.npz`.
The initial context checkpoint is `outputs/slip_context_sequence/best.pt`
(RGB4/0.5s, context history20/~2s). The RGB8 checkpoint can also generate a cache;
the action policy still has its own four-image observation contract.

```bash
cd /home/metafarmers/dkim/umi_ft
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python \
  tools/prepare_slip_action_context.py \
  --checkpoint outputs/slip_context_sequence/best.pt \
  --cache-dir outputs/slip_clip_features \
  --output outputs/slip_action/context_rgb4.npz
```

This reads a verified frozen CLIP cache and runs the frozen classifier at its
trained ~10 Hz cadence. History contains only past observation-model probabilities,
as in `SlipTemporalContextRuntime`. It never uses a manual class as the input
probability. The tool refuses to overwrite an existing cache. Once prepared,
reuse the cache; do not regenerate it before each training run.

The cache records the classifier SHA256, recording fingerprint, model settings,
and all three episode splits. Loading checks the checkpoint SHA and data identity.
Predictions can be held for up to six source frames, without crossing episode,
ignore, or cadence-gap boundaries. Both four-image and eight-image classifiers
share the class interface, but changing classifiers requires a new cache and
evaluation; identical output dimensions do not guarantee identical policy behavior.

`SlipActionDataset` uses the classifier's fixed **201/44/44 episode split**.
Every observation and the complete future action interval must stay outside
ignore and sensor-invalid regions. Context startup may exclude additional frames.
Normalization fits only eligible training inputs and targets. Held-out episodes
are never used for optimization or normalization.

## Train on physical GPUs 2 and 3

```bash
cd /home/metafarmers/dkim/umi_ft
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 --gpu_ids 2,3 \
  --mixed_precision no --main_process_port 29517 \
  train.py --config-name=train_slip_action_rgb4_workspace \
  training.num_epochs=100
```

This uses pretrained vision with newly trained F/T, fusion, context conditioning
and action modules. The former 786-D/TCP action checkpoint is not a resume point
for this 768-D, 12-token model. Keep the trained context classifier frozen.

Outputs: `outputs/slip_action/rgb4_paired_ft_no_tcp/`. Complete validation loss
is computed using EMA weights every epoch, with a fixed noise seed; the best
three validation checkpoints and `latest.ckpt` are saved under `checkpoints/`.
Optimizer state is retained for resume. The action checkpoint also records the
context cache metadata and split. Use validation for checkpoint selection and
reserve the 44 test episodes for subsequent evaluation of trajectory errors,
especially after error1/error2 transitions.

For a two-step wiring check, add the following overrides and choose a separate
output directory. This does not produce a trained checkpoint:

```text
training.num_epochs=1 training.max_train_steps=2 training.max_val_steps=2
dataloader.num_workers=0 dataloader.persistent_workers=false
val_dataloader.num_workers=0 val_dataloader.persistent_workers=false
checkpoint.save_last_ckpt=false checkpoint.topk.k=0
hydra.run.dir=outputs/slip_action/smoke_rgb4_paired_ft_no_tcp
```

The older `eval_real_indy_rg2*.py` observation builder does not implement this
paired F/T/no-TCP contract. Its model contract must not be bypassed: live evaluation
needs a corresponding observation builder using the same RGB anchors and force
windows. Training and offline smoke checks do not establish closed-loop performance.

## Offline evaluation of the best epoch-9 model

The completed 100-epoch run reached its lowest logged validation loss at epoch 9:
`checkpoints/epoch=0008-val_loss=0.02815.ckpt` (checkpoint epoch numbers start at zero).
`eval_slip_action.py` defaults to that checkpoint and its **EMA** weights. It restores
the checkpoint's normalizer without fitting anything on evaluation data, and disables
RandomCrop/ColorJitter. The CLIP weights come from the checkpoint; no Hugging Face
download or CLIP feature cache is needed for action inference.

```bash
cd /home/metafarmers/dkim/umi_ft
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python eval_slip_action.py \
  --checkpoint outputs/slip_action/rgb4_paired_ft_no_tcp/checkpoints/epoch=0008-val_loss=0.02815.ckpt \
  --split test --device cuda:0 --batch-size 32 --anchor-stride 6
```

Physical GPU 2 is `cuda:0` inside this process. The default test split contains
44 held-out episodes. Stride 6 selects the ~10 Hz episode-local anchor grid,
matching the nominal two-action/20 Hz replanning cadence. `--anchor-stride 1`
evaluates every eligible RGB anchor. `--episode 291` selects the original reviewed
episode number within the chosen split. `--max-samples 64` runs a diagnostic subset,
and is explicitly marked as partial evaluation in the report. It does not replace
the full selected-grid result. `--prediction-repeats 3` averages statistics over
three independently seeded diffusion samples. Reproducibility requires the same
seed, batch size, device and selected anchor grid.

The evaluator checks the action checkpoint's saved context class order, classifier
SHA256, data fingerprint, model settings and episode splits against the context cache.
By default it uses the same `outputs/slip_action/context_rgb4.npz` and classifier
`outputs/slip_context_sequence/best.pt` as training. Only the cached probabilities
enter the action network. The classifier `.pt` is read for identity checking, not
run inside each action prediction. All recorded data files referenced by the saved
config must remain available. `--context-cache` and `--context-checkpoint` can re-root
those two artifacts while retaining their identity. The existing recording fingerprint
also includes the RGB ZIP path/mtime, so relocating recording data requires addressing
that contract; changing a path alone does not bypass it.

Without `--output`, each run creates a new timestamped directory beside `checkpoints/`.
An explicitly supplied output directory must not already exist. Files are:

- `metrics.json`: overall, per-phase and per-episode metrics, checkpoint identity and coverage.
- `per_phase.csv`, `per_episode.csv`, `per_horizon.csv`: physical-unit error tables.
- `predictions.npz`: `[repeats, anchors,16,11]` predicted actions, `[anchors,16,11]`
  targets, source frames, timestamps, original episode IDs, manual phase IDs,
  context probabilities and their source frames. Phase IDs are evaluation metadata;
  they are never passed to the network.
- `errors_by_horizon.png`, `report.md`: readable results and error curves.
- `run_config.json`: exact evaluation settings, also present if a run fails before completion.

Metrics are position **Euclidean distance in mm**, rotation **SO(3) angle in degrees**,
gripper-width **absolute error in mm**, and signed-force-reference **absolute error in N**.
No overall score mixes these units. Main metrics exclude step zero (the current pose)
and cover future steps 1–15, approximately 0.05–0.75 seconds. Step zero and the first
future step are reported separately. Invalid predicted rotation bases are counted;
they are not silently converted to valid angles.

The hold-current-demo baseline keeps the current relative pose, demonstrated width
and demonstrated force unchanged throughout the horizon. It is a diagnostic baseline
with access to those current measurements. Low offline action errors do not establish
robot task success or recovery performance. Batch latency excludes sensor I/O and
context inference; it is not end-to-end real-time latency. The evaluator's noise loss
uses unaugmented RGB and therefore differs from the augmented validation loss used
to select the checkpoint. This script never connects to the robot.

## Relevant code

- `diffusion_policy/dataset/slip_action_dataset.py`: paired sampling, actions, splits, normalization.
- `diffusion_policy/model/vision/dual_ft_obs_encoder.py`: optional `per_rgb` F/T mode; legacy defaults retained.
- `tools/prepare_slip_action_context.py`: frozen causal probability cache.
- `diffusion_policy/config/task/slip_action_rgb4_paired_ft.yaml`: explicit input/action contract.
- `tests/test_slip_action.py`: causality, boundary exclusion, TCP independence, normalizer isolation, cached/runtime-equivalent history.
- `eval_slip_action.py`, `tests/test_slip_action_eval.py`: EMA restoration, held-out trajectory evaluation, unit-aware metrics and reproducibility checks.
