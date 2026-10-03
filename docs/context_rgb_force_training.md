# RGB + native F/T four-state context training

This Stage A classifier uses the existing labelled 284-episode dataset in
`three_dataset`, with class order `approach`, `turning`, `recovery`, `error`.
It uses no TCP position/orientation, gripper width, action targets, Qwen labels,
or model-predicted context sidecar. The action policy is not trained here.

## Architecture and observations

- RGB: two 224x224 observations at `[t-3, t]` on the RGB frame grid (about
  50 ms apart). Frozen pretrained `vit_base_patch16_clip_224.openai` produces
  768-D features, followed by a trainable projection to 128. Normalization
  uses the pretrained model's CLIP mean/std. No task-trained visual checkpoint
  is used to initialize the model.
- Native F/T: the latest 41 measurements at or before the RGB anchor, selected
  within the same episode. At 100 Hz this covers approximately 400 ms. Short
  startup histories repeat the first available sample. Both fingers retain
  `[Fx,Fy,Fz,Tx,Ty,Tz]` in their native sensor frames, including torque. The
  already bias-corrected `wrench_12d` is not tared again or rotated to TCP.
- Each finger has 18 feature channels: six raw axes, six trailing means, and
  six differences between the current mean and the mean five samples earlier.
  For native sample index `i`, `mean[i] = mean(raw[i-4:i+1])` and
  `delta[i] = mean[i] - mean[i-5]`. These are signed changes in N/Nm, not
  derivatives in N/s or Nm/s. The lag is approximately 50 ms at 100 Hz.
- Nine extra preceding samples supply the filter context; the remaining 32
  output timesteps cover the original approximately 310 ms. No future sample
  is used. All averaging/differencing stays FP32 under mixed precision.
- Feature construction lives in `diffusion_policy/context/ft_features.py` and
  runs inside the model for both training and checkpoint inference. Live callers
  supply `[B,41,6]` per finger, with the same episode-start padding. They must
  not pad independently at each 32-step observation window or smooth twice.
- Left and right use independent `CausalFTEncoder` CNNs with 18 input channels. Five
  stride-two layers summarize all 32 feature timesteps into one 128-D token each.
  This is learned from the context labels; the F/T CNNs have no pretrained weights.
- Two image tokens, two force-history tokens, and a terminal query enter a
  two-layer, four-head Transformer with hidden size 128 and FFN size 256.
  The final query produces four logits. Softmax gives four context probabilities.
- Trainable parameters: 475,492. Frozen visual parameters: 85,799,424.
  Both the CLIP weights and F/T normalization buffers are included in checkpoints.

## Data and supervision

The loader verifies RGB/label/force episode boundaries, source episode IDs,
timestamps, native channel order, and every stored causal force index. It
rejects anchors with unavailable force or force older than 12 ms. All 106,304
valid labels in the supplied dataset pass these checks; 79,958 invalid labels
are excluded. RGB remains compressed in the read-only ZIP with a separate
lazy store per DataLoader worker. TCP arrays are never read.

EP001–193 are V12-derived rule labels (weight 0.3); EP194–284 are manual
labels (weight 1.0). The training loss also uses inverse-frequency class
weights computed from training episodes. Normalization fits separate mean/std
for all 36 features (left raw/mean/delta, then right raw/mean/delta), using
only training episodes and the same causal feature transform. Padding and filter
state reset at each episode boundary. Validation/test metrics report all
selected labels and a separate manual-only subset.

Splits use seed 260904 and retain the previous observer's fixed test episodes
`[37,45,47,87,104,202,211,246,273,281]` (one-based). The remaining 274 episodes
are divided into 233 train and 41 validation episodes. No episode crosses splits.

| Split | approach | turning | recovery | error |
|---|---:|---:|---:|---:|
| Train | 32,303 | 41,531 | 5,164 | 6,548 |
| Validation | 5,660 | 7,819 | 1,329 | 1,478 |
| Test | 1,470 | 2,095 | 403 | 504 |

Validation macro F1 selects the best checkpoint. Full test evaluation happens
after training, using that checkpoint; test observations do not fit statistics
or affect checkpoint selection.

## Train on physical GPUs 2 and 3

From the repository root:

```bash
cd /home/metafarmers/dkim/umi_ft
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  /home/metafarmers/anaconda3/envs/umi/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  train_context_rgb_force.py \
  --config context/qwen/config/train_context_rgb_force.yaml
```

`torchrun` launches one DDP process per GPU. Visible `cuda:0` maps to physical
GPU 2 and `cuda:1` maps to physical GPU 3. Defaults are 30 epochs, 32 examples
per GPU (global batch 64), AdamW learning rate 3e-4, and bfloat16 mixed
precision. CLIP weights are already present in the local Hugging Face cache;
offline mode fails rather than silently initializing random visual weights.

The script refuses to overwrite an existing nonempty output directory. To
start a separate run, append `--set output_dir=outputs/context_rgb_force_4state_ft_features_run2`.
To continue the same run, append:

```bash
--resume outputs/context_rgb_force_4state_ft_features/context_encoder_last.pt
```

The main artifacts in `outputs/context_rgb_force_4state_ft_features/` are:

- `context_encoder_best.pt`, `context_encoder_last.pt`: complete model,
  optimizer, architecture, normalization, epoch, labels/splits, and config.
- `resolved_config.yaml`, `dataset_report.json`, `episode_splits.json`.
- `training_history.jsonl`: per-epoch loss and validation metrics.
- `test_eval/`: metrics, confusion matrix, timeline, confidence histogram,
  class distribution, transition statistics, and saved predictions.

For a short implementation check, use a separate output directory and append
`--smoke-batches 2`. This runs only two training/validation/test batches;
its metrics are not an estimate of classifier performance, and its checkpoint
cannot resume a full training run. A CPU-only data check is available with
`--dry-run --set device=cpu --set output_dir=/tmp/context_rgb_force_data_check`.

The feature mode, mean window, lag, input shapes, and normalization are saved
with the new `context_rgb_force_4state_ft_features_v2` checkpoint. It requires
a new training run; the old six-channel F/T checkpoint cannot resume this
18-channel model. Old raw-only checkpoints still reconstruct their original
32-by-6 observation contract. Dataset/model feature settings must agree, and
resume validation happens before replacing any saved run metadata.

This classifier is separate from the earlier pooled-RGB `ContextEncoder`.
Use `RGBForceContextEncoder.from_checkpoint(payload)` to reconstruct it without
downloading visual weights. Integrating its soft probabilities into Stage B
requires adapting that policy's context model loader; the older loader is not
compatible with this checkpoint schema.

For live camera/RG2-FT observation, use `eval_real_context_rgb_force.py` and the
[real-evaluation / checkpoint handoff guide](context_rgb_force_real_eval.md).
The standalone observer restores this checkpoint, preserves native causal
histories, and records predictions without sending robot/gripper commands.
