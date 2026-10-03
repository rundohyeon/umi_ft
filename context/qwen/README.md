# Qwen3.5-9B labeling

Run commands from `/home/metafarmers/dkim/umi_ft`. The new comparison config is
`context/qwen/config/context_labels_qwen35.yaml`; its outputs go to
`data/context_labels_v8_qwen35`. Class definitions follow v6:
approach / turning / finish / error; finish means a sharp right-finger **force
magnitude** decrease. Existing training and review defaults remain on v6 until
the new predictions have been reviewed.

The UMI interpreter reads the dataset and exports video. Qwen3.5 runs in a
separate Python 3.11 process with its own PyTorch and Transformers. Frames cross
the local process boundary as lossless PNGs in chronological order. No dataset
or images are uploaded. Inference uses local weights, BF16 on CUDA, SDPA,
greedy decoding, and `enable_thinking=False` for short JSON output. Actual
runtime versions are recorded in the labeling manifest. Flash Linear Attention
and causal-conv1d accelerate the recurrent attention layers; Triton kernels compile
on the first invocation for this GPU and are cached in `.triton_cache` here.
V8 uses `force_visual_v1`: Qwen returns only visual visibility, motion and
confidence. Numeric force decisions are computed from native causal samples.
Reasons are rendered from those measurements and visual enums, so Qwen cannot
write invented force numbers. The archived v7 config and results are preserved
as `config/context_labels_v7_qwen35.yaml` and `data/context_labels_v7_qwen35`.

Contact uses the median magnitude of the last 50 ms: enter at 0.5 N, exit at
0.3 N, and retain the previous contact state between thresholds. These are
user-selected **trial** values. Unknown, sparse or stale force never establishes
contact. Turning requires both measured contact and Qwen's engaged-rotation
observation. Background motion or a nonzero noise reading cannot suffice.
No contact alone does not establish approach; visual approach is required.

Recent force is compared with the median from 250 to 100 ms before the anchor.
A decrease of at least 1 N **and** 35% is a trial sharp-drop rule. A measured
sharp drop selects finish even if the lever is still rotating. Explicit lever
disappearance remains error. Old peaks outside this short window cannot keep
selecting finish after force stabilizes. The 2-second force maxima, averages and
derivatives are removed from Qwen's input to avoid confusing old peaks with the
current force; RGB, TCP rotation, gripper width and previous labels remain causal.

Rows retain `visual_evidence_json`, `force_evidence_json`, `decision_source`,
and the complete raw model responses. Rule-only finish decisions have confidence
0 (uncalibrated), and the video identifies them as force rules rather than showing
an invented model probability. All remain automatic labels requiring review.
Thresholds and the decision version are included in the resume manifest.

## Install

```bash
uv venv --python /home/metafarmers/anaconda3/envs/gemini-python311/bin/python context/qwen/.venv35
UV_CACHE_DIR=/home/metafarmers/dkim/.uv-cache uv pip install \
  --python context/qwen/.venv35/bin/python \
  --index-url https://pypi.org/simple -r context/qwen/requirements-qwen35.txt

context/qwen/.venv35/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir context/qwen/checkpoints/Qwen3.5-9B
```

Official model: <https://huggingface.co/Qwen/Qwen3.5-9B>.
The full weights occupy about 19.3 GB. Do not install the new requirements
into the original `umi` environment or the old `.venv` used for Qwen2.5-VL.

## Label and review one episode

Only physical GPU 3 is exposed here; it becomes `cuda:0` inside the process.
Use the original UMI Python for this command; it launches `.venv35` itself.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python -u scripts/generate_context_labels.py \
  --config context/qwen/config/context_labels_qwen35.yaml --device cuda:0 --episode 0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python tools/export_context_review_video.py \
  --config context/qwen/config/context_labels_qwen35.yaml --episodes 0 \
  --require-complete --output-dir outputs/context_label_review_v8_qwen35_ep00
```

Episode 0 has 568 RGB frames, with 95 causal predictions at stride 6. Video
holds the most recent prediction between anchors. Add `--resume` to continue
an interrupted labeling run with identical model/config/data; use a new output
directory for a different experiment. The exporter refuses to overwrite an
existing video directory.

For interactive review, use `CONTEXT_REVIEW_CONFIG=context/qwen/config/context_labels_qwen35.yaml`
with `streamlit run tools/review_context_labels.py` in the UMI environment.

## Fill unknown phases after labeling

For completed episodes, the optional offline rule fills only `unknown` labels:
before the first turning prediction -> approach; after the first finish prediction
-> finish. Unknowns between these boundaries and all explicit labels (including
error) stay unchanged. Missing turning/finish boundaries do not trigger their
respective rule; if both rules apply, finish wins. This assumes one task sequence
per episode. The approach fill uses a future phase boundary, so this is offline
postprocessing, not a change to causal Qwen inference.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python tools/postprocess_context_labels.py \
  --config context/qwen/config/context_labels_qwen35.yaml --episodes 1 \
  --output-dir data/context_labels_v8_qwen35_phase_fill_ep01

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python tools/export_context_review_video.py \
  --config context/qwen/config/context_labels_qwen35.yaml --episodes 1 \
  --labels-dir data/context_labels_v8_qwen35_phase_fill_ep01 --require-complete \
  --output-dir outputs/context_label_review_v8_qwen35_phase_fill_ep01
```

Original labels remain in their source directory. The separate derived directory
contains JSONL/Parquet labels with `raw_*` fields, original force/visual evidence,
the correction rule and boundary anchor. The manifest records source/output
hashes and offline provenance. Filled labels have uncalibrated confidence 0 and
are marked as sequence-rule corrections in the video; they are not human-reviewed
labels. Incomplete episodes and overwriting existing output directories are
rejected. To review this result in the GUI, set its **Label directory** to the
derived directory above. Training defaults do not switch automatically.

## Raw clip without a SLAM trajectory: GX011718

`session_260827/raw_videos/GX011718.MP4` resolves to
`demos/demo_C3531324983555_2026.08.27_12.11.15.314817/raw_video.mp4` in that session.
The existing dataset plan excludes this recording because `camera_trajectory.csv`
is missing. Its paired native sensor recording is
`session_260827/demos/demo_121113/rg2ft.csv`.

`tools/prepare_context_raw_clip.py` prepares a separate labeling-only NPZ with
all 527 video frames, original video-relative timestamps, native bias-corrected
wrenches and causally selected sensor gripper width. It fits the clock shift by
comparing paired ArUco tag separation with measured gripper width; force and
phase predictions are not used to fit alignment. CSV is already software-tared;
only a residual median bias from a quiet pre-video interval is removed. The
original `ft_offset.json` is not added back. Video preprocessing reuses UMI tag
inpainting, gripper masking and the 224x224 crop/resize.

For this clip, `t_video = t_csv - t_csv_first - 0.864 s`, with width correlation
0.9991 over 167 paired tag detections. This is a motion-based estimate, not a
hardware-verified clock alignment. Candidate shifts within 0.001 correlation of
the optimum span -0.904 to -0.840 s; this is a sensitivity range, not a confidence
interval. Source hashes, the fit and preprocessing are saved with the NPZ.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python tools/prepare_context_raw_clip.py \
  --video session_260827/raw_videos/GX011718.MP4 \
  --csv session_260827/demos/demo_121113/rg2ft.csv \
  --output data/context_recordings/GX011718.npz

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python scripts/generate_context_labels.py \
  --config context/qwen/config/context_labels_GX011718.yaml --device cuda:0 --episode 0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python tools/postprocess_context_labels.py \
  --config context/qwen/config/context_labels_GX011718.yaml --episodes 0 \
  --output-dir data/context_labels_v8_qwen35_GX011718_phase_fill

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python tools/export_context_review_video.py \
  --config context/qwen/config/context_labels_GX011718.yaml --episodes 0 \
  --labels-dir data/context_labels_v8_qwen35_GX011718_phase_fill --require-complete \
  --output-dir outputs/context_label_review_v8_qwen35_GX011718_phase_fill
```

The clip's local episode ID is **0**, distinct from episode 0 in the original
214-episode dataset. TCP position/rotation inputs are disabled explicitly; no
pose or training action targets are fabricated. This NPZ is an offline labeling
input, not a policy-training dataset. Raw-label outputs, phase-filled outputs and
review videos use separate directories; existing files are not overwritten.
