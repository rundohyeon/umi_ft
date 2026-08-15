# eval_real_indy transfer notes

> **LEGACY ONLY — DO NOT RUN THESE COMMANDS FOR THE 0709 MODEL.**
> They reference the old axis checkpoint, old `-m dataset_axis_newP`, and an
> obsolete machine path. Use `DEPLOY_0709_ROBOT_TCP.md` and the two
> `run_eval_0709_*.sh` wrappers instead. This file is retained only to document
> how the previous deployment behaved.

This file summarizes files and machine-side settings from the previous
`eval_real_indy.py` workflow.

## Current working directory

Run commands from:

```bash
cd /home/idim/dkim/umi/scripts/indy_umi
```

Primary command currently written in `umi/asdasd.txt`:

```bash
/home/idim/miniforge3/envs/umi/bin/python eval_real_indy.py \
  -i scripts/waypoints/epoch100_axis.ckpt \
  -o data/eval_run \
  -rc example/eval_robots_config_indy.yaml \
  -m data/axix_data_zarrfile/dataset_axis_newP.zarr.zip \
  --allow_rotation
```

Safer plan-only/audit command:

```bash
/home/idim/miniforge3/envs/umi/bin/python eval_real_indy.py \
  -i scripts/waypoints/epoch100_axis.ckpt \
  -o data/eval_run \
  -rc example/eval_robots_config_indy.yaml \
  --plan_only \
  --print_motion_debug \
  --print_model_input \
  --pose_eval_audit \
  --dataset_zarr data/axix_data_zarrfile/dataset_axis_newP.zarr.zip
```

## Must move to run on another machine

Use `transfer_eval_real_indy_minimal_files.txt` from repo root
`/home/idim/dkim`.

Core code:

- `umi/scripts/indy_umi/eval_real_indy.py`
- `umi/scripts/indy_umi/diffusion_policy/`
- `umi/scripts/indy_umi/umi/`
- `umi/scripts/indy_umi/scripts/`
- `umi/scripts/indy_umi/example/`
- `umi/scripts/indy_umi/assets/`
- `umi/scripts/indy_umi/slam_pipeline_latest/`
- `umi/scripts/indy_umi/conda_environment.yaml`

Current runtime data:

- `umi/scripts/indy_umi/scripts/waypoints/epoch100_axis.ckpt` (~1.3 GB)
- `umi/scripts/indy_umi/scripts/waypoints/rulebase_indy.yaml`
- `umi/scripts/indy_umi/example/eval_robots_config_indy.yaml`
- `umi/scripts/indy_umi/data/axix_data_zarrfile/dataset_axis_newP.zarr.zip` (~1.8 GB)

Approximate minimal transfer size: about 3.3 GB plus small code/config files.

## Optional data to move

Use `transfer_eval_real_indy_archive_files.txt` if you also want existing run
records and rendered debug videos.

- `umi/scripts/indy_umi/data/eval_run/` (~188 MB at time of writing)
- `umi/scripts/indy_umi/data/pose_render/` (~38 MB)

## Machine/hardware assumptions

The target machine must be on the robot network and have equivalent devices.

Current observed network/device state:

- Indy robot IP in config: `192.168.1.10`
- Current host Ethernet IP: `192.168.1.11/24`
- Current host Wi-Fi IP: `192.168.1.53/24`
- Direct Dynamixel gripper config uses `/dev/ttyUSB0`, id `1`, baud `57600`
- Current USB video devices: `/dev/video0`, `/dev/video1`, `/dev/video3`, `/dev/video4`
- USB capture currently visible: Elgato HD60 X

If `/dev/ttyUSB0` changes on the target machine, update:

```yaml
scripts/waypoints/rulebase_indy.yaml
```

specifically:

```yaml
gripper:
  port: /dev/ttyUSB0
```

If the robot network/IP changes, update:

```yaml
example/eval_robots_config_indy.yaml
```

specifically:

```yaml
robot_ip: "192.168.1.10"
```

## Environment

Current working Python:

```bash
/home/idim/miniforge3/envs/umi/bin/python
```

Required packages are not satisfied by the base `python3`; use/create the
`umi` conda environment. The existing environment spec is:

```bash
umi/scripts/indy_umi/conda_environment.yaml
```

On the target, typical setup is:

```bash
conda env create -f conda_environment.yaml
conda activate umi
```

or, if the environment already exists, install missing project dependencies into
that environment.

## Transfer commands

From `/home/idim/dkim`, replace `USER@HOST:/dest/dkim/` with the target.

Minimal runnable bundle:

```bash
rsync -avh --progress \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --files-from=umi/scripts/indy_umi/transfer_eval_real_indy_minimal_files.txt \
  /home/idim/dkim/ USER@HOST:/dest/dkim/
```

Runnable bundle plus logs/videos:

```bash
rsync -avh --progress \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --files-from=umi/scripts/indy_umi/transfer_eval_real_indy_archive_files.txt \
  /home/idim/dkim/ USER@HOST:/dest/dkim/
```

## Important caution

The checkpoint metadata says its original dataset path was:

```text
0624_axis/dataset.zarr.zip
```

The current commands explicitly pass:

```text
data/axix_data_zarrfile/dataset_axis_newP.zarr.zip
```

Keep passing `--dataset_zarr` or `-m` explicitly on the target unless the target
filesystem recreates the original `0624_axis/dataset.zarr.zip` path.
