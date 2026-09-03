# Indy + RG2-FT evaluation path

This is the RG2-FT-specific deployment path.  The preserved Dynamixel entry
point is `eval_real_indy_dynamixel.py`; do not use its YAML for RG2-FT.

## Control path

```text
eval_real_indy_rg2.py
  -> umi.real_world.umi_env.UmiEnv
  -> umi.real_world.rg2ft_controller.RG2FTController
  -> umi.real_world.rg2ft_protocol.RG2FTModbusClient
  -> OnRobot Compute Box (Modbus/TCP)
```

Policy and teleoperation actions use gripper width in metres.  RG2-FT register
units are 0.1 mm, so 0.0 / 0.05 / 0.1 m map to 0 / 500 / 1000.

The default hardware configuration is
`example/eval_robots_config_indy_rg2.yaml`:

- Indy: `192.168.1.10`
- OnRobot Compute Box: `192.168.2.1:502`
- Modbus device/slave ID: `65`
- RG2-FT width: `0..0.1 m`
- grip force: `20 N`
- startup auto-open: disabled

Confirm these addresses on the deployment machine before connecting.

## First-scene camera overlay

The launcher loads `data/dataset.zarr.zip` by default and overlays episode 0's
first `camera0_rgb` image at 50% opacity on the live camera pop-up.  Select a
different initial scene with `MATCH_EPISODE=<index>`, or provide another Zarr
dataset with `MATCH_DATASET=/path/to/dataset.zarr.zip`.

## F/T zero at startup

Evaluation software-tares both RG2-FT finger sensors automatically at startup.
It averages the latest 25 raw samples (about 0.25 seconds at 100 Hz) and
subtracts that 12-channel baseline from all policy and eval-recording F/T
values. Keep the unloaded gripper still while the program starts. Use
`--no_zero_ft_on_start` only when raw sensor values are intentionally required;
change the averaging window with `--ft_zero_samples N`.

## Python dependency

Install the additional pinned dependency in the full UMI eval environment:

```bash
python -m pip install -r requirements_rg2ft.txt
```

`rg2ft_protocol.py` supports pymodbus `unit`, `slave`, and current
`device_id` APIs.  The pinned/tested API is pymodbus 3.14.0.

## Safe first run

Set the checkpoint and Python environment, then run the launcher.  Its default
is plan-only: it connects to RG2-FT and reads width/F/T, but the controller does
not write a motion command until an explicit width waypoint is scheduled.

```bash
PYTHON_BIN=/path/to/umi/bin/python \
RG2_CHECKPOINT=/path/to/rg2ft_latest.ckpt \
./deploy_real_indy_rg2.sh
```

After validating printed actions, addresses, current width and emergency-stop
access, explicitly enable motion:

```bash
PYTHON_BIN=/path/to/umi/bin/python \
RG2_CHECKPOINT=/path/to/rg2ft_latest.ckpt \
RG2_ENABLE_MOTION=1 \
./deploy_real_indy_rg2.sh
```

## Safety behaviour

- Connecting alone does not open or close the gripper.
- Commands are clamped to the RG2-FT physical range.
- Intermediate/closed widths are refreshed to hold against spring opening.
- A fully-open target is released after arrival to avoid re-grip oscillation.
- Modbus response errors are detected; the controller reconnects after a
  runtime communication failure.
- Signed slightly-negative fully-closed readings are clipped to 0 m before
  entering policy observations.

Hardware-free tests:

```bash
python -m unittest tests.test_rg2ft_protocol tests.test_rg2ft_controller -v
```
