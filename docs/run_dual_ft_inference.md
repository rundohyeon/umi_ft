# Running dual-finger F/T Indy inference

Run all commands from `/home/oem/smh/umi_ft` in the `umi` environment. The
checkpoint must be the dual-F/T run, not an RGB-only checkpoint.

```bash
CKPT=data/outputs/2026.08.18/11.14.51_train_diffusion_unet_timm_umi_dual_ft_umi_dual_ft/checkpoints/latest.ckpt
```

## Inspect a checkpoint (no hardware access)

```bash
python eval_real_indy_rg2_dual_ft.py \
  --checkpoint "$CKPT" \
  --inspect-checkpoint
```

It must print `condition: [1, 800]`, `action: [1, 16, 10]`, both F/T shapes,
and `normalizer_owner: policy.predict_action`.

## Offline parity tests

```bash
PYTHONPATH=. pytest -q \
  tests/test_dual_ft_inference.py \
  tests/test_dual_ft_online_offline_parity.py
```

## Hardware dry-run (mandatory before actuation)

This starts the GoPro, Indy pose receiver, RG2-FT Modbus reader, history
assembly and policy. It **does not submit robot commands**.

```bash
python eval_real_indy_rg2_dual_ft.py \
  --checkpoint "$CKPT" \
  --robot-config example/eval_robots_config_indy_rg2.yaml \
  --n-action-steps 2 \
  --dry-run \
  --max-cycles 100 \
  --log-dir data/eval_dual_ft_dryrun
```

Before real use, confirm the console shows causal left/right F/T ages below
12 ms, finite `[1,32,6]` inputs, a `[1,800]` condition contract, and full
`[1,16,10]` predictions. Save the resulting logs and record the 100-cycle
latency/age summary manually; that requires the physical device and cannot be
claimed by offline tests.

## Real inference (explicit user action required)

```bash
python eval_real_indy_rg2_dual_ft.py \
  --checkpoint "$CKPT" \
  --robot-config example/eval_robots_config_indy_rg2.yaml \
  --n-action-steps 2 \
  --log-dir data/eval_dual_ft_real
```

The reference evaluator retains its operator-controlled start/stop UI. Ensure
the robot emergency stop is reachable. No force/admittance/impedance or
compliance controller is added: F/T is an observation only.
