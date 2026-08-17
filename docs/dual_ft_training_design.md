# Dual-F/T-conditioned Diffusion Policy implementation

## 범위

이 구현은 RGB+pose에 left/right native six-axis wrench history를 조건으로 추가한
position Diffusion Policy다. Force/torque feedback controller, impedance,
admittance 또는 force target predictor가 아니다.

구현은 기존 CLIP ViT vision encoder, image transform, relative pose preprocessing,
rotation-6D, 10D coordinate action 및 prediction horizon 16을 유지한다.

## Pipeline

```text
RGB [B,2,3,224,224]
  -> existing TimmObsEncoder vision path
  -> two visual tokens [B,2,768] --------------------+
                                                      |
left native F/T [B,32,6] -> independent CausalConv -->| left token
right native F/T [B,32,6] -> independent CausalConv ->| right token
                                                      |
  [CLS, rgb_old, rgb_anchor, left, right]             |
  -> TransformerEncoder fusion -> fused [B,768] ------+

existing low-dimensional pose/gripper [B,32]
  -> concat -> global condition [B,800]
  -> existing ConditionalUnet1D
  -> action [B,16,10]
```

각 F/T encoder는 five-stage left-padded causal convolution이다.

```text
6 -> 16 -> 32 -> 64 -> 128 -> 768
kernel=2, stride=2, LeakyReLU
```

Left/right encoder는 `share_ft_encoder: false`가 default이고 서로 다른 parameter를
가진다. 양쪽 wrench를 합하거나 평균내지 않으며 common tool/wrist frame으로
변환하지 않는다.

## Dataset sampling

`UmiDualFTDataset`은 explicit allowlist만 읽는다. RGB anchor는 TARGET의 기존
`current` sample에 해당하는 최신 RGB timestamp다. Pose/gripper는 같은 timestamp
grid이고, F/T는 episode별로 독립 causal lookup한다.

```python
idx = np.searchsorted(ft_timestamp, anchor, side="right") - 1
```

Left/right 중 어느 하나라도 `idx < 0`이면 anchor를 sampler index에 넣지 않는다.
한 causal sample이 존재한 뒤 history 시작이 episode 첫 sample보다 앞서는 slot은
earliest causal sample로 repeat-first padding한다. 모든 반환 sample은
`max(observation_timestamp) <= anchor_timestamp`를 assert한다.

RGB `ZipStore`는 dataset object 간 또는 DataLoader process 간 공유하지 않는다.
각 worker가 PID별로 nested prefix를 read-only lazy-open한다.

## Normalization

Left/right 각각 6 channel을 별도로 range-normalize한다. Min/max/mean/std는 seed 42
train episode 139개만 사용한다. Validation episode와 postprocessed force label은
통계에 포함하지 않는다. Normalizer state는 기존 policy checkpoint state에 함께
저장·복원된다. 실제 값은 [dataset report](0814_dataset_report.md)에 기록했다.

## Config

- task: `diffusion_policy/config/task/umi_dual_ft.yaml`
- train: `diffusion_policy/config/train_diffusion_unet_timm_umi_dual_ft_workspace.yaml`
- prediction horizon: 16
- actual action frequency: 19.980011909 Hz
- default execution `n_action_steps`: 2
- default replanning interval: 100.100 ms
- supported deployment ablation: 1, 2, 4, 8

Prediction과 execution은 별도 config다. 학습 target은 항상 `[B,16,10]`이고 배포가
prediction 앞부분 중 몇 step을 실제 실행할지만 `execution.n_action_steps`로 정한다.

```bash
python eval_real_indy_rg2.py ... --steps_per_inference 1
python eval_real_indy_rg2.py ... --steps_per_inference 2
python eval_real_indy_rg2.py ... --steps_per_inference 4
python eval_real_indy_rg2.py ... --steps_per_inference 8
```

CLI 값을 생략하면 새 checkpoint의 `execution.n_action_steps`를 사용한다. 실시간
`UmiEnv`는 RGB anchor 이하의 raw RG2-FT history를 직접 만들고 매 policy
iteration에 left/right latest timestamp와 age를 출력한다.

## 주요 파일

| 역할 | 파일 |
|---|---|
| nested ZIP read-only open | `diffusion_policy/common/nested_zarr.py` |
| JPEG-XL compatibility | `diffusion_policy/codecs/imagecodecs_numcodecs.py` |
| causal multirate dataset | `diffusion_policy/dataset/umi_dual_ft_dataset.py` |
| dual causal encoders/fusion | `diffusion_policy/model/vision/dual_ft_obs_encoder.py` |
| live causal history | `umi/real_world/rg2ft_obs.py`, `umi/real_world/umi_env.py` |
| deployment interface | `eval_real_indy_rg2.py` |
| reproducible audit | `scripts/inspect_dual_ft_multirate.py` |

## 검증

실행한 필수 검증:

- 원본 ZIP SHA-256 전/후 동일
- detected nested prefix read-only open
- worker 2개 DataLoader에서 process-local lazy open
- JPEG-XL 3 episode × 시작/중간/끝 decode
- future F/T가 earlier output/anchor에 선택되지 않음
- first F/T 전 anchor가 제외됨
- train-split left/right independent normalization
- legacy RGB+pose encoder forward
- RGB+pose+dual-F/T forward/backward
- checkpoint save/reload
- small-model one-batch AdamW optimizer smoke step
- actual ZIP + production 800D CLIP-ViT policy one-batch loss/backward

전체 장시간 training 및 실제 robot motion은 시작하지 않았다.

재현 명령:

```bash
PYTHONPATH=. python scripts/inspect_dual_ft_multirate.py
PYTHONPATH=. pytest -q tests/test_rg2ft_obs.py tests/test_dual_ft_policy.py
PYTHONPATH=. pytest -q tests/test_dual_ft_dataset_integration.py
```

## Deployment blocker

Recorder와 deployment parser는 동일 register order, force `/10`, torque `/100`을
사용하여 software convention은 일치한다. 다만 native sensor frame의 물리 축을
world/tool frame으로 해석하는 manufacturer/mounting 문서는 저장소에서 확정하지
못했다. 임의 transform은 추가하지 않았다. 센서 재장착이나 교체가 있었다면
left/right mounting orientation 확인 전 실제 motion deployment를 중단해야 한다.
