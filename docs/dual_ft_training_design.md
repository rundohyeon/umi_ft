# Dual-F/T-conditioned Diffusion Policy implementation

## 범위

이 구현은 RGB+pose에 left/right native six-axis wrench history를 조건으로 추가하고,
signed grasp-force **measurement label**을 예측하는 position Diffusion Policy다.
이 예측은 bounded proportional controller에서 gripper-width 보정 reference로
사용하며 RG2 force register에 직접 보내지 않는다. Impedance/admittance 또는
direct force-command policy는 아니다.

![Dual F/T-conditioned diffusion policy algorithm](assets/dual_ft_algorithm.svg)

Vision/F/T fusion, proprioception, diffusion timestep embedding과 grasp-force
action은 공개 UMI-FT 구조에 맞춘다. F/T 입력은 이 데이터의 100 Hz, 32 sample,
stride 1 receptive field를 유지한다.

## Pipeline

```text
RGB [B,2,3,224,224]
  -> existing TimmObsEncoder vision path
  -> two visual tokens [B,2,768] --------------------+
                                                      |
left native F/T [B,32,6] -> independent CausalConv -->| left token
right native F/T [B,32,6] -> independent CausalConv ->| right token
                                                      |
  [rgb_old, rgb_anchor, left, right]                  |
  -> TransformerEncoderLayer                          |
  -> flatten [B,4*768] -> Linear -> fused [B,768] ----+

pose-only proprioception (2 * (xyz3 + rotation6d6)) [B,18]
  -> concat -> global condition [B,786]
  -> existing ConditionalUnet1D
     (diffusion timestep embedding 32)
  -> action [B,16,11]
     (xyz3 + rotation6d6 + gripper width1 + grasp force1)
```

각 F/T encoder는 five-stage causal downsampling convolution이다.

```text
6 -> 16 -> 32 -> 64 -> 128 -> 768
kernel=2, stride=2, LeakyReLU
```

32 sample에 kernel 2 / stride 2를 다섯 번 적용할 때는 padding을 넣지 않아
`32 -> 16 -> 8 -> 4 -> 2 -> 1`로 줄인다. 기존의 매-layer left padding은 최종
토큰이 첫 timestep만 보게 만들었으므로 제거했다. 최종 토큰의 gradient가 32개
입력 timestep 모두에 도달하는 테스트가 이를 고정한다.

Left/right encoder는 `share_ft_encoder: false`가 default이고 서로 다른 parameter를
가진다. 양쪽 wrench를 합하거나 평균내지 않으며 common tool/wrist frame으로
변환하지 않는다.

현재 260827 task는 RGB/pose/gripper를 stock `dataset.zarr.zip`에서 읽고, F/T는
`dataset_force_sidecar.zarr/data/wrench_12d`만 읽는다. 이 key는 각 episode의
standing bias만 native 12축에서 제거한 값이다. axis permutation/sign flip/TCP
transform은 모두 금지한다. 11번째 action label은 `wrench_12d`의 측정값을 RGB
timestamp에 선형 정렬해 loader가 생성한다.

```text
0.5 * ((right native Fz - right episode-bias Fz)
     - (left native Fz  - left episode-bias Fz))
```

Dataset loader는 생성 label과 이 식의 전체-sample 최대 오차를 검사한다.

배포 시 width 보정식은 다음과 같다.

```text
F_measured = 0.5 * (right native Fz - left native Fz), after startup bias
error      = F_measured - clip(F_predicted, 0 N, 12 N)
correction = clip(0.0001 m/N * deadband(error, 0.5 N), -0.001 m, 0.001 m)
width_cmd  = clip(width_policy + correction, 0 m, 0.1 m)
```

측정 force가 reference보다 크면 gripper를 조금 열고, 작으면 조금 닫는다.
Controller 구현은 `umi/real_world/grasp_force_width_feedback.py`에 있다. Live
evaluation은 policy 시작 전에 unloaded native F/T sample의 unique timestamp와
stationarity를 검사해 startup bias를 추정한다. Calibration이 실패하거나 bias가
없는 경우에는 actuation 전에 fail-closed한다.

## Dataset sampling

`UmiDualFTDataset`은 explicit allowlist만 읽는다. RGB anchor는 TARGET의 기존
`current` sample에 해당하는 최신 RGB timestamp다. Pose, gripper action label,
grasp-force action label은 같은 timestamp grid이고, F/T는 episode별로 독립
causal lookup한다.

Offline pose는 stock ZIP의 `robot0_eef_pos`와
`robot0_eef_rot_axis_angle`을 그대로 사용하므로 quaternion 변환이 없다. Config의
`xyzw` marker는 live/legacy 7-D pose가 들어오는 경우의 convention을 고정하며,
과거의 잘못된 `wxyz` 해석으로 학습한 checkpoint는 evaluation에서 거부된다.

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

Left/right 각각 6 channel을 별도로 range-normalize한다. Action의 gripper width와
grasping force도 서로 독립적인 train-split 통계로 range-normalize한다. Validation
episode는 통계에 포함하지 않는다. Normalizer state는 policy checkpoint state에
함께 저장·복원된다.

## Optimization

Pretrained 여부는 전체 observation encoder가 아니라 실제 timm ViT에만
적용한다. AdamW parameter group은 다음처럼 서로 겹치지 않게 나뉜다.

- diffusion U-Net: `3e-4`
- pretrained ViT backbone: `3e-5`
- randomly initialized fusion Transformer layer: `1e-4`
- left/right F/T CNN, fusion projection, fusion position embedding과 그 밖의
  신규 observation module: `3e-4`

Warmup/cosine scheduler는 각 base LR에 같은 multiplier를 적용하므로 위 비율은
학습 내내 유지된다. 로그에는 기존 `lr` 외에도 `lr/<group-name>`을 기록한다.

## Config

- base task: `diffusion_policy/config/task/umi_dual_ft.yaml`
- current bias-only task: `diffusion_policy/config/task/umi_dual_ft_260827_bias_only.yaml`
- train: `diffusion_policy/config/train_diffusion_unet_timm_umi_dual_ft_workspace.yaml`
- prediction horizon: 16
- actual action frequency: 19.980011909 Hz
- default execution `n_action_steps`: 2
- default replanning interval: 100.100 ms
- supported deployment ablation: 1, 2, 4, 8

Training config의 default task는 현재 `umi_dual_ft_260827_bias_only`다. Prediction과
execution은 별도 config이며 학습 target은 항상 `[B,16,11]`이다. 11번째
measurement output은 현재 native Fz measurement에서 계산한 force와 비교되어
bounded width correction에만 사용되고 RG2 force register에는 전달되지 않는다.

`execution.n_action_steps`의 1/2/4/8 값은 live scheduling ablation용이다. 실행
단계에서는 선택된 pose, width, force-reference row가 동일한 timestamp를 유지한다.

## 주요 파일

| 역할 | 파일 |
|---|---|
| nested ZIP read-only open | `diffusion_policy/common/nested_zarr.py` |
| JPEG-XL compatibility | `diffusion_policy/codecs/imagecodecs_numcodecs.py` |
| causal multirate dataset | `diffusion_policy/dataset/umi_dual_ft_dataset.py` |
| dual causal encoders/fusion | `diffusion_policy/model/vision/dual_ft_obs_encoder.py` |
| hardware-free checkpoint contract | `diffusion_policy/common/dual_ft_contract.py` |
| offline metrics/provenance | `eval_dual_ft_offline.py` |
| live causal history | `umi/real_world/rg2ft_obs.py`, `umi/real_world/umi_env.py` |
| bounded force-to-width feedback | `umi/real_world/grasp_force_width_feedback.py` |
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
- F/T final token이 32개 timestep 모두를 사용하는 gradient test
- 786-D condition / 11-D action one-batch loss/backward와 checkpoint reload
- 260827 전체 label과 signed native-Fz 식 일치 (최대 오차 약 `9.3e-7 N`)
- 260827 `xyzw` pose를 원본 axis-angle과 전 sample 비교 (최대 회전 오차 `< 2e-7 rad`)
- bias-only dataset 214 episode, train 104,994 / validation 5,778 sample
- checkpoint state/schema/normalizer fail-closed inspection과 offline metric smoke test

장시간 training checkpoint는 생성되었다. 실제 robot motion commissioning은 이
software/code 검증 범위에 포함되지 않았다.

재현 명령:

```bash
PYTHONPATH=. python scripts/inspect_dual_ft_multirate.py
PYTHONPATH=. pytest -q tests/test_rg2ft_obs.py tests/test_dual_ft_policy.py
PYTHONPATH=. pytest -q tests/test_dual_ft_dataset_integration.py
```

## Deployment commissioning boundary

Live evaluator에는 startup bias calibration, causal F/T history, bounded
force-to-width feedback, deadline-aware scheduling, motion/F/T safety guard가 연결되어
있다. `plan_only`는 waypoint 제출을 막지만 robot/camera/gripper worker에는
연결하므로 passive sensor-only mode가 아니다. 실제 장비에서는 README 순서대로
checkpoint inspection과 offline evaluation을 먼저 수행하고, emergency stop을 준비한
상태에서 one-cycle dry-run 및 저속 commissioning을 거쳐야 한다. 11번째 action은
끝까지 width correction reference이며 direct RG2 force command로 사용하지 않는다.
