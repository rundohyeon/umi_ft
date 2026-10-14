# RGB4 slip action 모델 실물 로봇 테스트

실물용 진입점은 저장소 루트의 `eval_real_slip_action.py`다. 기존
`eval_real_indy_rg2.py`는 786차원 pose 입력 체크포인트용이므로 이 768차원
RGB4/paired-F/T/no-TCP 체크포인트에 사용하지 않는다.

## 필요한 모델 파일

학습을 수행한 원격 컴퓨터의 원본 경로는 다음 두 개다.

```text
/home/metafarmers/dkim/umi_ft/outputs/slip_action/rgb4_paired_ft_no_tcp/checkpoints/epoch=0008-val_loss=0.02815.ckpt
/home/metafarmers/dkim/umi_ft/outputs/slip_context_sequence/best.pt
```

첫 번째는 action policy의 epoch 9 EMA, normalizer, frozen CLIP 가중치를 포함한다.
두 번째는 action 학습 때 고정해서 사용한 **RGB4/0.5초 context classifier**다.
Action 체크포인트가 저장한 SHA-256과 context 파일의 SHA-256이 다르면 실물
스크립트가 실행을 거부한다. 최신 RGB8 `best.pt`로 바꾸면 안 된다.

현재 로봇 컴퓨터에는 두 파일을 다음 위치에 배치했다. 실물 실행 코드의 기본값도
이 경로를 사용한다.

```text
/home/idim/dkim/umi_ft/data/contextFT-4/epoch=0008-val_loss=0.02815.ckpt
/home/idim/dkim/umi_ft/data/contextFT-4/best.pt
```

확인한 SHA-256은 다음과 같다. Action 체크포인트 내부에도 context SHA가
`65f27d...fc67`로 기록되어 있어 두 파일의 결합이 일치한다.

```text
action  4d754330b165590ce5f4eca5df4f46effcb3cda8ecd990949bc0635e4c19c3df
context 65f27d843596c09f7a71172e641af5006334208ca42e8fea9636e0cf79e4fc67
```

`outputs/slip_action/context_rgb4.npz`와 학습 Zarr/라벨은 오프라인 평가에는
필요하지만 실시간 추론에는 필요하지 않다. 실시간 context는 카메라와 RG2-FT에서
직접 계산한다.

## 기존 배포 방식으로 실행

새 진입 스크립트는 `deploy_real_slip_action.sh`다. 기존 배포 명령의 환경변수와
underscore 옵션을 그대로 받는다. 다음 명령은 Docker 안의
`/ros2_ws/src/indy_umi_rg_ft`를 기준으로 실제 motion을 활성화한다.

```bash
PYTHON_BIN=python3 \
RG2_CHECKPOINT=/ros2_ws/src/indy_umi_rg_ft/data/contextFT-4/epoch=0008-val_loss=0.02815.ckpt \
VALVE_CLASSIFIER_CHECKPOINT=/ros2_ws/src/indy_umi_rg_ft/data/contextFT-4/best.pt \
MATCH_DATASET=/ros2_ws/src/indy_umi_rg_ft/data/context-3/dataset.zarr.zip \
EVAL_OUTPUT_DIR=/ros2_ws/src/indy_umi_rg_ft/data/eval_contextFT4 \
RG2_ENABLE_MOTION=1 \
ACTION_SCALE=1.0 \
RG2_FT_MAX_AGE_SEC=0.020 \
RG2_LATEST_FT_MAX_AGE_SEC=0.050 \
CONTEXT_MOMENTUM_PREVIOUS_WEIGHT=0.5 \
./deploy_real_slip_action.sh \
  --steps_per_inference 4 \
  --show_policy_image \
  --policy_input_audit \
  --print_policy_output
```

`CONTEXT_MOMENTUM_PREVIOUS_WEIGHT=0.5`는 context 모델의 최종 확률을 10 Hz
classifier cadence에서 `0.5 * 이전 사용값 + 0.5 * 현재 모델값`으로 섞는다.
causal context history가 초기화되면 이 값도 즉시 초기화된다. 실행 로그의
`model_probabilities`는 관성 적용 전 값이고 `probabilities`는 action 모델에 실제로
들어간 값이다. `0`으로 설정하면 추가 context 관성을 끈다.

터미널에 `EXECUTE`를 입력하는 단계는 없다. 모델 워밍업이 끝나면
`policy_input` 창에 **학습 Zarr 첫 프레임 / 현재 policy 입력 / 50:50 overlap**이
나온다. 기존 keyboard Teleop으로 위치를 맞춘 뒤 `c`를 누르면 새 센서 구간을
모으고 정책이 시작된다. OpenCV 창과 Docker 터미널 입력을 모두 받는다.

- `a/d`: x+/x-
- `s/w`: y+/y-
- `e/q`: z+/z-
- `j/l`: roll-/+
- `i/k`: pitch+/-
- `u/o`: yaw-/+
- `z/x`: gripper close/open
- `n/b`: 다음/이전 비교 episode
- `c`: Teleop 종료 후 정책 시작
- `Esc`: 종료

기본 한 번 입력은 위치 5 mm, 회전 0.05 rad, gripper 폭 10 mm다. 각 Teleop
명령에도 F/T freshness·overload 및 workspace/이동량 안전 검사를 적용한다.
정책 실행 중에는 OpenCV 창에서 `s`, `q`, 또는 `Esc`로 중단할 수 있다.

`RG2_ENABLE_MOTION=0`이면 같은 절차로 출력만 확인하고 waypoint는 보내지 않는다.
`AUTO_START_POLICY=1`을 추가하면 `c` 대기 없이 시작한다. `MATCH_DATASET`은 화면
정렬용으로만 쓰며 모델 입력에는 들어가지 않는다. 각 실행 로그는
`EVAL_OUTPUT_DIR/<실행시각>/`에 따로 저장된다.

시작 시 F/T 처리는 두 단계다.

1. 최근 25개의 100 Hz raw sample 평균으로 software tare한다. 기본 활성화다.
2. 설정 파일 기준 200개 sample로 residual startup bias를 검사하고 제거한다.

두 단계 동안 양쪽 손가락을 무부하로 두고 로봇과 그리퍼를 움직이지 않아야 한다.
기존 명령과의 호환성을 위해 `RG2_FT_MAX_AGE_SEC=0.020`은 받지만, 이 모델의
causal RGB/F/T 결합 한도는 학습 체크포인트 계약인 12 ms로 고정되어 있어 런타임에
늘리지 않는다. `RG2_LATEST_FT_MAX_AGE_SEC=0.050`은 Teleop과 policy 명령을 보내기
직전에 읽은 최신 F/T sample의 안전 한도다. 일시적으로 이 한도를 넘은 Teleop
입력은 해당 명령만 거부하고 Teleop 화면은 유지한다.

## 하드웨어 없이 모델 확인

두 모델의 구조와 SHA 결합을 하드웨어 없이 먼저 확인한다.

```bash
cd /home/idim/dkim/umi_ft
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
python eval_real_slip_action.py \
  --inspect --device cuda:0
```

파이썬 진입점을 직접 실행할 때 기본값은 카메라, Indy, RG2-FT에 연결하고 F/T
bias를 측정한 뒤 한 번 추론하는 plan-only 모드다. 후보 waypoint와 context는
저장하지만 로봇 명령은 보내지 않는다.
모델 로드 직후 context와 action을 한 번 합성 입력으로 워밍업한 다음 새로운
약 0.9초 센서 구간을 모은다. 따라서 첫 CUDA 실행 지연이 실물 입력의 age에
포함되지 않는다. 워밍업 횟수는 `--warmup-runs`로 바꿀 수 있다.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
python eval_real_slip_action.py \
  --robot-config example/eval_robots_config_indy_rg2.yaml \
  --device cuda:0 --max-cycles 1 --save-inputs
```

카메라의 보정된 촬영 시각부터 명령 직전까지의 총 age는 기본 400 ms를 넘으면
폐기한다. 이 가운데 설정된 정상 카메라 지연 125 ms를 뺀 queue/context/action
처리 age는 별도로 250 ms를 넘으면 폐기한다. wrapper에서는 각각
`RG2_MAX_RESULT_AGE_SEC`, `RG2_MAX_PROCESSING_AGE_SEC`로 설정한다.
`cycles.jsonl`의 `sensor_age_before_s`, `context_prepare_s`,
`action_inference_s`, `result_age_s`, `processing_age_s`로 어느 단계가 느린지
확인할 수 있다.

waypoint 안전검사에서 중단되면 거부된 입력과 전체 예측을
`rejected_input_<cycle>.npz`, `rejected_candidate_<cycle>.json`으로 남긴다.
최근 실행의 RGB, context 확률, F/T, waypoint 안전 여유와 종료 원인은 다음처럼
validation 영상으로 만들 수 있다.

```bash
python3 render_slip_action_validation.py
```

영상은 기본적으로 `exports/slip_action_validation/`에 생성된다.

Action 추론 중 지나간 카메라 frame은 다음 cycle에서 체크포인트의 0.1001초
주기에 맞춰 context 분류를 순서대로 보충한다. 따라서 동기식 action 추론이
10 Hz보다 느려도 이전 observation 확률 이력이 매 cycle 초기화되지 않는다.
터미널의 `context_history=<현재>/20`, `context_updates=<이번 보충 횟수>`로
누적 여부를 확인한다.

파이썬 진입점으로 직접 실제 waypoint를 보내려면 `--execute`를 사용한다.
문자열 확인은 없으며, 아래 예시는 `policy_input` 창에서 `c`를 눌렀을 때 한
cycle만 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
python eval_real_slip_action.py \
  --robot-config example/eval_robots_config_indy_rg2.yaml \
  --device cuda:0 --max-cycles 1 --action-scale 0.2 \
  --freeze-rotation --execute --wait-for-start-key --show-policy-image \
  --match-dataset data/context-3/dataset.zarr.zip
```

검사 후 전체 pose를 사용하는 연속 테스트는 `--freeze-rotation`을 빼고
`--action-scale 1.0 --max-cycles 100`처럼 지정한다. 출력은
`outputs/slip_action_real/<실행시각>/`에 저장된다. `--output`을 지정하면 그 경로
아래에 실행시각 폴더가 생성된다. `run.json`에는 모델 hash,
F/T bias, 안전 제한, 종료 상태가 있고 `cycles.jsonl`에는 실제 context 확률,
RGB/F/T 시각, `[16,11]` 원출력, 로봇 좌표 waypoint가 있다.

## 실시간 입력과 명령 변환

한 cycle에서 다음 순서를 사용한다.

1. 전처리가 끝난 60 Hz RGB 중 `[t-0.5, t-0.333, t-0.167, t]` 네 장을 선택한다.
2. 각 RGB 시각 이전의 마지막 RG2-FT 샘플부터 과거 41개를 좌우 각각 붙인다.
3. 같은 RGB/F/T 묶음으로 context 확률을 계산해
   `[approach,error1,error2,turning,valid]`를 만든다.
4. Action policy가 `[16,11]`을 예측한다. 0번은 현재 상태이므로 명령에서 제외하고
   1번 이후만 현재 실측 TCP에 대한 상대 pose로 변환한다.
5. 11번째 force 값은 RG2 force register에 쓰지 않고, 예측 width를 최대 1 mm
   보정하는 기준값으로만 쓴다.
6. RGB/F/T age, 최신 F/T overload, waypoint 간 이동/회전/width, 최소 높이와
   exclusion sphere 검사를 모두 통과한 명령만 제출한다.
