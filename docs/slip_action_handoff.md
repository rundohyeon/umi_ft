# Slip action: 다른 컴퓨터에서 실행하기

2026-10-12 기준. **현재 검증한 것은 저장 데이터에 대한 오프라인 action 평가다.**
`eval_slip_action.py`는 로봇에 연결하지 않는다. 이 문서는 현재 RGB4/paired-F/T/no-TCP
모델용이며 README의 과거 786-D/TCP 또는 A/T/R/E 모델 배포 명령과 혼용하지 않는다.

## 1. Git으로 가져올 코드

저장소 전체를 가져온다. `eval_slip_action.py` 하나만 복사하면 안 된다.
이 파일은 기존 `eval_dual_ft_offline.py`와 `diffusion_policy/`, `umi/`의 공통 코드를 사용한다.
원격에 필요한 변경을 포함한 commit이 있어야 하며, 새 컴퓨터에서는 fetch 후 그
branch/commit을 checkout해야 한다. **fetch만으로 현재 작업 파일이 바뀌지는 않는다.**
원본 컴퓨터의 미커밋 변경과 untracked 파일은 Git으로 전달되지 않는다.

특히 아래 파일을 함께 포함해야 한다:

- `eval_slip_action.py`
- `diffusion_policy/dataset/slip_action_dataset.py`
- `diffusion_policy/model/vision/dual_ft_obs_encoder.py`의 `per_rgb` 모드 변경
- `diffusion_policy/config/train_slip_action_rgb4_workspace.yaml`
- `diffusion_policy/config/task/slip_action_rgb4_paired_ft.yaml`
- `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`의 학습 변경
- `tools/prepare_slip_action_context.py`, 관련 tests 및 이 문서

새 컴퓨터에서 저장소 루트로 이동해 확인한다:

```bash
git rev-parse HEAD
git status --short
git ls-files --error-unmatch eval_slip_action.py \
  diffusion_policy/dataset/slip_action_dataset.py \
  diffusion_policy/config/task/slip_action_rgb4_paired_ft.yaml \
  docs/slip_action_handoff.md
```

## 2. 별도로 가져올 모델과 데이터

`.gitignore`에서 `outputs/`, `*.ckpt`, `*.pt`, `*.npz`, Zarr 데이터를 제외한다.
아래 **6개 항목**은 Git clone/fetch로 받지 못하므로 별도로 복사한다.
`dataset_force_sidecar.zarr`는 파일 하나가 아니라 내부 전체가 필요한 디렉터리다.

| 항목 | 원본 위치 | 용도 |
|---|---|---|
| Action 모델 | `outputs/slip_action/rgb4_paired_ft_no_tcp/checkpoints/epoch=0008-val_loss=0.02815.ckpt` | epoch 9의 EMA와 학습 정규화 포함, 약 2.07 GB |
| Context 모델 | `outputs/slip_context_sequence/best.pt` | RGB4/0.5초 + history20/2초, 약 349 MB |
| Context 예측 | `outputs/slip_action/context_rgb4.npz` | 학습 때 사용한 확률 및 split, 약 286 KB |
| RGB와 시연 pose/width | `/home/metafarmers/dkim/data/slip_processed_context/dataset.zarr.zip` | 약 4.99 GB |
| Native F/T | `/home/metafarmers/dkim/data/slip_processed_context/dataset_force_sidecar.zarr/` | 좌우 6축 및 timestamp |
| 수동 label | `/home/metafarmers/dkim/data/slip_processed_context/state_labels_v2.npz` | ignore 제외, 평가 구간·클래스 결정 |

`best.pt`는 **context 모델**이고 action 모델은 위 `.ckpt`이다.
`latest.ckpt`는 epoch 100이므로 best 대신 사용하지 않는다.
RGB8 context 모델은 이 action 모델을 학습할 때 사용한 모델이 아니므로 바꿔 끼우지 않는다.
Context `.pt`는 오프라인 eval에서 SHA256 확인에 사용하고, 실제 입력 확률은 `.npz`에서 읽는다.
CLIP 가중치는 action 체크포인트에 포함된다. 이 평가에는 Qwen, Hugging Face 모델 캐시,
`outputs/slip_clip_features` 및 다른 epoch 체크포인트가 필요 없다.

### 데이터 경로와 수정 시각

**현재 eval은 임의 경로로 옮긴 데이터에 바로 대응하지 않는다.** 저장된 설정은 위 데이터의
절대 경로를 사용한다. Fingerprint에는 RGB ZIP의 `Path.resolve()` 경로, 파일 크기,
나노초 단위 수정 시각, label 내용과 RGB/F/T timestamp·wrench가 들어간다.

- 코드 저장소 자체는 다른 위치에 두어도 된다. 모델 3개는 그 저장소의 `outputs/` 아래에 둔다.
- 데이터 3개는 `/home/metafarmers/dkim/data/slip_processed_context/`에서 접근 가능해야 한다.
  원래 절대 경로를 실제로 만들거나 그 경로에 bind mount한다. 다른 위치를 가리키는
  symlink는 `resolve()` 결과가 달라지므로 해결책이 아니다.
- ZIP을 재압축하거나 수정 시각이 사라지는 복사를 하지 않는다. 아래 PAX tar는
  나노초 수정 시각을 보존한다. 대상 파일시스템도 이를 지원해야 한다.
- 다른 데이터 경로를 지원하려면 검증된 데이터 이식 기능을 추가해야 한다.
  현재 CLI에는 `--dataset`/`--data-root` 옵션이 없다. YAML만 바꿔도 저장된 checkpoint의
  설정은 바뀌지 않는다. Fingerprint/SHA 검사를 삭제하거나 metadata를 임의로 고치지 않는다.

원본 컴퓨터에서 필요한 산출물만 묶는 예시다. `/tmp/slip_action_epoch9.tar`는 새 파일명으로
선택하고, 추가로 약 7.5 GB 이상의 여유 공간을 확보한다. 이 명령은 코드를 묶지 않는다.

```bash
cd /home/metafarmers/dkim
tar --format=pax -cf /tmp/slip_action_epoch9.tar \
  umi_ft/outputs/slip_action/rgb4_paired_ft_no_tcp/checkpoints/epoch=0008-val_loss=0.02815.ckpt \
  umi_ft/outputs/slip_context_sequence/best.pt \
  umi_ft/outputs/slip_action/context_rgb4.npz \
  data/slip_processed_context/dataset.zarr.zip \
  data/slip_processed_context/dataset_force_sidecar.zarr \
  data/slip_processed_context/state_labels_v2.npz
sha256sum /tmp/slip_action_epoch9.tar
```

새 컴퓨터로 archive를 복사한 뒤 SHA256을 비교한다. 원래 디렉터리에 쓰기 권한이 있고,
대상 산출물이 아직 없는 경우 다음처럼 풀 수 있다. `--keep-old-files`는 기존 파일을 덮어쓰지 않는다.

```bash
mkdir -p /home/metafarmers/dkim
tar --keep-old-files --no-same-owner -xf /path/to/slip_action_epoch9.tar \
  -C /home/metafarmers/dkim
```

저장소를 다른 위치에 clone했다면 추출한 `umi_ft/outputs/`의 모델 3개를 그 저장소 아래에
배치한다. 데이터 디렉터리는 위 절대 경로를 유지한다. `--context-cache`와
`--context-checkpoint`로 context 파일 2개의 경로만 별도로 지정할 수도 있다.

## 3. Python 환경과 사전 확인

저장소 루트에서 `conda_environment.yaml`로 환경을 만들고 활성화한다.
기존 `umi` 환경이 있으면 재생성하지 않고 버전을 확인한다. 시스템 의존성 설치는
[README Installation](../README.md#️-installation)을 참고한다.

```bash
conda env create -f conda_environment.yaml
conda activate umi
python -c "import torch, timm, diffusers; print(torch.__version__, timm.__version__, diffusers.__version__); print('CUDA:', torch.cuda.is_available())"
python eval_slip_action.py --help
```

원본에서 검증한 주요 버전은 Python 3.9, PyTorch 2.1.0+cu121,
torchvision 0.16.0+cu121, timm 0.9.7, diffusers 0.18.2,
huggingface-hub 0.25.2, NumPy 1.24.4, Zarr 2.16.1, numcodecs 0.11.0,
imagecodecs 2023.9.18이다. `conda_environment.yaml`은 전체 lockfile이 아니므로
새 환경에서 아래 진단 평가까지 통과하는지 확인한다. 특히 이 diffusers 버전에는
`huggingface-hub<0.26`이 필요하다. Qwen용 `.venv`는 사용하지 않는다.

## 4. GPU 2에서 빠른 검사 → 전체 평가

아래 명령은 clone한 **저장소 루트**, 활성화된 `umi` 환경에서 실행한다.
물리 GPU 2가 프로세스 안에서는 `cuda:0`이다. 새 컴퓨터의 GPU 번호가 다르면
`nvidia-smi`로 확인해 `CUDA_VISIBLE_DEVICES`만 조정한다.

먼저 64개 anchor로 모델 복원, 데이터 identity, 이미지 디코딩과 추론을 확인한다:

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python eval_slip_action.py \
  --checkpoint outputs/slip_action/rgb4_paired_ft_no_tcp/checkpoints/epoch=0008-val_loss=0.02815.ckpt \
  --weights ema_model --split test --device cuda:0 \
  --batch-size 32 --anchor-stride 6 --max-samples 64
```

전체 test 평가에서는 `--max-samples`를 제거한다:

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python eval_slip_action.py \
  --checkpoint outputs/slip_action/rgb4_paired_ft_no_tcp/checkpoints/epoch=0008-val_loss=0.02815.ckpt \
  --weights ema_model --split test --device cuda:0 \
  --batch-size 32 --anchor-stride 6
```

같은 원본 데이터라면 전체 결과는 **44개 episode, 2,495개 anchor**다.
원본에서 미래 step 1–15의 평균 오차는 위치 18.29 mm, 회전 6.19도,
width 2.58 mm, force 0.594 N이었다. 장치와 라이브러리가 달라지면 숫자는 조금 달라질 수 있다.
메모리 부족 시 batch를 줄일 수 있지만 diffusion 샘플이 달라져 완전히 같은 수치가 나오지는 않는다.

매번 체크포인트 상위 run 디렉터리에 `eval_epoch9_test_날짜_시간/`을 새로 만든다.
`report.md`, `metrics.json`, `errors_by_horizon.png`, phase/episode/horizon별 CSV,
`predictions.npz`를 확인한다. `--output`을 쓰면 아직 없는 폴더를 지정한다.
`--episode 291`은 test 중 원래 EP291만 선택한다.

## 5. 실물 로봇 연결 작업을 이어받는 사람에게

지원 현황: action 오프라인 평가 완료, slip context의 streaming runtime 존재,
**이 action 모델용 실물 로봇 연결은 미완료**다. 기존 `deploy_real_indy_rg2.sh`나
`eval_real_indy_rg2*.py`에 이 체크포인트만 넣어 실행하지 않는다.

구현 시 기준은 [action 입력·출력 계약](slip_action_training.md)과
[SlipTemporalContextRuntime 안내](slip_temporal_context.md)다:

- Action 입력은 RGB 4장/0.5초, 각 RGB 이전 좌우 F/T `[41,6]`, context `[1,5]`이다.
  Context 클래스 순서는 `approach, error1, error2, turning`, 마지막 값은 validity다.
- Context는 위 RGB4 체크포인트의 설정을 복원하며 과거 관측 확률 20개/약 2초를 사용한다.
  최신 학습 YAML의 RGB8 설정으로 덮어쓰지 않는다.
- 센서 버퍼는 약 0.9초가 필요하고 실물 F/T에는 시작 시 정적 bias 보정이 필요하다.
  TCP는 모델 입력에 넣지 않는다. 현재 pose는 상대 action을 로봇 목표로 변환할 때 사용한다.
- 출력은 `[16,11]`: 상대 xyz 3 + rotation6D 6 + width(m) 1 + force reference(N) 1.
  Step 0은 현재 시점, 나머지는 약 50 ms 간격이다. Force는 width feedback의 기준값이다.
- 현재 오프라인 오류와 policy batch 시간만으로 실물 성공률이나 전체 시스템의
  실시간 동작을 확인한 것은 아니다. 센서 동기화·context·action·제어 연결을 별도로 검증해야 한다.
