# Valve State Classifier v4 — portable inference bundle

이 폴더만 다른 컴퓨터로 복사하면 원래 학습 repository나 dataset 없이 추론할 수 있다.

## 폴더 구성

```text
valve_state_classifier_v4/
├── model/final.pt             # epoch 23 checkpoint
├── valve_state_classifier.py  # 모델 구조 + window/streaming inference
├── example_streaming.py       # 실시간 연결 예제
├── infer_npz.py               # NPZ 녹화 데이터 일괄 추론
├── verify_install.py          # checksum/load/forward 검증
├── README_FOR_CODEX.md        # 다른 컴퓨터 Codex용 통합 지침
├── MODEL_CARD.md              # 입력 규약, 구조, 성능, 제한사항
├── requirements.txt
├── environment.yml
└── manifest.sha256
```

다른 컴퓨터에서 Codex로 기존 모델에 합칠 때는 Codex에게 먼저 `README_FOR_CODEX.md` 전체를 읽도록 지시한다. 해당 문서에는 입력 단위, 센서 순서, causal timestamp, episode reset, warm-up, 통합 완료 조건이 정리되어 있다.

## 1. 환경 설치

CUDA 11.8 환경 예시:

```bash
conda env create -f environment.yml
conda activate valve-state-v4
```

이미 PyTorch가 설치된 컴퓨터에서는 해당 CUDA에 맞는 PyTorch를 유지하고 `numpy`와 `torchvision` 버전만 확인해도 된다. 원 학습 환경은 Python 3.9.18, PyTorch 2.1.0, torchvision 0.16.0, NumPy 1.24.4이다.

## 2. 복사 직후 검증

```bash
cd valve_state_classifier_v4
python verify_install.py --device cuda
```

GPU가 없으면 `--device cpu`를 사용한다. 아래 네 줄이 나오면 bundle이 정상이다.

```text
checkpoint SHA-256: OK
strict model load: OK
forward pass: OK
device: ...
```

전체 파일 checksum은 다음과 같이 확인한다.

```bash
sha256sum -c manifest.sha256
```

## 3. 실시간 모델에 연결

```python
from pathlib import Path
from valve_state_classifier import ValveStateRuntime

runtime = ValveStateRuntime(Path("model/final.pt"), device="cuda")

# 100 Hz F/T callback: 수신되는 모든 sample을 시간순으로 추가
runtime.append_wrench(ft_timestamp_s, wrench_12d)

# 약 60 Hz RGB/robot callback: 같은 시각까지의 F/T를 먼저 추가한 후 호출
result = runtime.predict(
    timestamp_s=rgb_timestamp_s,
    rgb=rgb_224x224_uint8,               # RGB 순서, BGR 아님
    position_m=robot_position_xyz,
    rotation_axis_angle_rad=robot_rotvec,
    gripper_width_m=gripper_width,
)

print(result.phase, result.confidence)
print(result.phase_probabilities)
print(result.error_reason)
```

새 episode가 시작되면 반드시 다음을 호출한다.

```python
runtime.reset()
```

모델은 16개 RGB 시점을 stride 4로 보므로 약 60개의 원본 RGB frame이 쌓여야 정상 window가 완성된다. 그 전에도 training과 동일하게 첫 frame을 반복해 예측하지만 `result.warmed_up`은 `False`이다.

## 4. 센서 입력 규약

- RGB: `(224,224,3)` uint8, **RGB channel order**, 약 59.94/60 Hz.
- Position: `(3,)`, metre.
- Rotation: `(3,)`, axis-angle/rotation-vector, radian.
- Gripper width: scalar, metre.
- Wrench: `(12,)`, `[L_Fx,L_Fy,L_Fz,L_Tx,L_Ty,L_Tz,R_Fx,R_Fy,R_Fz,R_Tx,R_Ty,R_Tz]`, N/Nm, 약 100 Hz.
- 모든 timestamp는 같은 monotonic clock 기준의 초 단위이고 strictly increasing이어야 한다.
- F/T sample을 RGB timestamp보다 미래에서 가져오면 안 된다.
- camera crop, 색 순서, robot frame, F/T 채널 순서와 calibration이 학습 때와 같아야 한다.

## 5. NPZ 일괄 추론

입력 NPZ key:

```text
rgb                         (N,224,224,3) uint8
rgb_timestamp_s             (N,)
position_m                  (N,3)
rotation_axis_angle_rad     (N,3)
gripper_width_m             (N,) or (N,1)
wrench_12d                  (M,12)
wrench_timestamp_s          (M,)
```

실행:

```bash
python infer_npz.py recording.npz --device cuda --output predictions.csv
```

`predictions.csv`에는 phase/reason ID, 이름, confidence, 각 class probability가 저장된다.
