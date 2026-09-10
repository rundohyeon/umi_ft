# `best_context.pt` 이식 / 추론 명세

대상 파일: `best_context.pt`  
체크포인트 스키마: `umi_causal_context_observer_v2_4state`  
출력 클래스 순서: `0=approach`, `1=turning`, `2=recovery`, `3=error`

이 파일은 로봇 동작을 생성하는 diffusion policy가 아니라, 현재 작업 맥락을
분류하는 **causal 4-state observer**다. 미래 프레임 또는 미래 F/T 샘플은
입력으로 사용하지 않는다.

## 새 컴퓨터로 함께 복사할 것

체크포인트만으로는 모델 클래스를 만들 수 없으므로, 최소한 아래 파일/폴더를
같은 프로젝트 구조로 옮긴다.

```text
best_context.pt
config/train_umi_context_manual_v12_4state_safe_single_gpu.yaml
config/task/umi_context_manual_v12_4state.yaml
force_cnn_umi/
diffusion_policy/
umi/
infer_deploy_context_logs.py                 # CSV/PNG 로그용 권장 실행기
```

Python, PyTorch, CUDA, `timm`, `hydra-core`, `omegaconf`, `numpy`, `opencv-python`,
`einops`가 필요하다. 학습 환경의 `umi` conda 환경과 같은 PyTorch/CUDA 조합을
권장한다.

`best_context.pt`에는 모델 가중치와 normalizer 통계가 포함되어 있다. 다만
`LinearNormalizer`는 입력 키별 버퍼를 동적으로 등록하므로, 로드 전에 아래와
동일한 입력 스키마로 모델을 instantiate 해야 한다. 제공된 추론기는 학습
dataset의 normalizer로 해당 버퍼를 먼저 만들고 checkpoint를 로드한다.

원본 학습 dataset 없이 완전히 독립 실행하려면, 동일한 키/shape의 normalizer
버퍼를 먼저 생성한 뒤 checkpoint의 `model_state_dict`를 strict load하는 작은
로더를 사용해야 한다. 따라서 가장 안전한 이식 방법은 위 프로젝트 코드와
config를 같이 옮기는 것이다.

## 모델 구조

```text
camera0_rgb (2 x 3 x 224 x 224) ─┐
TCP relative 10-D (2 x 10)       ├─ frozen CLIP ViT-B/16 RGB + proprio encoder
                                │
F/T history (2 x 50 x 12) ──────┴─ causal 1-D Force CNN (32 → 64 → 128)
                                             │
                               encoded features: 1,812-D
                                             │
                         Linear(1812,256) → SiLU → Dropout(0.10)
                                             │
                                      Linear(256,4) → logits
                                             │
                                    softmax → A/T/R/E probabilities
```

- Vision backbone: `vit_base_patch16_clip_224.openai`; 학습 당시 frozen.
- Force CNN: causal Conv1D, input 12 채널, 128-D feature, gate의 초기값 0.10.
- Context head: `1812 → 256 → 4`.
- checkpoint에는 모델 텐서 214개, optimizer state, epoch, validation metadata도
  들어 있다. 추론에는 `model_state_dict`, `context_phase_names`만 필요하다.

## 입력 텐서 계약

모든 텐서는 batch-first `float32`다. 배치 크기를 `B`라 하면 `obs`는 아래
dictionary여야 한다.

| 키 | shape | 의미 / 전처리 |
|---|---:|---|
| `camera0_rgb` | `(B, 2, 3, 224, 224)` | 현재와 과거 관측 RGB. `RGB`, 범위 `0..1`, channel-first. 모델 내부에서 ImageNet normalization을 적용한다. |
| `robot0_eef_pos` | `(B, 2, 3)` | TCP 위치. **절대 base 좌표를 그대로 넣지 말고**, 두 프레임 중 최신 TCP를 기준으로 한 상대 pose의 XYZ를 사용한다. 단위 m. |
| `robot0_eef_rot_axis_angle` | `(B, 2, 6)` | TCP 회전. 로봇이 준 axis-angle/rotvec 3-D를 pose matrix로 변환 후, 최신 TCP 기준 상대 회전을 6-D rotation representation으로 변환한다. 관절각도나 IMU gyro가 아니다. |
| `robot0_gripper_width` | `(B, 2, 1)` | 그리퍼 폭, m. |
| `robot0_ft_history` | `(B, 2, 50, 12)` | 각 RGB 시점 **이하**의 가장 최근 F/T 50개. 채널 순서: `left Fx,Fy,Fz,Tx,Ty,Tz,right Fx,Fy,Fz,Tx,Ty,Tz`; 힘 N, 토크 Nm. 미래 보간 금지. |
| `robot0_ft_history_valid` | `(B, 2, 50, 1)` | F/T history 유효 마스크. 시작 구간의 왼쪽 zero-padding은 `0`, 실제 샘플은 `1`. |

### 포즈와 IMU에 관한 핵심

이 모델은 물리 IMU의 가속도/자이로를 입력으로 사용하지 않는다. 로봇
controller/FK가 제공한 TCP `(x,y,z,rotation-vector)`에서 최근 pose 변화량을
만들어 쓴다. 2-frame horizon에서 현재 TCP가 기준 pose가 되므로 최신 pose의
상대값은 0이고, 약 3 프레임 전 TCP와의 상대 변위/회전이 최근 축 변화 정보를
제공한다.

### F/T 스케일

원시 입력은 위 단위의 물리값으로 제공한다. dataset normalizer와 model state가
학습 시 통계를 적용한다. 별도 전처리에서 임의로 N/Nm를 다시 곱하거나 나누지
말아야 한다. 학습 데이터 생성 기준의 nominal scale은 force `10 N`, torque
`0.1 Nm`였다.

## 출력 계약

```python
result = model.predict_context(obs)
result['context_logits']  # float32, (B, 4)
result['context_prob']    # softmax(logits), (B, 4)
result['context_pred']    # argmax(logits), int64, (B,)
```

`context_pred`는 가장 높은 확률의 클래스일 뿐이다. 특히 상위 확률이 낮거나
2개 이상이 비슷한 전이 구간은 `error`를 실제 안전 인터록 또는 실제 실패
확정으로 사용하면 안 된다. 예: 47.8% error / 34.1% approach는 error가
argmax이지만 확신 있는 error 판정은 아니다.

## PNG + CSV 배포 로그에서의 권장 실행

입력 로그 구조:

```text
logs_root/<episode>/context_inputs/
  context_frames.csv
  context_wrenches.csv
  images/frame_XXXXXXXX.png
```

```bash
cd <project-root>
CUDA_VISIBLE_DEVICES=0 python infer_deploy_context_logs.py \
  --checkpoint /path/to/best_context.pt \
  --logs-root /path/to/logs_root \
  --output-dir /path/to/inference_output \
  --device cuda --batch-size 64
```

출력 파일은 `frame_predictions.csv` (raw 확률 및 argmax),
`predicted_segments_smooth5.csv` (표시용 5-frame smoothing),
`inference_report.json`이다. `pred_raw`가 미래 정보를 사용하지 않은 실제
causal 출력이며, `pred_smooth`는 검수 화면용이므로 제어 입력으로 쓰지 않는다.

## 로드 시 점검 목록

1. `context_phase_names`가 정확히 `approach, turning, recovery, error` 순서인지 확인한다.
2. RGB는 BGR이 아닌 RGB인지, 정확히 224×224인지 확인한다.
3. TCP pose를 현재 TCP 기준 relative 10-D로 변환했는지 확인한다.
4. F/T 12채널 순서와 N/Nm 단위가 일치하는지 확인한다.
5. history mask에서 미래 샘플을 채우지 않았는지 확인한다.
6. `model.eval()`과 `torch.no_grad()`로 추론한다.

