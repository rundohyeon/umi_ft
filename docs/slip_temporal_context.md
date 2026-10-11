# Slip context: RGB별 F/T + 과거 context 확률

이 모델의 클래스 순서는 `0 approach / 1 error1 / 2 error2 / 3 turning`이다.
`data/slip_processed_context`의 검수 라벨로 학습하며, 기존 284개 데이터의
`approach / turning / recovery / error` 체크포인트와는 다른 모델이다.
기존 `best.pt`를 이 모델의 학습 재개나 policy observer 교체에 사용하지 않는다.

## 입력과 구조

현재 기본 설정은 **RGB 8장/약 1초 + context 확률 10개/약 1초, 추론 약 10 Hz**이다.
59.94 FPS 원본에서 RGB를 고르게 뽑으면 현재 frame 대비
`[60, 51, 43, 34, 26, 17, 9, 0]` 프레임 전이며, 양끝의 실제 간격은 1.001초다.
단순히 연속 8프레임을 사용하는 것이 아니다.
각 이미지마다 그 RGB timestamp 이전의 마지막 native F/T 41샘플을 붙인다.
RGB마다 좌우 각각 `[41,6]`이고 축은 `[Fx,Fy,Fz,Tx,Ty,Tz]`, 단위는 N/Nm이다.
이미 bias 보정된 sensor-frame `wrench_12d`를 사용한다. TCP와 gripper 입력은 없다.

RGB → frozen pretrained CLIP → 128차원 projection을 사용한다. 각 손가락의 F/T는
원본 6축, 과거 5샘플 평균, 그 평균과 5샘플 전 평균의 차이를 만들고 causal CNN을
통과한다. `[41,6]` 원본에서 `[32,18]` 특징을 얻으며 이미지 시점 이후 값을 쓰지 않는다.
여덟 시점의 RGB/왼손/오른손 토큰 24개와 query 1개가 Transformer에 들어간다.

이 관측 모델의 4개 클래스 확률을 약 0.1001초마다 하나씩 기록한다.
과거 10개, 총 1.001초를 64차원 GRU로 요약한다. GRU 입력은 확률 4개와
현재 시점 대비 sample age이며, 현재 확률은 과거 history에 포함하지 않는다.
현재 관측 특징과 GRU 결과로 현재 관측 logits를 보정한다. History가 없으면
관측 모델만으로 출력하고, 확률 history가 가득 찰 때까지 기다릴 필요는 없다.
관측 모델의 AMP 사용 여부와 관계없이 확률 GRU는 FP32로 계산한다
(현재 PyTorch 2.1 CUDA GRUCell의 BF16 지원 제한).

History에는 **보정 전 관측 모델의 확률**을 저장한다. 보정 후 결과를 재입력하는
재귀 구조가 아니므로, 보정 오류가 반복해서 자신을 강화하는 경로를 만들지 않는다.
이 선택은 학습/실시간 모두 동일하다.

기존 `outputs/slip_context_sequence/best.pt`는 RGB 4장/0.5초와 확률 20개/2초 모델이다.
그 체크포인트는 저장된 설정으로 계속 추론할 수 있다. 새 8장 모델은 별도 학습하며
이전 체크포인트를 `--resume`해 구조만 바꾸지 않는다. 이전 설정을 재현하려면
`--config outputs/slip_context_sequence/resolved_config.yaml`을 사용한다.

## 학습과 ignore

- 저장된 `-100`/invalid 프레임은 loss에 포함하지 않는다.
- 입력 RGB와 각 RGB의 41샘플 F/T도 현재 연속 valid 구간 안에 있어야 한다.
  가장 오래된 RGB가 약 1초 전이고 그 이미지의 F/T가 0.4초 더 과거를 보므로,
  첫 valid 라벨 뒤 약 1.4초부터 현재 관측을 만들 수 있다.
- 중간 ignore, episode 경계, 센서 누락 시 과거 context를 연결하지 않는다.
  부족한 history는 mask로 표시한다. Ignore 영상을 다른 라벨로 채우지 않는다.
- 학습 anchor도 6 RGB 프레임마다 하나, 약 10 Hz이다. 모든 훈련/평가 history는
  같은 episode의 과거 관측에서 계산한다. 원본 프레임은 삭제하지 않는다.
- 현재 관측 1개와 과거 관측 최대 10개를 함께 계산한다. 과거 확률은 `detach`해
  실제 클래스 확률 의미를 유지하고, 관측 분류에는 별도 cross-entropy loss를 준다.
  총 loss는 `현재 temporal CE + 0.5 × 관측 sequence CE`이다.
- 정답 클래스는 loss target으로만 사용한다. GRU에는 정답 label이나 one-hot을
  전달하지 않는다. 학습 시 history 일부를 가리고 일부 확률을 교란한다.
- F/T 정규화 통계는 train episode에서 실제 사용 가능한 RGB별 F/T 창으로만
  계산한다. Ignore와 validation/test 측정값은 통계에 들어가지 않는다.

현재 데이터 검사 결과: 289개 episode, RGB 201,472개, valid label 141,350개,
ignore 60,122개. 새 8장 설정에서 모든 입력 창이 유효한 프레임은 100,218개이며
10 Hz anchor는 16,702개이다. Train 201개, validation 44개, test 44개이며
실제 데이터 검사에서 이전 4장 모델과 **동일한 에피소드 분할**임을 확인했다.
입력 워밍업이 길어져 초반 평가 시점은 달라지므로, 두 모델의 정확한 비교에는
공통 유효 시점을 사용하는 것이 좋다.

| Split | approach | error1 | error2 | turning |
|---|---:|---:|---:|---:|
| Train | 2,499 | 1,578 | 616 | 6,685 |
| Validation | 614 | 342 | 161 | 1,552 |
| Test | 581 | 367 | 167 | 1,540 |

현재 클래스 가중치는 train anchor 분포에서 계산한다. Validation macro-F1으로
best checkpoint를 선택하고 test는 학습이 끝난 뒤 평가한다. 최종 모델과 관측 모델
단독 출력을 비교하여 macro-F1, 정답이 유지되는 구간의 예측 변동 횟수, 관측 가능한
error 전이 후 검출 지연 및 놓친 error 구간 수를 저장한다. 누락 프레임을 넘어
전이를 연결하지 않으며, 시간 지연의 기준은 10 Hz 평가 grid이다.

## 실행: 물리 GPU 2, 3

기본 설정은 `context/qwen/config/train_slip_context.yaml`이다.
학습을 시작하지 않고 데이터만 검사하려면:

```bash
cd /home/metafarmers/dkim/umi_ft
/home/metafarmers/anaconda3/envs/umi/bin/python train_slip_context.py --dry-run
```

고정 CLIP을 반복 실행하지 않도록 RGB 특징 캐시를 한 번 만든다.
이전 학습의 `outputs/slip_clip_features`가 있으면 그대로 재사용하므로 다시 만들 필요 없다.
캐시가 없을 때만 아래 준비 명령을 실행한다. 기존 캐시로 실행해도 검증 후 재사용한다.
아래 명령은 실제 pretrained CLIP이 로드되어야 성공하며 임의 가중치 대체는 없다.

```bash
cd /home/metafarmers/dkim/umi_ft
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_slip_context.py --prepare-cache-only
```

이후 100 epoch 학습:

```bash
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_slip_context.py
```

배치는 GPU당 8개이다. 샘플 하나가 최대 11개의 관측(각 RGB 8장)을 포함하므로 기존 모델의
배치와 계산량이 다르다. `--set batch_size=4`처럼 조정할 수 있다.
History 길이는 `--set dataset.history_steps=20`으로 바꿀 수 있고 model 설정도
함께 적용된다. 실험마다 새 `--set output_dir=outputs/다른이름`을 사용한다.
RGB 개수와 시간 범위는 각각 `dataset.rgb_horizon`, `dataset.rgb_span_s`이며 model에도
연동된다. 기본 출력 폴더는 `outputs/slip_context_rgb8_1s_history10`으로 기존 학습과 분리된다.

짧은 실행 검증은 위 학습 명령에
`--smoke-batches=1 --set output_dir=outputs/slip_sequence_smoke`를 추가한다.
Smoke checkpoint에는 표식을 남기며 full training 재개와 실시간 사용을 거부한다.

재개:

```bash
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_slip_context.py \
  --resume outputs/slip_context_rgb8_1s_history10/last.pt
```

## 산출물과 새 컴퓨터에서 사용할 때

`outputs/slip_clip_features`에는 `features.npy`, `vision.pt`, `manifest.json`을
저장한다. 특징은 `[201472,768]` FP32이며 invalid 행은 사용하지 않는다.
생성 완료 manifest와 두 파일의 SHA-256으로 캐시 무결성을 검사한다. 다른 데이터나
미완료 캐시는 학습에 쓰지 않으며 오류 시 새 `cache_dir`을 지정할 수 있다.
전체 캐시는 backbone을 포함해 약 1 GB이다. Cache 생성에는 라벨 fit이나
정규화 fit이 없고 pretrained encoder를 고정된 전처리로 실행하기만 한다.

`outputs/slip_context_rgb8_1s_history10`의 주요 결과:

- `best.pt`: validation macro-F1 최고 모델. CLIP, F/T 통계, GRU, 클래스 순서 포함.
- `last.pt`: optimizer를 포함한 학습 재개용 파일.
- `resolved_config.yaml`, `dataset_report.json`, `episode_splits.json`.
- `training_history.jsonl`, `test_comparison.json`, `test_eval/`.

실시간 추론에는 **새 `best.pt`와 해당 코드**를 옮기면 된다. 학습 feature cache나
데이터셋/Hugging Face 다운로드는 필요 없다. 스키마는
`slip_rgb_ft_context_sequence_v1`이다. 예전 `eval_real_context_rgb_force.py`와
Stage-B A/T/R/E policy adapter는 이 클래스를 해석하지 않으므로 연결하지 않는다.

새 API는 `umi.real_world.slip_temporal_context.SlipTemporalContextRuntime`이다.
액추에이터 명령을 보내지 않고 호출자가 제공하는 센서 버퍼에서 확률만 반환한다.

```python
from umi.real_world.slip_temporal_context import SlipTemporalContextRuntime
from umi.real_world.rgb_force_context import ObservationUnavailable

runtime = SlipTemporalContextRuntime('best.pt', device='cuda:0')
runtime.reset(task_start_timestamp=task_start_video_time)
# 약 10 Hz로 호출. RGB는 학습과 같은 전처리가 끝난 uint8 RGB [T,224,224,3].
# 새 8장 모델: RGB 최소 1초, F/T 최소 1.4초를 같은 video clock으로 유지한다.
# 실측 시각의 여유를 포함해 보관한다. 기존 4장 모델은 0.5초/0.9초다.
try:
    result = runtime.predict(rgb_times, rgb_frames, ft_times, wrench_12d,
                             anchor_timestamp=rgb_times[-1], now=current_video_time)
    probabilities = result['probabilities']
except ObservationUnavailable:
    # 아직 입력이 모자라거나 시각/센서 간격이 유효하지 않음.
    probabilities = None
```

Task가 시작될 때와 episode 전환 때 반드시 `reset()`한다. 실로봇에는 정답의
ignore mask가 없으므로 task 시작 시점은 호출 측에서 명시해야 한다.
비활성 구간에는 `active=False`를 전달해 history를 초기화한다. 일부 history가
없어도 현재 입력이 유효하면 예측하며, 잘못된 센서 입력을 error 클래스로 바꾸지 않는다.
시간 역전/중복은 거부하고, 약 10 Hz 주기에서 벗어나는 공백에서는 context history를 지운다.
입력 전처리와 bias 보정은 학습과 일치하도록 caller가 수행해야 한다.

8장 구조의 속도 점검: 물리 GPU 2번 RTX 3090, FP32, 워밍업 후 200회에서
평균 23.85 ms, 95백분위 24.35 ms였다. RGB 8장의 CLIP 추출, F/T, Transformer,
GRU와 CPU 결과 반환을 포함한다. 새 분류 head 학습 전의 속도 측정이며,
카메라 획득/resize와 행동 policy 실행은 포함하지 않았다. 목표 10 Hz의 주기는 100 ms다.

## 학습 후 비교 영상

새 학습이 끝나면 8장/10개 history 설정을 체크포인트에서 읽어 표시한다.
예전 4장 모델도 `--checkpoint outputs/slip_context_sequence/best.pt`로 지정할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python tools/export_slip_context_comparison.py \
  --checkpoint outputs/slip_context_rgb8_1s_history10/best.pt \
  --original-episode 291 \
  --output-dir outputs/slip_context_rgb8_1s_history10/comparison_EP291
```

정답·관측 단독·history 보정 후 출력, 확률, 각 RGB 입력, 양손 F/T를 표시한다.
이 비교는 같은 모델의 두 분기이며, 별도로 학습한 history 없는 모델과의 비교는 아니다.

## 테스트

실제 데이터와 pretrained CLIP을 사용한 두 번의 DDP 역전파 검증은 다음과 같다.
전체 캐시를 만들지 않고 필요한 이미지만 인코딩하며 체크포인트는 저장하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
/home/metafarmers/anaconda3/envs/umi/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tools/smoke_slip_context.py
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/metafarmers/anaconda3/envs/umi/bin/python -m pytest \
  tests/test_slip_temporal_context.py tests/test_context_rgb_force.py \
  tests/test_context_rgb_history.py -q
```

테스트는 RGB별 F/T 인과성, ignore/episode 경계, train-only 정규화, 확률 history
학습/실시간 일치, history 누락과 재설정, checkpoint taxonomy, 캐시, 짧은 학습 및 재개를 검증한다.
실제 분류 성능 개선 여부는 재학습 후 held-out episode의 비교 결과로 판단한다.
