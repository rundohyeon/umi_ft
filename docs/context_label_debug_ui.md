# 프레임별 RGB · F/T 라벨 디버거

현재 context classifier의 학습에 사용한 **284개 에피소드**를 확인하는 간단한
Streamlit UI입니다. RGB는 학습 데이터에 저장된 224×224 프레임을 연속 재생합니다.
Qwen 예측이나 모델 체크포인트, GPU 없이 사용할 수 있습니다.

## 실행

```bash
cd /home/metafarmers/dkim/umi_ft
/home/metafarmers/anaconda3/envs/umi/bin/python -m streamlit run \
  tools/debug_context_labels.py --server.address 127.0.0.1 --server.port 8502
```

브라우저에서 **http://localhost:8502** 접속. 첫 수동 라벨 에피소드인 **EP194**부터
열립니다. 서버 종료는 터미널에서 Ctrl-C입니다. 다른 컴퓨터에서는 저장소 루트에서
`conda activate umi` 후 해당 환경의 `python`으로 실행하세요. UI 추가 의존성이 없다면:

```bash
python -m pip install 'streamlit==1.49.1' 'plotly>=5.24,<7'
```

기본 데이터 위치는 저장소 루트의 다음 세 경로입니다. 코드를 `git fetch`로 받는
경우에도 이 데이터는 별도로 준비해야 합니다.

| 용도 | 경로 |
|---|---|
| 학습 RGB | `three_dataset/dataset.zarr.zip` |
| native F/T와 동기화 시간 | `three_dataset/dataset_force_sidecar.zarr/` |
| 기존 4종 라벨 | `three_dataset/canonical_context_supervision_v2_4state_284.npz` |

경로와 시작 에피소드, 수정 저장 파일은 Streamlit 옵션 뒤의 `--`로 지정할 수 있습니다.

```bash
python -m streamlit run tools/debug_context_labels.py --server.port 8502 -- \
  --dataset three_dataset/dataset.zarr.zip \
  --force-sidecar three_dataset/dataset_force_sidecar.zarr \
  --labels three_dataset/canonical_context_supervision_v2_4state_284.npz \
  --episode 194 --output outputs/context_label_debug/review.json
```

## 조작

| 조작 | 기능 |
|---|---|
| ← / → | 이전 / 다음 1프레임 |
| ◀ 10 / 10 ▶ 버튼 | 10프레임 이동 |
| Space | 재생 / 정지 |
| 슬라이더 / 프레임 번호 입력 | 원하는 프레임으로 이동 |
| 1 / 2 / 3 / 4 | approach / turning / recovery / error |
| U | 미지정 (검토 결과에서 valid=false) |
| [ / ] | 현재 프레임을 구간 시작 / 끝으로 지정 |
| Z / S | 마지막 수정 되돌리기 / 저장 |

에피소드는 **1부터**, 프레임은 **0부터** 표시합니다. `선택 구간` 모드에서는 시작과
끝 프레임을 **둘 다 포함**하여 라벨을 바꿉니다. 숫자키 1이 클래스 ID 0입니다.
에피소드 전환 시 재생 위치와 구간 선택이 초기화되며 수정 내역은 유지됩니다.
단축키는 입력칸에 타이핑 중일 때 작동하지 않습니다.

재생은 시간에 맞춘 미리보기로 약 0.1초마다 화면을 갱신하므로 빠른 배속에서는
일부 프레임을 건너뜁니다. 정확한 전수 검토에는 좌우 키 또는 0.1배속을 사용하세요.
기존 라벨에 영향을 받지 않고 보고 싶으면 사이드바의 라벨 표시를 끌 수 있습니다.

## F/T 해석

- 기본은 오른쪽 손가락이고 왼쪽도 함께 선택할 수 있습니다.
- Fx/Fy/Fz와 Tx/Ty/Tz는 부호를 유지하며, |F|와 |T|도 표시합니다.
- 원본은 학습에 쓰인 bias 보정 native 센서 좌표계 값입니다. TCP 좌표 변환은 없습니다.
- 힘은 N, 토크는 Nm 단위의 **별도 그래프**입니다. 빨간 세로선이 현재 RGB 시간입니다.
- 수치 표는 해당 RGB 시간 **이전의 가장 최근 F/T**이며 측정 시차를 함께 표시합니다.
  그래프에는 움직임을 검토할 수 있도록 전후 시간 구간을 함께 보여줍니다.
- 평균/변화량은 학습 코드의 동일한 함수를 사용합니다. 평균은 과거 5샘플,
  변화량은 현재 평균에서 5샘플 전 평균을 뺀 값입니다. 단위는 N/Nm로, 초당 변화율이 아닙니다.
- Δ 모드의 |ΔF|는 힘 벡터 변화량의 크기입니다. 증가/감소 방향은 각 축의 부호로 확인하세요.

## 저장

`S`를 누르면 기본적으로 `outputs/context_label_debug/review.json`에 **원본 대비 수정분**을
저장하고 다음 실행에서 복원합니다. 저장 전 수정은 브라우저 세션에만 있습니다.
원본 라벨을 수정하지 않으며, 저장한 JSON이 재학습에 자동 반영되지는 않습니다.
검토를 마친 후 별도로 학습 라벨 NPZ에 반영하는 단계가 필요합니다.

JSON에는 원본 라벨 파일의 SHA-256, 전체/에피소드 프레임 번호, 시간,
기존 라벨과 새 라벨이 들어 있습니다. 다른 데이터의 수정 파일은 불러오지 않으며,
다른 창이 저장 파일을 변경했을 때는 덮어쓰기를 막습니다.

## 원격 컴퓨터에서 브라우저로 보기

UI가 실행 중인 컴퓨터에 SSH 터널을 연결하고 로컬 브라우저의 localhost:8502를 여세요.
예를 들어 `idim-indy-114`에서 UI를 실행 중이라면 브라우저가 있는 컴퓨터에서:

```bash
ssh -F /dev/null -p 211 -N -L 8502:127.0.0.1:8502 idim@147.46.147.77
```

서버 주소/포트는 UI를 실제 실행한 컴퓨터에 맞추세요. X11 전달은 필요하지 않습니다.

## 검증

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_context_debug_review.py tests/test_context_rgb_force.py -q
```

테스트는 영상/F/T 인덱스, 학습용 평균·변화량과의 일치, 에피소드 경계,
프레임 이동, 구간 수정, 되돌리기, 저장/복원 및 원본 보존을 확인합니다.
