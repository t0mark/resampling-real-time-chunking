# GPU 서버 작업 요청 (2026-11-19, 2차)

원 논문 공식 코드대로 옮긴 RTC(`rtc_paper`)를 β 선택부터 본 실행까지 돌려줘. 1차 요청(β 선택, `rtc_vjp` 재실행)은 끝났고 결과는 병합했다.

## 왜 다시 돌리나
- 1차 β 선택에서 `rtc_vjp`는 β를 키울수록 prefix 오차가 커졌다(β5 0.0089 → β10 0.0114 → β20 0.0305). 가이던스가 목표를 지나치는(overshoot) 상태다.
- **원인:** 기존 `rtc_vjp`는 모든 step에 β√(1−ᾱ)를 고정으로 곱했다. 공식 코드(`real-time-chunking-kinetix/src/model.py`, `realtime_action`)는 flow 시간 τ에 따라 가중치를 w = min(β, (1−τ)/τ · (τ²+(1−τ)²)/(1−τ)²)로 조절한다.
  - 기존 구현은 노이즈가 큰 step에서 x̂0 보정량이 공식 코드보다 수십 배 크다.
- **새 `RtcPaperSampler`(`samplers/rtc_vjp.py`):** DDIM 상태를 flow 상태로 바꿔서 공식 코드의 식을 그대로 적용한다. 변환식은 x_τ = x_t/s, τ = √ᾱ/s, s = √ᾱ+√(1−ᾱ)이다.
- **CPU 확인 결과:** 초기 상태 4개, d=4 목표로 prefix 오차를 쟀다.

  | β | 0 | 5 | 20 | 100 |
  |---|---|---|---|---|
  | 기존 `rtc_vjp` | 0.255 | 0.234 | 0.215 | 1.51 (발산) |
  | 새 `rtc_paper` | 0.255 | 0.195 | 0.139 | 0.085 |

  `rtc_paper`는 β를 키울수록 오차가 줄고 발산하지 않는다. 실제 에피소드에서도 d=4로 3 에피소드를 돌려 정상 동작을 확인했다.

## 바뀐 파일 (pull로 받으면 됨)
- `src/rs_rtc/samplers/rtc_vjp.py`: `RtcPaperSampler` 추가. 기존 `RtcVjpSampler`는 그대로 둔다.
- `scripts/eval_gpu.py`: 방법 type에 `rtc_paper`를 추가했다.
- `configs/eval_gpu.yaml`: `rtc_paper` 방법을 추가했다(β=5는 공식 코드 기본값).
- `configs/sweep_beta.yaml`: `rtc_paper_b{5,10,20,50}`을 추가했다. 결과 폴더는 `sweep_beta_val/`이다.

## 할 일
1. **pull 후 테스트:** `pytest -q tests`
2. **β 선택:** `rtc_paper`만 돌린다. `rtc_vjp_b*`는 이미 끝났다. β 값마다 프로세스 1개로 병렬 실행한다(GPU 1장씩).
   ```bash
   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<id> python3 scripts/eval_gpu.py --config configs/sweep_beta.yaml --methods rtc_paper_b<β>
   ```
   - 고르는 규칙은 지난번과 같다. 성공률이 가장 높은 β를 고르고, 동률이면 `mean_prefix_cost`가 작은 쪽을 고른다. 발산한 β는 제외한다.
3. **본 실행:** `configs/eval_gpu.yaml`의 `rtc_paper` `guidance_weight`를 고른 β로 바꾸고 아래를 실행한다.
   ```bash
   python3 scripts/eval_gpu.py --methods rtc_paper
   ```
   - `--methods rtc_paper`를 꼭 지정한다. 지정하지 않으면 `rtc_vjp`도 다시 돈다.
   - 다른 작업이 없는 GPU 1장에서 프로세스 1개로 순차 실행한다. 생성 지연을 그대로 쓰기 위해서다.
4. **집계와 보고**
   - 실행: `python3 scripts/summarize_results.py --methods rtc_vjp rtc_paper --name rtc_gpu`
   - RESULT.md에 `rtc_paper` 섹션을 추가한다. β 선택 표(β별 성공률, 전환 변화량, prefix 오차, 발산 여부), 본 실행 표, 측정 조건을 적는다.
   - 기존 `rtc_vjp` 섹션은 "기존 이식(고정 가중치)"으로 이름만 바꿔서 남긴다.
5. **커밋·push 대상:** `results/metrics/sweep_beta_val/rtc_paper_*.csv`, `results/metrics/eval_async/rtc_paper_d*.csv`, `results/metrics/summary_rtc_gpu.csv`, `RESULT.md`

## 지킬 것
- 코드는 수정하지 않는다. `eval_gpu.yaml`의 `rtc_paper` `guidance_weight`만 바꾼다.
- 에러가 나면 우회하지 말고 원인을 RESULT.md에 적는다.
