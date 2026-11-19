# GPU 서버 작업 요청 (2026-11-19)

RTC(VJP) 비교군의 가이던스 가중치 β를 RS-RTC의 λ와 같은 절차로 고르고, RTC 본 실행을 다시 돌려줘.

## 왜 다시 돌리나
- RS-RTC는 검증 seed(200000~)에서 λ를 골랐는데, RTC는 β=5 하나만 썼다. 이대로면 비교가 공정하지 않다.
- 지난 결과에서 RTC의 chunk 전환 변화량이 naive와 거의 같았다.
- CPU에서 확인해 보니 β=5는 inpainting 오차를 가이던스가 없을 때보다 약 8%밖에 줄이지 못했다.
- β≥50에서는 일부 상태가 발산했다(β=1000은 NaN). 이 확인은 실제 에피소드보다 어려운 목표로 한 것이다.

## 그동안 바뀐 코드 (pull로 받으면 됨)
- `scripts/eval_*.py`: 맨 위에서 `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`를 기본 4로 설정한다.
  - 전에는 libgomp와 OpenBLAS가 프로세스마다 코어 수만큼 스레드를 띄워 spin했다. 그래서 동시 실행 시 생성 지연이 부풀려졌다.
  - 지난 GPU 지연 280 ms도 이 영향을 받았을 수 있다.
- `envs/async_executor.py`: 모든 방법에 `mean_prefix_cost`를 기록한다.
  - 생성된 chunk와 inpainting 목표 사이의 가중 오차이고, 정규화 공간 기준이다.
  - `summarize_results.py`도 이 값을 집계한다.
- `evaluation.py`: 설정의 `metrics_subdir`로 결과 폴더를 지정한다(기본 `eval_async`).
- `configs/sweep_beta.yaml` 추가: 검증 seed 200000~, d=4, 50 에피소드, β ∈ {5, 10, 20, 40, 80}.
  - 결과는 `results/metrics/sweep_beta_val/`에 저장된다.

## RESULT.md 정정
- "d = 0 은 inpainting 목표가 없어 가이던스 없이 생성된다"는 틀린 설명이다.
- d=0에서도 확정 구간은 index 0(이미 실행된 행동)이다. index 1~7은 감쇠 가중치로 이전 chunk에 맞춰지므로 가이던스가 작동한다.
- RTC와 RS-RTC가 d=0에서 둘 다 68%로 naive(84%)보다 낮은 것도 이 때문으로 보인다.

## 할 일
1. **pull 후 테스트:** `pytest -q tests`
2. **β 선택:** β 값마다 1개 프로세스로 병렬 실행한다(GPU 1장씩).
   ```bash
   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<id> python3 scripts/eval_gpu.py --config configs/sweep_beta.yaml --methods rtc_vjp_b<β>
   ```
   - 고르는 규칙은 λ와 같다. 성공률이 가장 높은 β를 고르고, 동률이면 `mean_prefix_cost`가 작은 쪽을 고른다.
   - NaN이나 발산이 나온 β는 제외한다.
   - 참고로 RS-RTC λ 선택 결과는 λ=1000에서 성공률 72%, 전환 변화량 18.9 px, prefix 오차 0.0057이었다.
3. **본 실행:** `configs/eval_gpu.yaml`의 `guidance_weight`를 고른 β로 바꾸고 `python3 scripts/eval_gpu.py`를 실행한다.
   - **프로세스 1개로 순차 실행한다.** 다른 작업이 없는 GPU 1장에서 돌린다. 그래야 `generation_p50_ms`를 RTC의 GPU 지연으로 그대로 쓸 수 있다.
   - 예상 소요 시간은 5 × 50 에피소드 × 약 11 s ≈ 45분이다.
   - 기존 `results/metrics/eval_async/rtc_vjp_d*.csv`를 덮어쓴다.
4. **집계와 보고**
   - 실행: `python3 scripts/summarize_results.py --methods rtc_vjp --name rtc_vjp_gpu`
   - RESULT.md에 β 선택 표(β별 성공률, 전환 변화량, prefix 오차, 발산 여부), 본 실행 표, 측정 조건(GPU 모델, 단독 실행 여부)을 적는다.
   - 위 d=0 설명도 고친다.
5. **커밋·push 대상:** `results/metrics/sweep_beta_val/`, `results/metrics/eval_async/rtc_vjp_d*.csv`, `results/metrics/summary_rtc_vjp_gpu.csv`, `RESULT.md`

## 지킬 것
- 코드는 수정하지 않는다. `eval_gpu.yaml`의 `guidance_weight`만 바꾼다.
- 에러가 나면 우회하지 말고 원인을 RESULT.md에 적는다.
