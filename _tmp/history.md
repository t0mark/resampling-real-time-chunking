# 작업 히스토리

## 2026-11-19 GPU 비교군(RTC VJP) 평가

- 환경: H200 × 8, GPU 0~3 사용 (사용자 허용), 4·5 타 사용자, 6·7 미사용
- `docker compose -f docker/gpu/docker-compose.yml build/up` → 이미지 t0mark251/rs_rtc_gpu, 컨테이너 airlab_hw_rtc_gpu
- `pip install -e` 후 pytest: test_ddim.py 가 diffusers 필요 → 컨테이너 내 diffusers==0.35.2 설치 → 6 passed
  - Dockerfile 에 diffusers==0.35.2 반영 (재빌드는 안 함)
- 동작 확인: d=4, 2 ep → 정상 (에피소드당 ~11 s, 생성 p50 ~288 ms)
- 본 실행: configs/eval_gpu.yaml (50 ep, seed 100000~), 지연별 GPU 분할
  - GPU0: d=0,7 / GPU1: d=1 / GPU2: d=2 / GPU3: d=4
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<id>` 로 docker exec
  - 조합마다 generator 재초기화 → 병렬 분할해도 순차 실행과 동일 결과
- 결과: results/metrics/eval_async/rtc_vjp_d{0,1,2,4,7}.csv
- 완료 (4개 프로세스 모두 exit 0, d=7 은 GPU0 에서 d=0 다음 순서로 ~17:59 종료)
- 집계: summarize_results.py --methods rtc_vjp --name rtc_vjp_gpu → results/metrics/summary_rtc_vjp_gpu.csv

| d | SR % [95% CI] | max reward | 경계 변화량 px | 내부 변화량 px | 비율 | 생성 p50 ms |
|---|---|---|---|---|---|---|
| 0 | 68.0 [54.2, 79.2] | 0.960 | 9.91 | 9.19 | 1.08 | 279.0 |
| 1 | 72.0 [58.3, 82.5] | 0.970 | 13.12 | 9.24 | 1.42 | 286.6 |
| 2 | 58.0 [44.2, 70.6] | 0.981 | 16.73 | 9.38 | 1.78 | 286.3 |
| 4 | 66.0 [52.2, 77.6] | 0.994 | 21.10 | 9.57 | 2.21 | 298.3 |
| 7 | 46.0 [33.0, 59.6] | 0.949 | 33.15 | 9.13 | 3.63 | 275.0 |

## 2026-11-19 GPU.md 요청: β 선택 + RTC 재실행

- 코드 변경(OMP 스레드 제한, prefix_cost, metrics_subdir, sweep_beta.yaml)은 이미 워크스페이스에 반영돼 있었다 (git 명령 사용 안 함)
- pytest 6 passed
- β sweep (검증 seed 200000~, d=4, 50 ep): GPU0 b5→b80 순차, GPU1 b10, GPU2 b20, GPU3 b40 병렬
- β sweep 결과 (d=4, seed 200000~, NaN 없음): b5 76% / b10 54% / b20 62% / b40 50% / b80 28% (b80 전환 197 px, prefix 1.42 → 발산)
  - summarize_results.py 는 eval_async 만 읽으므로 sweep_beta_val 은 pandas 로 직접 집계 (코드 수정 금지)
  - 선택 β=5 (최고 SR) → eval_gpu.yaml 변경 없음
- 본 실행: GPU0 단독, 프로세스 1개 순차 (d=0,1,2,4,7)
- 본 실행 완료 (exit 0, 18:50~19:26). 품질 지표 이전과 동일, 단독 p50 293.6~298.4 ms
- summary_rtc_vjp_gpu.csv 재생성, RESULT.md 갱신 (β 선택 표, 본 실행 표, 측정 조건, d=0 설명 정정)
- 커밋·push 는 CLAUDE.md 규칙(git 사용 금지)으로 하지 않음 → 사용자에게 대상 안내
- 결과 파일 소유자가 root (컨테이너에서 생성)
