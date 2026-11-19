# RS-RTC: NPU에서 역전파 없이 하는 Real-Time Chunking

## 연구 내용

- **배경:** Rebellions ATOM-Max NPU(RBLN SDK)는 정적 그래프 컴파일과 추론 전용 실행만 지원한다. 역전파가 안 된다.
- **RTC (arXiv 2506.07339):** 비동기 action chunking 방법.
  - 이전 chunk 중 실행이 확정된 앞부분(prefix)을 inpainting 목표로 두고 새 chunk를 생성한다.
  - UNet을 거치는 역전파(VJP) 가이던스를 쓰므로 NPU에서 실행할 수 없다.
- **RS-RTC (제안 방법):** VJP 가이던스를 gradient 없는 particle resampling(SMC)으로 바꾼다.
  - NPU 고정 batch에 particle N개를 넣는다.
  - 매 DDIM step마다 끝점 x̂0를 추정하고, prefix와의 가중 오차 cost를 계산한다.
  - 가중치를 exp(−λ_k·cost)로 갱신한다. λ_k = λ_max·(k+1)/K.
  - ESS가 기준 아래로 떨어지면 systematic resampling을 한다.
  - DDIM η 노이즈로 particle 다양성을 유지한다.
  - 마지막에 cost가 가장 작은 particle을 고른다.
- **실험 설정**
  - 정책: 공개 사전학습 가중치 `lerobot/diffusion_pusht_keypoints` (Diffusion Policy, DDIM 10 step, horizon 16, 실행 8).
  - 환경: PushT 키포인트 관측.
  - 지연 d(제어 스텝)는 시뮬레이션으로 재현하므로 하드웨어 속도와 무관하다.
  - d ∈ {0, 1, 2, 4, 7}, 50 에피소드, 모든 방법이 같은 seed를 쓴다.
- **비교군**
  - 주 비교: RS-RTC(NPU) vs RTC VJP(GPU, 같은 코드·seed·초기 노이즈).
  - 보조 비교: naive async, BID(best-of-N).
- **지표:** 성공률(Wilson CI), 최대 보상, chunk 전환 시 행동 변화량(boundary jump px), 내부 변화량, 생성 지연 p50.
- **주장**
  - 품질: 역전파 없이 RTC와 비슷한 inpainting 품질을 낸다.
  - 실시간성: 하드웨어별 실측 지연은 따로 보고한다.

## 폴더 구조

```
rs_rtc/
├── configs/            # 실험 설정 (eval_npu, eval_gpu, eval_sync, bench_latency, sweep_lambda, sweep_lambda_eta)
├── scripts/
│   ├── eval_npu.py           # 역전파 없는 방법(naive, replacement, BID, RS-RTC) NPU 비동기 평가
│   ├── eval_gpu.py           # RTC VJP 가이던스 CUDA 비동기 평가
│   ├── eval_sync.py          # 사전학습 정책 동기 실행 평가 (가중치·정규화·DDIM·NPU 이식 검증)
│   ├── bench_latency.py      # UNet 1 step 그래프 batch N별 NPU 컴파일·정합성·지연 측정
│   └── summarize_results.py  # eval_async CSV를 방법 × 지연별로 집계
├── src/rs_rtc/
│   ├── evaluation.py   # 방법 1개 × 지연 1개 조합의 에피소드 반복과 CSV 저장
│   ├── envs/           # pusht.py: gym-pusht 래퍼 / async_executor.py: 지연 d 비동기 chunk 실행기
│   ├── npu/            # compile.py: RBLN 정적 그래프 컴파일 / runtime.py: CPU·NPU 공통 UNet 실행 래퍼
│   ├── policies/       # diffusion_unet.py: 조건부 1D UNet / pretrained.py: 사전학습 가중치 로딩·정규화
│   ├── samplers/       # base.py: 공통 요소·기본 DDIM 샘플러 / ddim.py: DDIM 스케줄·갱신
│   │                   # rs_rtc.py: 제안 방법 / rtc_vjp.py: RTC VJP / bid.py: BID / baselines.py: 확정 구간 치환
│   └── utils/          # logging.py: 콘솔·results/logs 동시 기록 로거
├── tests/              # test_ddim.py: diffusers DDIM 일치 확인 / test_samplers.py: 샘플러·실행기 단위 테스트
├── docker/gpu/         # GPU 평가용 Dockerfile, docker-compose.yml
├── docs/               # experiment_design.md: 실험 설계
└── results/
    ├── compiled/       # NPU 컴파일 결과 (.rbln)
    ├── metrics/        # eval_async(본 실험 CSV), sweep_lambda 등 지표
    └── logs/           # 실행 로그
```

## 실행 방법

- **NPU:** 상위 폴더 `hw/scripts/`의 `rbln_docker.sh`로 컨테이너 기동 후, `/workspace/rs_rtc`에서 실행
  - `CONTAINER_NAME=airlab_hw_rtc scripts/rbln_docker.sh up`
  - `python3 scripts/eval_npu.py` (컨테이너 `airlab_hw_rtc` 내부)
- **GPU:** `docker/gpu/docker-compose.yml` 상단의 사용법 주석을 따른다.
- **집계:** `python3 scripts/summarize_results.py`
