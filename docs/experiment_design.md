# RS-RTC 실험 설계 (초안 B)

가칭: **NPU에서 역전파 없이 하는 Real-Time Chunking — Resampling 기반 inpainting (RS-RTC)**

- 하드웨어: Rebellions ATOM-Max (RBLN-CA25) × 32, RBLN SDK 0.10.3 (정적 그래프, 추론 전용), CPU 128코어, torch 2.10 CPU, GPU 없음
- 학습: CPU / 추론: NPU(v_θ 1 step 그래프) + 호스트(가중치·resampling)
- 1차 벤치마크: gym-pusht 0.1.8, `obs_type="environment_state_agent_pos"`, 데이터 `lerobot/pusht_keypoints` (206 ep, 25650 frame, 10 Hz)

---

## 0. 표기와 텐서 shape

| 기호 | 의미 | shape / 값 |
|---|---|---|
| o_t | 관측 (agent_pos 2 + environment_state 16) × T_o=2 프레임 | [B, 2, 18] → flatten [B, 36] |
| A | action chunk (정규화 [-1,1]) | [H, Da] = [16, 2] |
| A_i (particle) | particle 집합 | [N, 16, 2] |
| τ_k | flow 시간, τ_k = k/K, k=0..K−1 | δ = 1/K |
| v_θ(A^τ, o, τ) | 속도장 (NPU 1 step 그래프, batch N 고정) | 입력 [N,16,2],[N,36],[N] → 출력 [N,16,2] |
| Â | 끝점 추정 Â = A^τ + (1−τ)v | [N, 16, 2] |
| Y | 이전 chunk의 미실행 구간을 현재 시점으로 shift한 목표 | [16, 2] (i ≥ H−s 는 0 패딩) |
| W | soft mask | [16] |
| d | 추론 지연 (제어 step 수) | 정수, 0 ≤ d ≤ H−s |
| s | 실행 horizon (cycle 주기) | 8 |
| c_i | particle cost | [N] |
| log w | 누적 log-weight | [N] |
| a(i) | resampling 조상 index | [N] int64 |
| C | 사용 NPU 카드 수 | {1,2,4,8,16,32} |

### 0.1 비동기 실행 타이밍 (모든 비동기 비교군 공통)

| 구간 | 실행되는 행동 |
|---|---|
| cycle 시작 t_c | o_{t_c} 관측, 추론 시작. Y_i = A_prev[s+i] (i < H−s), 그 외 0 |
| t_c … t_c+d−1 | 이전 chunk A_prev[s … s+d−1] 실행 (확정 구간, 새 chunk index 0…d−1 에 대응) |
| t_c+d … t_c+s+d−1 | 새 chunk A_new[d … s+d−1] 실행. 다음 cycle은 t_c+s 에 시작 |

- 제약: d ≤ s ≤ H−d (RTC 논문 조건). H=16, s=8 → d ≤ 8.
- 동기 실행(비교군 1): 추론 동안 환경 정지, A_new[0 … s−1] 실행.

### 0.2 soft mask W (RTC Eq.5 그대로 사용)

| 구간 | W_i |
|---|---|
| i < d | 1 |
| d ≤ i < H−s | c_i (e^{c_i} − 1)/(e − 1), c_i = (H−s−i)/(H−s−d+1) |
| i ≥ H−s | 0 |

---

## 1. RS-RTC 알고리즘 (실험 대상 정의)

| 단계 | 연산 | 실행 위치 | 수식 / shape |
|---|---|---|---|
| 초기화 | ε ~ N(0,I), A^0 = ε, log w = 0 | 호스트 | [N,16,2] |
| k-1 | v = v_θ(A^τk, o, τ_k) | NPU (C장 분할 시 카드당 [N/C,16,2]) | [N,16,2] |
| k-2 | Â = A^τk + (1−τ_k) v | 호스트 | [N,16,2] |
| k-3 | c = Σ_i W_i‖Y_i − Â_i‖² / Σ_i W_i | 호스트 | [N] |
| k-4 | log w += −λ_k c^(k) + λ_{k−1} c^(k−1) (λ_{−1}c^(−1)=0) | 호스트 | [N] |
| k-5 | ESS = (Σw)²/Σw² < γN 이면 systematic resampling, 상태·c^(k)·Â gather 후 log w = 0 | 호스트 | a: [N] |
| k-6 (옵션 R) | Â[:, :d] ← Y[:d] (replacement) | 호스트 | |
| k-7 | ε̂ = (A^τk − τ_k Â)/(1−τ_k) | 호스트 | [N,16,2] |
| k-8 | A^{τk+δ} = (τ_k+δ)Â + (1−τ_k−δ)[√(1−η²) ε̂ + η ε_new] | 호스트 | η=0 → Euler 와 동일 |
| 종료 | i* = argmax_i (log w_i − λ_{K−1}c_i^(K−1) 포함 최종 potential), A* = A^1_{i*}, A*[:d] ← Y[:d] | 호스트 | [16,2] |

- 다중 카드 모드: **global** (전 카드 particle 을 호스트에서 모아 N 전체로 resampling) / **island** (카드별 N/C 로 독립 resampling, 마지막에 카드 간 argmax).
- NPU 호출 수/chunk = K (카드당), host↔device 왕복 = K 회.

---

## 2. 하이퍼파라미터 기본값

| 항목 | 기본값 | sweep 범위 | 근거 |
|---|---|---|---|
| T_o (관측 프레임) | 2 | – | Diffusion Policy(DP) PushT: obs 2 / action 8 / prediction 16 [S1][S3] |
| H (chunk 길이) | 16 | (확장 시 32) | DP PushT prediction horizon 16, LeRobot `diffusion_pusht_keypoints` horizon 16 [S3] |
| s (실행 horizon) | 8 | – | DP action horizon 8, LeRobot n_action_steps 8 [S3] |
| d (지연) | {0,1,2,4,8} | 고정 d 모드 / 실측 환산 모드 | RTC 시뮬레이션: H=8 에서 d=0~4 (d ≤ H−s 상한까지) [S2]. 본 설계 H−s=8 이 상한 |
| Δt (제어 주기) | 100 ms | – | 데이터 10 Hz |
| K (Euler step) | 10 | {5, 10} | LeRobot DP keypoint num_inference_steps 10 [S3]; RTC 는 n=5 [S2] → 5 도 보조 측정 |
| N (particle) | 64 | {1,4,16,64,256,1024} | phase0 지연 곡선의 batch 집합과 일치 |
| λ 스케줄 | λ_k = λ_max · τ_k (선형 증가) | {상수, τ_k, τ_k²} × λ_max ∈ {10, 30, 100, 300, 1000} | 정규화 행동 공간에서 c ≈ 0.01 (오차 0.1) 일 때 λc ≈ 1 이 되도록 λ_max=100 |
| η (재잡음) | 0.3 | {0, 0.1, 0.3, 0.5, 1.0} | η=0 은 결정적 ODE (복제 particle 붕괴 대조군) |
| ESS 임계 γ | 0.5 | {매 step, 0.5, 0.25, 안 함} + 고정 스케줄 {매 2 step, 마지막 3 step} | SMC 관례 γ=0.5 |
| replacement 옵션 R | off | {off, on} | RePaint 결합 효과 분리 |
| RTC β (가이던스 clip) | 5 | {1, 2, 5, 10} | RTC Table 4: β=5 [S2] |
| BID N | 64 (RS-RTC 와 NFE 맞춤) | = RS-RTC N | RTC 논문 BID 비교: N=32, mode size 3 [S2] |
| 에피소드 최대 길이 | 300 step | – | LeRobot PushT env episode_length 300 [S3][S4] |
| 성공 판정 | max coverage ≥ 0.95 | – | LeRobot PushT 성공 정의 [S4] |

---

## 3. 지표 정의

| 지표 | 수식 / 정의 | 단위 |
|---|---|---|
| 성공률 SR | (1/E) Σ_e 1[max_t cov_{e,t} ≥ 0.95], 95% Wilson 구간 | % |
| 최대 coverage | (1/E) Σ_e max_t cov_{e,t}, cov = |블록 ∩ 목표| / |목표| | [0,1] |
| 경계 불연속 J_b | 각 chunk 전환 시점 t_sw 에서 ‖a_{t_sw} − a_{t_sw−1}‖₂ (픽셀, 0~512 좌표) 평균·p95 | px |
| 정규화 불연속 J_r | J_b / mean_{t∉경계} ‖a_t − a_{t−1}‖₂ | 배수 |
| 경계 jerk | ‖a_{t+1} − 2a_t + a_{t−1}‖₂, t ∈ [t_sw−1, t_sw+1] 평균 | px |
| inpainting 오차 | D_hard = mean_{i<d}‖A*_i − Y_i‖ (최종 덮어쓰기 전), D_soft = Σ_{d≤i<H−s} W_i‖A*_i − Y_i‖ / Σ W_i | px |
| 모드 전환 | 전환 event m = 1[D_ov > θ], D_ov = mean_{d≤i<H−s}‖A*_i − Y_i‖. θ=20 px 기본, {10,20,40} 민감도 보고. 지표 = event 수 / chunk 수 | % |
| ESS | (Σ_i w_i)² / Σ_i w_i², 각 step 기록, N 대비 비율 ESS/N | [1/N,1] |
| 조상 다양성 | 최종 step 고유 조상 수 U = |{a_root(i)}| / N | [1/N,1] |
| particle 다양성 | 최종 Â 의 평균 pairwise 거리 (1/N(N−1)) Σ_{i≠j}‖Â_i − Â_j‖_F | px |
| resampling 횟수 | chunk 당 resampling 발생 step 수 | 회 |
| NPU 호출 수 | chunk 당 v_θ 호출 = K (카드당). RTC(VJP): forward K + backward K (CPU) | 회 |
| NFE | chunk 당 총 particle-forward 수 = N·K | 회 |
| wall time | chunk 1개 추론 시간 p50 / p99 (≥1000 chunk), 분해: NPU 실행 / host↔device 전송 / host resampling | ms |
| 환산 지연 | d_meas = ⌈T_p99 / Δt⌉ | step |

---

## 4. 연구 질문

| RQ | 질문 |
|---|---|
| RQ1 | 동일 wall-time(=동일 환산 d) 조건에서 RS-RTC(NPU) 는 RTC(VJP, CPU) 대비 성공률·불연속에서 동등 이상인가 |
| RQ2 | d 가 커질수록 RTC(VJP) 수준(비열등 마진 2%p)에 도달하는 데 필요한 최소 N*(d) 는 어떻게 증가하는가 |
| RQ3 | resampling 스케줄(γ, 고정 스케줄), λ 스케줄·λ_max, η, replacement 옵션이 ESS·다양성·성공률에 주는 영향 |
| RQ4 | 총 N 고정 시 카드 수 C 와 분할 방식(global vs island)이 wall time 과 품질에 주는 영향 |

### 4.1 비교군

| ID | 이름 | 생성 방식 | 실행 위치 | 앞 d 개 처리 |
|---|---|---|---|---|
| B1 | 동기 실행 | Euler K, N=1 | NPU | 해당 없음 (환경 정지) |
| B2 | 단순 비동기 | Euler K, N=1 | NPU | 무시 (새 chunk 그대로) |
| B3 | 최종 덮어쓰기 | Euler K, N=1 | NPU | A*[:d] ← Y[:d] |
| B4 | RePaint | Euler K, N=1, 매 step Â[:d] ← Y[:d] | NPU | 매 step 치환 + 최종 덮어쓰기 |
| B5 | BID (backward) | Euler K, N 개 독립 생성 후 argmin_i D_soft 선택 | NPU | 최종 덮어쓰기 |
| B5' | BID (full, 선택) | B5 + 약한 checkpoint 대비 forward contrast | NPU | 최종 덮어쓰기 |
| B6 | RTC (VJP) | ΠGDM 가이던스, β=5 | CPU autograd | 가이던스 + 최종 덮어쓰기 |
| P | RS-RTC | 1절 | NPU (+호스트) | 가중치/resampling + 최종 덮어쓰기 |
| P+R | RS-RTC + replacement | 1절 + k-6 | NPU | 매 step 치환 |

### 4.2 RQ1 — 동일 wall-time 비교

| 항목 | 내용 |
|---|---|
| 독립변수 | 방법 {B1~B6, P, P+R}, 지연 모드 {고정 d=4, 실측 환산 d_meas} |
| 고정변수 | 정책 checkpoint, H=16, s=8, K=10, 초기 상태 집합, 노이즈 seed |
| RS-RTC 설정 | N = phase0 에서 K step 총 p99 가 RTC(VJP, CPU, N=1) p99 이하인 최대 N (wall-time 일치) |
| 측정 지표 | SR, 최대 coverage, J_b, 모드 전환율, D_hard, D_soft, wall time p50/p99, NPU 호출 수 |
| seed·에피소드 | 정책 학습 seed 3 × 평가 500 에피소드 (env seed 100000~100499) |

결과 표 양식 (지연 모드별 1개씩)

| 방법 | N | d | T p50 (ms) | T p99 (ms) | SR (%) [95% CI] | max cov | J_b (px) | 모드 전환 (%) | D_soft (px) |
|---|---|---|---|---|---|---|---|---|---|
| B1 동기 | 1 | – | | | | | | | |
| B2 단순 비동기 | 1 | | | | | | | | |
| B3 최종 덮어쓰기 | 1 | | | | | | | | |
| B4 RePaint | 1 | | | | | | | | |
| B5 BID | | | | | | | | | |
| B6 RTC (VJP, CPU) | 1 | | | | | | | | |
| P RS-RTC | | | | | | | | | |
| P+R | | | | | | | | | |

### 4.3 RQ2 — d 대비 필요 N

| 항목 | 내용 |
|---|---|
| 독립변수 | d ∈ {1,2,4,8}, N ∈ {1,4,16,64,256,1024} |
| 고정변수 | λ_max=100 선형, η=0.3, γ=0.5, R=off, K=10, C=1 |
| 비교 기준 | 같은 d 의 B6 (RTC VJP) SR, J_b |
| 정의 | N*(d) = min{N : SR_P(N,d) ≥ SR_B6(d) − 2%p 및 J_b,P ≤ 1.1·J_b,B6} |
| 측정 지표 | SR, J_b, 모드 전환율, ESS/N 평균, U, wall time |
| seed·에피소드 | 정책 seed 1 (주 checkpoint) × 200 에피소드 (env seed 100000~100199); N*(d) 후보 지점은 3 seed × 500 재확인 |

결과 표 양식 (셀: SR % / J_b px)

| d \ N | 1 | 4 | 16 | 64 | 256 | 1024 | B6 RTC | N*(d) |
|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| 4 | | | | | | | | |
| 8 | | | | | | | | |

### 4.4 RQ3 — resampling 스케줄·λ·η

| 항목 | 내용 |
|---|---|
| 독립변수 (one-factor-at-a-time, 기본값 기준) | (a) 스케줄 {매 step, γ=0.5, γ=0.25, 마지막 3 step, 안 함(IS only)} (b) λ 형태 {상수, τ, τ²} × λ_max {10,30,100,300,1000} (c) η {0,0.1,0.3,0.5,1.0} (d) R {off,on} |
| 고정변수 | d=4, N=64, K=10, C=1 |
| 측정 지표 | SR, J_b, D_soft, ESS/N (step 별 곡선), U, particle 다양성, resampling 횟수 |
| seed·에피소드 | 정책 seed 1 × 200 에피소드. 튜닝용 초기 상태(env seed 0~199)와 평가용(100000~)은 분리 |

결과 표 양식

| 요인 | 수준 | SR (%) | J_b (px) | D_soft (px) | 평균 ESS/N | 최종 U | 다양성 (px) | resampling 횟수 |
|---|---|---|---|---|---|---|---|---|
| 스케줄 | 매 step | | | | | | | |
| | γ=0.5 | | | | | | | |
| | γ=0.25 | | | | | | | |
| | 마지막 3 step | | | | | | | |
| | 안 함 | | | | | | | |
| λ (형태, λ_max) | (τ, 100) 등 15 조합 | | | | | | | |
| η | 0 / 0.1 / 0.3 / 0.5 / 1.0 | | | | | | | |
| R | off / on | | | | | | | |

### 4.5 RQ4 — 다중 카드 분할

| 항목 | 내용 |
|---|---|
| 독립변수 | C ∈ {1,2,4,8,16,32}, 분할 {global, island}, 총 N ∈ {256, 1024} |
| 고정변수 | 카드당 batch N/C 로 각각 컴파일, d=4, λ·η·γ 기본값 |
| 측정 지표 | wall time p50/p99 및 분해(NPU / 전송 / gather), SR, J_b, ESS/N, U |
| seed·에피소드 | 지연 측정 1000 chunk (환경 없이 고정 관측 replay), 품질 200 에피소드 |

결과 표 양식

| 총 N | C | N/C | 분할 | NPU (ms) | 전송 (ms) | resampling (ms) | T p50 | T p99 | SR (%) | J_b | ESS/N | U |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 1 | 256 | – | | | | | | | | | |
| 256 | 4 | 64 | global | | | | | | | | | |
| 256 | 4 | 64 | island | | | | | | | | | |
| … | | | | | | | | | | | | |
| 1024 | 32 | 32 | island | | | | | | | | | |

---

## 5. 단계별 계획과 판단 기준

| 단계 | 작업 | 완료 기준 | 실패 판단 기준 → 조치 |
|---|---|---|---|
| phase0 | v_θ 1 step 그래프 NPU 컴파일 (batch N ∈ {1,16,64,256,1024}), 지연 곡선, CPU 정합성, 호스트 resampling 왕복 비용 | 5개 batch 모두 컴파일; 상대오차 ‖v_npu − v_cpu‖/‖v_cpu‖ ≤ 1e-2; K=10 적분 후 chunk 최대오차 ≤ 1 px; N=64 기준 K step 총 p99 ≤ 400 ms (d_meas ≤ 4) | N=64 에서 K step 총 시간 ≥ s·Δt = 800 ms → 비동기 성립 불가, 모델 축소 또는 K=5; 컴파일 실패 시 UNet → MLP 계열로 아키텍처 교체 |
| phase1 | PushT keypoint flow-matching chunk 정책 CPU 학습 (seed 3개), 동기 실행 평가 | 동기 B1: 평균 max coverage ≥ 0.85, SR ≥ 50% (500 ep) | coverage < 0.7 이 학습 설정 조정 2회 후에도 지속 → 데이터 전처리·정규화 점검 후 아키텍처 변경 |
| phase2 | 고정 d / 실측 환산 d 비동기 평가, B1~B6·P 전체 및 RQ1~RQ4 | 4.2~4.5 표 전부 채움; B2 대비 P 의 J_b 감소가 d ≥ 2 에서 95% CI 상 유의 | d ≥ 2 에서 N=1024 로도 P 가 B3·B4 대비 SR·J_b 모두 개선 없음 → 가설 기각, 원인(ESS 붕괴/λ 스케일) 분석 보고 |
| phase3 (후순위) | LIBERO + π0 계열 확장 | phase2 완료 기준 충족 후 착수 | – |

---

## 6. 위험 요인과 대응

| 위험 | 관측 신호 | 대응 |
|---|---|---|
| particle 붕괴 (선행 연구 보고) | ESS/N < 0.05 지속, U → 1/N | η 증가, λ_max 감소, τ² 형태로 초기 가중치 완화, γ 낮춤 |
| λ 스케일 부적합 | 모든 particle 가중치 균일(무효) 또는 1개 독점 | c 를 Σ W 로 정규화한 정의 유지, λ_max grid 로 ESS/N 0.2~0.8 구간 선택 |
| NPU 수치 정밀도 차이 | phase0 정합성 기준 미달 | 컴파일 dtype 설정 확인, 문제 연산 분리해 원인 연산 수정 |
| 정적 batch 제약 | N/C 조합마다 재컴파일 필요 | 실험 전 필요한 batch 목록을 확정해 일괄 컴파일, 아티팩트 재사용 |
| 호스트 왕복 비용 지배 | 전송+gather 가 NPU 시간 이상 | 상태 텐서를 [N,32] 로 평탄화, 전송 횟수 = K 로 고정, island 모드 비교 |
| 실측 지연이 Δt=100 ms 보다 작아 d_meas = 0 | 실측 모드에서 비교 무의미 | 고정 d 모드를 주 결과로 사용, 실측 모드는 보조 표로 보고 |
| d ≤ H−s=8 상한 | d=8 에서 soft 구간 없음 | d=8 은 hard 구간 비교로 해석; 필요 시 H=32, s=16 추가 실험 |
| PushT 에서 모드 전환이 드묾 | B2 의 모드 전환율 < 5% | θ 민감도 보고, 접촉 직전 구간만 분리 집계 |
| 비교군 B6 의 CPU 지연이 과도 | RTC p99 ≫ NPU | 고정 d 모드에서 품질만 비교 (RQ1 지연 모드 a), 실측 모드에서 별도 보고 |

---

## 7. 공정 비교 원칙

| 원칙 | 구체 규칙 |
|---|---|
| 동일 정책 | 모든 비교군은 같은 checkpoint, 같은 K, 같은 관측 전처리 사용 |
| 동일 초기 상태 | env seed 목록 고정 (튜닝 0~199, 평가 100000~100499), 모든 방법 동일 목록 |
| 동일 노이즈 | particle 0 의 초기 ε 를 B1~B4·B6 의 ε 와 동일 seed 로 생성 |
| 연산량 표기 | 모든 표에 NFE(N·K)와 wall time 을 함께 기재; B5 와 P 는 NFE 일치 |
| 동일 튜닝 예산 | P (λ_max, η, γ) 와 B6 (β) 모두 튜닝용 초기 상태에서 동일 수(≤ 15 조합)로 탐색 후 평가 집합에서 1회 평가 |
| 지연 동일 적용 | 고정 d 모드에서는 모든 비동기 방법에 동일 d; 실측 모드에서는 각 방법 자기 p99 로 d_meas 산출 |
| 통계 | SR 은 Wilson 95% CI, 연속 지표는 에피소드 단위 bootstrap 1000회 95% CI, 정책 seed 간 평균 ± 표준편차 |
| 측정 환경 | wall time 은 단독 실행(다른 NPU·CPU 부하 없음), warm-up 50 chunk 제외 후 측정 |

---

## 출처

- [S1] Chi et al., Diffusion Policy, arXiv 2303.04137 — PushT T_o=2, T_a=8, T_p=16 (https://arxiv.org/pdf/2303.04137)
- [S2] Black et al., Real-Time Execution of Action Chunking Flow Policies, arXiv 2506.07339 — Eq.2~5, Table 4 (시뮬레이션 H=8, n=5, β=5; 실기 H=50, s_min=25), BID N=32, 2048 rollout/점 (https://arxiv.org/html/2506.07339v2)
- [S3] LeRobot `diffusion_pusht_keypoints` config — horizon 16, n_obs_steps 2, n_action_steps 8, num_inference_steps 10, eval 50 ep, episode_length 300 (https://huggingface.co/lerobot/diffusion_pusht_keypoints/blob/main/config.yaml)
- [S4] LeRobot `diffusion_pusht` 모델 카드 — 성공 = max overlap ≥ 0.95, 500 ep 평가 (https://huggingface.co/lerobot/diffusion_pusht)
