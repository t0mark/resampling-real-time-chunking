# 논문 구성 정리: RS-RTC

## 핵심 내용

- **문제**
  - action chunking 정책은 chunk를 비동기로 생성해야 추론 지연 동안에도 로봇이 멈추지 않는다.
  - 비동기로 생성하면 새 chunk가 이전 chunk와 어긋나서, chunk가 바뀔 때 행동이 튄다.
  - RTC는 이 문제를 역전파(VJP) 가이던스 inpainting으로 푼다. 새 chunk의 앞부분을 이미 실행이 확정된 행동에 맞춘다.
  - 추론 전용 NPU(Rebellions ATOM-Max)는 역전파를 지원하지 않아서 RTC를 실행할 수 없다.
- **제안: RS-RTC**
  - 역전파 가이던스를 particle resampling으로 바꾼다. 순전파만 쓰므로 NPU에서 실행된다.
  - 동작 방식
    1. particle N개를 고정 batch로 함께 역확산한다.
    2. 매 step 예측 x̂₀와 확정 구간 사이의 가중 오차로 가중치를 매긴다 (potential −λ·cost).
    3. ESS가 임계값보다 작으면 systematic resampling을 한다.
    4. η > 0인 DDIM으로 복제된 particle을 다시 갈라지게 한다.
    5. 마지막에 오차가 가장 작은 particle을 고른다.
- **주장**
  - RS-RTC는 역전파 없이 NPU에서 RTC와 같은 성공률을 낸다.
  - chunk 전환은 RTC보다 더 매끄럽다.
  - 생성 속도는 RTC보다 2.6배 빠르다.
- **주요 수치** (PushT, 지연 d별 500 에피소드)
  - **성공률:** 모든 d에서 RTC와 95% CI가 겹친다. 예를 들어 d = 4에서 RS-RTC 65.0%, RTC 61.8%다.
  - **prefix 오차와 전환 변화량:** 모든 d에서 RS-RTC가 가장 작다. d = 7에서 prefix 오차는 0.0149 대 0.0168, 전환 변화량은 30.7 px 대 32.7 px다.
  - **생성 시간:** RS-RTC는 NPU에서 약 110 ms, RTC는 H200에서 약 287 ms다.
  - **chunk 없는 원본:** 매 스텝 재추론하면 성공률이 15.8%다. chunk 단위 실행이 필요하다는 근거다.

## 전체 구성

1. **Introduction**
   - 비동기 chunk 실행의 필요성을 설명한다.
   - RTC의 한계(역전파 필요)와 추론 전용 가속기의 확산을 짚는다.
   - 기여는 세 가지다.
     - 역전파 없는 inpainting 방법(RS-RTC)
     - NPU에서의 실행
     - RTC와 같은 성공률에 더 매끄러운 전환과 더 빠른 생성
2. **Related Work**
   - action chunking과 diffusion policy
   - 비동기 실행: naive async, BID, RTC
   - diffusion guidance와 SMC/particle 방법 (역전파 없는 가이던스)
   - 추론 전용 가속기에서의 정책 실행
3. **Background**
   - DDIM 샘플링
   - 비동기 실행 모델: 실행 주기 s, 지연 d, 확정 구간
   - RTC의 VJP 가이던스와 τ 의존 가중치 w
4. **Method: RS-RTC**
   - 확정 구간 비용 c와 potential, 증분 가중치
   - ESS 기준 systematic resampling, η > 0 DDIM, 최종 선택
   - 고정 batch NPU 컴파일과 호스트 측 index gather
   - 알고리즘 박스 1개
5. **Experiments**
   - **설정**
     - 정책: `lerobot/diffusion_pusht_keypoints`
     - 환경: gym-pusht. 성공 기준은 겹침 ≥ 0.95다.
     - 지연 d = 0, 1, 2, 4, 7, 지연별 500 에피소드, seed 공통
     - 하드웨어: RTC는 H200, 나머지는 ATOM-Max
   - **비교군:** naive async, BID (N = 64), RTC (원 논문 공식 코드의 식, β = 5), chunk 없는 원본
   - **지표**
     - 성공률 (Wilson 95% CI)
     - 평균 최대 보상
     - 전환 변화량 (px)
     - prefix 오차
     - 생성 시간 p50
   - **주 결과:** Tab 1, Fig 1
   - **비용과 하드웨어:** Tab 2, Fig 4 (N에 따른 생성 시간. N = 256부터 RTC보다 느리다)
   - **동작 예시:** Fig 2 (chunk 전환 시계열과 후보 64개), Fig 3 (시간별 목표 겹침과 MuJoCo 프레임)
   - **하이퍼파라미터 선택:** Tab 3. λ와 β는 검증 seed, d = 4에서 골랐다.
6. **Discussion / Limitations**
   - 성공률 향상은 없다. 이득은 전환의 매끄러움과 실행 가능성(NPU)에 있다.
   - PushT 단일 과제에서만 평가했고, 지연은 시뮬레이션이다.
   - RTC와 RS-RTC는 측정 장치가 달라서 생성 시간 비교는 장치까지 포함한 비교다.
   - N이 커지면 생성 시간이 늘어난다 (Fig 4).
   - 후속 과제: Kinetix 등 동적 벤치마크, 실제 로봇
7. **Conclusion**

## 그림과 표 (`_tmp/paper/`)

| 번호 | 파일 | 내용 | 절 |
|---|---|---|---|
| Tab 1 | `tables/tab1_main` | 지연 d × 방법 4개: 성공률, 최대 보상, 전환 변화량, prefix 오차 | 5 주 결과 |
| Tab 2 | `tables/tab2_latency` | 장치, 역전파 여부, 생성 시간, 추론 횟수, d = 0 성공률 | 5 비용 |
| Tab 3 | `tables/tab3_tuning` | λ, β 선택 | 5 또는 부록 |
| Fig 1 | `figures/fig1_delay_sweep` | d에 따른 성공률, 전환 변화량, prefix 오차 | 5 주 결과 |
| Fig 2 | `figures/fig2_switch_trajectory` | chunk 전환 전후 행동 시계열 (RTC vs RS-RTC) | 5 동작 예시 |
| Fig 3 | `figures/fig3_render_frames` | 목표 겹침 시계열과 같은 시각의 MuJoCo 프레임 (ep0) | 5 동작 예시 |
| Fig 4 | `figures/fig4_latency_scaling` | particle 수 N에 따른 NPU 생성 시간과 RTC 기준선 | 5 비용 |
| Fig 0 | `figures/fig0_method_overview` | 방법 개념도: (a) 비동기 실행과 확정 구간, (b) RS-RTC 역확산 1 step (초안) | 4 |

- **다시 생성하기:** 컨테이너 안 `rs_rtc/` 폴더에서 아래 두 명령을 실행한다.
  - `python3 _tmp/paper/make_tables.py`
  - `python3 _tmp/paper/make_figures.py`
- **Fig 2, Fig 3:** 에피소드 하나를 보여주는 사례다. 성공률 차이의 근거로 쓰지 않는다.
