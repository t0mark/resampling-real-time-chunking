"""비동기 실행 평가 공통 요소: 방법 1개 × 지연 1개 조합의 에피소드 반복과 CSV 저장.

NPU 평가 (scripts/eval_npu.py) 와 GPU 평가 (scripts/eval_gpu.py) 가 함께 사용한다.
결과: results/metrics/{설정 metrics_subdir, 기본 eval_async}/{방법}_d{지연}.csv (에피소드별)
"""
import csv
import logging

import numpy as np
import torch

from rs_rtc import RESULTS_DIR
from rs_rtc.envs.async_executor import AsyncChunkExecutor
from rs_rtc.envs.pusht import PushTEnvironment

METRICS_ROOT = RESULTS_DIR / "metrics"
DEFAULT_METRICS_SUBDIR = "eval_async"

LOGGER = logging.getLogger("rs_rtc")


def save_episode_rows(episode_rows, method_name, delay_steps, metrics_dir):
    """조합 1개의 에피소드별 결과를 CSV 로 저장."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{method_name}_d{delay_steps}.csv"
    with open(metrics_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)
    LOGGER.info("결과 저장: %s", metrics_path)


def evaluate_combination(method_name, chunk_generator, delay_steps, policy, config, episode_count):
    """방법 1개 × 지연 1개 조합을 평가하고 요약을 로그로 남긴다."""
    environment = PushTEnvironment(policy.n_obs_steps, config["max_episode_steps"])
    executor = AsyncChunkExecutor(environment, policy, chunk_generator, delay_steps)
    generator = torch.Generator().manual_seed(config["sampler_seed"])

    # 에피소드 반복
    episode_rows = []
    for episode_index in range(episode_count):
        episode_result = executor.run_episode(config["seed_start"] + episode_index, generator)
        episode_rows.append({"method": method_name, "delay": delay_steps, "episode": episode_index, **episode_result})
        LOGGER.info("[%s d=%d] 에피소드 %d/%d: 성공=%d, 최대 보상=%.3f, 전환 변화량=%.2f px, 내부 변화량=%.2f px",
                    method_name, delay_steps, episode_index + 1, episode_count, episode_result["success"],
                    episode_result["max_reward"], episode_result["boundary_jump_px"],
                    episode_result["interior_step_px"])

    # 요약
    metrics_dir = METRICS_ROOT / config.get("metrics_subdir", DEFAULT_METRICS_SUBDIR)
    save_episode_rows(episode_rows, method_name, delay_steps, metrics_dir)
    LOGGER.info("요약 [%s d=%d, %d 에피소드]: 성공률 %.1f %%, 평균 최대 보상 %.3f, 전환 변화량 %.2f px, "
                "내부 변화량 %.2f px, 생성 p50 %.1f ms", method_name, delay_steps, episode_count,
                np.mean([row["success"] for row in episode_rows]) * 100,
                np.mean([row["max_reward"] for row in episode_rows]),
                np.nanmean([row["boundary_jump_px"] for row in episode_rows]),
                np.nanmean([row["interior_step_px"] for row in episode_rows]),
                np.mean([row["generation_p50_ms"] for row in episode_rows]))
