"""비동기 실행 평가 공통 요소: 방법 1개 × 지연 1개 조합의 에피소드 반복과 CSV 저장.

비동기 실행 평가 (scripts/eval_async.py) 가 사용한다.
결과: results/metrics/{설정 metrics_subdir, 기본 eval_async (PushT) / eval_async_{task}}/{방법}_d{지연}.csv (에피소드별)

렌더 기록 (render_dir 지정 시): {render_dir}/{방법}_d{지연}_ep{에피소드}.npz (영상은 scripts/render/ 에서 생성)
npz 항목 (좌표는 PushT 작업 공간 0~512 px, T = 실행 스텝 수, M = chunk 전환 횟수, H = chunk 길이, N = 후보 수):
    env_seed, delay, execute_steps, success, max_reward   에피소드 정보 (execute_steps = cycle 당 실행 스텝 s)
    goal_pose [3]                               목표 T 자세 (x, y, 각도 rad)
    executed_actions [T, 2]                     실행한 행동 (끝단 목표 좌표)
    agent_positions [T+1, 2]                    끝단 위치 (0 = 초기 상태, t = t 번째 행동 실행 후)
    block_poses [T+1, 3]                        T 블록 자세 (x, y, 각도 rad)
    switch_steps [M]                            새 chunk 의 첫 행동이 실행되는 실행 스텝 index
    previous_plans [M, H, 2], new_plans [M, H, 2]  전환 시점의 이전 chunk / 새 chunk 계획 경로
    candidate_plans [M, N, H, 2]                새 chunk 를 고른 후보 경로 (RS-RTC particle, 후보가 없으면 N = 0)
    previous_plan_offsets [M], new_plan_offsets [M]  계획 index j 는 실행 스텝 offset + j 에 대응
"""
import csv
import logging

import numpy as np
import torch

from rs_rtc import RESULTS_DIR
from rs_rtc.envs.async_executor import AsyncChunkExecutor
from rs_rtc.envs.pusht import PushTEnvironment
from rs_rtc.policies.pretrained import PUSHT_POLICY_NAME, load_pretrained_policy

METRICS_ROOT = RESULTS_DIR / "metrics"
DEFAULT_METRICS_SUBDIR = "eval_async"
RENDER_DIR = RESULTS_DIR / "render"
RENDER_ARRAY_KEYS = ("executed_actions", "agent_positions", "block_poses", "switch_steps", "previous_plans",
                     "new_plans", "candidate_plans", "previous_plan_offsets", "new_plan_offsets")

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


def save_render_record(render_record, episode_result, delay_steps, execute_steps, goal_pose, file_stem, render_dir):
    """에피소드 1개의 렌더 기록 (궤적, chunk 계획·후보 경로) 을 npz 로 저장."""
    render_dir.mkdir(parents=True, exist_ok=True)
    record_path = render_dir / f"{file_stem}.npz"
    record_arrays = {key: np.asarray(render_record[key]) for key in RENDER_ARRAY_KEYS}
    np.savez_compressed(record_path, env_seed=episode_result["env_seed"], delay=delay_steps,
                        execute_steps=execute_steps, success=episode_result["success"],
                        max_reward=episode_result["max_reward"], goal_pose=goal_pose, **record_arrays)
    LOGGER.info("렌더 기록 저장: %s (스텝 %d, 전환 %d 회, 후보 %d 개)", record_path.name,
                len(render_record["executed_actions"]), len(render_record["switch_steps"]),
                record_arrays["candidate_plans"].shape[1] if len(render_record["switch_steps"]) else 0)


def load_task_policy(config):
    """설정의 task (생략 시 pusht) 에 맞는 사전학습 정책 로딩 (robomimic 은 num_inference_steps 로 DDIM step 지정)."""
    task_name = config.get("task", PUSHT_POLICY_NAME)
    if task_name == PUSHT_POLICY_NAME:
        return load_pretrained_policy()
    from rs_rtc.policies.pretrained_robomimic import load_robomimic_policy

    return load_robomimic_policy(task_name, config["num_inference_steps"])


def build_environment(policy, config):
    """정책 task 에 맞는 환경 (max_episode_steps 생략 시 robomimic 은 체크포인트 평가 설정값)."""
    if policy.name == PUSHT_POLICY_NAME:
        return PushTEnvironment(policy.n_obs_steps, config["max_episode_steps"])
    from rs_rtc.envs.robomimic_env import RobomimicEnvironment

    return RobomimicEnvironment(policy.config, config.get("max_episode_steps", policy.config["max_episode_steps"]))


def evaluate_combination(method_name, chunk_generator, delay_steps, policy, config, episode_count, render_dir=None,
                         execute_steps=None, frozen_only=False):
    """방법 1개 × 지연 1개 조합을 평가하고 요약을 로그로 남긴다 (render_dir 지정 시 에피소드별 렌더 기록 저장)."""
    environment = build_environment(policy, config)
    executor = AsyncChunkExecutor(environment, policy, chunk_generator, delay_steps, execute_steps,
                                  render_dir is not None, frozen_only)
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

        # 렌더 기록 저장
        if render_dir is not None:
            save_render_record(executor.render_record, episode_result, delay_steps, executor.execute_steps,
                               environment.goal_pose, f"{method_name}_d{delay_steps}_ep{episode_index}", render_dir)

    # 요약
    default_subdir = DEFAULT_METRICS_SUBDIR if policy.name == PUSHT_POLICY_NAME else f"{DEFAULT_METRICS_SUBDIR}_{policy.name}"
    metrics_dir = METRICS_ROOT / config.get("metrics_subdir", default_subdir)
    save_episode_rows(episode_rows, method_name, delay_steps, metrics_dir)
    LOGGER.info("요약 [%s d=%d, %d 에피소드]: 성공률 %.1f %%, 평균 최대 보상 %.3f, 전환 변화량 %.2f px, "
                "내부 변화량 %.2f px, 생성 p50 %.1f ms", method_name, delay_steps, episode_count,
                np.mean([row["success"] for row in episode_rows]) * 100,
                np.mean([row["max_reward"] for row in episode_rows]),
                np.nanmean([row["boundary_jump_px"] for row in episode_rows]),
                np.nanmean([row["interior_step_px"] for row in episode_rows]),
                np.mean([row["generation_p50_ms"] for row in episode_rows]))
