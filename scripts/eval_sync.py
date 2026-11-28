"""사전학습 Diffusion Policy 를 동기 실행으로 평가 (가중치·정규화·DDIM·NPU 이식 검증, task: pusht | robomimic).

동기 실행: n_action_steps 마다 chunk 를 생성하고 그동안 환경은 정지.
chunk 의 index (n_obs_steps − 1) 부터 n_action_steps 개를 실행 (lerobot 과 동일).
공개 기준값
    PushT (lerobot eval_info.json, 500 에피소드): 성공률 71.0 %, 평균 최대 보상 0.969
    robomimic (Diffusion Policy 체크포인트 이름의 test_mean_score, DDPM 100 step): square_ph 1.000, tool_hang_ph 0.818

결과: results/metrics/eval_sync_{backend}.csv (PushT) / eval_sync_{task}_{backend}.csv, 로그 results/logs/eval_sync_*.log
실행 (컨테이너 내부, rs_rtc/ 에서):
    python3 scripts/eval_sync.py
    python3 scripts/eval_sync.py --backend cpu --episodes 10
    python3 scripts/eval_sync.py --config configs/eval_sync_robomimic.yaml --task tool_hang_ph
"""
import os

# OpenMP·OpenBLAS 스레드 풀 크기 제한 (numpy·torch·rebel import 전에 설정해야 적용됨, 외부 지정값 우선)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import csv
import time

import numpy as np
import torch
import yaml

from rs_rtc import PROJECT_ROOT, RESULTS_DIR
from rs_rtc.evaluation import build_environment, load_task_policy
from rs_rtc.npu.compile import compile_if_missing
from rs_rtc.backends import CpuDenoiser, NpuDenoiserRuntime
from rs_rtc.policies.pretrained import PUSHT_POLICY_NAME
from rs_rtc.samplers.base import DdimChunkSampler
from rs_rtc.samplers.ddim import DdimSchedule
from rs_rtc.utils.logging import setup_logger

COMPILED_DIR = RESULTS_DIR / "compiled"
METRICS_DIR = RESULTS_DIR / "metrics"

LOGGER = setup_logger("rs_rtc", "eval_sync")


def parse_arguments():
    parser = argparse.ArgumentParser(description="사전학습 정책 동기 실행 평가")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_sync.yaml"))
    parser.add_argument("--task", default=None, help="설정의 task 대체 (pusht, square_ph, tool_hang_ph)")
    parser.add_argument("--backend", choices=["cpu", "npu"], default=None, help="설정 파일 값 대체")
    parser.add_argument("--episodes", type=int, default=None, help="설정 파일 값 대체")
    return parser.parse_args()


def build_denoiser(policy, backend, npu_device):
    """backend 에 맞는 노이즈 예측기 (NPU 는 batch 1 그래프)."""
    if backend == "cpu":
        return CpuDenoiser(policy.unet)
    compiled_path = COMPILED_DIR / f"{policy.name}_unet_b1.rbln"
    compile_if_missing(policy.unet, 1, compiled_path)
    return NpuDenoiserRuntime(compiled_path, npu_device, 1)


def run_episode(environment, policy, chunk_sampler, env_seed, generator):
    """에피소드 1개를 동기 실행하고 (최대 보상, 성공, step 수, chunk 생성 시간 목록) 반환."""
    environment.reset(env_seed)
    first_action_index = policy.n_obs_steps - 1
    chunk_latency_ms = []

    while not environment.is_done:
        # chunk 생성 (환경 정지)
        global_cond = policy.build_global_cond(*environment.observation_history())
        generation_start = time.perf_counter()
        normalized_chunk = chunk_sampler.sample(global_cond, generator)
        chunk_latency_ms.append((time.perf_counter() - generation_start) * 1e3)

        # 실행 구간 n_action_steps 개를 환경 단위로 실행
        pixel_chunk = policy.action_normalizer.unnormalize(normalized_chunk[0])
        for action_index in range(first_action_index, first_action_index + policy.n_action_steps):
            environment.step(pixel_chunk[action_index].numpy())
            if environment.is_done:
                break
    return environment.max_reward, environment.is_success, environment.step_count, chunk_latency_ms


def save_episode_rows(episode_rows, policy_name, backend):
    """에피소드별 결과를 CSV 로 저장 (PushT 외 task 는 파일 이름에 task 포함)."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    file_stem = f"eval_sync_{backend}" if policy_name == PUSHT_POLICY_NAME else f"eval_sync_{policy_name}_{backend}"
    metrics_path = METRICS_DIR / f"{file_stem}.csv"
    with open(metrics_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)
    LOGGER.info("결과 저장: %s", metrics_path)


def main():
    arguments = parse_arguments()

    # 설정 로딩
    with open(arguments.config) as config_file:
        config = yaml.safe_load(config_file)
    if arguments.task is not None:
        config["task"] = arguments.task
    backend = arguments.backend or config["backend"]
    episode_count = arguments.episodes or config["episodes"]
    LOGGER.info("설정: %s, task=%s, backend=%s, episodes=%d", arguments.config, config.get("task", PUSHT_POLICY_NAME),
                backend, episode_count)

    # 정책·샘플러·환경 구성
    policy = load_task_policy(config)
    schedule = DdimSchedule(policy.config["num_train_timesteps"], policy.config["num_inference_steps"],
                            policy.config["clip_sample_range"])
    denoiser = build_denoiser(policy, backend, config["npu_device"])
    chunk_sampler = DdimChunkSampler(denoiser, schedule, policy.horizon, policy.unet.action_dim,
                                     policy.unet.time_embed_dim, config["eta"])
    environment = build_environment(policy, config)
    generator = torch.Generator().manual_seed(config["sampler_seed"])

    # 에피소드 반복
    episode_rows = []
    all_latency_ms = []
    for episode_index in range(episode_count):
        env_seed = config["seed_start"] + episode_index
        max_reward, is_success, step_count, chunk_latency_ms = run_episode(environment, policy, chunk_sampler,
                                                                           env_seed, generator)
        all_latency_ms.extend(chunk_latency_ms)
        episode_rows.append({"episode": episode_index, "env_seed": env_seed, "max_reward": round(max_reward, 4),
                             "success": int(is_success), "steps": step_count})
        success_rate = np.mean([row["success"] for row in episode_rows]) * 100
        LOGGER.info("에피소드 %d/%d: seed=%d, 최대 보상=%.3f, 성공=%s, step=%d | 누적 성공률 %.1f %%",
                    episode_index + 1, episode_count, env_seed, max_reward, is_success, step_count, success_rate)

    # 요약
    save_episode_rows(episode_rows, policy.name, backend)
    LOGGER.info("요약 (%s, %d 에피소드): 성공률 %.1f %%, 평균 최대 보상 %.3f, chunk 생성 p50=%.1f ms / p99=%.1f ms",
                backend, episode_count, np.mean([row["success"] for row in episode_rows]) * 100,
                np.mean([row["max_reward"] for row in episode_rows]),
                np.percentile(all_latency_ms, 50), np.percentile(all_latency_ms, 99))


if __name__ == "__main__":
    main()
