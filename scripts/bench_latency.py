"""사전학습 UNet 1 step 그래프를 batch N 별로 NPU 컴파일하고 정합성·지연을 측정한다.

측정 항목 (batch N 마다):
    1. 컴파일 시간, NPU 함수 수, 그래프 분할 사유, 장치 메모리
    2. 정합성: 1 step ε 상대오차 ‖ε_npu − ε_cpu‖ / ‖ε_cpu‖, K step DDIM 샘플링 후 chunk 최대 절대오차
    3. 지연: NPU 1회 호출 p50/p99, K step 샘플링 전체(호스트 DDIM 갱신 포함) p50/p99, CPU 1회 호출 p50
결과: results/metrics/bench_latency.csv, 로그 results/logs/bench_latency_*.log

실행 (컨테이너 내부, rs_rtc/ 에서):
    python3 scripts/bench_latency.py
    python3 scripts/bench_latency.py --batch-sizes 1 64
"""
import argparse
import csv
import time
from functools import partial

import numpy as np
import torch
import yaml

from rs_rtc import PROJECT_ROOT, RESULTS_DIR
from rs_rtc.npu.compile import compile_if_missing, inspect_compiled_model
from rs_rtc.npu.runtime import CpuDenoiser, NpuDenoiserRuntime
from rs_rtc.policies.pretrained import load_pretrained_policy
from rs_rtc.samplers.base import DdimChunkSampler
from rs_rtc.samplers.ddim import DdimSchedule
from rs_rtc.utils.logging import setup_logger

COMPILED_DIR = RESULTS_DIR / "compiled"
METRICS_PATH = RESULTS_DIR / "metrics" / "bench_latency.csv"

LOGGER = setup_logger("rs_rtc", "bench_latency")


def parse_arguments():
    parser = argparse.ArgumentParser(description="NPU 지연·정합성 측정")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "bench_latency.yaml"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None, help="설정 파일의 batch 목록 대체")
    return parser.parse_args()


def sample_random_inputs(policy, batch_size, generator):
    """정규화 공간의 무작위 입력 (x_t, 시간 특징, 조건) 을 만든다."""
    noisy_action = torch.randn(batch_size, policy.horizon, policy.unet.action_dim, generator=generator)
    time_features = torch.randn(batch_size, policy.unet.time_embed_dim, generator=generator).clamp(-1.0, 1.0)
    global_cond = torch.rand(batch_size, policy.unet.obs_dim, generator=generator) * 2.0 - 1.0
    return noisy_action, time_features, global_cond


def measure_parity(policy, cpu_sampler, npu_sampler, batch_size, sample_count, seed):
    """NPU 와 CPU 의 1 step ε 상대오차, K step 샘플링 chunk 최대오차를 측정."""
    relative_errors = []
    chunk_max_errors = []
    for sample_index in range(sample_count):
        input_generator = torch.Generator().manual_seed(seed + sample_index)
        noisy_action, time_features, global_cond = sample_random_inputs(policy, batch_size, input_generator)

        # 1 step 상대오차
        cpu_epsilon = cpu_sampler.denoiser(noisy_action, time_features, global_cond)
        npu_epsilon = npu_sampler.denoiser(noisy_action, time_features, global_cond)
        relative_errors.append((torch.linalg.norm(npu_epsilon - cpu_epsilon) / torch.linalg.norm(cpu_epsilon)).item())

        # K step 샘플링 chunk 오차 (같은 초기 노이즈 seed)
        cpu_chunk = cpu_sampler.sample(global_cond, torch.Generator().manual_seed(seed))
        npu_chunk = npu_sampler.sample(global_cond, torch.Generator().manual_seed(seed))
        chunk_max_errors.append((npu_chunk - cpu_chunk).abs().max().item())
    return max(relative_errors), max(chunk_max_errors)


def measure_latency_ms(timed_function, warmup_iterations, measure_iterations):
    """워밍업 후 호출별 지연(ms) 배열을 측정."""
    # 워밍업
    for _ in range(warmup_iterations):
        timed_function()

    # 호출별 지연 측정
    latency_ms = np.empty(measure_iterations)
    for iteration_index in range(measure_iterations):
        call_start = time.perf_counter()
        timed_function()
        latency_ms[iteration_index] = (time.perf_counter() - call_start) * 1e3
    return latency_ms


def bench_batch_size(policy, schedule, batch_size, config):
    """batch 1개에 대해 컴파일 → 점검 → 정합성 → 지연 측정 후 결과 행을 반환."""
    LOGGER.info("===== batch N = %d =====", batch_size)

    # 컴파일·점검·로드
    compiled_path = COMPILED_DIR / f"pusht_unet_b{batch_size}.rbln"
    compile_start = time.perf_counter()
    compile_if_missing(policy.unet, batch_size, compiled_path)
    compile_seconds = time.perf_counter() - compile_start
    inspection = inspect_compiled_model(compiled_path)
    npu_denoiser = NpuDenoiserRuntime(compiled_path, config["npu_device"], batch_size)

    # CPU·NPU 샘플러
    sampler_arguments = (schedule, policy.horizon, policy.unet.action_dim, policy.unet.time_embed_dim)
    cpu_sampler = DdimChunkSampler(CpuDenoiser(policy.unet), *sampler_arguments)
    npu_sampler = DdimChunkSampler(npu_denoiser, *sampler_arguments)

    # 정합성
    max_relative_error, max_chunk_error = measure_parity(policy, cpu_sampler, npu_sampler, batch_size,
                                                         config["parity_samples"], config["seed"])
    LOGGER.info("정합성: 1 step ε 상대오차 max=%.2e, %d step chunk 최대오차=%.2e (정규화 단위)",
                max_relative_error, schedule.num_inference_steps, max_chunk_error)

    # NPU 1회 호출, K step 샘플링 전체, CPU 1회 호출 지연
    latency_generator = torch.Generator().manual_seed(config["seed"])
    noisy_action, time_features, global_cond = sample_random_inputs(policy, batch_size, latency_generator)
    sampling_iterations = max(config["measure_iterations"] // schedule.num_inference_steps, 10)
    npu_call_ms = measure_latency_ms(partial(npu_denoiser, noisy_action, time_features, global_cond),
                                     config["warmup_iterations"], config["measure_iterations"])
    npu_sampling_ms = measure_latency_ms(partial(npu_sampler.sample, global_cond, latency_generator),
                                         config["warmup_iterations"] // 4, sampling_iterations)
    cpu_call_ms = measure_latency_ms(partial(cpu_sampler.denoiser, noisy_action, time_features, global_cond),
                                     2, config["cpu_measure_iterations"])

    result_row = {
        "batch_size": batch_size,
        "compile_seconds": round(compile_seconds, 1),
        "device_func_count": inspection["device_func_count"],
        "graph_split_count": len(inspection["host_op_reasons"]),
        "device_alloc_mb": round(inspection["device_alloc_bytes"] / 2**20, 1),
        "max_relative_error": f"{max_relative_error:.2e}",
        "max_chunk_error": f"{max_chunk_error:.2e}",
        "npu_call_p50_ms": round(float(np.percentile(npu_call_ms, 50)), 3),
        "npu_call_p99_ms": round(float(np.percentile(npu_call_ms, 99)), 3),
        "npu_sampling_p50_ms": round(float(np.percentile(npu_sampling_ms, 50)), 3),
        "npu_sampling_p99_ms": round(float(np.percentile(npu_sampling_ms, 99)), 3),
        "cpu_call_p50_ms": round(float(np.percentile(cpu_call_ms, 50)), 3),
    }
    LOGGER.info("지연: NPU 호출 p50=%.3f / p99=%.3f ms, %d step 샘플링 p50=%.3f / p99=%.3f ms, CPU 호출 p50=%.3f ms",
                result_row["npu_call_p50_ms"], result_row["npu_call_p99_ms"], schedule.num_inference_steps,
                result_row["npu_sampling_p50_ms"], result_row["npu_sampling_p99_ms"], result_row["cpu_call_p50_ms"])
    return result_row


def save_result_rows(result_rows):
    """결과 행을 CSV 로 저장."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)
    LOGGER.info("결과 저장: %s", METRICS_PATH)


def main():
    arguments = parse_arguments()

    # 설정 로딩
    with open(arguments.config) as config_file:
        config = yaml.safe_load(config_file)
    batch_sizes = arguments.batch_sizes or config["batch_sizes"]
    LOGGER.info("설정: %s, batch=%s", arguments.config, batch_sizes)

    # 사전학습 정책과 DDIM 스케줄
    policy = load_pretrained_policy()
    schedule = DdimSchedule(policy.config["num_train_timesteps"], policy.config["num_inference_steps"],
                            policy.config["clip_sample_range"])

    # batch 별 측정
    result_rows = [bench_batch_size(policy, schedule, batch_size, config) for batch_size in batch_sizes]
    save_result_rows(result_rows)


if __name__ == "__main__":
    main()
