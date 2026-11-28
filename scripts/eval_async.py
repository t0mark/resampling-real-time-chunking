"""비동기 실행 평가: 방법 × 지연 d 조합을 지정한 backend (npu | cuda | cpu) 에서 평가한다.

backend 우선순위: --backend > 방법별 backend > 설정 전체 backend. RTC (역전파) 는 cuda·cpu 에서만 실행된다.
결과: results/metrics/{설정 metrics_subdir, 기본 eval_async}/{방법}_d{지연}.csv (에피소드별), 로그 results/logs/eval_async_*.log
실행 (rs_rtc/ 에서, NPU 는 컨테이너 airlab_hw_rtc, GPU 는 airlab_hw_rtc_gpu 내부):
    python3 scripts/eval_async.py                                                  # 설정의 모든 조합
    python3 scripts/eval_async.py --methods rs_rtc_n64 --delays 4 --episodes 20 --device-index 3
    python3 scripts/eval_async.py --methods rtc_paper rs_rtc_n64_gpu               # GPU 방법 (GPU 컨테이너)
    python3 scripts/eval_async.py --methods rtc_paper --backend cpu --episodes 1   # GPU 없이 동작 확인
    python3 scripts/eval_async.py --config configs/render.yaml --methods rs_rtc_n64 --render  # 렌더 기록(npz) 저장
"""
import os

# OpenMP·OpenBLAS 스레드 풀 크기 제한 (numpy·torch·rebel import 전에 설정해야 적용됨, 외부 지정값 우선)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse

import torch
import yaml

from rs_rtc import PROJECT_ROOT
from rs_rtc.backends import BACKENDS
from rs_rtc.evaluation import RENDER_DIR, evaluate_combination, load_task_policy
from rs_rtc.samplers.ddim import DdimSchedule
from rs_rtc.samplers.factory import build_sampler
from rs_rtc.utils.logging import setup_logger

LOGGER = setup_logger("rs_rtc", "eval_async")


def parse_arguments():
    parser = argparse.ArgumentParser(description="비동기 실행 평가")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_async.yaml"))
    parser.add_argument("--task", default=None, help="설정의 task 대체 (pusht, square_ph, tool_hang_ph)")
    parser.add_argument("--methods", nargs="+", default=None, help="평가할 방법 이름 (기본: 설정 전체)")
    parser.add_argument("--delays", type=int, nargs="+", default=None, help="지연 목록 대체")
    parser.add_argument("--episodes", type=int, default=None, help="에피소드 수 대체")
    parser.add_argument("--backend", choices=BACKENDS, default=None, help="모든 방법의 실행 장치 대체")
    parser.add_argument("--device-index", type=int, default=None, help="NPU·CUDA 장치 번호 대체")
    parser.add_argument("--cpu-threads", type=int, default=None, help="torch CPU 스레드 수 대체")
    parser.add_argument("--render", action="store_true", help="에피소드별 궤적과 chunk 계획·후보 경로를 results/render/ 에 npz 로 저장")
    return parser.parse_args()


def log_device_info(backend, device_index):
    """CUDA 사용 시 GPU 이름과 torch·CUDA 버전 기록."""
    if backend == "cuda":
        LOGGER.info("GPU: %s, torch %s, CUDA %s", torch.cuda.get_device_name(device_index), torch.__version__,
                    torch.version.cuda)


def main():
    arguments = parse_arguments()

    # 설정 로딩
    with open(arguments.config) as config_file:
        config = yaml.safe_load(config_file)
    if arguments.task is not None:
        config["task"] = arguments.task
    method_names = arguments.methods or list(config["methods"].keys())
    default_delays = arguments.delays if arguments.delays is not None else config["delays"]
    episode_count = arguments.episodes or config["episodes"]
    device_index = arguments.device_index if arguments.device_index is not None else config["device_index"]
    cpu_threads = arguments.cpu_threads or config["cpu_threads"]
    LOGGER.info("설정: %s, task=%s, 방법=%s, 지연=%s, 에피소드=%d, 장치 번호=%d, CPU 스레드=%d", arguments.config,
                config.get("task", "pusht"), method_names, default_delays, episode_count, device_index, cpu_threads)

    # 병렬 실행 시 호스트 연산 경합을 막기 위해 CPU 스레드 수 제한
    torch.set_num_threads(cpu_threads)

    # 렌더 기록 저장 위치 (--render 일 때만)
    render_dir = RENDER_DIR if arguments.render else None
    LOGGER.info("렌더 기록 저장 위치: %s", render_dir)

    # 정책과 DDIM 스케줄
    policy = load_task_policy(config)
    schedule = DdimSchedule(policy.config["num_train_timesteps"], policy.config["num_inference_steps"],
                            policy.config["clip_sample_range"])

    # 방법 × 지연 조합 평가
    for method_name in method_names:
        method_config = config["methods"][method_name]

        # backend: --backend > 방법별 > 설정 전체
        backend = arguments.backend or method_config.get("backend", config["backend"])
        LOGGER.info("방법 %s: type=%s, backend=%s", method_name, method_config["type"], backend)
        log_device_info(backend, device_index)
        chunk_generator = build_sampler(method_config, policy, schedule, backend, device_index)

        # 지연 목록: --delays > 방법별 delays > 설정 전체 delays
        method_delays = default_delays if arguments.delays is not None else method_config.get("delays", default_delays)
        for delay_steps in method_delays:
            evaluate_combination(method_name, chunk_generator, delay_steps, policy, config, episode_count,
                                 render_dir, method_config.get("execute_steps"),
                                 method_config.get("frozen_only", False))


if __name__ == "__main__":
    main()
