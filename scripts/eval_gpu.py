"""GPU 비동기 실행 평가: 역전파(VJP) 가이던스 RTC 를 CUDA autograd 로 평가한다 (RBLN SDK 불필요).

결과: results/metrics/eval_async/{방법}_d{지연}.csv (에피소드별), 로그 results/logs/eval_gpu_*.log
실행 (GPU 컨테이너 airlab_hw_rtc_gpu 내부, rs_rtc/ 에서):
    python3 scripts/eval_gpu.py                                       # 설정의 모든 조합
    python3 scripts/eval_gpu.py --delays 4 --episodes 5               # 동작 확인
    python3 scripts/eval_gpu.py --device cpu                          # GPU 없이 확인
"""
import os

# OpenMP·OpenBLAS 스레드 풀 크기 제한 (numpy·torch·rebel import 전에 설정해야 적용됨, 외부 지정값 우선)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse

import torch
import yaml

from rs_rtc import PROJECT_ROOT
from rs_rtc.evaluation import evaluate_combination
from rs_rtc.policies.pretrained import load_pretrained_policy
from rs_rtc.samplers.ddim import DdimSchedule
from rs_rtc.samplers.rtc_vjp import RtcVjpSampler
from rs_rtc.utils.logging import setup_logger

LOGGER = setup_logger("rs_rtc", "eval_gpu")


def parse_arguments():
    parser = argparse.ArgumentParser(description="GPU 비동기 실행 평가 (RTC 역전파)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_gpu.yaml"))
    parser.add_argument("--methods", nargs="+", default=None, help="평가할 방법 이름 (기본: 설정 전체)")
    parser.add_argument("--delays", type=int, nargs="+", default=None, help="지연 목록 대체")
    parser.add_argument("--episodes", type=int, default=None, help="에피소드 수 대체")
    parser.add_argument("--device", default=None, help="UNet 연산 장치 대체 (cuda, cuda:1, cpu)")
    parser.add_argument("--cpu-threads", type=int, default=None, help="torch CPU 스레드 수 대체")
    return parser.parse_args()


def build_chunk_generator(method_config, policy, schedule, device):
    """방법 설정으로 역전파 가이던스 chunk 생성기를 만든다."""
    sampler_arguments = (schedule, policy.horizon, policy.unet.action_dim, policy.unet.time_embed_dim)
    return RtcVjpSampler(policy.unet, *sampler_arguments, method_config["eta"], method_config["guidance_weight"],
                         device)


def main():
    arguments = parse_arguments()

    # 설정 로딩
    with open(arguments.config) as config_file:
        config = yaml.safe_load(config_file)
    method_names = arguments.methods or list(config["methods"].keys())
    delays = arguments.delays if arguments.delays is not None else config["delays"]
    episode_count = arguments.episodes or config["episodes"]
    device = arguments.device or config["device"]
    cpu_threads = arguments.cpu_threads or config["cpu_threads"]
    LOGGER.info("설정: %s, 방법=%s, 지연=%s, 에피소드=%d, 장치=%s, CPU 스레드=%d", arguments.config, method_names,
                delays, episode_count, device, cpu_threads)

    # 장치 정보 기록
    if device.startswith("cuda"):
        LOGGER.info("GPU: %s, torch %s, CUDA %s", torch.cuda.get_device_name(torch.device(device)),
                    torch.__version__, torch.version.cuda)
    torch.set_num_threads(cpu_threads)

    # 정책과 DDIM 스케줄
    policy = load_pretrained_policy()
    schedule = DdimSchedule(policy.config["num_train_timesteps"], policy.config["num_inference_steps"],
                            policy.config["clip_sample_range"])

    # 방법 × 지연 조합 평가
    for method_name in method_names:
        chunk_generator = build_chunk_generator(config["methods"][method_name], policy, schedule, device)
        for delay_steps in delays:
            evaluate_combination(method_name, chunk_generator, delay_steps, policy, config, episode_count)


if __name__ == "__main__":
    main()
