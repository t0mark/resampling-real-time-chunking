"""NPU 비동기 실행 평가: 역전파 없는 방법 (naive, replacement, BID, RS-RTC) 을 NPU 고정 batch 로 평가한다.

결과: results/metrics/eval_async/{방법}_d{지연}.csv (에피소드별), 로그 results/logs/eval_npu_*.log
실행 (NPU 컨테이너 airlab_hw_rtc 내부, rs_rtc/ 에서):
    python3 scripts/eval_npu.py                                       # 설정의 모든 조합
    python3 scripts/eval_npu.py --methods rs_rtc_n64 --delays 4 --episodes 20 --npu-device 3
"""
import argparse

import torch
import yaml

from rs_rtc import PROJECT_ROOT, RESULTS_DIR
from rs_rtc.evaluation import evaluate_combination
from rs_rtc.npu.compile import compile_if_missing
from rs_rtc.npu.runtime import NpuDenoiserRuntime
from rs_rtc.policies.pretrained import load_pretrained_policy
from rs_rtc.samplers.base import DdimChunkSampler
from rs_rtc.samplers.baselines import ReplacementSampler
from rs_rtc.samplers.bid import BidSampler
from rs_rtc.samplers.ddim import DdimSchedule
from rs_rtc.samplers.rs_rtc import RsRtcSampler
from rs_rtc.utils.logging import setup_logger

COMPILED_DIR = RESULTS_DIR / "compiled"
NPU_SAMPLER_CLASSES = {"vanilla": DdimChunkSampler, "replacement": ReplacementSampler, "bid": BidSampler}

LOGGER = setup_logger("rs_rtc", "eval_npu")


def parse_arguments():
    parser = argparse.ArgumentParser(description="NPU 비동기 실행 평가")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_npu.yaml"))
    parser.add_argument("--methods", nargs="+", default=None, help="평가할 방법 이름 (기본: 설정 전체)")
    parser.add_argument("--delays", type=int, nargs="+", default=None, help="지연 목록 대체")
    parser.add_argument("--episodes", type=int, default=None, help="에피소드 수 대체")
    parser.add_argument("--npu-device", type=int, default=None, help="NPU 장치 번호 대체")
    parser.add_argument("--cpu-threads", type=int, default=None, help="torch CPU 스레드 수 대체")
    return parser.parse_args()


def build_chunk_generator(method_config, policy, schedule, npu_device):
    """방법 설정으로 NPU 고정 batch chunk 생성기를 만든다."""
    sampler_arguments = (schedule, policy.horizon, policy.unet.action_dim, policy.unet.time_embed_dim)
    method_type = method_config["type"]

    # NPU 고정 batch 런타임 (없으면 컴파일)
    particle_count = method_config["particles"]
    compiled_path = COMPILED_DIR / f"pusht_unet_b{particle_count}.rbln"
    compile_if_missing(policy.unet, particle_count, compiled_path)
    denoiser = NpuDenoiserRuntime(compiled_path, npu_device, particle_count)

    # 방법별 샘플러
    if method_type == "rs_rtc":
        return RsRtcSampler(denoiser, *sampler_arguments, method_config["eta"], method_config["guidance_strength"],
                            method_config["ess_threshold"], method_config["use_replacement"])
    return NPU_SAMPLER_CLASSES[method_type](denoiser, *sampler_arguments, method_config["eta"])


def main():
    arguments = parse_arguments()

    # 설정 로딩
    with open(arguments.config) as config_file:
        config = yaml.safe_load(config_file)
    method_names = arguments.methods or list(config["methods"].keys())
    delays = arguments.delays if arguments.delays is not None else config["delays"]
    episode_count = arguments.episodes or config["episodes"]
    npu_device = arguments.npu_device if arguments.npu_device is not None else config["npu_device"]
    cpu_threads = arguments.cpu_threads or config["cpu_threads"]
    LOGGER.info("설정: %s, 방법=%s, 지연=%s, 에피소드=%d, NPU=%d, CPU 스레드=%d", arguments.config, method_names,
                delays, episode_count, npu_device, cpu_threads)

    # 병렬 실행 시 호스트 연산 경합을 막기 위해 CPU 스레드 수 제한
    torch.set_num_threads(cpu_threads)

    # 정책과 DDIM 스케줄
    policy = load_pretrained_policy()
    schedule = DdimSchedule(policy.config["num_train_timesteps"], policy.config["num_inference_steps"],
                            policy.config["clip_sample_range"])

    # 방법 × 지연 조합 평가
    for method_name in method_names:
        chunk_generator = build_chunk_generator(config["methods"][method_name], policy, schedule, npu_device)
        for delay_steps in delays:
            evaluate_combination(method_name, chunk_generator, delay_steps, policy, config, episode_count)


if __name__ == "__main__":
    main()
