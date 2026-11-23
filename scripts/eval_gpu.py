"""GPU 비동기 실행 평가: 역전파(VJP) 가이던스 RTC 를 CUDA autograd 로, RS-RTC 를 CUDA 고정 batch 로 평가한다 (RBLN SDK 불필요).

결과: results/metrics/eval_async/{방법}_d{지연}.csv (에피소드별), 로그 results/logs/eval_gpu_*.log
실행 (GPU 컨테이너 airlab_hw_rtc_gpu 내부, rs_rtc/ 에서):
    python3 scripts/eval_gpu.py                                       # 설정의 모든 조합
    python3 scripts/eval_gpu.py --delays 4 --episodes 5               # 동작 확인
    python3 scripts/eval_gpu.py --device cpu                          # GPU 없이 확인
    python3 scripts/eval_gpu.py --config configs/render.yaml --methods rtc_paper --render   # 렌더 기록(npz) 저장
    python3 scripts/eval_gpu.py --methods rs_rtc_n64_gpu --delays 4                        # RS-RTC GPU 실행
"""
import os

# OpenMP·OpenBLAS 스레드 풀 크기 제한 (numpy·torch·rebel import 전에 설정해야 적용됨, 외부 지정값 우선)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse

import torch
import yaml

from rs_rtc import PROJECT_ROOT
from rs_rtc.evaluation import RENDER_DIR, evaluate_combination
from rs_rtc.policies.pretrained import load_pretrained_policy
from rs_rtc.samplers.ddim import DdimSchedule
from rs_rtc.samplers.rs_rtc import RsRtcSampler
from rs_rtc.samplers.rtc_vjp import RtcPaperSampler, RtcVjpSampler
from rs_rtc.utils.logging import setup_logger

RTC_SAMPLER_CLASSES = {"rtc_vjp": RtcVjpSampler, "rtc_paper": RtcPaperSampler}

LOGGER = setup_logger("rs_rtc", "eval_gpu")


class GpuDenoiser:
    """UNet 을 CUDA 에서 고정 batch 로 실행 (입출력은 호스트 CPU 텐서, NPU 런타임과 같은 호출 방식)."""

    def __init__(self, unet, device, batch_size):
        self._device = torch.device(device)
        self._unet = unet.to(self._device)
        self._batch_size = batch_size
        LOGGER.info("GPU 노이즈 예측기: device=%s, batch=%d", self._device, batch_size)

    @property
    def batch_size(self):
        return self._batch_size

    def __call__(self, noisy_action, time_features, global_cond):
        # 고정 batch 확인 (NPU 정적 그래프와 같은 조건)
        if noisy_action.shape[0] != self._batch_size:
            raise ValueError(f"입력 batch {noisy_action.shape[0]} ≠ 설정 batch {self._batch_size}")

        # 장치로 전송 후 실행, 결과를 호스트로 복사
        with torch.no_grad():
            epsilon = self._unet(noisy_action.to(self._device), time_features.to(self._device),
                                 global_cond.to(self._device))
        return epsilon.cpu()


def parse_arguments():
    parser = argparse.ArgumentParser(description="GPU 비동기 실행 평가 (RTC 역전파)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_gpu.yaml"))
    parser.add_argument("--methods", nargs="+", default=None, help="평가할 방법 이름 (기본: 설정 전체)")
    parser.add_argument("--delays", type=int, nargs="+", default=None, help="지연 목록 대체")
    parser.add_argument("--episodes", type=int, default=None, help="에피소드 수 대체")
    parser.add_argument("--device", default=None, help="UNet 연산 장치 대체 (cuda, cuda:1, cpu)")
    parser.add_argument("--cpu-threads", type=int, default=None, help="torch CPU 스레드 수 대체")
    parser.add_argument("--render", action="store_true", help="에피소드별 궤적과 chunk 계획·후보 경로를 results/render/ 에 npz 로 저장")
    return parser.parse_args()


def build_chunk_generator(method_config, policy, schedule, device):
    """방법 설정으로 chunk 생성기를 만든다 (RTC 역전파 가이던스 또는 RS-RTC)."""
    sampler_arguments = (schedule, policy.horizon, policy.unet.action_dim, policy.unet.time_embed_dim)

    # RS-RTC: particle 수만큼 고정 batch 로 GPU 에서 실행
    if method_config["type"] == "rs_rtc":
        denoiser = GpuDenoiser(policy.unet, device, method_config["particles"])
        return RsRtcSampler(denoiser, *sampler_arguments, method_config["eta"], method_config["guidance_strength"],
                            method_config["ess_threshold"], method_config["use_replacement"])

    # RTC: 역전파 가이던스 샘플러
    sampler_class = RTC_SAMPLER_CLASSES[method_config["type"]]
    return sampler_class(policy.unet, *sampler_arguments, method_config["eta"], method_config["guidance_weight"], device)


def main():
    arguments = parse_arguments()

    # 설정 로딩
    with open(arguments.config) as config_file:
        config = yaml.safe_load(config_file)
    method_names = arguments.methods or list(config["methods"].keys())
    default_delays = arguments.delays if arguments.delays is not None else config["delays"]
    episode_count = arguments.episodes or config["episodes"]
    device = arguments.device or config["device"]
    cpu_threads = arguments.cpu_threads or config["cpu_threads"]
    LOGGER.info("설정: %s, 방법=%s, 지연=%s, 에피소드=%d, 장치=%s, CPU 스레드=%d", arguments.config, method_names,
                default_delays, episode_count, device, cpu_threads)

    # 장치 정보 기록
    if device.startswith("cuda"):
        LOGGER.info("GPU: %s, torch %s, CUDA %s", torch.cuda.get_device_name(torch.device(device)),
                    torch.__version__, torch.version.cuda)
    torch.set_num_threads(cpu_threads)

    # 렌더 기록 저장 위치 (--render 일 때만)
    render_dir = RENDER_DIR if arguments.render else None
    LOGGER.info("렌더 기록 저장 위치: %s", render_dir)

    # 정책과 DDIM 스케줄
    policy = load_pretrained_policy()
    schedule = DdimSchedule(policy.config["num_train_timesteps"], policy.config["num_inference_steps"],
                            policy.config["clip_sample_range"])

    # 방법 × 지연 조합 평가
    for method_name in method_names:
        method_config = config["methods"][method_name]
        chunk_generator = build_chunk_generator(method_config, policy, schedule, device)

        # 지연 목록: --delays > 방법별 delays > 설정 전체 delays
        method_delays = default_delays if arguments.delays is not None else method_config.get("delays", default_delays)
        for delay_steps in method_delays:
            evaluate_combination(method_name, chunk_generator, delay_steps, policy, config, episode_count,
                                 render_dir, method_config.get("execute_steps"))


if __name__ == "__main__":
    main()
