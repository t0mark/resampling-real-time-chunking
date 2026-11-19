"""샘플러·실행기 단위 테스트 (가짜 노이즈 예측기 사용, NPU 불필요)."""
import torch

from rs_rtc.envs.async_executor import build_prefix_weights
from rs_rtc.samplers.base import DdimChunkSampler, InpaintingTarget, weighted_inpainting_cost
from rs_rtc.samplers.ddim import DdimSchedule
from rs_rtc.samplers.rs_rtc import RsRtcSampler, systematic_resample

HORIZON = 16
ACTION_DIM = 2
TIME_EMBED_DIM = 128
OBS_DIM = 36
PARTICLE_COUNT = 64


class ZeroDenoiser:
    """항상 ε̂ = 0 을 내는 가짜 예측기 (x̂_0 = x_t / √ᾱ_t, 분포가 넓게 퍼짐)."""

    def __init__(self, batch_size):
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    def __call__(self, noisy_action, time_features, global_cond):
        return torch.zeros_like(noisy_action)


def build_schedule():
    return DdimSchedule(100, 10, 1.0)


def build_target(frozen_count):
    prefix_actions = torch.full((HORIZON, ACTION_DIM), 0.3)
    prefix_weights = build_prefix_weights(HORIZON, frozen_count, HORIZON - 1 - 8)
    return InpaintingTarget(prefix_actions, prefix_weights, frozen_count)


def test_prefix_weights_shape_and_range():
    prefix_weights = build_prefix_weights(HORIZON, 5, 7)
    assert torch.all(prefix_weights[:5] == 1.0)
    assert torch.all((prefix_weights[5:8] > 0.0) & (prefix_weights[5:8] < 1.0))
    assert torch.all(prefix_weights[8:] == 0.0)
    assert torch.all(prefix_weights[5:8][1:] < prefix_weights[5:8][:-1])


def test_systematic_resample_keeps_heavy_particle():
    normalized_weights = torch.zeros(8)
    normalized_weights[3] = 1.0
    ancestor_indices = systematic_resample(normalized_weights, torch.Generator().manual_seed(0))
    assert torch.all(ancestor_indices == 3)


def test_rs_rtc_lowers_cost_against_vanilla():
    schedule = build_schedule()
    target = build_target(frozen_count=5)
    global_cond = torch.zeros(1, OBS_DIM)
    sampler_arguments = (schedule, HORIZON, ACTION_DIM, TIME_EMBED_DIM)

    # 같은 seed 로 단순 DDIM 과 RS-RTC 비교
    vanilla_sampler = DdimChunkSampler(ZeroDenoiser(PARTICLE_COUNT), *sampler_arguments, 0.0)
    rs_rtc_sampler = RsRtcSampler(ZeroDenoiser(PARTICLE_COUNT), *sampler_arguments, 0.5, 100.0, 0.5, False)
    vanilla_chunk, _ = vanilla_sampler.generate(global_cond, target, torch.Generator().manual_seed(0))
    guided_chunk, guided_stats = rs_rtc_sampler.generate(global_cond, target, torch.Generator().manual_seed(0))

    vanilla_cost = float(weighted_inpainting_cost(vanilla_chunk.unsqueeze(0), target))
    guided_cost = float(weighted_inpainting_cost(guided_chunk.unsqueeze(0), target))
    assert guided_stats["resample_count"] > 0
    assert guided_cost < vanilla_cost


def test_rs_rtc_without_guidance_never_resamples():
    schedule = build_schedule()
    target = build_target(frozen_count=3)
    rs_rtc_sampler = RsRtcSampler(ZeroDenoiser(PARTICLE_COUNT), schedule, HORIZON, ACTION_DIM, TIME_EMBED_DIM,
                                  0.0, 0.0, 0.5, False)
    _, guided_stats = rs_rtc_sampler.generate(torch.zeros(1, OBS_DIM), target, torch.Generator().manual_seed(0))
    assert guided_stats["resample_count"] == 0
    assert guided_stats["min_ess"] == PARTICLE_COUNT
