"""샘플러 공통 요소: inpainting 목표, 가중 cost, 표준 DDIM chunk 샘플러 (가이던스 없음).

모든 샘플러는 generate(global_cond [1, obs_dim], target 또는 None, generator) → (chunk [H, Da], 통계 dict) 를 제공한다.
chunk 는 정규화 [-1, 1] 공간이다.
"""
import torch

from rs_rtc.policies.diffusion_unet import sinusoidal_time_features


class InpaintingTarget:
    """이전 chunk 에서 온 목표 Y [H, Da], 일치 가중치 W [H], 확정 구간 길이 (index 0 … frozen_count−1)."""

    def __init__(self, prefix_actions, prefix_weights, frozen_count):
        self._prefix_actions = prefix_actions
        self._prefix_weights = prefix_weights
        self._frozen_count = frozen_count

    @property
    def prefix_actions(self):
        return self._prefix_actions

    @property
    def prefix_weights(self):
        return self._prefix_weights

    @property
    def frozen_count(self):
        return self._frozen_count


def weighted_inpainting_cost(clean_samples, target):
    """particle 별 가중 제곱오차 c_i = Σ_j W_j ‖Y_j − x̂_{i,j}‖² / Σ_j W_j, [N, H, Da] → [N]."""
    squared_error = ((clean_samples - target.prefix_actions.unsqueeze(0)) ** 2).sum(dim=-1)
    return (squared_error * target.prefix_weights.unsqueeze(0)).sum(dim=-1) / target.prefix_weights.sum()


class DdimChunkSampler:
    """노이즈 예측기와 DDIM 스케줄로 chunk 를 생성 (목표는 무시하는 단순 비동기 비교군)."""

    def __init__(self, denoiser, schedule, horizon, action_dim, time_embed_dim, eta=0.0):
        self._denoiser = denoiser
        self._schedule = schedule
        self._horizon = horizon
        self._action_dim = action_dim
        self._time_embed_dim = time_embed_dim
        self._eta = eta
        self._last_candidates = None

    @property
    def last_candidates(self):
        """마지막 generate 의 후보 chunk [N, H, Da] (정규화 공간, 후보를 쓰지 않는 방법은 None)."""
        return self._last_candidates

    @property
    def schedule(self):
        return self._schedule

    @property
    def denoiser(self):
        return self._denoiser

    @property
    def batch_size(self):
        """NPU 고정 batch (CPU 는 1)."""
        return self._denoiser.batch_size or 1

    def time_features(self, step_index, batch_size):
        """step_index 의 timestep 사인 특징 [batch_size, time_embed_dim] (호스트 fp32)."""
        timesteps = self._schedule.timesteps[step_index].repeat(batch_size)
        return sinusoidal_time_features(timesteps, self._time_embed_dim)

    def initial_noise(self, batch_size, generator):
        """x_T ~ N(0, I), [batch_size, H, Da]."""
        return torch.randn(batch_size, self._horizon, self._action_dim, generator=generator)

    def sample(self, global_cond, generator):
        """조건 [B, obs_dim] 마다 chunk 1개씩 생성해 [B, H, Da] 반환."""
        batch_size = global_cond.shape[0]
        noisy_sample = self.initial_noise(batch_size, generator)

        # K step 역확산
        for step_index in range(self._schedule.num_inference_steps):
            epsilon = self._denoiser(noisy_sample, self.time_features(step_index, batch_size), global_cond)
            noisy_sample, _ = self._schedule.step(noisy_sample, epsilon, step_index, self._eta, generator)
        return noisy_sample

    def generate(self, global_cond, target, generator):
        """batch 전체를 생성하고 0번 chunk 를 반환 (목표 무시)."""
        chunks = self.sample(global_cond.expand(self.batch_size, -1), generator)
        return chunks[0], {"denoiser_calls": self._schedule.num_inference_steps}
