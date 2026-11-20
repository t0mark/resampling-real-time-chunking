"""제안 방법 RS-RTC: 역전파 없이 particle resampling 으로 하는 inpainting.

particle N 개를 고정 batch 로 함께 역확산한다. step k 마다
    1. ε̂ = UNet(x_t) (NPU, batch N)
    2. x̂_0 추정, 목표 cost c_i = Σ W_j ‖Y_j − x̂_{i,j}‖² / Σ W_j
    3. potential G_k = −λ_k c_i, 증분 log-weight = G_k − G_{k−1}
    4. ESS < γN 이면 systematic resampling (호스트 index gather), log-weight 초기화
    5. (선택) 확정 구간 치환, DDIM 갱신 (η > 0 이면 복제 particle 이 갈라짐)
마지막에 cost 가 가장 작은 particle 을 고른다.
"""
import torch

from rs_rtc.samplers.base import DdimChunkSampler, weighted_inpainting_cost


def systematic_resample(normalized_weights, generator):
    """systematic resampling: 정규화 가중치 [N] → 조상 index [N]."""
    particle_count = normalized_weights.shape[0]
    offsets = (torch.rand(1, generator=generator) + torch.arange(particle_count)) / particle_count
    cumulative_weights = torch.cumsum(normalized_weights, dim=0)
    cumulative_weights[-1] = 1.0
    return torch.searchsorted(cumulative_weights, offsets).clamp(max=particle_count - 1)


class RsRtcSampler(DdimChunkSampler):
    """Resampling 기반 RTC."""

    def __init__(self, denoiser, schedule, horizon, action_dim, time_embed_dim, eta, guidance_strength,
                 ess_threshold, use_replacement):
        super().__init__(denoiser, schedule, horizon, action_dim, time_embed_dim, eta)
        self._guidance_strength = guidance_strength
        self._ess_threshold = ess_threshold
        self._use_replacement = use_replacement

    def potential_scale(self, step_index):
        """λ_k = λ_max · (k + 1) / K (역확산이 진행될수록 강해짐)."""
        return self._guidance_strength * (step_index + 1) / self.schedule.num_inference_steps

    def generate(self, global_cond, target, generator):
        # 목표가 없으면 (첫 chunk) 표준 DDIM
        self._last_candidates = None
        if target is None:
            return super().generate(global_cond, target, generator)

        particle_count = self.batch_size
        batch_cond = global_cond.expand(particle_count, -1)
        noisy_sample = self.initial_noise(particle_count, generator)
        log_weights = torch.zeros(particle_count)
        previous_potential = torch.zeros(particle_count)
        step_count = self.schedule.num_inference_steps
        resample_count = 0
        ess_history = []

        for step_index in range(step_count):
            # 1~2. 노이즈 예측, 끝점 추정, 목표 cost
            epsilon = self.denoiser(noisy_sample, self.time_features(step_index, particle_count), batch_cond)
            clean_sample = self.schedule.predict_clean_sample(noisy_sample, epsilon, step_index)
            current_potential = -self.potential_scale(step_index) * weighted_inpainting_cost(clean_sample, target)

            # 3. 증분 log-weight
            log_weights = log_weights + current_potential - previous_potential
            previous_potential = current_potential

            # 4. ESS 확인 후 resampling (마지막 step 제외)
            normalized_weights = torch.softmax(log_weights, dim=0)
            effective_sample_size = float(1.0 / (normalized_weights ** 2).sum())
            ess_history.append(round(effective_sample_size, 1))
            if step_index < step_count - 1 and effective_sample_size < self._ess_threshold * particle_count:
                ancestor_indices = systematic_resample(normalized_weights, generator)
                clean_sample = clean_sample[ancestor_indices]
                epsilon = epsilon[ancestor_indices]
                previous_potential = previous_potential[ancestor_indices]
                log_weights = torch.zeros(particle_count)
                resample_count += 1

            # 5. (선택) 확정 구간 치환 후 DDIM 갱신
            if self._use_replacement:
                clean_sample[:, :target.frozen_count] = target.prefix_actions[:target.frozen_count]
            noisy_sample = self.schedule.step_from_clean(clean_sample, epsilon, step_index, self._eta, generator)

        # cost 가 가장 작은 particle 선택
        final_costs = weighted_inpainting_cost(noisy_sample, target)
        best_index = int(torch.argmin(final_costs))
        self._last_candidates = noisy_sample.clone()
        unique_particle_count = int(torch.unique(noisy_sample.flatten(start_dim=1), dim=0).shape[0])
        return noisy_sample[best_index], {"denoiser_calls": step_count, "resample_count": resample_count,
                                          "selected_cost": float(final_costs[best_index]),
                                          "unique_particles": unique_particle_count,
                                          "min_ess": min(ess_history)}
