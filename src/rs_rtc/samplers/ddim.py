"""DDIM 노이즈 스케줄과 1 step 갱신 (diffusers DDIMScheduler 와 같은 식, lerobot 기본 설정 기준).

설정: beta_schedule = squaredcos_cap_v2, set_alpha_to_one = True, timestep_spacing = leading, ε 예측, x̂_0 clip.
step k 에서 (x_t, ε̂) →
    x̂_0 = clip((x_t − √(1−ᾱ_t) ε̂) / √ᾱ_t)
    σ_t = η √((1−ᾱ_prev)/(1−ᾱ_t) · (1 − ᾱ_t/ᾱ_prev))
    x_prev = √ᾱ_prev x̂_0 + √(1−ᾱ_prev−σ_t²) ε̂ + σ_t z
"""
import math

import torch

MAX_BETA = 0.999


def squaredcos_alphas_cumprod(num_train_timesteps):
    """squaredcos_cap_v2 β 스케줄의 누적곱 ᾱ_t [num_train_timesteps]."""
    # ᾱ(t) = cos²((t + 0.008)/1.008 · π/2) 에서 β 를 구하고 상한 적용
    betas = []
    for timestep in range(num_train_timesteps):
        alpha_bar_now = math.cos((timestep / num_train_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar_next = math.cos(((timestep + 1) / num_train_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        betas.append(min(1.0 - alpha_bar_next / alpha_bar_now, MAX_BETA))
    return torch.cumprod(1.0 - torch.tensor(betas, dtype=torch.float64), dim=0).float()


class DdimSchedule:
    """추론 timestep 목록과 DDIM 갱신식."""

    def __init__(self, num_train_timesteps, num_inference_steps, clip_sample_range):
        self._num_inference_steps = num_inference_steps
        self._clip_sample_range = clip_sample_range
        self._alphas_cumprod = squaredcos_alphas_cumprod(num_train_timesteps)

        # leading 간격 timestep (예: 100/10 → [90, 80, ..., 0])
        step_ratio = num_train_timesteps // num_inference_steps
        self._step_ratio = step_ratio
        self._timesteps = (torch.arange(num_inference_steps) * step_ratio).flip(0)

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def num_inference_steps(self):
        return self._num_inference_steps

    def alpha_cumprod_pair(self, step_index):
        """step_index 의 (ᾱ_t, ᾱ_prev). 마지막 step 의 ᾱ_prev 는 1."""
        timestep = int(self._timesteps[step_index])
        previous_timestep = timestep - self._step_ratio
        alpha_cumprod = self._alphas_cumprod[timestep]
        previous_alpha_cumprod = self._alphas_cumprod[previous_timestep] if previous_timestep >= 0 else torch.tensor(1.0)
        return alpha_cumprod, previous_alpha_cumprod

    def clip_clean_sample(self, clean_sample):
        """x̂_0 를 [−clip, clip] 범위로 자름."""
        return clean_sample.clamp(-self._clip_sample_range, self._clip_sample_range)

    def predict_clean_sample(self, noisy_sample, epsilon, step_index):
        """x_t, ε̂ 로 끝점 x̂_0 를 추정하고 clip."""
        alpha_cumprod, _ = self.alpha_cumprod_pair(step_index)
        clean_sample = (noisy_sample - (1.0 - alpha_cumprod).sqrt() * epsilon) / alpha_cumprod.sqrt()
        return self.clip_clean_sample(clean_sample)

    def step_from_clean(self, clean_sample, epsilon, step_index, eta, generator=None):
        """주어진 x̂_0, ε̂ 로 x_prev 를 계산 (η > 0 이면 새 노이즈 주입)."""
        # 분산 항
        alpha_cumprod, previous_alpha_cumprod = self.alpha_cumprod_pair(step_index)
        variance = (1.0 - previous_alpha_cumprod) / (1.0 - alpha_cumprod) * (1.0 - alpha_cumprod / previous_alpha_cumprod)
        noise_std = eta * variance.clamp_min(0.0).sqrt()

        # 결정적 부분 + 확률적 부분
        direction_scale = (1.0 - previous_alpha_cumprod - noise_std ** 2).clamp_min(0.0).sqrt()
        previous_sample = previous_alpha_cumprod.sqrt() * clean_sample + direction_scale * epsilon
        if eta > 0:
            fresh_noise = torch.randn(clean_sample.shape, generator=generator)
            previous_sample = previous_sample + noise_std * fresh_noise
        return previous_sample

    def step(self, noisy_sample, epsilon, step_index, eta=0.0, generator=None):
        """표준 DDIM 1 step: x_t → (x_prev, x̂_0)."""
        clean_sample = self.predict_clean_sample(noisy_sample, epsilon, step_index)
        previous_sample = self.step_from_clean(clean_sample, epsilon, step_index, eta, generator)
        return previous_sample, clean_sample
