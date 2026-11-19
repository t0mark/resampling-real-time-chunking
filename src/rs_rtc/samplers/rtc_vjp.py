"""RTC 원본 방식 비교군 (arXiv 2506.07339): 역전파(VJP) 가이던스 inpainting. CPU 또는 CUDA autograd (NPU 불가).

DDIM ε 예측 모델에 맞춘 형태:
    x̂_0(x_t) = (x_t − √(1−ᾱ_t) ε̂(x_t)) / √ᾱ_t
    g = ∇_{x_t} ½ Σ_j W_j ‖Y_j − x̂_{0,j}‖²      (UNet 을 거치는 역전파)
    ε̃ = ε̂ + β √(1−ᾱ_t) g                        (score 에 −g 를 더한 것과 같음)
    ε̃ 로 x̂_0 를 다시 추정해 DDIM 갱신

UNet 계산만 지정 장치에서 수행하고, 노이즈 생성과 DDIM 갱신은 CPU 에서 수행한다 (장치와 무관하게 같은 난수열).
"""
import torch

from rs_rtc.samplers.base import DdimChunkSampler


class RtcVjpSampler(DdimChunkSampler):
    """역전파 가이던스 (batch 1, CPU 또는 CUDA)."""

    def __init__(self, unet, schedule, horizon, action_dim, time_embed_dim, eta, guidance_weight, device="cpu"):
        super().__init__(None, schedule, horizon, action_dim, time_embed_dim, eta)
        self._device = torch.device(device)
        self._unet = unet.to(self._device)
        self._guidance_weight = guidance_weight

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def unguided_epsilon(self, noisy_sample, step_index, global_cond):
        """가이던스 없이 장치에서 ε̂ 을 계산해 CPU 로 반환."""
        with torch.no_grad():
            epsilon = self._unet(noisy_sample.to(self._device), self.time_features(step_index, 1).to(self._device),
                                 global_cond.to(self._device))
        return epsilon.cpu()

    def guided_epsilon(self, noisy_sample, step_index, global_cond, target):
        """UNet 을 거치는 VJP 로 가이던스를 더한 ε̃ 을 장치에서 계산해 CPU 로 반환."""
        alpha_cumprod, _ = self.schedule.alpha_cumprod_pair(step_index)
        alpha_cumprod = alpha_cumprod.to(self._device)
        noisy_sample = noisy_sample.detach().to(self._device).requires_grad_(True)
        prefix_actions = target.prefix_actions.to(self._device)
        prefix_weights = target.prefix_weights.to(self._device)

        # 끝점 추정과 가중 오차 (clip 없이 미분)
        epsilon = self._unet(noisy_sample, self.time_features(step_index, 1).to(self._device),
                             global_cond.to(self._device))
        clean_sample = (noisy_sample - (1.0 - alpha_cumprod).sqrt() * epsilon) / alpha_cumprod.sqrt()
        squared_error = ((prefix_actions - clean_sample[0]) ** 2).sum(dim=-1)
        inpainting_loss = 0.5 * (prefix_weights * squared_error).sum()

        # 역전파로 x_t 에 대한 기울기
        loss_gradient, = torch.autograd.grad(inpainting_loss, noisy_sample)
        guided = epsilon + self._guidance_weight * (1.0 - alpha_cumprod).sqrt() * loss_gradient
        return guided.detach().cpu()

    def generate(self, global_cond, target, generator):
        noisy_sample = self.initial_noise(1, generator)

        # K step 역확산 (목표가 없으면 가이던스 없음)
        for step_index in range(self.schedule.num_inference_steps):
            if target is None:
                epsilon = self.unguided_epsilon(noisy_sample, step_index, global_cond)
            else:
                epsilon = self.guided_epsilon(noisy_sample, step_index, global_cond, target)
            noisy_sample, _ = self.schedule.step(noisy_sample, epsilon, step_index, self._eta, generator)
        return noisy_sample[0], {"denoiser_calls": self.schedule.num_inference_steps}


class RtcPaperSampler(RtcVjpSampler):
    """원 논문 공식 코드 (real-time-chunking-kinetix, realtime_action) 의 가이던스를 DDIM 에 그대로 옮긴 형태.

    VP 상태 x_t = √ᾱ x_0 + √(1−ᾱ) ε 를 flow 상태로 바꿔 계산한다 (τ: 0 = 노이즈, 1 = 데이터):
        s = √ᾱ + √(1−ᾱ),  x_τ = x_t / s,  τ = √ᾱ / s
        x̂_1 = x̂_0 (clip 없음),  c = W ⊙ (Y − x̂_1),  J_τᵀ c = s · (∂x̂_0/∂x_t)ᵀ c
        w = min(β, (1−τ)/τ · (τ² + (1−τ)²) / (1−τ)²)
        v' = v + w J_τᵀ c  →  x̂_0' = x̂_0 + (1−τ) w J_τᵀ c
    x̂_0' 와 일치하는 ε' = (x_t − √ᾱ x̂_0') / √(1−ᾱ) 로 DDIM 갱신한다.
    """

    def guidance_scale(self, step_index):
        """step 의 flow 시간 τ, 상태 배율 s, 논문 가이던스 가중치 w."""
        alpha_cumprod, _ = self.schedule.alpha_cumprod_pair(step_index)
        signal_scale = float(alpha_cumprod.sqrt())
        noise_scale = float((1.0 - alpha_cumprod).sqrt())
        state_scale = signal_scale + noise_scale
        flow_time = signal_scale / state_scale
        inverse_r2 = (flow_time ** 2 + (1.0 - flow_time) ** 2) / (1.0 - flow_time) ** 2
        guidance_weight = min((1.0 - flow_time) / flow_time * inverse_r2, self._guidance_weight)
        return flow_time, state_scale, guidance_weight

    def guided_epsilon(self, noisy_sample, step_index, global_cond, target):
        """논문 가중치로 x̂_0 를 보정하고 그에 맞는 ε' 을 CPU 로 반환."""
        alpha_cumprod, _ = self.schedule.alpha_cumprod_pair(step_index)
        alpha_cumprod = alpha_cumprod.to(self._device)
        flow_time, state_scale, guidance_weight = self.guidance_scale(step_index)
        noisy_sample = noisy_sample.detach().to(self._device).requires_grad_(True)
        prefix_actions = target.prefix_actions.to(self._device)
        prefix_weights = target.prefix_weights.to(self._device)

        # 끝점 추정 (clip 없음)
        epsilon = self._unet(noisy_sample, self.time_features(step_index, 1).to(self._device),
                             global_cond.to(self._device))
        clean_sample = (noisy_sample - (1.0 - alpha_cumprod).sqrt() * epsilon) / alpha_cumprod.sqrt()

        # 가중 오차를 x_t 로 역전파한 VJP 를 flow 상태 기준으로 변환
        weighted_error = (prefix_weights[:, None] * (prefix_actions - clean_sample[0])).detach()
        vjp_on_state, = torch.autograd.grad(clean_sample[0], noisy_sample, grad_outputs=weighted_error)
        flow_correction = state_scale * vjp_on_state

        # 보정된 끝점과 일치하는 ε'
        guided_clean = clean_sample.detach() + (1.0 - flow_time) * guidance_weight * flow_correction
        guided_epsilon = (noisy_sample.detach() - alpha_cumprod.sqrt() * guided_clean) / (1.0 - alpha_cumprod).sqrt()
        return guided_epsilon.cpu()
