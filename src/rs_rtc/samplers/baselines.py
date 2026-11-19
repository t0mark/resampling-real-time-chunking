"""역전파 없는 단순 inpainting 비교군: step 별 확정 구간 치환 (RePaint 방식)."""
from rs_rtc.samplers.base import DdimChunkSampler


class ReplacementSampler(DdimChunkSampler):
    """매 step 끝점 추정 x̂_0 의 확정 구간(index < frozen_count)을 목표 Y 로 치환한 뒤 DDIM 갱신."""

    def generate(self, global_cond, target, generator):
        # 목표가 없으면 (첫 chunk) 표준 DDIM
        if target is None:
            return super().generate(global_cond, target, generator)

        batch_size = self.batch_size
        batch_cond = global_cond.expand(batch_size, -1)
        noisy_sample = self.initial_noise(batch_size, generator)
        frozen_count = target.frozen_count

        # K step 역확산, 매 step 확정 구간 치환
        for step_index in range(self.schedule.num_inference_steps):
            epsilon = self.denoiser(noisy_sample, self.time_features(step_index, batch_size), batch_cond)
            clean_sample = self.schedule.predict_clean_sample(noisy_sample, epsilon, step_index)
            clean_sample[:, :frozen_count] = target.prefix_actions[:frozen_count]
            noisy_sample = self.schedule.step_from_clean(clean_sample, epsilon, step_index, self._eta, generator)
        return noisy_sample[0], {"denoiser_calls": self.schedule.num_inference_steps}
