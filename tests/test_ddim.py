"""DDIM 스케줄·갱신식이 diffusers DDIMScheduler (lerobot 설정) 와 같은지 확인."""
import torch
from diffusers import DDIMScheduler

from rs_rtc.samplers.ddim import DdimSchedule

NUM_TRAIN_TIMESTEPS = 100
NUM_INFERENCE_STEPS = 10
CLIP_SAMPLE_RANGE = 1.0


def build_reference_scheduler():
    """lerobot Diffusion Policy 가 쓰는 diffusers 설정."""
    reference_scheduler = DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, beta_start=0.0001, beta_end=0.02,
                                        beta_schedule="squaredcos_cap_v2", clip_sample=True,
                                        clip_sample_range=CLIP_SAMPLE_RANGE, prediction_type="epsilon")
    reference_scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    return reference_scheduler


def test_timesteps_match():
    schedule = DdimSchedule(NUM_TRAIN_TIMESTEPS, NUM_INFERENCE_STEPS, CLIP_SAMPLE_RANGE)
    assert torch.equal(schedule.timesteps, build_reference_scheduler().timesteps)


def test_deterministic_step_matches():
    schedule = DdimSchedule(NUM_TRAIN_TIMESTEPS, NUM_INFERENCE_STEPS, CLIP_SAMPLE_RANGE)
    reference_scheduler = build_reference_scheduler()
    generator = torch.Generator().manual_seed(0)

    # 같은 (x_t, ε̂) 로 모든 step 의 x_prev 비교
    for step_index, timestep in enumerate(reference_scheduler.timesteps):
        noisy_sample = torch.randn(4, 16, 2, generator=generator)
        epsilon = torch.randn(4, 16, 2, generator=generator)
        previous_sample, _ = schedule.step(noisy_sample, epsilon, step_index, eta=0.0)
        reference_sample = reference_scheduler.step(epsilon, timestep, noisy_sample).prev_sample
        assert torch.allclose(previous_sample, reference_sample, atol=1e-5)
