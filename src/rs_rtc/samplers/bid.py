"""BID 방식 비교군 (arXiv 2408.17355 의 backward coherence 선택만 사용).

batch N 개 chunk 를 독립 생성한 뒤, 이전 chunk 목표와의 가중 cost 가 가장 작은 chunk 를 고른다.
생성 도중 교정은 없다 (RS-RTC 와 같은 N·K 호출 수).
"""
import torch

from rs_rtc.samplers.base import DdimChunkSampler, weighted_inpainting_cost


class BidSampler(DdimChunkSampler):
    """생성 후 선택 (best-of-N)."""

    def generate(self, global_cond, target, generator):
        chunks = self.sample(global_cond.expand(self.batch_size, -1), generator)

        # 목표가 없으면 (첫 chunk) 0번 chunk
        if target is None:
            return chunks[0], {"denoiser_calls": self.schedule.num_inference_steps}

        # 목표와 가장 일치하는 chunk 선택
        candidate_costs = weighted_inpainting_cost(chunks, target)
        best_index = int(torch.argmin(candidate_costs))
        return chunks[best_index], {"denoiser_calls": self.schedule.num_inference_steps,
                                    "selected_cost": float(candidate_costs[best_index])}
