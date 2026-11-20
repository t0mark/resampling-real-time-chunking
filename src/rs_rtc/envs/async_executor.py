"""추론 지연 d 스텝을 시뮬레이션 시간에서 재현하는 비동기 chunk 실행기.

chunk index j 는 시각 (생성 시작 t_c) − (n_obs_steps − 1) + j 에 대응한다 (PushT: n_obs_steps = 2 → t_c − 1 + j).
cycle 주기는 s (= n_action_steps) 스텝. cycle 시작 t_c 에 관측 o_{t_c} 로 새 chunk 생성을 시작하고,
    t_c … t_c+d−1      : 이전 chunk index s+o … s+o+d−1 실행 (o = n_obs_steps − 1, 생성 중)
    t_c+d … t_c+s−1    : 새 chunk index o+d … o+s−1 실행
새 chunk 의 index 0 … o+d−1 은 이미 실행됐거나 실행이 확정된 구간 (목표 Y_j = 이전 chunk[j + s]).
d = 0 이면 동기 실행과 같다. 첫 chunk 는 동기로 생성한다. 제약: o + s + d − 1 ≤ H − 1.
s 는 기본으로 정책의 n_action_steps 이고 execute_steps 로 바꿀 수 있다 (1 = 매 스텝 재추론). 제약: d < s.
render_enabled 이면 매 스텝 끝단·블록 상태와 chunk 전환 시 계획·후보 경로를 render_record 에 기록한다.
"""
import logging
import math
import time

import numpy as np
import torch

from rs_rtc.samplers.base import InpaintingTarget, weighted_inpainting_cost

LOGGER = logging.getLogger("rs_rtc.envs")


def build_prefix_weights(horizon, frozen_count, overlap_end_index):
    """일치 가중치 W [H]: 확정 구간 1, overlap_end 까지 RTC 식 지수 감쇠, 이후 0."""
    prefix_weights = torch.zeros(horizon)
    prefix_weights[:frozen_count] = 1.0
    decay_length = overlap_end_index + 1 - (frozen_count - 1)
    for action_index in range(frozen_count, overlap_end_index + 1):
        decay_ratio = (overlap_end_index + 1 - action_index) / decay_length
        prefix_weights[action_index] = decay_ratio * (math.exp(decay_ratio) - 1.0) / (math.e - 1.0)
    return prefix_weights


class AsyncChunkExecutor:
    """고정 지연 d 로 chunk 생성기를 비동기 실행하고 에피소드 지표를 기록."""

    def __init__(self, environment, policy, chunk_generator, delay_steps, execute_steps=None, render_enabled=False):
        self._environment = environment
        self._policy = policy
        self._chunk_generator = chunk_generator
        self._delay_steps = delay_steps
        self._execute_steps = execute_steps or policy.n_action_steps
        self._render_enabled = render_enabled
        self._first_action_index = policy.n_obs_steps - 1
        self._horizon = policy.horizon

        # 목표 구간: 이전 chunk 와 시간이 겹치는 index 0 … H−1−s
        self._overlap_end_index = self._horizon - 1 - self._execute_steps
        self._frozen_count = self._first_action_index + delay_steps
        if self._first_action_index + self._execute_steps + delay_steps - 1 > self._horizon - 1:
            raise ValueError(f"지연 d={delay_steps} 가 chunk 길이를 넘음 (o + s + d − 1 ≤ H − 1)")
        if delay_steps >= self._execute_steps:
            raise ValueError(f"지연 d={delay_steps} 가 실행 스텝 수 s={self._execute_steps} 이상 (생성 1개씩 순차라 d < s)")
        self._prefix_weights = build_prefix_weights(self._horizon, self._frozen_count, self._overlap_end_index)
        self._render_record = None

    @property
    def execute_steps(self):
        """cycle 당 실행 스텝 수 s."""
        return self._execute_steps

    @property
    def render_record(self):
        """마지막 에피소드의 렌더 기록 (렌더링이 꺼져 있으면 None)."""
        return self._render_record

    def create_render_record(self):
        """빈 렌더 기록 (스텝별 상태, 전환별 계획·후보 경로)."""
        return {"agent_positions": [], "block_poses": [], "switch_steps": [], "previous_plans": [], "new_plans": [],
                "candidate_plans": [], "previous_plan_offsets": [], "new_plan_offsets": []}

    def record_render_state(self):
        """현재 끝단 위치와 블록 자세를 렌더 기록에 추가."""
        self._render_record["agent_positions"].append(self._environment.agent_position)
        self._render_record["block_poses"].append(self._environment.block_pose)

    def record_render_switch(self, switch_step, cycle_start_step, previous_pixel_chunk, new_pixel_chunk):
        """chunk 전환 시점, 이전/새 chunk 계획 경로 (index j ↔ 실행 스텝 offset + j), 후보 경로를 렌더 기록에 추가."""
        new_plan_offset = cycle_start_step - self._first_action_index
        self._render_record["switch_steps"].append(switch_step)
        self._render_record["previous_plans"].append(previous_pixel_chunk.numpy())
        self._render_record["new_plans"].append(new_pixel_chunk.numpy())

        # 후보 경로 (후보를 쓰지 않는 방법은 0 개)
        candidate_chunks = self._chunk_generator.last_candidates
        if candidate_chunks is None:
            candidate_plans = np.zeros((0, self._horizon, new_pixel_chunk.shape[-1]), dtype=np.float32)
        else:
            candidate_plans = self._policy.action_normalizer.unnormalize(candidate_chunks).numpy()
        self._render_record["candidate_plans"].append(candidate_plans)
        self._render_record["previous_plan_offsets"].append(new_plan_offset - self._execute_steps)
        self._render_record["new_plan_offsets"].append(new_plan_offset)

    def build_target(self, previous_chunk):
        """이전 chunk 를 현재 cycle 시각으로 s 만큼 당긴 목표 Y 와 가중치 W."""
        prefix_actions = torch.zeros_like(previous_chunk)
        overlap_length = self._overlap_end_index + 1
        prefix_actions[:overlap_length] = previous_chunk[self._execute_steps:self._execute_steps + overlap_length]
        return InpaintingTarget(prefix_actions, self._prefix_weights, self._frozen_count)

    def generate_chunk(self, target, generator, generation_records):
        """현재 관측으로 chunk 를 생성하고 (정규화 chunk, 픽셀 chunk) 반환, 생성 기록 추가."""
        agent_pos_history, keypoint_history = self._environment.observation_history()
        global_cond = self._policy.build_global_cond(agent_pos_history, keypoint_history)
        generation_start = time.perf_counter()
        normalized_chunk, generation_stats = self._chunk_generator.generate(global_cond, target, generator)
        generation_stats["latency_ms"] = (time.perf_counter() - generation_start) * 1e3

        # 생성 chunk 와 inpainting 목표의 가중 오차 (모든 방법 공통 지표)
        if target is not None:
            generation_stats["prefix_cost"] = float(weighted_inpainting_cost(normalized_chunk.unsqueeze(0), target)[0])
        generation_records.append(generation_stats)
        return normalized_chunk, self._policy.action_normalizer.unnormalize(normalized_chunk)

    def execute_actions(self, pixel_chunk, action_indices, executed_actions):
        """chunk 의 지정 index 행동을 차례로 실행 (에피소드가 끝나면 중단)."""
        for action_index in action_indices:
            if self._environment.is_done:
                return
            pixel_action = pixel_chunk[action_index].numpy()
            self._environment.step(pixel_action)
            executed_actions.append(pixel_action)

            # 렌더링 시 실행 후 상태 기록
            if self._render_record is not None:
                self.record_render_state()

    def run_episode(self, env_seed, generator):
        """에피소드 1개 실행 후 지표 dict 반환."""
        self._environment.reset(env_seed)
        self._render_record = self.create_render_record() if self._render_enabled else None
        if self._render_record is not None:
            self.record_render_state()
        generation_records = []
        executed_actions = []
        boundary_step_indices = []
        first_index = self._first_action_index

        # 첫 chunk: 동기 생성 후 index o … o+s−1 실행
        previous_chunk, previous_pixel_chunk = self.generate_chunk(None, generator, generation_records)
        self.execute_actions(previous_pixel_chunk, range(first_index, first_index + self._execute_steps),
                             executed_actions)

        # 비동기 cycle 반복
        while not self._environment.is_done:
            # 새 chunk 생성 (관측은 cycle 시작 시각)
            cycle_start_step = len(executed_actions)
            target = self.build_target(previous_chunk)
            new_chunk, new_pixel_chunk = self.generate_chunk(target, generator, generation_records)

            # 생성 중 d 스텝: 이전 chunk 의 나머지 실행
            self.execute_actions(previous_pixel_chunk,
                                 range(self._execute_steps + first_index,
                                       self._execute_steps + first_index + self._delay_steps), executed_actions)

            # 새 chunk 로 전환 (전환 시점 index 기록)
            boundary_step_indices.append(len(executed_actions))
            if self._render_record is not None:
                self.record_render_switch(len(executed_actions), cycle_start_step, previous_pixel_chunk,
                                          new_pixel_chunk)
            self.execute_actions(new_pixel_chunk, range(first_index + self._delay_steps,
                                                        first_index + self._execute_steps), executed_actions)
            previous_chunk, previous_pixel_chunk = new_chunk, new_pixel_chunk

        if self._render_record is not None:
            self._render_record["executed_actions"] = list(executed_actions)
        return self.summarize_episode(env_seed, executed_actions, boundary_step_indices, generation_records)

    def summarize_episode(self, env_seed, executed_actions, boundary_step_indices, generation_records):
        """에피소드 지표: 성공, 최대 보상, chunk 전환 시 행동 변화량, inpainting 오차, 생성 통계."""
        # 연속 행동 간 변화량 (픽셀)
        action_array = np.asarray(executed_actions)
        step_changes = np.linalg.norm(np.diff(action_array, axis=0), axis=1)
        boundary_changes = [step_changes[index - 1] for index in boundary_step_indices
                            if 0 < index < len(action_array)]
        boundary_set = set(boundary_step_indices)
        interior_changes = [step_changes[index - 1] for index in range(1, len(action_array))
                            if index not in boundary_set]

        # 생성 통계 평균
        latency_ms = [record["latency_ms"] for record in generation_records]
        guided_records = generation_records[1:]
        episode_result = {
            "env_seed": env_seed,
            "success": int(self._environment.is_success),
            "max_reward": round(self._environment.max_reward, 4),
            "steps": self._environment.step_count,
            "boundary_jump_px": round(float(np.mean(boundary_changes)), 3) if boundary_changes else float("nan"),
            "interior_step_px": round(float(np.mean(interior_changes)), 3) if interior_changes else float("nan"),
            "generation_p50_ms": round(float(np.percentile(latency_ms, 50)), 2),
            "chunk_count": len(generation_records),
        }
        for stat_name in ("prefix_cost", "resample_count", "unique_particles", "min_ess", "selected_cost"):
            stat_values = [record[stat_name] for record in guided_records if stat_name in record]
            if stat_values:
                episode_result[f"mean_{stat_name}"] = round(float(np.mean(stat_values)), 4)
        return episode_result
