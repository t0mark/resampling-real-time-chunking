"""gym-pusht 환경 래퍼: 관측 이력 관리, 에피소드 지표(최대 보상, 성공) 기록."""
from collections import deque

import gym_pusht  # noqa: F401  (환경 등록)
import gymnasium as gym
import numpy as np
import torch

PUSHT_ENV_ID = "gym_pusht/PushT-v0"
PUSHT_OBS_TYPE = "environment_state_agent_pos"


class PushTEnvironment:
    """keypoint 상태 관측 PushT. 최근 n_obs_steps 개 관측을 이력으로 유지."""

    def __init__(self, n_obs_steps, max_episode_steps):
        self._n_obs_steps = n_obs_steps
        self._max_episode_steps = max_episode_steps
        self._env = gym.make(PUSHT_ENV_ID, obs_type=PUSHT_OBS_TYPE, max_episode_steps=max_episode_steps)
        self._agent_pos_history = deque(maxlen=n_obs_steps)
        self._keypoint_history = deque(maxlen=n_obs_steps)
        self._step_count = 0
        self._max_reward = 0.0
        self._is_success = False
        self._is_done = False

    @property
    def step_count(self):
        return self._step_count

    @property
    def max_reward(self):
        return self._max_reward

    @property
    def is_success(self):
        return self._is_success

    @property
    def is_done(self):
        return self._is_done

    def reset(self, seed):
        """초기 관측을 n_obs_steps 번 복제해 이력을 채운다 (lerobot 평가와 동일)."""
        observation, _ = self._env.reset(seed=seed)
        self._step_count = 0
        self._max_reward = 0.0
        self._is_success = False
        self._is_done = False
        for _ in range(self._n_obs_steps):
            self._append_observation(observation)

    def _append_observation(self, observation):
        self._agent_pos_history.append(np.asarray(observation["agent_pos"], dtype=np.float32))
        self._keypoint_history.append(np.asarray(observation["environment_state"], dtype=np.float32))

    def observation_history(self):
        """관측 이력 (agent_pos [1, T_o, 2], keypoint [1, T_o, 16]) 픽셀 단위 텐서."""
        agent_pos_history = torch.from_numpy(np.stack(self._agent_pos_history)).unsqueeze(0)
        keypoint_history = torch.from_numpy(np.stack(self._keypoint_history)).unsqueeze(0)
        return agent_pos_history, keypoint_history

    def step(self, pixel_action):
        """픽셀 단위 행동 [2] 1 step 실행 후 지표 갱신."""
        observation, reward, terminated, truncated, info = self._env.step(np.asarray(pixel_action, dtype=np.float32))
        self._append_observation(observation)
        self._step_count += 1
        self._max_reward = max(self._max_reward, float(reward))
        self._is_success = self._is_success or bool(info["is_success"])
        self._is_done = bool(terminated or truncated)
