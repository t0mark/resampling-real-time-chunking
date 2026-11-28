"""robomimic (robosuite) low-dim 환경 래퍼: PushTEnvironment 와 같은 인터페이스.

환경 설정은 Diffusion Policy 학습 데이터셋(hdf5)의 env_args 로 robosuite 환경을 직접 만들고,
관측 키는 robomimic 규칙 (object-state → object) 을 따른다. 초기 배치는 np.random seed 로 정한다 (DP 평가와 동일).
행동: 정책 출력 [위치 3, 회전 6D 6, 그리퍼 1] → 환경 입력 [위치 3, 축-각 3, 그리퍼 1] (절대 자세 OSC 제어).
공식 환경 (robosuite 1.2 포크, mujoco-py) 기준. 시뮬레이션 불안정 예외는 에피소드 실패로 기록하고 종료한다 (robomimic 평가와 동일).
"""
import json
import logging
import os
from collections import deque

# MuJoCo 렌더링 백엔드 (robosuite import 전에 설정해야 적용됨, 외부 지정값 우선)
os.environ.setdefault("MUJOCO_GL", "osmesa")

import h5py
import numpy as np
import robosuite
import torch
from mujoco_py.builder import MujocoException
from scipy.spatial.transform import Rotation

from rs_rtc.policies.pretrained_robomimic import ROBOMIMIC_ASSET_DIR

LOGGER = logging.getLogger("rs_rtc.envs")
DATASET_DIR = ROBOMIMIC_ASSET_DIR / "datasets"


def rotation_6d_to_axis_angle(rotation_6d):
    """회전 6D [N, 6] (행렬 앞 두 행) → 축-각 [N, 3] (Gram-Schmidt 정규직교화)."""
    first_row = rotation_6d[:, :3] / np.linalg.norm(rotation_6d[:, :3], axis=-1, keepdims=True)
    second_row = rotation_6d[:, 3:] - np.sum(first_row * rotation_6d[:, 3:], axis=-1, keepdims=True) * first_row
    second_row = second_row / np.linalg.norm(second_row, axis=-1, keepdims=True)
    third_row = np.cross(first_row, second_row)
    rotation_matrix = np.stack([first_row, second_row, third_row], axis=-2)
    return Rotation.from_matrix(rotation_matrix).as_rotvec()


def policy_action_to_env_action(policy_action):
    """정책 행동 [10] (위치, 회전 6D, 그리퍼) → 환경 행동 [7] (위치, 축-각, 그리퍼)."""
    policy_action = np.asarray(policy_action, dtype=np.float64).reshape(1, -1)
    axis_angle = rotation_6d_to_axis_angle(policy_action[:, 3:9])
    return np.concatenate([policy_action[:, :3], axis_angle, policy_action[:, 9:]], axis=-1)[0]


class RobomimicEnvironment:
    """low-dim 관측 robomimic 과제. 최근 n_obs_steps 개 관측을 이력으로 유지."""

    def __init__(self, policy_config, max_episode_steps):
        self._n_obs_steps = policy_config["n_obs_steps"]
        self._obs_keys = policy_config["obs_keys"]
        self._abs_action = policy_config["abs_action"]
        self._max_episode_steps = max_episode_steps
        self._env = self.create_env(DATASET_DIR / policy_config["dataset_file"], self._abs_action)
        self._obs_history = deque(maxlen=self._n_obs_steps)
        self._step_count = 0
        self._max_reward = 0.0
        self._is_success = False
        self._is_done = False
        self._is_sim_unstable = False

    @staticmethod
    def create_env(dataset_path, abs_action):
        """데이터셋 env_args 로 robosuite 환경 생성 (렌더링·카메라 관측 없음, abs_action 이면 절대 자세 제어)."""
        with h5py.File(dataset_path, "r") as dataset_file:
            env_args = json.loads(dataset_file["data"].attrs["env_args"])
        env_kwargs = env_args["env_kwargs"]
        env_kwargs.update(has_renderer=False, has_offscreen_renderer=False, ignore_done=True, use_object_obs=True,
                          use_camera_obs=False)
        env_kwargs["controller_configs"]["control_delta"] = not abs_action
        env = robosuite.make(env_args["env_name"], **env_kwargs)
        LOGGER.info("robosuite 환경 생성: %s (데이터셋 %s, 절대 자세 제어=%s)", env_args["env_name"], dataset_path.name,
                    abs_action)
        return env

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

    @property
    def is_sim_unstable(self):
        return self._is_sim_unstable

    def reset(self, seed):
        """seed 로 초기 배치를 정하고 초기 관측을 n_obs_steps 번 복제해 이력을 채운다 (DP 평가와 동일)."""
        np.random.seed(seed)
        observation = self._env.reset()
        self._step_count = 0
        self._max_reward = 0.0
        self._is_success = False
        self._is_done = False
        self._is_sim_unstable = False
        for _ in range(self._n_obs_steps):
            self.append_observation(observation)

    def append_observation(self, observation):
        """관측 dict 에서 obs_keys 순서로 이어 붙인 벡터를 이력에 추가 (object = robosuite object-state)."""
        observation = {**observation, "object": observation["object-state"]}
        self._obs_history.append(np.concatenate([np.asarray(observation[key], dtype=np.float32).reshape(-1)
                                                 for key in self._obs_keys]))

    def observation_history(self):
        """관측 이력 (obs [1, T_o, obs_dim],) 텐서 tuple."""
        return (torch.from_numpy(np.stack(self._obs_history)).unsqueeze(0),)

    def step(self, policy_action):
        """정책 행동 [10] 1 step 실행 후 지표 갱신 (성공 또는 최대 step 도달 시 종료)."""
        env_action = policy_action_to_env_action(policy_action) if self._abs_action else policy_action

        # 시뮬레이션 불안정 시 실패로 종료
        try:
            observation, reward, _, _ = self._env.step(env_action)
        except MujocoException as sim_error:
            LOGGER.warning("시뮬레이션 불안정으로 에피소드 실패 처리 (step %d): %s", self._step_count, sim_error)
            self._is_sim_unstable = True
            self._is_done = True
            return

        self.append_observation(observation)
        self._step_count += 1
        self._max_reward = max(self._max_reward, float(reward))
        self._is_success = self._is_success or bool(self._env._check_success())
        self._is_done = self._is_success or self._step_count >= self._max_episode_steps
