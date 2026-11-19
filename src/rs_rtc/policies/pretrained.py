"""HuggingFace 사전학습 Diffusion Policy (lerobot/diffusion_pusht_keypoints) 로딩과 min-max 정규화."""
import json
import logging

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from rs_rtc.policies.diffusion_unet import DiffusionUnet1D

LOGGER = logging.getLogger("rs_rtc.policies")

PUSHT_KEYPOINTS_REPO = "lerobot/diffusion_pusht_keypoints"
CHECKPOINT_UNET_PREFIX = "diffusion.unet."

# 원본 가중치 이름 중 우리 모듈에서 private(_ 접두) 속성으로 둔 이름
PRIVATE_MODULE_NAMES = {"diffusion_step_encoder", "down_modules", "mid_modules", "up_modules", "final_conv",
                        "cond_encoder", "conv1", "conv2", "residual_conv", "block"}


class MinMaxNormalizer:
    """lerobot min_max 정규화: [min, max] ↔ [-1, 1]."""

    def __init__(self, value_min, value_max):
        self._value_min = value_min
        self._value_range = (value_max - value_min).clamp_min(1e-8)

    def normalize(self, raw_value):
        return (raw_value - self._value_min) / self._value_range * 2.0 - 1.0

    def unnormalize(self, normalized_value):
        return (normalized_value + 1.0) / 2.0 * self._value_range + self._value_min


class PretrainedDiffusionPolicy:
    """사전학습 UNet, 정규화기, 정책 설정(horizon, n_obs_steps 등)을 묶은 객체."""

    def __init__(self, unet, policy_config, agent_pos_normalizer, keypoint_normalizer, action_normalizer):
        self._unet = unet
        self._policy_config = policy_config
        self._agent_pos_normalizer = agent_pos_normalizer
        self._keypoint_normalizer = keypoint_normalizer
        self._action_normalizer = action_normalizer

    @property
    def unet(self):
        return self._unet

    @property
    def config(self):
        return self._policy_config

    @property
    def horizon(self):
        return self._policy_config["horizon"]

    @property
    def n_obs_steps(self):
        return self._policy_config["n_obs_steps"]

    @property
    def n_action_steps(self):
        return self._policy_config["n_action_steps"]

    @property
    def action_normalizer(self):
        return self._action_normalizer

    def build_global_cond(self, agent_pos_history, keypoint_history):
        """관측 이력 [B, T_o, 2], [B, T_o, 16] (픽셀) → 정규화·펼친 조건 [B, T_o × 18]."""
        normalized_agent_pos = self._agent_pos_normalizer.normalize(agent_pos_history)
        normalized_keypoints = self._keypoint_normalizer.normalize(keypoint_history)
        return torch.cat([normalized_agent_pos, normalized_keypoints], dim=-1).flatten(start_dim=1)


def convert_checkpoint_key(checkpoint_key):
    """원본 가중치 이름을 우리 모듈 이름으로 변환 (예: down_modules.0.0.conv1.block.0 → _down_modules.0.0._conv1._block.0)."""
    name_parts = checkpoint_key[len(CHECKPOINT_UNET_PREFIX):].split(".")
    return ".".join(f"_{part}" if part in PRIVATE_MODULE_NAMES else part for part in name_parts)


def load_pretrained_policy(repo_id=PUSHT_KEYPOINTS_REPO):
    """HF 에서 설정·가중치를 받아 UNet 과 정규화기를 구성한다."""
    # 설정과 가중치 다운로드
    with open(hf_hub_download(repo_id, "config.json")) as config_file:
        policy_config = json.load(config_file)
    checkpoint = load_file(hf_hub_download(repo_id, "model.safetensors"))
    LOGGER.info("사전학습 정책 다운로드: %s", repo_id)

    # UNet 구성 (관측 차원 = n_obs_steps × (agent_pos + keypoint))
    per_step_obs_dim = (policy_config["input_shapes"]["observation.state"][0]
                        + policy_config["input_shapes"]["observation.environment_state"][0])
    unet = DiffusionUnet1D(
        action_dim=policy_config["output_shapes"]["action"][0],
        horizon=policy_config["horizon"],
        obs_dim=policy_config["n_obs_steps"] * per_step_obs_dim,
        down_dims=tuple(policy_config["down_dims"]),
        time_embed_dim=policy_config["diffusion_step_embed_dim"],
        kernel_size=policy_config["kernel_size"],
        group_count=policy_config["n_groups"],
    )

    # UNet 가중치 로드 (누락·잉여 키가 있으면 오류)
    unet_state_dict = {convert_checkpoint_key(key): tensor for key, tensor in checkpoint.items()
                       if key.startswith(CHECKPOINT_UNET_PREFIX)}
    unet.load_state_dict(unet_state_dict, strict=True)
    unet.eval()
    parameter_count = sum(parameter.numel() for parameter in unet.parameters())
    LOGGER.info("UNet 가중치 로드 완료: %.1f M 파라미터", parameter_count / 1e6)

    # 정규화기 구성
    agent_pos_normalizer = MinMaxNormalizer(checkpoint["normalize_inputs.buffer_observation_state.min"],
                                            checkpoint["normalize_inputs.buffer_observation_state.max"])
    keypoint_normalizer = MinMaxNormalizer(checkpoint["normalize_inputs.buffer_observation_environment_state.min"],
                                           checkpoint["normalize_inputs.buffer_observation_environment_state.max"])
    action_normalizer = MinMaxNormalizer(checkpoint["unnormalize_outputs.buffer_action.min"],
                                         checkpoint["unnormalize_outputs.buffer_action.max"])
    return PretrainedDiffusionPolicy(unet, policy_config, agent_pos_normalizer, keypoint_normalizer,
                                     action_normalizer)
