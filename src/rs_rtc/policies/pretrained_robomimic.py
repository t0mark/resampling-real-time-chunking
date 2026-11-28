"""Diffusion Policy 공식 robomimic low-dim 체크포인트 로딩과 선형 정규화.

체크포인트: assets/robomimic/checkpoints/{task}.ckpt (scripts/download_robomimic.sh)
행동 (abs_action): [위치 3, 회전 6D 6, 그리퍼 1] = 10 차원. 환경에는 [위치 3, 축-각 3, 그리퍼 1] 로 변환해 넣는다.
"""
import logging

import dill
import torch

from rs_rtc import PROJECT_ROOT
from rs_rtc.policies.diffusion_unet import DiffusionUnet1D
from rs_rtc.policies.pretrained import PRIVATE_MODULE_NAMES

LOGGER = logging.getLogger("rs_rtc.policies")

ROBOMIMIC_ASSET_DIR = PROJECT_ROOT / "assets" / "robomimic"
CHECKPOINT_UNET_PREFIX = "model."
DEFAULT_INFERENCE_STEPS = 10
CLIP_SAMPLE_RANGE = 1.0

# 원본 가중치 이름 조각 → 우리 모듈 이름 조각
MODULE_NAME_REPLACEMENTS = {"blocks.0.": "conv1.", "blocks.1.": "conv2.", ".2.conv.": ".2."}


class LinearNormalizer:
    """Diffusion Policy 선형 정규화: normalized = raw × scale + offset."""

    def __init__(self, scale, offset):
        self._scale = scale
        self._offset = offset

    def normalize(self, raw_value):
        return raw_value * self._scale + self._offset

    def unnormalize(self, normalized_value):
        return (normalized_value - self._offset) / self._scale


class RobomimicDiffusionPolicy:
    """robomimic 사전학습 UNet, 관측·행동 정규화기, 정책 설정을 묶은 객체 (PushT 정책과 같은 인터페이스)."""

    def __init__(self, unet, policy_config, obs_normalizer, action_normalizer):
        self._unet = unet
        self._policy_config = policy_config
        self._obs_normalizer = obs_normalizer
        self._action_normalizer = action_normalizer

    @property
    def name(self):
        return self._policy_config["name"]

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

    def build_global_cond(self, obs_history):
        """관측 이력 [B, T_o, obs_dim] → 정규화·펼친 조건 [B, T_o × obs_dim]."""
        return self._obs_normalizer.normalize(obs_history).flatten(start_dim=1)


def convert_checkpoint_key(checkpoint_key):
    """원본 가중치 이름을 우리 모듈 이름으로 변환 (예: down_modules.0.0.blocks.0.block.0 → _down_modules.0.0._conv1._block.0)."""
    module_key = checkpoint_key[len(CHECKPOINT_UNET_PREFIX):]
    for original_name, converted_name in MODULE_NAME_REPLACEMENTS.items():
        module_key = module_key.replace(original_name, converted_name)
    return ".".join(f"_{part}" if part in PRIVATE_MODULE_NAMES else part for part in module_key.split("."))


def load_robomimic_policy(task_name, num_inference_steps=DEFAULT_INFERENCE_STEPS):
    """체크포인트에서 설정·EMA 가중치·정규화기를 읽어 정책을 구성한다."""
    # 체크포인트 로드 (설정은 OmegaConf, 가중치는 state_dict)
    checkpoint_path = ROBOMIMIC_ASSET_DIR / "checkpoints" / f"{task_name}.ckpt"
    payload = torch.load(checkpoint_path, map_location="cpu", pickle_module=dill, weights_only=False)
    training_config = payload["cfg"]
    LOGGER.info("robomimic 체크포인트 로드: %s", checkpoint_path.name)

    # 정책 설정 (DDIM 스케줄·실행기·환경이 쓰는 값)
    policy_config = {
        "name": task_name,
        "horizon": int(training_config.horizon),
        "n_obs_steps": int(training_config.n_obs_steps),
        "n_action_steps": int(training_config.n_action_steps),
        "num_train_timesteps": int(training_config.policy.noise_scheduler.num_train_timesteps),
        "num_inference_steps": num_inference_steps,
        "clip_sample_range": CLIP_SAMPLE_RANGE,
        "obs_keys": list(training_config.task.obs_keys),
        "abs_action": bool(training_config.task.abs_action),
        "max_episode_steps": int(training_config.task.env_runner.max_steps),
        "dataset_file": str(training_config.task.dataset_path).split("datasets/", 1)[1],
    }
    LOGGER.info("정책 설정: %s", policy_config)

    # UNet 구성
    model_config = training_config.policy.model
    unet = DiffusionUnet1D(
        action_dim=int(model_config.input_dim),
        horizon=policy_config["horizon"],
        obs_dim=int(model_config.global_cond_dim),
        down_dims=tuple(model_config.down_dims),
        time_embed_dim=int(model_config.diffusion_step_embed_dim),
        kernel_size=int(model_config.kernel_size),
        group_count=int(model_config.n_groups),
    )

    # EMA 가중치 로드 (누락·잉여 키가 있으면 오류)
    ema_state_dict = payload["state_dicts"]["ema_model"]
    unet_state_dict = {convert_checkpoint_key(key): tensor for key, tensor in ema_state_dict.items()
                       if key.startswith(CHECKPOINT_UNET_PREFIX)}
    unet.load_state_dict(unet_state_dict, strict=True)
    unet.eval()
    parameter_count = sum(parameter.numel() for parameter in unet.parameters())
    LOGGER.info("UNet EMA 가중치 로드 완료: %.1f M 파라미터", parameter_count / 1e6)

    # 정규화기 구성
    obs_normalizer = LinearNormalizer(ema_state_dict["normalizer.params_dict.obs.scale"],
                                      ema_state_dict["normalizer.params_dict.obs.offset"])
    action_normalizer = LinearNormalizer(ema_state_dict["normalizer.params_dict.action.scale"],
                                         ema_state_dict["normalizer.params_dict.action.offset"])
    return RobomimicDiffusionPolicy(unet, policy_config, obs_normalizer, action_normalizer)
