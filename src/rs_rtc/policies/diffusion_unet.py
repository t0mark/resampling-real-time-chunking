"""Diffusion Policy 조건부 1D UNet (lerobot DiffusionConditionalUnet1d 와 같은 구조, 노이즈 ε 예측).

NPU 정밀도 문제로 timestep 사인 임베딩은 그래프 밖(호스트, fp32)에서 계산해 입력으로 받는다.
    noisy_action:   [B, H, Da]   정규화 [-1, 1] 공간의 x_t
    time_features:  [B, Dt]      sinusoidal_time_features(timestep) 결과
    global_cond:    [B, obs_dim] 정규화된 관측 (n_obs_steps × [agent_pos 2, keypoint 16] 펼침)
    출력 epsilon:   [B, H, Da]
"""
import math

import torch
import torch.nn as nn


def sinusoidal_time_features(timesteps, feature_dim):
    """정수 timestep [B] 을 사인·코사인 특징 [B, feature_dim] 으로 변환 (호스트 fp32 계산)."""
    half_dim = feature_dim // 2
    frequency_exponent = torch.arange(half_dim, dtype=torch.float32) * (-math.log(10000.0) / (half_dim - 1))
    phase = timesteps.float().unsqueeze(-1) * torch.exp(frequency_exponent).unsqueeze(0)
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class Mish(nn.Module):
    """Mish 활성화 x · tanh(softplus(x)) (RBLN 이 지원하는 기본 연산으로 구성)."""

    def forward(self, feature):
        return feature * torch.tanh(nn.functional.softplus(feature))


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish."""

    def __init__(self, in_channels, out_channels, kernel_size, group_count):
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(group_count, out_channels),
            Mish(),
        )

    def forward(self, feature):
        return self._block(feature)


class ConditionalResidualBlock1D(nn.Module):
    """조건 벡터로 FiLM(scale, bias) 변조를 하는 1D residual block."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size, group_count):
        super().__init__()
        self._out_channels = out_channels
        self._conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, group_count)
        self._conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, group_count)

        # 조건 벡터 → 채널별 scale, bias
        self._cond_encoder = nn.Sequential(Mish(), nn.Linear(cond_dim, out_channels * 2))

        # 채널 수가 다르면 1x1 conv 로 residual 경로를 맞춘다
        if in_channels != out_channels:
            self._residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self._residual_conv = nn.Identity()

    def forward(self, feature, cond_vector):
        # 첫 conv 후 FiLM 변조
        hidden = self._conv1(feature)
        film_params = self._cond_encoder(cond_vector).unsqueeze(-1)
        hidden = film_params[:, :self._out_channels] * hidden + film_params[:, self._out_channels:]

        # 두 번째 conv 와 residual 합
        hidden = self._conv2(hidden)
        return hidden + self._residual_conv(feature)


class DiffusionUnet1D(nn.Module):
    """action chunk 의 시간축을 1D 시퀀스로 보는 조건부 UNet 노이즈 예측기."""

    def __init__(self, action_dim, horizon, obs_dim, down_dims, time_embed_dim, kernel_size, group_count):
        super().__init__()
        self._action_dim = action_dim
        self._horizon = horizon
        self._obs_dim = obs_dim
        self._time_embed_dim = time_embed_dim

        # 시간 특징 MLP (0번 사인 임베딩은 호스트에서 계산하므로 Identity, 가중치 index 는 원본과 같게 유지)
        self._diffusion_step_encoder = nn.Sequential(
            nn.Identity(),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        cond_dim = time_embed_dim + obs_dim

        # 인코더 경로: 레벨마다 residual block 2개 + (마지막 제외) 2배 다운샘플
        channel_dims = [action_dim] + list(down_dims)
        level_count = len(down_dims)
        self._down_modules = nn.ModuleList()
        for level_index in range(level_count):
            in_channels = channel_dims[level_index]
            out_channels = channel_dims[level_index + 1]
            is_last_level = level_index == level_count - 1
            self._down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(in_channels, out_channels, cond_dim, kernel_size, group_count),
                ConditionalResidualBlock1D(out_channels, out_channels, cond_dim, kernel_size, group_count),
                nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1) if not is_last_level else nn.Identity(),
            ]))

        # 가운데 블록
        bottleneck_channels = down_dims[-1]
        self._mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(bottleneck_channels, bottleneck_channels, cond_dim, kernel_size, group_count),
            ConditionalResidualBlock1D(bottleneck_channels, bottleneck_channels, cond_dim, kernel_size, group_count),
        ])

        # 디코더 경로: skip 연결과 합친 뒤 residual block 2개 + 2배 업샘플
        self._up_modules = nn.ModuleList()
        reversed_dims = list(reversed(down_dims))
        for level_index in range(level_count - 1):
            in_channels = reversed_dims[level_index]
            out_channels = reversed_dims[level_index + 1]
            self._up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(in_channels * 2, out_channels, cond_dim, kernel_size, group_count),
                ConditionalResidualBlock1D(out_channels, out_channels, cond_dim, kernel_size, group_count),
                nn.ConvTranspose1d(out_channels, out_channels, 4, stride=2, padding=1),
            ]))

        # 출력 투영
        self._final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, group_count),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def horizon(self):
        return self._horizon

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def time_embed_dim(self):
        return self._time_embed_dim

    def input_info(self, batch_size):
        """RBLN 컴파일용 고정 입력 shape 목록."""
        return [
            ("noisy_action", [batch_size, self._horizon, self._action_dim], "float32"),
            ("time_features", [batch_size, self._time_embed_dim], "float32"),
            ("global_cond", [batch_size, self._obs_dim], "float32"),
        ]

    def forward(self, noisy_action, time_features, global_cond):
        # 조건 벡터 구성
        cond_vector = torch.cat([self._diffusion_step_encoder(time_features), global_cond], dim=-1)

        # [B, H, Da] → [B, Da, H] (채널 우선)
        feature = noisy_action.transpose(1, 2)

        # 인코더: 각 레벨 출력을 skip 으로 저장
        skip_features = []
        for first_block, second_block, downsample in self._down_modules:
            feature = first_block(feature, cond_vector)
            feature = second_block(feature, cond_vector)
            skip_features.append(feature)
            feature = downsample(feature)

        # 가운데 블록
        for mid_block in self._mid_modules:
            feature = mid_block(feature, cond_vector)

        # 디코더: 가장 깊은 skip 은 가운데 블록 출력과 같은 해상도
        for (first_block, second_block, upsample), skip_feature in zip(self._up_modules,
                                                                      reversed(skip_features[1:])):
            feature = torch.cat([feature, skip_feature], dim=1)
            feature = first_block(feature, cond_vector)
            feature = second_block(feature, cond_vector)
            feature = upsample(feature)

        # 출력 투영 후 [B, H, Da] 로 복원
        epsilon = self._final_conv(feature)
        return epsilon.transpose(1, 2)
