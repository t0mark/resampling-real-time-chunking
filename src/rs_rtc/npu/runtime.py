"""노이즈 예측기(UNet)를 CPU 또는 NPU 에서 같은 호출 방식으로 실행하는 래퍼."""
import logging

import rebel
import torch

LOGGER = logging.getLogger("rs_rtc.npu")


class CpuDenoiser:
    """PyTorch UNet 을 CPU 에서 실행 (임의 batch)."""

    def __init__(self, unet):
        self._unet = unet

    @property
    def batch_size(self):
        return None

    def __call__(self, noisy_action, time_features, global_cond):
        with torch.no_grad():
            return self._unet(noisy_action, time_features, global_cond)


class NpuDenoiserRuntime:
    """고정 batch 로 컴파일된 UNet 그래프 1개를 NPU 장치 1개에서 실행."""

    def __init__(self, compiled_path, device_index, batch_size):
        self._compiled_path = compiled_path
        self._device_index = device_index
        self._batch_size = batch_size
        self._runtime = rebel.Runtime(str(compiled_path), tensor_type="pt", device=device_index)
        LOGGER.info("런타임 로드: %s (device=%d, batch=%d)", compiled_path.name, device_index, batch_size)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device_index(self):
        return self._device_index

    def __call__(self, noisy_action, time_features, global_cond):
        # 정적 그래프이므로 입력 batch 크기 확인
        if noisy_action.shape[0] != self._batch_size:
            raise ValueError(f"입력 batch {noisy_action.shape[0]} ≠ 컴파일 batch {self._batch_size}")

        epsilon = self._runtime.run(noisy_action.contiguous(), time_features.contiguous(), global_cond.contiguous())
        return torch.as_tensor(epsilon).float()
