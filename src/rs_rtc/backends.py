"""노이즈 예측기(UNet) 실행 장치: CPU, CUDA, NPU 를 같은 호출 방식으로 실행하는 래퍼.

모든 래퍼는 호스트 CPU 텐서를 받아 호스트 CPU 텐서 ε̂ 를 돌려준다.
rebel (RBLN SDK) 은 NPU 를 쓸 때만 import 하므로 SDK 가 없는 GPU 환경에서도 이 모듈을 쓸 수 있다.
"""
import logging

import torch

from rs_rtc import RESULTS_DIR
from rs_rtc.utils.cuda_graph import CudaGraphFunction

LOGGER = logging.getLogger("rs_rtc.backends")
COMPILED_DIR = RESULTS_DIR / "compiled"
BACKENDS = ("npu", "cuda", "cpu")


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


class TorchDenoiser:
    """UNet 을 torch 장치 (cuda 또는 cpu) 에서 고정 batch 로 실행 (NPU 런타임과 같은 호출 방식)."""

    def __init__(self, unet, device, batch_size, use_cuda_graph=False):
        self._device = torch.device(device)
        self._unet = unet.to(self._device)
        self._batch_size = batch_size
        self._use_cuda_graph = use_cuda_graph
        self._forward_graph = None
        LOGGER.info("torch 노이즈 예측기: device=%s, batch=%d, CUDA Graph=%s", self._device, batch_size,
                    self._use_cuda_graph)

    @property
    def batch_size(self):
        return self._batch_size

    def forward_on_device(self, noisy_action, time_features, global_cond):
        """장치 텐서로 ε̂ 계산 (CUDA Graph 캡처 대상)."""
        with torch.no_grad():
            return self._unet(noisy_action, time_features, global_cond)

    def __call__(self, noisy_action, time_features, global_cond):
        # 고정 batch 확인 (NPU 정적 그래프와 같은 조건)
        if noisy_action.shape[0] != self._batch_size:
            raise ValueError(f"입력 batch {noisy_action.shape[0]} ≠ 설정 batch {self._batch_size}")

        # 장치로 전송 후 실행 (CUDA Graph 사용 시 그래프 재생), 결과를 호스트로 복사
        device_inputs = (noisy_action.to(self._device), time_features.to(self._device), global_cond.to(self._device))
        if not self._use_cuda_graph:
            return self.forward_on_device(*device_inputs).cpu()
        if self._forward_graph is None:
            self._forward_graph = CudaGraphFunction(self.forward_on_device, device_inputs,
                                                    f"unet_b{self._batch_size}")
        return self._forward_graph(*device_inputs).cpu()


class NpuDenoiserRuntime:
    """고정 batch 로 컴파일된 UNet 그래프 1개를 NPU 장치 1개에서 실행."""

    def __init__(self, compiled_path, device_index, batch_size):
        import rebel

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


def torch_device_name(backend, device_index):
    """backend 와 장치 번호로 torch 장치 이름 (cuda:0, cpu)."""
    return f"cuda:{device_index}" if backend == "cuda" else "cpu"


def build_denoiser(backend, unet, model_name, batch_size, device_index=0, use_cuda_graph=False):
    """backend 에 맞는 고정 batch 노이즈 예측기 (NPU 는 {model_name}_unet_b{batch}.rbln, 없으면 컴파일)."""
    if backend not in BACKENDS:
        raise ValueError(f"알 수 없는 backend: {backend} (가능: {BACKENDS})")

    # NPU: 고정 batch 그래프 로드
    if backend == "npu":
        from rs_rtc.npu.compile import compile_if_missing

        compiled_path = COMPILED_DIR / f"{model_name}_unet_b{batch_size}.rbln"
        compile_if_missing(unet, batch_size, compiled_path)
        return NpuDenoiserRuntime(compiled_path, device_index, batch_size)

    # CUDA·CPU: torch 고정 batch 실행
    return TorchDenoiser(unet, torch_device_name(backend, device_index), batch_size, use_cuda_graph)
