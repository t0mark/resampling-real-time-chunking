"""고정 shape GPU 함수를 CUDA Graph 로 캡처해 재생 (커널 디스패치 오버헤드 제거, NPU 정적 그래프·JAX jit 과 같은 조건)."""
import logging

import torch

LOGGER = logging.getLogger("rs_rtc.utils")


class CudaGraphFunction:
    """function(*정적 입력) 을 1회 캡처하고, 호출 시 입력을 정적 버퍼에 복사한 뒤 재생한다.

    반환값은 그래프의 정적 출력 버퍼이므로 다음 호출 전에 복사(.cpu() 등)해서 써야 한다.
    """

    def __init__(self, function, example_inputs, name, warmup_iterations=3):
        self._name = name
        self._static_inputs = [example_input.detach().clone() for example_input in example_inputs]

        # 캡처 전 별도 stream 에서 warm-up (cuDNN 알고리즘·메모리 할당 확정)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup_iterations):
                function(*self._static_inputs)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        # 그래프 캡처
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = function(*self._static_inputs)
        torch.cuda.synchronize()
        LOGGER.info("CUDA Graph 캡처 완료: %s (입력 shape %s)", name,
                    [tuple(static_input.shape) for static_input in self._static_inputs])

    def __call__(self, *inputs):
        # 입력을 정적 버퍼로 복사 후 재생
        for static_input, new_input in zip(self._static_inputs, inputs):
            static_input.copy_(new_input)
        self._graph.replay()
        return self._static_output
