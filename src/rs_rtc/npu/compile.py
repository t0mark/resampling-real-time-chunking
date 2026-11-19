"""노이즈 예측기를 RBLN 정적 그래프로 컴파일·저장하고 컴파일 결과를 점검한다."""
import logging
import time

import rebel

LOGGER = logging.getLogger("rs_rtc.npu")


def compile_denoiser_model(denoiser_model, batch_size, output_path):
    """고정 batch 크기로 컴파일해 .rbln 으로 저장하고 컴파일 시간(초)을 반환."""
    # 평가 모드로 고정 후 컴파일
    denoiser_model.eval()
    LOGGER.info("컴파일 시작: batch=%d → %s", batch_size, output_path)
    compile_start = time.perf_counter()
    compiled_model = rebel.compile_from_torch(denoiser_model, input_info=denoiser_model.input_info(batch_size))
    compile_seconds = time.perf_counter() - compile_start

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiled_model.save(str(output_path))
    LOGGER.info("컴파일 완료: batch=%d, %.1f 초", batch_size, compile_seconds)
    return compile_seconds


def compile_if_missing(denoiser_model, batch_size, output_path):
    """컴파일 산출물이 없을 때만 컴파일한다."""
    if output_path.exists():
        LOGGER.info("기존 컴파일 결과 사용: %s", output_path)
        return
    compile_denoiser_model(denoiser_model, batch_size, output_path)


def inspect_compiled_model(compiled_path):
    """저장된 .rbln 의 서브그래프 통계로 NPU 함수 수, host op(CPU fallback), 장치 메모리를 확인."""
    # 메타데이터의 서브그래프 통계 집계
    model_metadata = rebel.RBLNCompiledModel.inspect(str(compiled_path))
    subgraph_stats = model_metadata["subgraph"]["stats"]
    device_func_count = sum(stat["device_func_count"] for stat in subgraph_stats)
    host_op_reasons = {}
    for stat in subgraph_stats:
        host_op_reasons.update(stat["host_op_reason"])

    inspection = {
        "device_func_count": device_func_count,
        "host_op_reasons": host_op_reasons,
        "device_alloc_bytes": sum(model_metadata["alloc_per_node"]),
    }
    LOGGER.info("점검: device_func=%d, host_op=%s, alloc=%.1f MB", device_func_count,
                host_op_reasons or "없음", inspection["device_alloc_bytes"] / 2**20)
    return inspection
