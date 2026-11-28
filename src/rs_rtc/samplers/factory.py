"""방법 설정 → chunk 샘플러 생성 (방법 로직과 실행 장치를 분리)."""
from rs_rtc.backends import build_denoiser, torch_device_name
from rs_rtc.samplers.base import DdimChunkSampler
from rs_rtc.samplers.baselines import ReplacementSampler
from rs_rtc.samplers.bid import BidSampler
from rs_rtc.samplers.rs_rtc import RsRtcSampler
from rs_rtc.samplers.rtc_vjp import RtcPaperSampler, RtcVjpSampler

# 순전파만 쓰는 방법 (모든 backend) / 역전파가 필요한 방법 (cuda·cpu 만)
FORWARD_SAMPLER_CLASSES = {"vanilla": DdimChunkSampler, "replacement": ReplacementSampler, "bid": BidSampler}
GRADIENT_SAMPLER_CLASSES = {"rtc_vjp": RtcVjpSampler, "rtc_paper": RtcPaperSampler}


def build_sampler(method_config, policy, schedule, backend, device_index=0):
    """방법 설정과 backend 로 chunk 샘플러를 만든다."""
    sampler_arguments = (schedule, policy.horizon, policy.unet.action_dim, policy.unet.time_embed_dim)
    method_type = method_config["type"]

    # CUDA Graph 는 cuda backend 에서만 사용
    use_cuda_graph = method_config.get("cuda_graph", False) and backend == "cuda"

    # 역전파 가이던스 (RTC): UNet 을 직접 받아 autograd 로 계산
    if method_type in GRADIENT_SAMPLER_CLASSES:
        if backend == "npu":
            raise ValueError(f"{method_type} 은 역전파가 필요해 NPU 에서 실행할 수 없음 (backend: cuda 또는 cpu)")
        device = torch_device_name(backend, device_index)
        if method_type == "rtc_paper":
            return RtcPaperSampler(policy.unet, *sampler_arguments, method_config["eta"],
                                   method_config["guidance_weight"], device, use_cuda_graph)
        return RtcVjpSampler(policy.unet, *sampler_arguments, method_config["eta"], method_config["guidance_weight"],
                             device)

    # 순전파만 쓰는 방법: 고정 batch 노이즈 예측기
    denoiser = build_denoiser(backend, policy.unet, policy.name, method_config["particles"], device_index,
                              use_cuda_graph)
    if method_type == "rs_rtc":
        return RsRtcSampler(denoiser, *sampler_arguments, method_config["eta"], method_config["guidance_strength"],
                            method_config["ess_threshold"], method_config["use_replacement"],
                            method_config.get("kalman_move", False), method_config.get("kalman_scale", 1.0))
    if method_type in FORWARD_SAMPLER_CLASSES:
        return FORWARD_SAMPLER_CLASSES[method_type](denoiser, *sampler_arguments, method_config["eta"])
    raise ValueError(f"알 수 없는 방법 type: {method_type}")
