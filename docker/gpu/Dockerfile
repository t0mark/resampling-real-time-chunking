# RS-RTC GPU 평가 이미지: 역전파(VJP) RTC 비교군을 CUDA 에서 실행 (RBLN SDK 불필요)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1

# PushT 렌더링(opencv, pygame) 런타임 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 평가 의존성 (NPU 컨테이너와 같은 버전)
RUN pip install \
        numpy==2.2.6 \
        pyyaml==6.0.3 \
        pandas==2.3.3 \
        pyarrow==25.0.1 \
        huggingface-hub==0.36.2 \
        safetensors==0.7.0 \
        gymnasium==1.4.0 \
        gym-pusht==0.1.8 \
        diffusers==0.35.2 \
        pytest

WORKDIR /workspace/rs_rtc
CMD ["bash"]
