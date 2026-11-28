#!/usr/bin/env bash
# Diffusion Policy robomimic low-dim 체크포인트와 데이터셋 다운로드 (assets/robomimic/)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${PROJECT_DIR}/assets/robomimic"
SERVER_URL="https://diffusion-policy.cs.columbia.edu/data"

# task 이름 → 공개 체크포인트 파일명
declare -A CHECKPOINT_FILES=(
    ["square_ph"]="epoch%3D1750-test_mean_score%3D1.000.ckpt"
    ["tool_hang_ph"]="epoch%3D0850-test_mean_score%3D0.818.ckpt"
)

mkdir -p "${ASSET_DIR}/checkpoints"

# 체크포인트 다운로드 (이미 있으면 건너뜀)
for task_name in "${!CHECKPOINT_FILES[@]}"; do
    checkpoint_path="${ASSET_DIR}/checkpoints/${task_name}.ckpt"
    if [[ -f "${checkpoint_path}" ]]; then
        echo "[download] 체크포인트 있음: ${checkpoint_path}"
        continue
    fi
    echo "[download] 체크포인트: ${task_name}"
    curl -fL --retry 3 -o "${checkpoint_path}.part" \
        "${SERVER_URL}/experiments/low_dim/${task_name}/diffusion_policy_cnn/train_0/checkpoints/${CHECKPOINT_FILES[${task_name}]}"
    mv "${checkpoint_path}.part" "${checkpoint_path}"
done

# 데이터셋 다운로드 (환경 설정 읽기용)
dataset_zip="${ASSET_DIR}/robomimic_lowdim.zip"
if [[ ! -f "${dataset_zip}" ]]; then
    echo "[download] 데이터셋: robomimic_lowdim.zip"
    curl -fL --retry 3 -o "${dataset_zip}.part" "${SERVER_URL}/training/robomimic_lowdim.zip"
    mv "${dataset_zip}.part" "${dataset_zip}"
fi

# 과제별 절대 자세 데이터셋만 압축 해제 (assets/robomimic/datasets/{task}/ph/low_dim_abs.hdf5)
for task_name in "${!CHECKPOINT_FILES[@]}"; do
    dataset_file="${task_name%_ph}/ph/low_dim_abs.hdf5"
    if [[ -f "${ASSET_DIR}/datasets/${dataset_file}" ]]; then
        echo "[download] 데이터셋 있음: ${dataset_file}"
        continue
    fi
    echo "[download] 압축 해제: ${dataset_file}"
    unzip -q -o "${dataset_zip}" "robomimic/datasets/${dataset_file}" -d "${ASSET_DIR}/extract_tmp"
    mkdir -p "$(dirname "${ASSET_DIR}/datasets/${dataset_file}")"
    mv "${ASSET_DIR}/extract_tmp/robomimic/datasets/${dataset_file}" "${ASSET_DIR}/datasets/${dataset_file}"
    rm -rf "${ASSET_DIR}/extract_tmp"
done
echo "[download] 완료: ${ASSET_DIR}"
