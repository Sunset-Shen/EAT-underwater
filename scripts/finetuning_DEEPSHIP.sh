#!/bin/bash
set -euo pipefail

# Usage: bash scripts/finetuning_DEEPSHIP.sh [GPU_ID] [BATCH_SIZE] [UPDATE_FREQ] [MAX_UPDATE]
GPU_ID="${1:-0}"
BATCH_SIZE="${2:-8}"
UPDATE_FREQ="${3:-6}"
MAX_UPDATE="${4:-4000}"

EAT_ROOT="/hy-tmp/eat_work/EAT"
FAIRSEQ_ROOT="/hy-tmp/eat_work/fairseq"
MANIFEST_DIR="/hy-tmp/exp/eat/manifests/deepship"
RUN_ROOT="/hy-tmp/exp/eat/runs/deepship_baseline"
PRETRAIN_CKPT="/hy-tmp/model_zoo/eat_checkpoints/EAT-base_epoch30_pt.pt"
if [[ ! -f "${MANIFEST_DIR}/label_descriptors.csv" ]]; then
  echo "[ERROR] ${MANIFEST_DIR}/label_descriptors.csv not found. Please generate manifest first."
  exit 1
fi
NUM_CLASSES=$(wc -l < "${MANIFEST_DIR}/label_descriptors.csv")

mkdir -p "${RUN_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

cd "${FAIRSEQ_ROOT}"
python fairseq_cli/hydra_train.py \
    --config-dir "${EAT_ROOT}/config" \
    --config-name finetuning_deepship \
    common.user_dir="${EAT_ROOT}" \
    checkpoint.save_dir="${RUN_ROOT}" \
    dataset.batch_size="${BATCH_SIZE}" \
    optimization.update_freq="[${UPDATE_FREQ}]" \
    optimization.max_update="${MAX_UPDATE}" \
    task.data="${MANIFEST_DIR}" \
    model.model_path="${PRETRAIN_CKPT}" \
    model.num_classes="${NUM_CLASSES}"
