#!/bin/bash
set -euo pipefail
GPU_ID="${1:-0}"
BATCH_SIZE="${2:-8}"
UPDATE_FREQ="${3:-6}"
EAT_ROOT="/hy-tmp/eat_work/EAT"
FAIRSEQ_ROOT="/hy-tmp/eat_work/fairseq"
MANIFEST_DIR="/hy-tmp/exp/eat/manifests/deepship"
if [[ ! -f "${MANIFEST_DIR}/label_descriptors.csv" ]]; then
  echo "[ERROR] ${MANIFEST_DIR}/label_descriptors.csv not found."
  exit 1
fi
NUM_CLASSES=$(wc -l < "${MANIFEST_DIR}/label_descriptors.csv")
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
cd "${FAIRSEQ_ROOT}"
python fairseq_cli/hydra_train.py \
  --config-dir "${EAT_ROOT}/config" \
  --config-name finetuning_deepship_uteat \
  common.user_dir="${EAT_ROOT}" \
  dataset.batch_size="${BATCH_SIZE}" \
  optimization.update_freq="[${UPDATE_FREQ}]" \
  model.num_classes="${NUM_CLASSES}"
