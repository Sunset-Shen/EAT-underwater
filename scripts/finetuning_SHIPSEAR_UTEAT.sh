#!/bin/bash
set -euo pipefail
GPU_ID="${1:-0}"
BATCH_SIZE="${2:-8}"
UPDATE_FREQ="${3:-6}"
MAX_UPDATE="${4:-8000}"
FRESH_START="${5:-1}"
EAT_ROOT="/hy-tmp/eat_work/EAT"
FAIRSEQ_ROOT="/hy-tmp/eat_work/fairseq"
MANIFEST_DIR="/hy-tmp/exp/eat/manifests/shipsear"
RUN_ROOT="/hy-tmp/exp/eat/runs/shipsear_uteat"
FORMAL_PRETRAIN_CKPT="/hy-tmp/exp/eat/runs/shipsear_uteat_formal_pretrain/checkpoint_last.pt"
LEGACY_PRETRAIN_CKPT="/hy-tmp/exp/eat/runs/shipsear_uteat_pretrain/checkpoint_last.pt"
CONFIG_NAME="finetuning_shipsear_uteat"

if [[ -f "${FORMAL_PRETRAIN_CKPT}" ]]; then
  PRETRAIN_CKPT="${FORMAL_PRETRAIN_CKPT}"
elif [[ -f "${LEGACY_PRETRAIN_CKPT}" ]]; then
  PRETRAIN_CKPT="${LEGACY_PRETRAIN_CKPT}"
  echo "[WARN] Using legacy pretrain checkpoint: ${PRETRAIN_CKPT}"
else
  echo "[ERROR] Missing UT-EAT pretrain checkpoint."
  echo "[ERROR] Expected formal: ${FORMAL_PRETRAIN_CKPT}"
  echo "[ERROR] Fallback legacy: ${LEGACY_PRETRAIN_CKPT}"
  exit 1
fi

mkdir -p "${RUN_ROOT}"
if [[ "${FRESH_START}" == "1" ]]; then
  rm -f "${RUN_ROOT}"/checkpoint*.pt
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[UTEAT-FINETUNE] config-name=${CONFIG_NAME}"
echo "[UTEAT-FINETUNE] checkpoint.save_dir=${RUN_ROOT}"
echo "[UTEAT-FINETUNE] model.model_path=${PRETRAIN_CKPT}"
echo "[UTEAT-FINETUNE] optimization.max_update=${MAX_UPDATE}"
echo "[UTEAT-FINETUNE] dataset.batch_size=${BATCH_SIZE}"
echo "[UTEAT-FINETUNE] optimization.update_freq=[${UPDATE_FREQ}]"

cd "${FAIRSEQ_ROOT}"
python fairseq_cli/hydra_train.py \
  --config-dir "${EAT_ROOT}/config" \
  --config-name "${CONFIG_NAME}" \
  common.user_dir="${EAT_ROOT}" \
  task.data="${MANIFEST_DIR}" \
  checkpoint.save_dir="${RUN_ROOT}" \
  model.model_path="${PRETRAIN_CKPT}" \
  optimization.max_update="${MAX_UPDATE}" \
  dataset.batch_size="${BATCH_SIZE}" \
  optimization.update_freq="[${UPDATE_FREQ}]"
