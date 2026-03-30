#!/bin/bash
set -euo pipefail
GPU_ID="${1:-0}"
BATCH_SIZE="${2:-8}"
UPDATE_FREQ="${3:-2}"
FRESH_START="${4:-1}"
EAT_ROOT="/hy-tmp/eat_work/EAT"
FAIRSEQ_ROOT="/hy-tmp/eat_work/fairseq"
MANIFEST_DIR="/hy-tmp/exp/eat/manifests/deepship"
RUN_ROOT="/hy-tmp/exp/eat/runs/deepship_uteat_smoke_pretrain"
CONFIG_NAME="pretraining_deepship_uteat_smoke"
MAX_UPDATE=100
mkdir -p "${RUN_ROOT}"
if [[ "${FRESH_START}" == "1" ]]; then
  rm -f "${RUN_ROOT}"/checkpoint*.pt
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
echo "[UTEAT-SMOKE] config-name=${CONFIG_NAME}"
echo "[UTEAT-SMOKE] checkpoint.save_dir=${RUN_ROOT}"
echo "[UTEAT-SMOKE] optimization.max_update=${MAX_UPDATE}"
echo "[UTEAT-SMOKE] checkpoint.restore_file=${RUN_ROOT}/checkpoint_last.pt"
cd "${FAIRSEQ_ROOT}"
python fairseq_cli/hydra_train.py \
  --config-dir "${EAT_ROOT}/config" \
  --config-name "${CONFIG_NAME}" \
  common.user_dir="${EAT_ROOT}" \
  task.data="${MANIFEST_DIR}" \
  checkpoint.save_dir="${RUN_ROOT}" \
  optimization.max_update="${MAX_UPDATE}" \
  common.log_file="${RUN_ROOT}/train.log" \
  dataset.batch_size="${BATCH_SIZE}" \
  optimization.update_freq="[${UPDATE_FREQ}]"
python "${EAT_ROOT}/utils/export_run_artifacts.py" --run-dir "${RUN_ROOT}" --tag deepship_uteat_smoke_pretrain
