#!/bin/bash
set -euo pipefail
GPU_ID="${1:-0}"
BATCH_SIZE="${2:-8}"
UPDATE_FREQ="${3:-2}"
EAT_ROOT="/hy-tmp/eat_work/EAT"
FAIRSEQ_ROOT="/hy-tmp/eat_work/fairseq"
RUN_ROOT="/hy-tmp/exp/eat/runs/deepship_uteat_formal_pretrain"
mkdir -p "${RUN_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
cd "${FAIRSEQ_ROOT}"
python fairseq_cli/hydra_train.py \
  --config-dir "${EAT_ROOT}/config" \
  --config-name pretraining_deepship_uteat_formal \
  common.user_dir="${EAT_ROOT}" \
  common.log_file="${RUN_ROOT}/train.log" \
  dataset.batch_size="${BATCH_SIZE}" \
  optimization.update_freq="[${UPDATE_FREQ}]"
python "${EAT_ROOT}/utils/export_run_artifacts.py" --run-dir "${RUN_ROOT}" --tag deepship_uteat_formal_pretrain
