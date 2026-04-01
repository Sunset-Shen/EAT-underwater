#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
BATCH_SIZE_PRETRAIN="${2:-8}"
BATCH_SIZE_FINETUNE="${3:-8}"
PRETRAIN_UPDATE_FREQ="${4:-2}"
FINETUNE_UPDATE_FREQ="${5:-6}"
PRETRAIN_MAX_UPDATE="${6:-10000}"
FINETUNE_MAX_UPDATE="${7:-4000}"

EAT_ROOT="${EAT_ROOT:-/hy-tmp/eat_work/EAT}"
FAIRSEQ_ROOT="${FAIRSEQ_ROOT:-/hy-tmp/eat_work/fairseq}"
MANIFEST_DIR="${MANIFEST_DIR:-/hy-tmp/exp/eat/manifests/shipsear}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

run_pretrain() {
  local variant="$1"
  local config_name="$2"
  local run_root="/hy-tmp/exp/eat/runs/ablation_shipsear_${variant}_pretrain"
  mkdir -p "${run_root}"
  cd "${FAIRSEQ_ROOT}"
  python fairseq_cli/hydra_train.py \
    --config-dir "${EAT_ROOT}/config/ablation" \
    --config-name "${config_name}" \
    common.user_dir="${EAT_ROOT}" \
    task.data="${MANIFEST_DIR}" \
    checkpoint.save_dir="${run_root}" \
    optimization.max_update="${PRETRAIN_MAX_UPDATE}" \
    dataset.batch_size="${BATCH_SIZE_PRETRAIN}" \
    optimization.update_freq="[${PRETRAIN_UPDATE_FREQ}]" \
    common.log_file="${run_root}/train.log"
}

run_finetune() {
  local variant="$1"
  local pretrain_ckpt="/hy-tmp/exp/eat/runs/ablation_shipsear_${variant}_pretrain/checkpoint_last.pt"
  local run_root="/hy-tmp/exp/eat/runs/ablation_shipsear_${variant}_finetune"
  mkdir -p "${run_root}"
  cd "${FAIRSEQ_ROOT}"
  python fairseq_cli/hydra_train.py \
    --config-dir "${EAT_ROOT}/config" \
    --config-name finetuning_shipsear_uteat \
    common.user_dir="${EAT_ROOT}" \
    task.data="${MANIFEST_DIR}" \
    checkpoint.save_dir="${run_root}" \
    model.model_path="${pretrain_ckpt}" \
    optimization.max_update="${FINETUNE_MAX_UPDATE}" \
    dataset.batch_size="${BATCH_SIZE_FINETUNE}" \
    optimization.update_freq="[${FINETUNE_UPDATE_FREQ}]" \
    common.log_file="${run_root}/train.log"
}

# UT-backbone only
run_pretrain "ut_backbone" "pretraining_shipsear_ablation_ut_backbone"
run_finetune "ut_backbone"

# UT-backbone + mask
run_pretrain "ut_mask" "pretraining_shipsear_ablation_ut_mask"
run_finetune "ut_mask"

# UT-backbone + loss
run_pretrain "ut_loss" "pretraining_shipsear_ablation_ut_loss"
run_finetune "ut_loss"

# UT-EAT full
run_pretrain "ut_full" "pretraining_shipsear_ablation_ut_full"
run_finetune "ut_full"

echo "[DONE] ablation shipsear training finished"
