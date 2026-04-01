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
  local physical_loss_weight="$2"
  local mask_strategy="$3"
  local run_root="/hy-tmp/exp/eat/runs/ablation_shipsear_${variant}_pretrain"
  mkdir -p "${run_root}"
  cd "${FAIRSEQ_ROOT}"
  python fairseq_cli/hydra_train.py \
    --config-dir "${EAT_ROOT}/config" \
    --config-name "pretraining_shipsear_uteat_formal" \
    common.user_dir="${EAT_ROOT}" \
    task.data="${MANIFEST_DIR}" \
    checkpoint.save_dir="${run_root}" \
    common.tensorboard_logdir="/hy-tmp/exp/eat/logs/ablation_shipsear_${variant}_pretrain/tb" \
    model.backbone_variant=uteat \
    model.physical_loss_weight="${physical_loss_weight}" \
    model.modalities.image.mask_strategy="${mask_strategy}" \
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
run_pretrain "ut_backbone" "0.0" "baseline"
run_finetune "ut_backbone"

# UT-backbone + mask
run_pretrain "ut_mask" "0.0" "uteat_phys"
run_finetune "ut_mask"

# UT-backbone + loss
run_pretrain "ut_loss" "0.2" "baseline"
run_finetune "ut_loss"

# UT-EAT full
run_pretrain "ut_full" "0.2" "uteat_phys"
run_finetune "ut_full"

echo "[DONE] ablation shipsear training finished"
