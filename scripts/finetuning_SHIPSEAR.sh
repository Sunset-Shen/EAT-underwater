#!/bin/bash
set -euo pipefail

# Usage (example):
#   bash scripts/finetuning_SHIPSEAR.sh 0 8 6 4000
# args: GPU_ID BATCH_SIZE UPDATE_FREQ MAX_UPDATE

GPU_ID="${1:-0}"
BATCH_SIZE="${2:-8}"
UPDATE_FREQ="${3:-6}"
MAX_UPDATE="${4:-4000}"

EAT_ROOT="/hy-tmp/eat_work/EAT"
FAIRSEQ_ROOT="/hy-tmp/eat_work/fairseq"
MANIFEST_DIR="/hy-tmp/exp/eat/manifests/shipsear"
RUN_ROOT="/hy-tmp/exp/eat/runs/shipsear_baseline"
PRETRAIN_CKPT="/hy-tmp/model_zoo/eat_checkpoints/EAT-base_epoch30_pt.pt"

mkdir -p "${RUN_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

cd "${FAIRSEQ_ROOT}"
python fairseq_cli/hydra_train.py \
    --config-dir "${EAT_ROOT}/config" \
    --config-name finetuning_shipsear \
    common.user_dir="${EAT_ROOT}" \
    common.log_interval=50 \
    common.seed=42 \
    checkpoint.save_dir="${RUN_ROOT}" \
    checkpoint.best_checkpoint_metric=accuracy \
    checkpoint.maximize_best_checkpoint_metric=true \
    dataset.train_subset=train \
    dataset.valid_subset=valid \
    dataset.batch_size="${BATCH_SIZE}" \
    optimization.update_freq="[${UPDATE_FREQ}]" \
    criterion.log_keys=['correct'] \
    task.data="${MANIFEST_DIR}" \
    task.audio_mae=true \
    task.downsr_16hz=true \
    task.target_length=1024 \
    task.esc50_eval=true \
    task.roll_aug=true \
    task.noise=false \
    optimization.max_update="${MAX_UPDATE}" \
    optimization.lr=[0.00005] \
    optimizer.groups.default.lr_float=0.00005 \
    optimizer.groups.default.lr_scheduler.warmup_updates=800 \
    model.model_path="${PRETRAIN_CKPT}" \
    model.num_classes=12 \
    model.esc50_eval=true \
    model.audio_mae=true \
    model.target_length=1024 \
    model.mixup=0.0 \
    model.mask_ratio=0.2 \
    model.label_smoothing=0.1 \
    model.prediction_mode=PredictionMode.CLS_TOKEN
