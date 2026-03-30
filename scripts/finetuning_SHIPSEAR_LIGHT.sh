#!/bin/bash
set -euo pipefail

GPU_ID="${1:-0}"
EAT_ROOT="/hy-tmp/eat_work/EAT"
FAIRSEQ_ROOT="/hy-tmp/eat_work/fairseq"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${FAIRSEQ_ROOT}"
python fairseq_cli/hydra_train.py -m \
  --config-dir "${EAT_ROOT}/config" \
  --config-name finetuning_shipsear_light \
  common.user_dir="${EAT_ROOT}" \
  checkpoint.save_dir=/hy-tmp/exp/eat/runs/shipsear_light_finetune \
  task.data=/hy-tmp/exp/eat/manifests/shipsear
