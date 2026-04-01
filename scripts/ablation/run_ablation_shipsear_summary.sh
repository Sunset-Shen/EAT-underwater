#!/usr/bin/env bash
set -euo pipefail

EAT_ROOT="${EAT_ROOT:-/hy-tmp/eat_work/EAT}"
RESULT_DIR="${RESULT_DIR:-/hy-tmp/exp/eat/results/ablation_shipsear}"
MANIFEST_DIR="${MANIFEST_DIR:-/hy-tmp/exp/eat/manifests/shipsear}"
FIGURES_DIR="${FIGURES_DIR:-/hy-tmp/exp/eat/results/compare_v1/figures}"

mkdir -p "${RESULT_DIR}/eval"

run_eval() {
  local variant="$1"
  local ckpt="$2"
  python "${EAT_ROOT}/utils/eval_manifest_classification.py" \
    --checkpoint "${ckpt}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --dataset-name shipsear \
    --model-name "${variant}" \
    --subset test \
    --output "${RESULT_DIR}/eval/${variant}_eval.json"
}

run_eval ut_backbone /hy-tmp/exp/eat/runs/ablation_shipsear_ut_backbone_finetune/checkpoint_best.pt
run_eval ut_mask /hy-tmp/exp/eat/runs/ablation_shipsear_ut_mask_finetune/checkpoint_best.pt
run_eval ut_loss /hy-tmp/exp/eat/runs/ablation_shipsear_ut_loss_finetune/checkpoint_best.pt
run_eval ut_full /hy-tmp/exp/eat/runs/ablation_shipsear_ut_full_finetune/checkpoint_best.pt

python "${EAT_ROOT}/utils/summarize_ablation_shipsear.py" \
  --output-dir "${RESULT_DIR}" \
  --compare-main-csv /hy-tmp/exp/eat/results/compare_v1/summary/compare_main.csv

python "${EAT_ROOT}/utils/plot_ablation_shipsear.py" \
  --ablation-dir "${RESULT_DIR}" \
  --figures-dir "${FIGURES_DIR}"

echo "[DONE] ablation summary: ${RESULT_DIR}"
echo "[DONE] ablation figures: ${FIGURES_DIR}"
