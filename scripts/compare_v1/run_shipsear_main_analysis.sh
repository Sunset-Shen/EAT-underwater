#!/usr/bin/env bash
set -euo pipefail

EAT_ROOT="${EAT_ROOT:-/hy-tmp/eat_work/EAT}"
ANALYSIS_DIR="${ANALYSIS_DIR:-/hy-tmp/exp/eat/results/compare_v1/shipsear/analysis}"
FIGURES_DIR="${FIGURES_DIR:-/hy-tmp/exp/eat/results/compare_v1/figures}"

BASELINE_CKPT="${BASELINE_CKPT:-/hy-tmp/exp/eat/runs/shipsear_baseline/checkpoint_best.pt}"
UTEAT_CKPT="${UTEAT_CKPT:-/hy-tmp/exp/eat/runs/shipsear_uteat/checkpoint_best.pt}"
MANIFEST_DIR="${MANIFEST_DIR:-/hy-tmp/exp/eat/manifests/shipsear}"
COMPARE_MAIN_CSV="${COMPARE_MAIN_CSV:-/hy-tmp/exp/eat/results/compare_v1/summary/compare_main.csv}"
PRETRAIN_LOG="${PRETRAIN_LOG:-/hy-tmp/exp/eat/results/compare_v1/shipsear_formal_pretrain_2026-03-31_164952.log}"
BASELINE_FINETUNE_LOG="${BASELINE_FINETUNE_LOG:-/hy-tmp/exp/eat/results/compare_v1/train_shipsear_eatbase_2026-03-31_115626.log}"
UTEAT_FINETUNE_LOG="${UTEAT_FINETUNE_LOG:-/hy-tmp/exp/eat/results/compare_v1/shipsear_uteat_finetune_2026-04-01_005407.log}"

python "${EAT_ROOT}/utils/export_shipsear_classification_analysis.py" \
  --manifest-dir "${MANIFEST_DIR}" \
  --baseline-checkpoint "${BASELINE_CKPT}" \
  --uteat-checkpoint "${UTEAT_CKPT}" \
  --analysis-dir "${ANALYSIS_DIR}"

python "${EAT_ROOT}/utils/plot_shipsear_main_analysis.py" \
  --compare-main-csv "${COMPARE_MAIN_CSV}" \
  --pretrain-log "${PRETRAIN_LOG}" \
  --baseline-finetune-log "${BASELINE_FINETUNE_LOG}" \
  --uteat-finetune-log "${UTEAT_FINETUNE_LOG}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --figures-dir "${FIGURES_DIR}"

echo "[DONE] analysis csv: ${ANALYSIS_DIR}"
echo "[DONE] figures: ${FIGURES_DIR}"
