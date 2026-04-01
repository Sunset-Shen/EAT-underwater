#!/usr/bin/env bash
set -euo pipefail

EAT_ROOT="${EAT_ROOT:-/hy-tmp/eat_work/EAT}"
RESULT_ROOT="${RESULT_ROOT:-/hy-tmp/exp/eat/results/compare_v1}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RESULT_ROOT}/lightweighting_analysis}"
FIGURES_DIR="${FIGURES_DIR:-${RESULT_ROOT}/figures}"
COMPARE_MAIN_CSV="${COMPARE_MAIN_CSV:-${RESULT_ROOT}/summary/compare_main.csv}"

python "${EAT_ROOT}/utils/build_lightweighting_analysis.py" \
  --compare-main-csv "${COMPARE_MAIN_CSV}" \
  --results-root "${RESULT_ROOT}" \
  --output-dir "${ANALYSIS_DIR}"

python "${EAT_ROOT}/utils/plot_lightweighting_analysis.py" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --figures-dir "${FIGURES_DIR}"

echo "[DONE] lightweighting tables: ${ANALYSIS_DIR}"
echo "[DONE] lightweighting figures: ${FIGURES_DIR}"
