#!/usr/bin/env bash
set -euo pipefail

# Full pipeline: pretrain+finetune then summary/plots
bash "$(dirname "$0")/run_ablation_shipsear_train.sh" "$@"
bash "$(dirname "$0")/run_ablation_shipsear_summary.sh"
