#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
THROUGHPUT_BS="${2:-8}"
BASELINE_MAX_UPDATE="${3:-4000}"

EAT_ROOT="${EAT_ROOT:-/hy-tmp/eat_work/EAT}"
FAIRSEQ_ROOT="${FAIRSEQ_ROOT:-/hy-tmp/eat_work/fairseq}"
RESULT_ROOT="${RESULT_ROOT:-/hy-tmp/exp/eat/results/compare_v1}"
PRETRAIN_BASE="${PRETRAIN_BASE:-/hy-tmp/model_zoo/eat_checkpoints/EAT-base_epoch30_pt.pt}"

SHIP_MANIFEST="${SHIP_MANIFEST:-/hy-tmp/exp/eat/manifests/shipsear}"
DEEP_MANIFEST="${DEEP_MANIFEST:-/hy-tmp/exp/eat/manifests/deepship}"

SHIP_EAT_RUN="${SHIP_EAT_RUN:-/hy-tmp/exp/eat/runs/shipsear_baseline}"
DEEP_EAT_RUN="${DEEP_EAT_RUN:-/hy-tmp/exp/eat/runs/deepship_baseline}"
SHIP_UT_RUN="${SHIP_UT_RUN:-/hy-tmp/exp/eat/runs/shipsear_uteat}"
DEEP_UT_RUN="${DEEP_UT_RUN:-/hy-tmp/exp/eat/runs/deepship_uteat}"

mkdir -p "${RESULT_ROOT}"/{shipsear,deepship}/eat_base/{train,eval,profile} "${RESULT_ROOT}"/{shipsear,deepship}/ut_eat/{eval,profile} "${RESULT_ROOT}/summary"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

train_if_missing() {
  local ckpt="$1"
  local train_cmd="$2"
  if [[ -f "${ckpt}" ]]; then
    echo "[REUSE] checkpoint exists: ${ckpt}"
  else
    echo "[TRAIN] missing checkpoint -> ${ckpt}"
    eval "${train_cmd}"
  fi
}

choose_ckpt() {
  local run_dir="$1"
  if [[ -f "${run_dir}/checkpoint_best.pt" ]]; then
    echo "${run_dir}/checkpoint_best.pt"
  elif [[ -f "${run_dir}/checkpoint_last.pt" ]]; then
    echo "${run_dir}/checkpoint_last.pt"
  else
    echo ""
  fi
}

run_eval_and_profile() {
  local dataset="$1"
  local model_tag="$2"
  local manifest="$3"
  local ckpt="$4"
  local init_ckpt="$5"
  local out_base="${RESULT_ROOT}/${dataset}/${model_tag}"

  python "${EAT_ROOT}/utils/eval_manifest_classification.py" \
    --checkpoint "${ckpt}" \
    --manifest-dir "${manifest}" \
    --dataset-name "${dataset}" \
    --model-name "${model_tag}" \
    --init-checkpoint "${init_ckpt}" \
    --output "${out_base}/eval/eval_metrics.json"

  python "${EAT_ROOT}/utils/profile_manifest_classification.py" \
    --checkpoint "${ckpt}" \
    --manifest-dir "${manifest}" \
    --dataset-name "${dataset}" \
    --model-name "${model_tag}" \
    --throughput-batch-size "${THROUGHPUT_BS}" \
    --fallback-throughput-batch-size 4 \
    --output "${out_base}/profile/profile_metrics.json"
}

# 1) ShipsEar / EAT-base
train_if_missing "${SHIP_EAT_RUN}/checkpoint_best.pt" "bash ${EAT_ROOT}/scripts/finetuning_SHIPSEAR.sh ${GPU_ID} 8 6 ${BASELINE_MAX_UPDATE}"
SHIP_EAT_CKPT="$(choose_ckpt "${SHIP_EAT_RUN}")"
if [[ -z "${SHIP_EAT_CKPT}" ]]; then
  echo "[ERROR] ShipsEar EAT-base checkpoint not found under ${SHIP_EAT_RUN}" >&2
  exit 1
fi
run_eval_and_profile "shipsear" "eat_base" "${SHIP_MANIFEST}" "${SHIP_EAT_CKPT}" "${PRETRAIN_BASE}"

# 2) ShipsEar / UT-EAT
SHIP_UT_CKPT="$(choose_ckpt "${SHIP_UT_RUN}")"
if [[ -z "${SHIP_UT_CKPT}" ]]; then
  echo "[WARN] ShipsEar UT-EAT finetuned checkpoint missing. Minimal rerun:" | tee "${RESULT_ROOT}/shipsear/ut_eat/retrain_required.txt"
  echo "bash ${EAT_ROOT}/scripts/finetuning_SHIPSEAR_UTEAT.sh ${GPU_ID} 8 6" | tee -a "${RESULT_ROOT}/shipsear/ut_eat/retrain_required.txt"
else
  run_eval_and_profile "shipsear" "ut_eat" "${SHIP_MANIFEST}" "${SHIP_UT_CKPT}" "${SHIP_UT_RUN}/(from_uteat_pretrain)"
fi

# 3) DeepShip / EAT-base
train_if_missing "${DEEP_EAT_RUN}/checkpoint_best.pt" "bash ${EAT_ROOT}/scripts/finetuning_DEEPSHIP.sh ${GPU_ID} 8 6 ${BASELINE_MAX_UPDATE}"
DEEP_EAT_CKPT="$(choose_ckpt "${DEEP_EAT_RUN}")"
if [[ -z "${DEEP_EAT_CKPT}" ]]; then
  echo "[ERROR] DeepShip EAT-base checkpoint not found under ${DEEP_EAT_RUN}" >&2
  exit 1
fi
run_eval_and_profile "deepship" "eat_base" "${DEEP_MANIFEST}" "${DEEP_EAT_CKPT}" "${PRETRAIN_BASE}"

# 4) DeepShip / UT-EAT
DEEP_UT_CKPT="$(choose_ckpt "${DEEP_UT_RUN}")"
if [[ -z "${DEEP_UT_CKPT}" ]]; then
  echo "[WARN] DeepShip UT-EAT finetuned checkpoint missing. Minimal rerun:" | tee "${RESULT_ROOT}/deepship/ut_eat/retrain_required.txt"
  echo "bash ${EAT_ROOT}/scripts/finetuning_DEEPSHIP_UTEAT.sh ${GPU_ID} 8 6" | tee -a "${RESULT_ROOT}/deepship/ut_eat/retrain_required.txt"
else
  run_eval_and_profile "deepship" "ut_eat" "${DEEP_MANIFEST}" "${DEEP_UT_CKPT}" "${DEEP_UT_RUN}/(from_uteat_pretrain)"
fi

python "${EAT_ROOT}/utils/summarize_compare_v1.py" \
  --results-root "${RESULT_ROOT}" \
  --output-dir "${RESULT_ROOT}/summary"

echo "[DONE] compare_v1 outputs at: ${RESULT_ROOT}"
