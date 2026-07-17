#!/usr/bin/env bash
# Run MPC planning on all (or selected) ManiSkill comparison checkpoints
# and summarize success_rate.
#
# Usage (on the training server, from dino_bsmpc/):
#   export DATASET_DIR=/path/to/data          # must match training
#   bash scripts/run_maniskill_planning.sh
#
# Knobs:
#   TASKS="pickcube pushcube"
#   EXPS="bisim_baseline bisim_id_id"
#   SEEDS="0 1 2"
#   N_EVALS=10
#   GOAL_H=5
#   GOAL_SOURCE=dset          # or random_state
#   MODEL_EPOCH=latest
#   CKPT_BASE=.               # repo root containing outputs/
#   OUT_ROOT=plan_outputs/maniskill_eval
#   PYTHON=python

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/data}"
CKPT_BASE="${CKPT_BASE:-${ROOT}}"
TASKS="${TASKS:-pickcube pushcube}"
EXPS="${EXPS:-bisim_baseline bisim_id_id}"
SEEDS="${SEEDS:-0 1 2}"
N_EVALS="${N_EVALS:-10}"
GOAL_H="${GOAL_H:-5}"
GOAL_SOURCE="${GOAL_SOURCE:-dset}"
MODEL_EPOCH="${MODEL_EPOCH:-latest}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/plan_outputs/maniskill_eval}"
export DATASET_DIR
export WANDB_MODE="${WANDB_MODE:-disabled}"

_extract() {
  # $1=file_or_dir  $2=regex that ends with a number
  local target="$1" pat="$2"
  if [[ -d "$target" ]]; then
    if command -v rg >/dev/null 2>&1; then
      rg -oN "$pat" "$target" 2>/dev/null | tail -1 | grep -oE '[0-9.eE+-]+$' || true
    else
      grep -RhoE "$pat" "$target" 2>/dev/null | tail -1 | grep -oE '[0-9.eE+-]+$' || true
    fi
  elif [[ -f "$target" ]]; then
    if command -v rg >/dev/null 2>&1; then
      rg -oN "$pat" "$target" 2>/dev/null | tail -1 | grep -oE '[0-9.eE+-]+$' || true
    else
      grep -oE "$pat" "$target" 2>/dev/null | tail -1 | grep -oE '[0-9.eE+-]+$' || true
    fi
  fi
}

mkdir -p "${OUT_ROOT}"
SUMMARY="${OUT_ROOT}/planning_summary.csv"
echo "task,exp,seed,model_name,returncode,success_rate,mean_place_err,plan_dir" > "${SUMMARY}"

echo "=== ManiSkill planning eval ==="
echo "CKPT_BASE=${CKPT_BASE}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "TASKS=${TASKS} EXPS=${EXPS} SEEDS=${SEEDS}"
echo "n_evals=${N_EVALS} goal_H=${GOAL_H} goal_source=${GOAL_SOURCE} epoch=${MODEL_EPOCH}"

for TASK in ${TASKS}; do
  for EXP in ${EXPS}; do
    for SEED in ${SEEDS}; do
      MODEL_NAME="${TASK}_comparison/${EXP}/seed_${SEED}"
      CKPT_DIR="${CKPT_BASE}/outputs/${MODEL_NAME}/checkpoints"
      RUN_DIR="${CKPT_BASE}/outputs/${MODEL_NAME}"

      if [[ ! -f "${RUN_DIR}/hydra.yaml" ]]; then
        echo "[skip] missing hydra.yaml: ${RUN_DIR}"
        continue
      fi

      MODEL_EPOCH_USE="${MODEL_EPOCH}"
      if [[ ! -f "${CKPT_DIR}/model_${MODEL_EPOCH}.pth" ]]; then
        if [[ -f "${CKPT_DIR}/model_latest.pth" ]]; then
          MODEL_EPOCH_USE=latest
        else
          ALT="$(ls -1 "${CKPT_DIR}"/model_*.pth 2>/dev/null | sort -V | tail -1 || true)"
          if [[ -n "${ALT}" ]]; then
            MODEL_EPOCH_USE="$(basename "${ALT}" | sed -E 's/model_([0-9]+)\.pth/\1/')"
          else
            echo "[skip] no checkpoint in ${CKPT_DIR}"
            continue
          fi
        fi
      fi

      echo ""
      echo ">>> planning ${MODEL_NAME} (epoch=${MODEL_EPOCH_USE})"
      PLAN_DIR="${OUT_ROOT}/${TASK}_${EXP}_seed${SEED}"
      PLAN_LOG="${OUT_ROOT}/${TASK}_${EXP}_seed${SEED}.log"
      mkdir -p "${PLAN_DIR}"

      set +e
      ${PYTHON} plan.py --config-name plan_maniskill.yaml \
        "ckpt_base_path=${CKPT_BASE}" \
        "model_name=${MODEL_NAME}" \
        "model_epoch=${MODEL_EPOCH_USE}" \
        "n_evals=${N_EVALS}" \
        "goal_H=${GOAL_H}" \
        "goal_source=${GOAL_SOURCE}" \
        "hydra.run.dir=${PLAN_DIR}" \
        > "${PLAN_LOG}" 2>&1
      rc=$?
      set -e

      SR="$(_extract "${PLAN_DIR}" '"final_eval/success_rate": ?[0-9.eE+-]+')"
      if [[ -z "${SR}" ]]; then
        SR="$(_extract "${PLAN_LOG}" 'Success rate:[[:space:]]*[0-9.eE+-]+')"
      fi
      if [[ -z "${SR}" ]]; then
        SR="$(_extract "${PLAN_DIR}" '"success_rate": ?[0-9.eE+-]+')"
      fi
      PE="$(_extract "${PLAN_DIR}" '"final_eval/mean_place_err": ?[0-9.eE+-]+')"
      if [[ -z "${PE}" ]]; then
        PE="$(_extract "${PLAN_LOG}" '"final_eval/mean_place_err": ?[0-9.eE+-]+')"
      fi
      [[ -z "${SR}" ]] && SR=nan
      [[ -z "${PE}" ]] && PE=nan

      echo "${TASK},${EXP},${SEED},${MODEL_NAME},${rc},${SR},${PE},${PLAN_DIR}" >> "${SUMMARY}"
      echo "    rc=${rc} success_rate=${SR} mean_place_err=${PE}"
      echo "    log: ${PLAN_LOG}"
    done
  done
done

echo ""
echo "=== summary written to ${SUMMARY} ==="
column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
