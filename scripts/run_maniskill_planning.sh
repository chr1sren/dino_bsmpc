#!/usr/bin/env bash
# Run MPC planning on ManiSkill comparison checkpoints → success_rate summary.
#
# Usage (on the training server, from dino_bsmpc/):
#   export DATASET_DIR=/path/to/data
#   bash scripts/run_maniskill_planning.sh
#
# IMPORTANT: default N_EVALS=1. Opening many ManiSkill/SAPIEN envs at once
# frequently segfaults (rc=139). Raise only after a single-env run works.
#
# Knobs:
#   TASKS="pickcube pushcube"
#   EXPS="bisim_baseline bisim_id_id"
#   SEEDS="0 1 2"
#   N_EVALS=1
#   GOAL_H=5
#   GOAL_SOURCE=dset
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
N_EVALS="${N_EVALS:-1}"
GOAL_H="${GOAL_H:-5}"
GOAL_SOURCE="${GOAL_SOURCE:-dset}"
MODEL_EPOCH="${MODEL_EPOCH:-latest}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/plan_outputs/maniskill_eval}"
export DATASET_DIR
export WANDB_MODE="${WANDB_MODE:-disabled}"

_extract() {
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

_find_ckpt_epoch() {
  # prints epoch tag to use with model_${tag}.pth, or empty if none
  local ckpt_dir="$1" want="$2"
  if [[ -f "${ckpt_dir}/model_${want}.pth" ]]; then
    echo "${want}"
    return
  fi
  if [[ -f "${ckpt_dir}/model_latest.pth" ]]; then
    echo "latest"
    return
  fi
  local alt
  alt="$(ls -1 "${ckpt_dir}"/model_*.pth 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "${alt}" ]]; then
    basename "${alt}" | sed -E 's/model_([0-9]+)\.pth/\1/'
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
echo "(tip: if you see rc=139 segfault, keep N_EVALS=1; inspect *.log for Traceback on rc=1)"

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
      if [[ ! -d "${CKPT_DIR}" ]]; then
        echo "[skip] no checkpoints/ dir: ${CKPT_DIR}"
        echo "       (training may not have saved weights here — check that run)"
        continue
      fi

      MODEL_EPOCH_USE="$(_find_ckpt_epoch "${CKPT_DIR}" "${MODEL_EPOCH}")"
      if [[ -z "${MODEL_EPOCH_USE}" ]]; then
        echo "[skip] empty checkpoints/: ${CKPT_DIR}"
        ls -la "${CKPT_DIR}" || true
        continue
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

      if [[ ${rc} -ne 0 ]]; then
        echo "    ---- last 40 lines of log ----"
        tail -n 40 "${PLAN_LOG}" || true
        echo "    -------------------------------"
      fi
    done
  done
done

echo ""
echo "=== summary written to ${SUMMARY} ==="
column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
