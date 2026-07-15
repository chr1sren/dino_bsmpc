#!/usr/bin/env bash
# Full ManiSkill experiments: PushCube + PickCube, baseline vs ID-ID.
#
# Usage (on the training server):
#   export DATASET_DIR=/path/to/data
#   export CKPT_BASE=/path/to/outputs   # optional
#   bash scripts/run_full_maniskill_experiments.sh
#
# Env knobs:
#   TASKS="pushcube pickcube"   # default both
#   N_EPISODES=200
#   EPOCHS=100
#   BATCH_SIZE=32
#   SEEDS="0 1 2"
#   EXPERIMENTS="bisim_baseline bisim_id_id"   # add ablations if needed
#   SKIP_COLLECT=0|1
#   ABLACTIONS=0|1   # if 1, also run id_target_only + id_supervision_only
#   ID_LAMBDA=0.05   # ID-target weight for id_id / id_target_only
#   ID_OMEGA=0.1     # ID-supervision weight
#   SKIP_PLOT=0|1    # if 0 (default), generate eval plots for all tasks at the end
#   PYTHON=python

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/data}"
CKPT_BASE="${CKPT_BASE:-${ROOT}/outputs}"
TASKS="${TASKS:-pushcube pickcube}"
N_EPISODES="${N_EPISODES:-200}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEEDS="${SEEDS:-0 1 2}"
IMG_SIZE="${IMG_SIZE:-128}"
FRAMESKIP="${FRAMESKIP:-5}"
NUM_HIST="${NUM_HIST:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
ABLACTIONS="${ABLACTIONS:-0}"
ID_LAMBDA="${ID_LAMBDA:-0.05}"
ID_OMEGA="${ID_OMEGA:-0.1}"
SKIP_PLOT="${SKIP_PLOT:-0}"
export DATASET_DIR
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ "${ABLACTIONS}" == "1" ]]; then
  EXPERIMENTS="${EXPERIMENTS:-bisim_baseline bisim_id_id bisim_id_target_only bisim_id_supervision_only}"
else
  EXPERIMENTS="${EXPERIMENTS:-bisim_baseline bisim_id_id}"
fi

mkdir -p "${DATASET_DIR}" "${CKPT_BASE}"

echo "=== ManiSkill full experiments ==="
echo "DATASET_DIR=${DATASET_DIR}"
echo "CKPT_BASE=${CKPT_BASE}"
echo "TASKS=${TASKS}"
echo "N_EPISODES=${N_EPISODES} EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE}"
echo "SEEDS=${SEEDS}"
echo "EXPERIMENTS=${EXPERIMENTS}"
echo "img_size=${IMG_SIZE} frameskip=${FRAMESKIP} num_hist=${NUM_HIST}"

for TASK in ${TASKS}; do
  DATA_PATH="${DATASET_DIR}/${TASK}_v1"
  OUT_DIR="${CKPT_BASE}/${TASK}_comparison"

  if [[ "${SKIP_COLLECT}" != "1" ]]; then
    if [[ -f "${DATA_PATH}/train/states.pth" && -f "${DATA_PATH}/val/states.pth" ]]; then
      echo "[${TASK}] data already exists at ${DATA_PATH} (rm to re-collect)"
    else
      echo "[${TASK}] collecting ${N_EPISODES} episodes -> ${DATA_PATH}"
      # ManiSkill/SAPIEN often segfaults on process teardown after a successful save.
      set +e
      ${PYTHON} scripts/collect_maniskill_data.py \
        --task "${TASK}" \
        --out "${DATA_PATH}" \
        --n_episodes "${N_EPISODES}" \
        --max_steps 50 \
        --image_size "${IMG_SIZE}" \
        --random_fraction 0.2 \
        --motion_plan_fraction 0.3
      collect_rc=$?
      set -e
      if [[ ! -f "${DATA_PATH}/train/states.pth" || ! -f "${DATA_PATH}/val/states.pth" ]]; then
        echo "[${TASK}] ERROR: collect failed (exit=${collect_rc}) and data missing" >&2
        exit 1
      fi
      if [[ ${collect_rc} -ne 0 ]]; then
        echo "[${TASK}] warning: collect exited ${collect_rc} but data looks complete; continuing"
      fi
    fi
  else
    echo "[${TASK}] SKIP_COLLECT=1, using ${DATA_PATH}"
  fi

  if [[ ! -f "${DATA_PATH}/train/states.pth" ]]; then
    echo "[${TASK}] ERROR: missing ${DATA_PATH}/train/states.pth" >&2
    exit 1
  fi

  echo "[${TASK}] training comparison -> ${OUT_DIR}"
  # shellcheck disable=SC2086
  ${PYTHON} scripts/run_maniskill_comparison.py \
    --task "${TASK}" \
    --data_dir "${DATA_PATH}" \
    --out_dir "${OUT_DIR}" \
    --seeds ${SEEDS} \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --img_size "${IMG_SIZE}" \
    --frameskip "${FRAMESKIP}" \
    --num_hist "${NUM_HIST}" \
    --num_workers "${NUM_WORKERS}" \
    --id_lambda "${ID_LAMBDA}" \
    --id_omega "${ID_OMEGA}" \
    --experiments ${EXPERIMENTS}

  echo "[${TASK}] done. Summary: ${OUT_DIR}/comparison_summary.csv"
done

if [[ "${SKIP_PLOT}" != "1" ]]; then
  echo "=== generating eval plots for all tasks under ${CKPT_BASE} ==="
  ${PYTHON} scripts/plot_maniskill_training.py \
    --outputs_dir "${CKPT_BASE}" \
    --out_dir "${CKPT_BASE}/plots" || echo "warning: plotting failed (matplotlib installed?)"
fi

echo "=== all tasks finished ==="
