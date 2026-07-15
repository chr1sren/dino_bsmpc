#!/usr/bin/env bash
# Single-task helper: train one VARIANT on pickcube or pushcube.
#
# Usage:
#   export DATASET_DIR=/path/to/data
#   TASK=pushcube VARIANT=id_id bash scripts/run_maniskill_train.sh
#   TASK=pickcube VARIANT=baseline bash scripts/run_maniskill_train.sh
#
# VARIANT: baseline | id_id | id_target_only | id_supervision_only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
TASK="${TASK:?set TASK=pickcube|pushcube}"
VARIANT="${VARIANT:-id_id}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMG_SIZE="${IMG_SIZE:-128}"
FRAMESKIP="${FRAMESKIP:-5}"
NUM_HIST="${NUM_HIST:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/data}"
DATA_PATH="${DATA_PATH:-${DATASET_DIR}/${TASK}_v1}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/${TASK}_${VARIANT}_seed${SEED}}"
export DATASET_DIR
export WANDB_MODE="${WANDB_MODE:-disabled}"

case "${VARIANT}" in
  baseline)
    ID_FLAGS=(model.train_bisim_id_id=false id_lambda=0.0 id_omega=0.0)
    ;;
  id_id)
    ID_FLAGS=(model.train_bisim_id_id=true id_lambda="${ID_LAMBDA:-0.05}" id_omega="${ID_OMEGA:-0.1}")
    ;;
  id_target_only)
    ID_FLAGS=(model.train_bisim_id_id=true id_lambda="${ID_LAMBDA:-0.05}" id_omega=0.0)
    ;;
  id_supervision_only)
    ID_FLAGS=(model.train_bisim_id_id=true id_lambda=0.0 id_omega=0.1)
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 1
    ;;
esac

mkdir -p "${RUN_DIR}"
rm -f "${RUN_DIR}/training_loss_log.csv"

echo "TASK=${TASK} VARIANT=${VARIANT} SEED=${SEED}"
echo "DATA_PATH=${DATA_PATH}"
echo "RUN_DIR=${RUN_DIR}"

${PYTHON} train.py \
  "env=${TASK}" \
  "training.seed=${SEED}" \
  "training.epochs=${EPOCHS}" \
  "training.batch_size=${BATCH_SIZE}" \
  "img_size=${IMG_SIZE}" \
  "frameskip=${FRAMESKIP}" \
  "num_hist=${NUM_HIST}" \
  num_pred=1 \
  bisim_memory_buffer_size=1000 \
  bisim_comparison_size=200 \
  debug=False \
  "env.num_workers=${NUM_WORKERS}" \
  "env.dataset.data_path=${DATA_PATH}" \
  "hydra.run.dir=${RUN_DIR}" \
  "ckpt_base_path=${RUN_DIR}" \
  "${ID_FLAGS[@]}"
