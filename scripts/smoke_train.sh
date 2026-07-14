#!/usr/bin/env bash
# 基础训练 smoke：跑 baseline 或 ID-ID，观察 loss 是否稳定下降、无 NaN/爆炸。
# 预计 1–2 小时（取决于 GPU 与数据量）；数据不足时先用合成数据。
#
# 用法:
#   bash scripts/smoke_train.sh                  # baseline，数据默认写到 ./data
#   VARIANT=id_id bash scripts/smoke_train.sh    # bisim-id-id
#   export DATASET_DIR=$HOME/datasets            # 可选：真实数据根目录
#
# 可调:
#   EPOCHS=40 BATCH_SIZE=8 IMG_SIZE=128 FRAMESKIP=1 NUM_HIST=3

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
DEFAULT_DATASET_DIR="$(pwd)/data"

# 忽略文档占位路径 /path/to/data，否则会 PermissionError: '/path'
if [[ -z "${DATASET_DIR:-}" || "${DATASET_DIR}" == "/path/to/data"* || "${DATASET_DIR}" == *"path/to"* ]]; then
  export DATASET_DIR="${DEFAULT_DATASET_DIR}"
  echo "[smoke] DATASET_DIR unset/placeholder; using ${DATASET_DIR}"
fi
export WANDB_MODE="${WANDB_MODE:-disabled}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMG_SIZE="${IMG_SIZE:-128}"
# Default frameskip=1 for synthetic smoke; frameskip=5 needs traj_len > 26
FRAMESKIP="${FRAMESKIP:-1}"
NUM_HIST="${NUM_HIST:-3}"
VARIANT="${VARIANT:-baseline}"   # baseline | id_id
DATA_PATH="${DATASET_DIR}/pickcube_v1"
MIN_TRAJ_LEN=$(( FRAMESKIP * (2 + NUM_HIST) + 2 ))

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python"
fi

need_data=0
if [[ ! -f "${DATA_PATH}/train/states.pth" ]]; then
  need_data=1
else
  # Regenerate if existing synthetic trajs are too short for openloop
  cur_len="$($PYTHON - <<PY
import torch
s=torch.load("${DATA_PATH}/train/states.pth")
print(int(s.shape[1]))
PY
)"
  if [[ "${cur_len}" -lt "${MIN_TRAJ_LEN}" ]]; then
    echo "[smoke] existing traj_len=${cur_len} < required ${MIN_TRAJ_LEN}; regenerating"
    need_data=1
  fi
fi

if [[ "${need_data}" -eq 1 ]]; then
  echo "[smoke] generating synthetic data at ${DATA_PATH} (max_steps=$(( MIN_TRAJ_LEN + 20 )))"
  mkdir -p "${DATA_PATH}"
  $PYTHON scripts/generate_synthetic_pickcube_data.py \
    --out "${DATA_PATH}" \
    --n_episodes 20 \
    --max_steps $(( MIN_TRAJ_LEN + 20 ))
fi

EXTRA_ARGS=()
case "$VARIANT" in
  id_id)
    EXTRA_ARGS+=(model.train_bisim_id_id=true id_lambda=0.1 id_omega=0.1)
    RUN_TAG="pickcube_id_id_smoke"
    ;;
  *)
    EXTRA_ARGS+=(model.train_bisim_id_id=false id_lambda=0.0 id_omega=0.0)
    RUN_TAG="pickcube_baseline_smoke"
    ;;
esac

echo "[smoke] variant=${VARIANT} epochs=${EPOCHS} batch=${BATCH_SIZE} frameskip=${FRAMESKIP} data=${DATA_PATH}"

$PYTHON train.py \
  env=pickcube \
  img_size="${IMG_SIZE}" \
  frameskip="${FRAMESKIP}" \
  num_hist="${NUM_HIST}" \
  num_pred=1 \
  training.epochs="${EPOCHS}" \
  training.batch_size="${BATCH_SIZE}" \
  training.save_every_x_epoch=5 \
  bisim_memory_buffer_size=0 \
  env.num_workers=0 \
  env.dataset.data_path="${DATA_PATH}" \
  hydra.run.dir="outputs/smoke/${RUN_TAG}" \
  "${EXTRA_ARGS[@]}"

echo "[smoke] done. Check outputs/smoke/${RUN_TAG}/training_loss_log.csv"
echo "  关注: train_loss / val_loss 应整体下降，bisim_loss 与 id_loss 不应出现 NaN 或持续爆炸。"
