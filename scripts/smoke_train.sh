#!/usr/bin/env bash
# 基础训练 smoke：跑 baseline 或 ID-ID，观察 loss 是否稳定下降、无 NaN/爆炸。
# 预计 1–2 小时（取决于 GPU/MPS 与数据量）；数据不足时先用合成数据。
#
# 用法:
#   export DATASET_DIR=/path/to/data
#   bash scripts/smoke_train.sh                  # baseline
#   VARIANT=id_id bash scripts/smoke_train.sh    # bisim-id-id
#
# 可调环境变量:
#   EPOCHS=40 BATCH_SIZE=8 IMG_SIZE=128 FRAMESKIP=5

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
export DATASET_DIR="${DATASET_DIR:-$(pwd)/data}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMG_SIZE="${IMG_SIZE:-128}"
FRAMESKIP="${FRAMESKIP:-5}"
VARIANT="${VARIANT:-baseline}"   # baseline | id_id
DATA_PATH="${DATASET_DIR}/pickcube_v1"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="/opt/miniconda3/envs/csc420/bin/python"
fi

# 准备数据
if [[ ! -f "${DATA_PATH}/train/states.pth" ]]; then
  echo "[smoke] no data at ${DATA_PATH}; generating synthetic fallback"
  $PYTHON scripts/generate_synthetic_pickcube_data.py --out "${DATA_PATH}" --n_episodes 20
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

echo "[smoke] variant=${VARIANT} epochs=${EPOCHS} batch=${BATCH_SIZE} data=${DATA_PATH}"

$PYTHON train.py \
  env=pickcube \
  img_size="${IMG_SIZE}" \
  frameskip="${FRAMESKIP}" \
  num_hist=3 \
  num_pred=1 \
  training.epochs="${EPOCHS}" \
  training.batch_size="${BATCH_SIZE}" \
  training.save_every_x_epoch=5 \
  bisim_memory_buffer_size=0 \
  env.num_workers=4 \
  env.dataset.data_path="${DATA_PATH}" \
  hydra.run.dir="outputs/smoke/${RUN_TAG}" \
  "${EXTRA_ARGS[@]}"

echo "[smoke] done. Check outputs/smoke/${RUN_TAG}/training_loss_log.csv"
echo "  关注: train_loss / val_loss 应整体下降，bisim_loss 与 id_loss 不应出现 NaN 或持续爆炸。"
