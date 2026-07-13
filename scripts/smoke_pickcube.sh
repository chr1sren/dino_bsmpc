#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="/opt/miniconda3/envs/csc420/bin/python"
fi

export DATASET_DIR="${DATASET_DIR:-$(pwd)/data}"
export WANDB_MODE=disabled

echo "=== ID-bisim unit tests ==="
$PYTHON scripts/test_id_bisim.py

echo "=== PickCube data (ManiSkill if available, else synthetic) ==="
if $PYTHON -c "import mani_skill" 2>/dev/null; then
  $PYTHON scripts/collect_pickcube_data.py \
    --out "${DATASET_DIR}/pickcube_v1" \
    --n_episodes 2 \
    --max_steps 15
else
  echo "mani_skill not installed; using synthetic data"
  $PYTHON scripts/generate_synthetic_pickcube_data.py \
    --out "${DATASET_DIR}/pickcube_v1" \
    --n_episodes 6
fi

echo "=== Dataset load smoke ==="
$PYTHON - <<'PY'
import os
from datasets.img_transforms import default_transform
from datasets.pickcube_dset import load_pickcube_slice_train_val
data_path = os.path.join(os.environ["DATASET_DIR"], "pickcube_v1")
ds, traj = load_pickcube_slice_train_val(
    default_transform(128), data_path=data_path, frameskip=1, num_hist=1, num_pred=1
)
obs, act, state, info = ds["valid"][0]
print("visual", obs["visual"].shape, "proprio", obs["proprio"].shape, "act", act.shape, "state", state.shape)
PY

if $PYTHON -c "import mani_skill" 2>/dev/null; then
  echo "=== Wrapper API smoke ==="
  $PYTHON - <<'PY'
import gym
import numpy as np
import env  # noqa
e = gym.make("pickcube")
obs, st = e.reset()
obs2, st2 = e.prepare(0, st)
obses, states = e.rollout(0, st, np.random.randn(5, 4).astype("float32") * 0.1)
print("visual", obses["visual"].shape, "states", states.shape)
print(e.eval_state(st, states[-1]))
e.close()
PY
else
  echo "Skipping ManiSkill wrapper smoke (mani_skill not installed)"
fi

echo "=== 1-epoch WM train smoke ==="
$PYTHON train.py env=pickcube img_size=128 frameskip=1 num_hist=1 num_pred=1 \
  training.epochs=1 training.batch_size=2 debug=True \
  bisim_memory_buffer_size=0 env.num_workers=0 env.dataset.data_path="${DATASET_DIR}/pickcube_v1"

echo "=== ID-ID 1-epoch smoke ==="
$PYTHON train.py env=pickcube img_size=128 frameskip=1 num_hist=1 num_pred=1 \
  training.epochs=1 training.batch_size=2 debug=True \
  model.train_bisim_id_id=true id_lambda=0.1 id_omega=0.1 \
  bisim_memory_buffer_size=0 env.num_workers=0 env.dataset.data_path="${DATASET_DIR}/pickcube_v1"

echo "Smoke tests completed."
