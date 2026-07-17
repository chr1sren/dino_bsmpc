#!/usr/bin/env bash
# Inventory comparison run folders: which have checkpoints / logs / CSV.
#
# Usage (from dino_bsmpc/ on the server):
#   bash scripts/list_maniskill_runs.sh
#   CKPT_BASE=/path/to/dino_bsmpc bash scripts/list_maniskill_runs.sh
#
# Writes: outputs/run_inventory.csv  (and prints a table)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_BASE="${CKPT_BASE:-${ROOT}}"
OUT_CSV="${OUT_CSV:-${CKPT_BASE}/outputs/run_inventory.csv}"

mkdir -p "$(dirname "${OUT_CSV}")"
echo "task,exp,seed,has_hydra,has_train_csv,n_epochs,ckpt_files,preferred_ckpt,ckpt_size_mb,plan_ready" > "${OUT_CSV}"

shopt -s nullglob
for run_dir in "${CKPT_BASE}"/outputs/*_comparison/bisim_*/seed_*; do
  [[ -d "${run_dir}" ]] || continue
  seed_dir="$(basename "${run_dir}")"
  exp="$(basename "$(dirname "${run_dir}")")"
  task="$(basename "$(dirname "$(dirname "${run_dir}")")" | sed 's/_comparison$//')"
  seed="${seed_dir#seed_}"

  has_hydra=0
  [[ -f "${run_dir}/hydra.yaml" ]] && has_hydra=1

  has_csv=0
  n_epochs=0
  if [[ -f "${run_dir}/training_loss_log.csv" ]]; then
    has_csv=1
    n_epochs=$(($(wc -l < "${run_dir}/training_loss_log.csv") - 1))
  fi

  ckpt_dir="${run_dir}/checkpoints"
  ckpt_files=""
  preferred=""
  size_mb=0
  plan_ready=0
  if [[ -d "${ckpt_dir}" ]]; then
    names=()
    for f in "${ckpt_dir}"/model_*.pth; do
      [[ -f "$f" ]] || continue
      names+=("$(basename "$f")")
    done
    if ((${#names[@]} > 0)); then
      ckpt_files="$(IFS='|'; echo "${names[*]}")"
      for tag in model_final.pth model_latest.pth; do
        if [[ -f "${ckpt_dir}/${tag}" ]]; then
          preferred="${tag}"
          break
        fi
      done
      if [[ -z "${preferred}" ]]; then
        preferred="$(ls -1 "${ckpt_dir}"/model_*.pth | sort -V | tail -1 | xargs -n1 basename)"
      fi
      size_mb="$(du -m "${ckpt_dir}/${preferred}" 2>/dev/null | awk '{print $1}')"
      plan_ready=1
    fi
  fi
  [[ -z "${ckpt_files}" ]] && ckpt_files="-"
  [[ -z "${preferred}" ]] && preferred="-"

  echo "${task},${exp},${seed},${has_hydra},${has_csv},${n_epochs},${ckpt_files},${preferred},${size_mb},${plan_ready}" >> "${OUT_CSV}"
done

echo "=== run inventory ==="
if command -v column >/dev/null 2>&1; then
  column -t -s, "${OUT_CSV}"
else
  cat "${OUT_CSV}"
fi
echo ""
echo "plan_ready=1 means a .pth exists and can be used for planning."
echo "Wrote ${OUT_CSV}"

# quick counts
n_total=$(($(wc -l < "${OUT_CSV}") - 1))
n_ready="$(awk -F, 'NR>1 && $10==1 {c++} END{print c+0}' "${OUT_CSV}")"
n_missing="$(awk -F, 'NR>1 && $10==0 {c++} END{print c+0}' "${OUT_CSV}")"
echo "total_runs=${n_total}  plan_ready=${n_ready}  missing_ckpt=${n_missing}"
