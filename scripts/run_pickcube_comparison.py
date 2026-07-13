#!/usr/bin/env python3
"""Run PickCube baseline vs ID-ID vs ablations across multiple seeds."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = {
    "bisim_baseline": {
        "model.train_bisim_id_id": "false",
        "id_lambda": "0.0",
        "id_omega": "0.0",
    },
    "bisim_id_id": {
        "model.train_bisim_id_id": "true",
        "id_lambda": "0.1",
        "id_omega": "0.1",
    },
    "bisim_id_target_only": {
        "model.train_bisim_id_id": "true",
        "id_lambda": "0.1",
        "id_omega": "0.0",
    },
    "bisim_id_supervision_only": {
        "model.train_bisim_id_id": "true",
        "id_lambda": "0.0",
        "id_omega": "0.1",
    },
}


def run_train(name, seed, epochs, batch_size, data_dir, out_dir, extra_overrides):
    run_dir = out_dir / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "env=pickcube",
        f"training.seed={seed}",
        f"training.epochs={epochs}",
        f"training.batch_size={batch_size}",
        "img_size=128",
        "frameskip=1",
        "num_hist=1",
        "num_pred=1",
        "bisim_memory_buffer_size=0",
        "bisim_comparison_size=20",
        "debug=True",
        "env.num_workers=0",
        f"env.dataset.data_path={data_dir}",
        f"hydra.run.dir={run_dir}",
    ]
    for key, value in extra_overrides.items():
        cmd.append(f"{key}={value}")

    env = os.environ.copy()
    env.setdefault("DATASET_DIR", str(Path(data_dir).parent))
    env["WANDB_MODE"] = "disabled"
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    log_path = run_dir / "train.log"
    log_path.write_text(result.stdout + "\n" + result.stderr)
    return {
        "name": name,
        "seed": seed,
        "returncode": result.returncode,
        "log_path": str(log_path),
    }


def parse_training_csv(run_dir):
    csv_path = Path(run_dir) / "training_loss_log.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    keys = [
        "train_loss", "val_loss", "train_bisim_loss", "val_bisim_loss",
        "train_id_loss", "val_id_loss", "train_bisim_id_l1", "val_bisim_id_l1",
        "train_z_loss", "val_z_loss",
    ]
    out = {}
    for k in keys:
        if k in last and last[k] not in (None, ""):
            try:
                out[k] = float(last[k])
            except ValueError:
                pass
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/pickcube_v1")
    parser.add_argument("--out_dir", type=str, default="outputs/pickcube_comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS.keys()))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for name in args.experiments:
        if name not in EXPERIMENTS:
            print(f"Skipping unknown experiment: {name}")
            continue
        for seed in args.seeds:
            run_meta = run_train(
                name=name,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                data_dir=str(Path(args.data_dir).resolve()),
                out_dir=out_dir,
                extra_overrides=EXPERIMENTS[name],
            )
            metrics = parse_training_csv(Path(run_meta["log_path"]).parent)
            run_meta.update(metrics)
            results.append(run_meta)

    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = out_dir / "comparison_summary.csv"
    if results:
        fieldnames = sorted({k for row in results for k in row.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
