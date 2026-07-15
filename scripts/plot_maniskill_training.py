#!/usr/bin/env python3
"""Plot & analyze ManiSkill comparison logs in outputs/*_comparison.

Produces, per task:
  1. <task>_curves.png        : per-metric mean±std curves (log-y), baseline vs id_id.
  2. <task>_fair_compare.png  : the APPLES-TO-APPLES panel (world-model quality +
                                bisim internals) zoomed to the converged region.
  3. <task>_final_bars.png    : grouped bar chart of final-epoch metrics -> the verdict.
Optionally per-seed curve dumps with --per_seed.

Why this layout: total_loss is NOT comparable across baseline/id_id because id_id
adds ID + larger bisim terms. The fair signal is z_proprio_loss (next-state pred)
and, ultimately, downstream MPC success. These plots make that explicit.

Usage:
  python scripts/plot_maniskill_training.py
  python scripts/plot_maniskill_training.py --outputs_dir outputs --out_dir outputs/plots --per_seed
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# (train_key, val_key, title, comparable_across_configs?)
METRICS = [
    ("train_loss", "val_loss", "total loss (NOT comparable)", False),
    ("train_z_proprio_loss", "val_z_proprio_loss", "z_proprio loss (world-model, FAIR)", True),
    ("train_bisim_loss", "val_bisim_loss", "bisim loss", False),
    ("train_bisim_z_dist", "val_bisim_z_dist", "bisim z_dist", False),
    ("train_id_loss", "val_id_loss", "ID supervision (id_id only)", False),
    ("train_bisim_id_l1", "val_bisim_id_l1", "ID target L1 (id_id only)", False),
]

# metrics used for the "fair" figure and the final-bar verdict
FAIR_KEYS = [
    ("val_z_proprio_loss", "world-model val (FAIR)"),
    ("train_z_proprio_loss", "world-model train (FAIR)"),
    ("val_bisim_z_dist", "bisim z_dist (val)"),
    ("val_bisim_var_loss", "bisim variance (val)"),
]

EXP_STYLE = {
    "bisim_baseline": {"color": "#1f77b4", "label": "baseline"},
    "bisim_id_id": {"color": "#d62728", "label": "id_id"},
    "bisim_id_target_only": {"color": "#2ca02c", "label": "id_target_only"},
    "bisim_id_supervision_only": {"color": "#9467bd", "label": "id_sup_only"},
}


def style_for(exp):
    return EXP_STYLE.get(exp, {"color": "gray", "label": exp})


def load_csv(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    data = defaultdict(list)
    for r in rows:
        data["epoch"].append(int(float(r["epoch"])))
        for k, v in r.items():
            if k == "epoch":
                continue
            data[k].append(np.nan if v in (None, "") else float(v))
    return {k: np.asarray(v, dtype=float) for k, v in data.items()}


def discover_runs(outputs_dir: Path):
    runs = []
    for csv_path in sorted(outputs_dir.glob("*_comparison/bisim_*/seed_*/training_loss_log.csv")):
        seed = csv_path.parent.name.replace("seed_", "")
        exp = csv_path.parent.parent.name
        task = csv_path.parent.parent.parent.name.replace("_comparison", "")
        runs.append({"task": task, "exp": exp, "seed": seed, "path": csv_path, "data": load_csv(csv_path)})
    return runs


def stack_over_seeds(task_runs, exp, key):
    """Return (T, mean, std) reindexed by logged-epoch order (handles resumed runs)."""
    curves = []
    for r in task_runs:
        if r["exp"] != exp:
            continue
        d = r["data"]
        if key not in d:
            continue
        curves.append(d[key])
    if not curves:
        return None
    T = max(len(c) for c in curves)
    arr = np.full((len(curves), T), np.nan)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
    return np.arange(T), np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _autoscale_y(ax, values):
    """Log-scale if all-positive and spans >1 decade, else linear; robust to zeros/NaN."""
    vals = np.concatenate([v[np.isfinite(v)] for v in values if v is not None and len(v)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    pos = vals[vals > 0]
    if pos.size and pos.min() > 0 and (pos.max() / pos.min() > 20):
        ax.set_yscale("log")


def plot_curves(task, task_runs, out_path: Path):
    exps = [e for e in EXP_STYLE if any(r["exp"] == e for r in task_runs)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, (tk, vk, title, fair) in zip(axes, METRICS):
        collected = []
        for exp in exps:
            sty = style_for(exp)
            tr = stack_over_seeds(task_runs, exp, tk)
            va = stack_over_seeds(task_runs, exp, vk)
            if tr is not None:
                xs, m, s = tr
                ax.plot(xs, m, color=sty["color"], lw=2, label=f"{sty['label']} train")
                ax.fill_between(xs, m - s, m + s, color=sty["color"], alpha=0.15)
                collected.append(m)
            if va is not None:
                xs, m, s = va
                ax.plot(xs, m, color=sty["color"], lw=1.8, ls="--", label=f"{sty['label']} val")
                ax.fill_between(xs, m - s, m + s, color=sty["color"], alpha=0.08)
                collected.append(m)
        _autoscale_y(ax, collected)
        ax.set_title(title, fontsize=11, color=("#0a7d0a" if fair else "black"))
        ax.set_xlabel("epoch index")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{task}: training curves (mean±std over seeds, log-y where useful)", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_fair(task, task_runs, out_path: Path):
    """Zoom into the converged region for the comparable metrics."""
    exps = [e for e in EXP_STYLE if any(r["exp"] == e for r in task_runs)]
    fig, axes = plt.subplots(1, len(FAIR_KEYS), figsize=(5 * len(FAIR_KEYS), 4.2), constrained_layout=True)
    if len(FAIR_KEYS) == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, FAIR_KEYS):
        collected = []
        for exp in exps:
            sty = style_for(exp)
            res = stack_over_seeds(task_runs, exp, key)
            if res is None:
                continue
            xs, m, s = res
            ax.plot(xs, m, color=sty["color"], lw=2, label=sty["label"])
            ax.fill_between(xs, m - s, m + s, color=sty["color"], alpha=0.15)
            collected.append(m)
        # zoom to last 60% of epochs (converged region)
        if collected:
            T = max(len(m) for m in collected)
            lo = int(T * 0.4)
            ax.set_xlim(lo, T - 1)
            seg = np.concatenate([m[lo:][np.isfinite(m[lo:])] for m in collected])
            if seg.size:
                pad = 0.1 * (seg.max() - seg.min() + 1e-9)
                ax.set_ylim(seg.min() - pad, seg.max() + pad)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch index (converged region)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"{task}: fair comparison (converged zoom)", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def final_stats(task_runs, exp, key):
    finals = []
    for r in task_runs:
        if r["exp"] != exp:
            continue
        d = r["data"]
        if key in d and np.isfinite(d[key][-1]):
            finals.append(d[key][-1])
    if not finals:
        return None
    return float(np.mean(finals)), float(np.std(finals))


def plot_final_bars(task, task_runs, out_path: Path):
    keys = [
        ("val_z_proprio_loss", "WM val\n(FAIR↓)"),
        ("train_z_proprio_loss", "WM train\n(FAIR↓)"),
        ("val_bisim_z_dist", "bisim z_dist\n(val)"),
        ("val_bisim_var_loss", "bisim var\n(val)"),
        ("val_id_loss", "ID val\n(id_id)"),
        ("train_id_loss", "ID train\n(id_id)"),
    ]
    exps = [e for e in EXP_STYLE if any(r["exp"] == e for r in task_runs)]
    x = np.arange(len(keys))
    width = 0.8 / max(len(exps), 1)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, exp in enumerate(exps):
        sty = style_for(exp)
        means, stds = [], []
        for key, _ in keys:
            st = final_stats(task_runs, exp, key)
            means.append(st[0] if st else 0.0)
            stds.append(st[1] if st else 0.0)
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                      color=sty["color"], label=sty["label"], alpha=0.85)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width * (len(exps) - 1) / 2)
    ax.set_xticklabels([k[1] for k in keys], fontsize=9)
    ax.set_ylabel("final-epoch value (mean±std over seeds)")
    ax.set_title(f"{task}: final metrics — FAIR panels are WM train/val (lower=better)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_per_seed(task_runs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in task_runs:
        d = r["data"]
        xs = np.arange(len(d["epoch"]))
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        axes = axes.ravel()
        for ax, (tk, vk, title, _) in zip(axes, METRICS):
            if tk in d:
                ax.plot(xs, d[tk], lw=2, label="train")
            if vk in d:
                ax.plot(xs, d[vk], lw=2, ls="--", label="val")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(f"index (epoch {int(d['epoch'][0])}->{int(d['epoch'][-1])})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"{r['task']} | {r['exp']} | seed {r['seed']}")
        fig.savefig(out_dir / f"{r['task']}_{r['exp']}_seed{r['seed']}.png", dpi=130)
        plt.close(fig)


def print_verdict(runs):
    by_task = defaultdict(list)
    for r in runs:
        by_task[r["task"]].append(r)
    print("\n=== FAIR verdict (final-epoch mean over seeds; lower is better) ===")
    for task, task_runs in by_task.items():
        b = final_stats(task_runs, "bisim_baseline", "val_z_proprio_loss")
        i = final_stats(task_runs, "bisim_id_id", "val_z_proprio_loss")
        if b and i:
            delta = (i[0] - b[0]) / b[0] * 100
            verdict = "id_id ~ baseline" if abs(delta) < 5 else ("id_id WORSE" if delta > 0 else "id_id BETTER")
            print(f"  {task:9s} val_z_proprio: baseline={b[0]:.4f}  id_id={i[0]:.4f}  ({delta:+.1f}%)  -> {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--per_seed", action="store_true")
    args = parser.parse_args()

    runs = discover_runs(args.outputs_dir)
    if not runs:
        raise SystemExit(f"No training_loss_log.csv under {args.outputs_dir}")

    by_task = defaultdict(list)
    for r in runs:
        by_task[r["task"]].append(r)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for task, task_runs in by_task.items():
        plot_curves(task, task_runs, args.out_dir / f"{task}_curves.png")
        plot_fair(task, task_runs, args.out_dir / f"{task}_fair_compare.png")
        plot_final_bars(task, task_runs, args.out_dir / f"{task}_final_bars.png")
        if args.per_seed:
            plot_per_seed(task_runs, args.out_dir / "per_seed")

    print_verdict(runs)
    print(f"\nDone. Figures in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
