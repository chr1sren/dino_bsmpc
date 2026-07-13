#!/usr/bin/env python3
"""Generate synthetic PickCube-like offline data when ManiSkill is unavailable."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def make_trajectory(episode_seed, max_steps=20, image_size=128):
    rng = np.random.RandomState(episode_seed)
    states = []
    proprios = []
    visuals = []
    actions = []
    for t in range(max_steps + 1):
        tcp = rng.uniform(-0.2, 0.2, size=3).astype(np.float32)
        tcp[2] = abs(tcp[2]) + 0.05
        obj = rng.uniform(-0.15, 0.15, size=3).astype(np.float32)
        obj[2] = 0.02
        goal = rng.uniform(-0.15, 0.15, size=3).astype(np.float32)
        goal[2] = 0.02
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        grip = np.array([rng.rand()], dtype=np.float32)
        state = np.concatenate([tcp, quat, obj, goal, grip], axis=0)
        proprio = np.concatenate([tcp, grip], axis=0)
        frame = (rng.rand(image_size, image_size, 3) * 255).astype(np.uint8)
        states.append(state)
        proprios.append(proprio)
        visuals.append(frame)
        if t < max_steps:
            actions.append(rng.uniform(-1, 1, size=4).astype(np.float32))
    if actions:
        actions.append(actions[-1].copy())
    return {
        "states": np.stack(states),
        "proprios": np.stack(proprios),
        "visuals": visuals,
        "actions": np.stack(actions),
        "env_info": {
            "episode_seed": episode_seed,
            "goal_pos": states[0][10:13].copy(),
            "control_mode": "pd_ee_delta_pos",
            "image_size": image_size,
            "success": False,
            "synthetic": True,
        },
    }


def save_split(trajectories, out_dir, split_name):
    out_dir = Path(out_dir) / split_name
    obs_dir = out_dir / "obses"
    obs_dir.mkdir(parents=True, exist_ok=True)
    seq_lengths = [len(t["visuals"]) for t in trajectories]
    max_t = max(seq_lengths)
    states = torch.zeros(len(trajectories), max_t, 14)
    proprios = torch.zeros(len(trajectories), max_t, 4)
    actions = torch.zeros(len(trajectories), max_t, 4)
    env_infos = []
    for i, traj in enumerate(trajectories):
        T = seq_lengths[i]
        states[i, :T] = torch.from_numpy(traj["states"])
        proprios[i, :T] = torch.from_numpy(traj["proprios"])
        actions[i, :T] = torch.from_numpy(traj["actions"])
        torch.save(torch.from_numpy(np.stack(traj["visuals"])), obs_dir / f"episode_{i:03d}.pth")
        env_infos.append(traj["env_info"])
    torch.save(states, out_dir / "states.pth")
    torch.save(proprios, out_dir / "proprios.pth")
    torch.save(actions, out_dir / "actions.pth")
    with open(out_dir / "seq_lengths.pkl", "wb") as f:
        pickle.dump(seq_lengths, f)
    with open(out_dir / "env_info.pkl", "wb") as f:
        pickle.dump(env_infos, f)
    if split_name == "train":
        stacked = actions.reshape(-1, 4)
        stats = {
            "action_mean": stacked.mean(dim=0),
            "action_std": stacked.std(dim=0).clamp(min=1e-6),
            "state_mean": states.reshape(-1, 14).mean(dim=0),
            "state_std": states.reshape(-1, 14).std(dim=0).clamp(min=1e-6),
            "proprio_mean": proprios.reshape(-1, 4).mean(dim=0),
            "proprio_std": proprios.reshape(-1, 4).std(dim=0).clamp(min=1e-6),
        }
        torch.save(stats, out_dir / "stats.pth")
    print(f"Saved synthetic {split_name}: {len(trajectories)} episodes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/pickcube_v1")
    parser.add_argument("--n_episodes", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--base_seed", type=int, default=0)
    args = parser.parse_args()
    trajs = [make_trajectory(args.base_seed + i, args.max_steps, args.image_size) for i in range(args.n_episodes)]
    n_val = max(1, args.n_episodes // 5)
    save_split(trajs[:-n_val], args.out, "train")
    save_split(trajs[-n_val:], args.out, "val")


if __name__ == "__main__":
    main()
