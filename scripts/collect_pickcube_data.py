#!/usr/bin/env python3
"""Collect PickCube-v1 offline trajectories for dino_bsmpc world-model training."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env  # noqa: F401 — register pickcube
from env.maniskill.pickcube_wrapper import PickCubeWrapper
from env.maniskill.state_utils import extract_pickcube_state, state_to_proprio


class PickCubeHeuristicPolicy:
    """Phased pick-and-place heuristic with optional action noise."""

    def __init__(self, action_low, action_high, noise_std=0.05):
        self.action_low = action_low
        self.action_high = action_high
        self.noise_std = noise_std
        self.phase = 0

    def reset(self):
        self.phase = 0

    def __call__(self, env_unwrapped, rng):
        state = extract_pickcube_state(env_unwrapped)
        tcp = state[:3]
        obj = state[7:10]
        goal = state[10:13]
        action = np.zeros(4, dtype=np.float32)

        dist_tcp_obj = float(np.linalg.norm(tcp - obj))
        dist_obj_goal = float(np.linalg.norm(obj - goal))

        if self.phase == 0:
            target = obj + np.array([0.0, 0.0, 0.08], dtype=np.float32)
            action[:3] = np.clip((target - tcp) * 5.0, -1.0, 1.0)
            action[3] = 1.0
            if dist_tcp_obj < 0.04:
                self.phase = 1
        elif self.phase == 1:
            action[:3] = np.clip((obj - tcp) * 8.0, -1.0, 1.0)
            action[3] = -1.0
            if dist_tcp_obj < 0.02:
                self.phase = 2
        elif self.phase == 2:
            target = obj + np.array([0.0, 0.0, 0.12], dtype=np.float32)
            action[:3] = np.clip((target - tcp) * 6.0, -1.0, 1.0)
            action[3] = -1.0
            if tcp[2] - obj[2] > 0.08:
                self.phase = 3
        elif self.phase == 3:
            target = goal + np.array([0.0, 0.0, 0.10], dtype=np.float32)
            action[:3] = np.clip((target - tcp) * 5.0, -1.0, 1.0)
            action[3] = -1.0
            if dist_obj_goal < 0.04:
                self.phase = 4
        else:
            action[:3] = np.clip((goal - tcp) * 4.0, -1.0, 1.0)
            action[3] = 1.0

        if self.noise_std > 0:
            action = action + rng.normal(0, self.noise_std, size=action.shape).astype(np.float32)
        return np.clip(action, self.action_low, self.action_high)


def try_motion_plan_episode(env, seed, max_steps):
    """Attempt ManiSkill official motion-planning demo if installed."""
    try:
        from mani_skill.examples.motionplanning.panda.run import solve_pick_cube
        actions = solve_pick_cube(env._ms._env, seed=seed)
        if actions is None or len(actions) == 0:
            return None
        return np.asarray(actions, dtype=np.float32)[:max_steps]
    except Exception:
        pass
    try:
        from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve
        result = solve(env._ms._env, seed=seed)
        if result is None:
            return None
        if isinstance(result, dict) and 'actions' in result:
            actions = result['actions']
        else:
            actions = result
        return np.asarray(actions, dtype=np.float32)[:max_steps]
    except Exception:
        return None


def collect_episode(env, policy, episode_seed, max_steps, use_random=False, rng=None):
    env.seed(episode_seed)
    obs, state = env.reset()
    sim_state = env._ms.get_sim_state()

    visuals = [obs['visual'].copy()]
    proprios = [obs['proprio'].copy()]
    states = [state.copy()]
    actions = []
    success = False

    for _ in range(max_steps):
        if use_random:
            action = env.action_space.sample()
        else:
            action = policy(env._ms._env.unwrapped, rng)
        next_obs, reward, done, info = env.step(action)
        actions.append(action.astype(np.float32))
        visuals.append(next_obs['visual'].copy())
        proprios.append(next_obs['proprio'].copy())
        states.append(info['state'].copy())
        success = success or bool(info.get('success', False))
        obs = next_obs
        if done:
            break

    env_info = {
        'episode_seed': episode_seed,
        'goal_pos': states[0][10:13].copy(),
        'control_mode': 'pd_ee_delta_pos',
        'image_size': env._ms._image_size,
        'success': success,
    }
    if sim_state is not None:
        env_info['sim_state'] = np.asarray(sim_state)

    return {
        'visuals': visuals,
        'proprios': np.stack(proprios, axis=0),
        'states': np.stack(states, axis=0),
        'actions': np.stack(actions, axis=0) if actions else np.zeros((0, 4), dtype=np.float32),
        'env_info': env_info,
    }


def collect_episode_from_planned_actions(env, planned_actions, episode_seed):
    env.seed(episode_seed)
    obs, state = env.reset()
    sim_state = env._ms.get_sim_state()
    visuals = [obs['visual'].copy()]
    proprios = [obs['proprio'].copy()]
    states = [state.copy()]
    actions = []
    success = False
    noise = np.random.RandomState(episode_seed)

    for t, action in enumerate(planned_actions):
        action = np.asarray(action, dtype=np.float32)
        action = action + noise.normal(0, 0.02, size=action.shape).astype(np.float32)
        action = np.clip(action, env._ms._action_low, env._ms._action_high)
        next_obs, reward, done, info = env.step(action)
        actions.append(action)
        visuals.append(next_obs['visual'].copy())
        proprios.append(next_obs['proprio'].copy())
        states.append(info['state'].copy())
        success = success or bool(info.get('success', False))
        obs = next_obs
        if done:
            break

    env_info = {
        'episode_seed': episode_seed,
        'goal_pos': states[0][10:13].copy(),
        'control_mode': 'pd_ee_delta_pos',
        'image_size': env._ms._image_size,
        'success': success,
        'source': 'motion_plan',
    }
    if sim_state is not None:
        env_info['sim_state'] = np.asarray(sim_state)
    return {
        'visuals': visuals,
        'proprios': np.stack(proprios, axis=0),
        'states': np.stack(states, axis=0),
        'actions': np.stack(actions, axis=0) if actions else np.zeros((0, 4), dtype=np.float32),
        'env_info': env_info,
    }


def save_split(trajectories, out_dir, split_name):
    out_dir = Path(out_dir) / split_name
    obs_dir = out_dir / 'obses'
    obs_dir.mkdir(parents=True, exist_ok=True)

    seq_lengths = []
    states_list = []
    actions_list = []
    proprios_list = []
    env_infos = []

    for i, traj in enumerate(trajectories):
        T = len(traj['visuals'])
        seq_lengths.append(T)
        states_list.append(torch.from_numpy(traj['states']))
        proprios_list.append(torch.from_numpy(traj['proprios']))
        if traj['actions'].shape[0] > 0:
            actions_list.append(torch.from_numpy(traj['actions']))
        else:
            actions_list.append(torch.zeros(0, 4))

        video_path = obs_dir / f'episode_{i:03d}.mp4'
        imageio.mimsave(video_path, traj['visuals'], fps=10)

        env_infos.append(traj['env_info'])

    max_t = max(seq_lengths)
    state_dim = states_list[0].shape[-1]
    action_dim = 4
    proprio_dim = proprios_list[0].shape[-1]

    states = torch.zeros(len(trajectories), max_t, state_dim)
    proprios = torch.zeros(len(trajectories), max_t, proprio_dim)
    actions = torch.zeros(len(trajectories), max_t, action_dim)
    for i, length in enumerate(seq_lengths):
        states[i, :length] = states_list[i]
        proprios[i, :length] = proprios_list[i]
        actions[i, :length] = actions_list[i][:length]

    torch.save(states, out_dir / 'states.pth')
    torch.save(proprios, out_dir / 'proprios.pth')
    torch.save(actions, out_dir / 'actions.pth')
    with open(out_dir / 'seq_lengths.pkl', 'wb') as f:
        pickle.dump(seq_lengths, f)
    with open(out_dir / 'env_info.pkl', 'wb') as f:
        pickle.dump(env_infos, f)

    if split_name == 'train':
        train_actions = []
        for i, length in enumerate(seq_lengths):
            train_actions.append(actions[i, :length])
        stacked = torch.vstack(train_actions)
        stats = {
            'action_mean': stacked.mean(dim=0),
            'action_std': stacked.std(dim=0).clamp(min=1e-6),
            'state_mean': states.reshape(-1, state_dim).mean(dim=0),
            'state_std': states.reshape(-1, state_dim).std(dim=0).clamp(min=1e-6),
            'proprio_mean': proprios.reshape(-1, proprio_dim).mean(dim=0),
            'proprio_std': proprios.reshape(-1, proprio_dim).std(dim=0).clamp(min=1e-6),
        }
        torch.save(stats, out_dir / 'stats.pth')

    print(f"Saved {split_name}: {len(trajectories)} episodes to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='data/pickcube_v1')
    parser.add_argument('--n_episodes', type=int, default=40)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--base_seed', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--random_fraction', type=float, default=0.2)
    parser.add_argument('--motion_plan_fraction', type=float, default=0.3)
    args = parser.parse_args()

    env = PickCubeWrapper(image_size=args.image_size, reconfiguration_freq=1)
    policy = PickCubeHeuristicPolicy(env._ms._action_low, env._ms._action_high, noise_std=0.05)
    rng = np.random.RandomState(args.base_seed)

    trajectories = []
    for ep in range(args.n_episodes):
        episode_seed = args.base_seed + ep
        use_random = rng.rand() < args.random_fraction
        use_motion = (not use_random) and (rng.rand() < args.motion_plan_fraction)

        if use_motion:
            planned = try_motion_plan_episode(env, episode_seed, args.max_steps)
            if planned is not None and len(planned) > 0:
                traj = collect_episode_from_planned_actions(env, planned, episode_seed)
            else:
                policy.reset()
                traj = collect_episode(env, policy, episode_seed, args.max_steps, use_random=False, rng=rng)
        elif use_random:
            traj = collect_episode(env, policy, episode_seed, args.max_steps, use_random=True, rng=rng)
        else:
            policy.reset()
            traj = collect_episode(env, policy, episode_seed, args.max_steps, use_random=False, rng=rng)

        trajectories.append(traj)
        print(
            f"episode {ep:03d} seed={episode_seed} T={len(traj['visuals'])} "
            f"success={traj['env_info'].get('success', False)} "
            f"source={'random' if use_random else ('motion' if use_motion else 'heuristic')}"
        )

    env.close()

    n_val = max(1, int(args.n_episodes * args.val_ratio))
    n_train = args.n_episodes - n_val
    train_trajs = trajectories[:n_train]
    val_trajs = trajectories[n_train:]

    out_root = Path(args.out)
    save_split(train_trajs, out_root, 'train')
    save_split(val_trajs, out_root, 'val')


if __name__ == '__main__':
    main()
