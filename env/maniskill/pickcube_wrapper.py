"""PickCube-v1 wrapper with dino_bsmpc planning API."""

import numpy as np

from env.maniskill.pixel_env import ManiSkillPixelEnv
from env.maniskill.state_utils import (
    PROPRIO_DIM,
    STATE_DIM,
    extract_pickcube_state,
    state_to_proprio,
)
from utils import aggregate_dct


class PickCubeWrapper:
    """ManiSkill PickCube env exposing prepare/rollout/eval_state for MPC."""

    def __init__(
        self,
        env_id='PickCube-v1',
        image_size=128,
        show_goal=True,
        control_mode='pd_ee_delta_pos',
        reward_mode='dense',
        max_episode_steps=50,
        sim_backend='auto',
        reconfiguration_freq=0,
        sensor_cam_eye_pos=None,
        sensor_cam_target_pos=None,
        seed=0,
    ):
        if sensor_cam_eye_pos is None:
            sensor_cam_eye_pos = [0.45, 0.35, 0.25]
        if sensor_cam_target_pos is None:
            sensor_cam_target_pos = [0.0, 0.0, 0.1]
        self._ms = ManiSkillPixelEnv(
            env_id=env_id,
            seed=seed,
            image_size=image_size,
            frame_skip=1,
            show_goal=show_goal,
            control_mode=control_mode,
            reward_mode=reward_mode,
            max_episode_steps=max_episode_steps,
            sim_backend=sim_backend,
            reconfiguration_freq=reconfiguration_freq,
            sensor_cam_eye_pos=sensor_cam_eye_pos,
            sensor_cam_target_pos=sensor_cam_target_pos,
        )
        self.action_dim = self._ms.action_dim
        self.state_dim = STATE_DIM
        self.proprio_dim = PROPRIO_DIM
        self._saved_sim_state = None
        self._episode_seed = seed
        self._goal_pos = None
        self._sim_state = None

    @property
    def observation_space(self):
        return self._ms.observation_space

    @property
    def action_space(self):
        return self._ms.action_space

    def seed(self, seed):
        self._episode_seed = seed
        self._ms.seed(seed)

    def _make_obs(self, rgb_hwc):
        state = extract_pickcube_state(self._ms._env.unwrapped)
        proprio = state_to_proprio(state)
        return {'visual': rgb_hwc, 'proprio': proprio}, state

    def reset(self):
        rgb, _ = self._ms.reset(seed=self._episode_seed)
        if self._sim_state is not None:
            self._ms.set_sim_state(self._sim_state)
            try:
                obs_dict = self._ms._env.unwrapped.get_obs()
                rgb = self._ms._process_obs(obs_dict)
            except Exception:
                pass
        obs, state = self._make_obs(rgb)
        self._goal_pos = state[10:13].copy()
        self._saved_sim_state = self._ms.get_sim_state()
        return obs, state

    def step(self, action):
        rgb, reward, done, info_ms = self._ms.step(action)
        obs, state = self._make_obs(rgb)
        info = {
            'state': state,
            'success': info_ms.get('success', False),
            'is_obj_placed': info_ms.get('is_obj_placed', False),
            'is_robot_static': info_ms.get('is_robot_static', False),
            'ms_success_terminated': info_ms.get('ms_success_terminated', False),
            'truncated': info_ms.get('truncated', False),
        }
        return obs, reward, done, info

    def prepare(self, seed, init_state, sim_state=None):
        self.seed(seed)
        self._sim_state = sim_state
        obs, state = self.reset()
        if init_state is not None:
            state = np.asarray(init_state, dtype=np.float32)
        return obs, state

    def step_multiple(self, actions):
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions, sim_state=None):
        obs, state = self.prepare(seed, init_state, sim_state=sim_state)
        obses, _, _, infos = self.step_multiple(actions)
        for k in obses:
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos['state']])
        return obses, states

    def eval_state(self, goal_state, cur_state):
        goal_state = np.asarray(goal_state, dtype=np.float32)
        cur_state = np.asarray(cur_state, dtype=np.float32)
        obj_pos = cur_state[7:10]
        goal_pos = cur_state[10:13]
        place_err = float(np.linalg.norm(obj_pos - goal_pos))
        state_dist = float(np.linalg.norm(goal_state - cur_state))
        success = place_err < 0.025
        return {
            'success': success,
            'place_err': place_err,
            'state_dist': state_dist,
        }

    def update_env(self, env_info):
        if env_info is None:
            return
        if 'episode_seed' in env_info:
            self._episode_seed = int(env_info['episode_seed'])
        if 'goal_pos' in env_info:
            self._goal_pos = np.asarray(env_info['goal_pos'], dtype=np.float32)
        if 'sim_state' in env_info and env_info['sim_state'] is not None:
            self._sim_state = np.asarray(env_info['sim_state'])

    def sample_random_init_goal_states(self, seed):
        rs = np.random.RandomState(seed)
        init_seed = int(rs.randint(0, 1_000_000))
        goal_seed = int(rs.randint(0, 1_000_000))
        _, init_state = self.prepare(init_seed, None)
        _, goal_state = self.prepare(goal_seed, None)
        return init_state, goal_state

    def close(self):
        self._ms.close()
