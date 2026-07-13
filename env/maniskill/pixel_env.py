"""Standalone ManiSkill pixel env (adapted from BiSimBad/maniskill_gym)."""

import os
import platform
import sys

import gymnasium
import numpy as np

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

_EE_CONTROL_MODES = frozenset({
    'pd_ee_delta_pos',
    'pd_ee_delta_pose',
    'pd_ee_target_delta_pos',
    'pd_ee_target_delta_pose',
})


def _ensure_windows_dll_paths():
    if platform.system() != 'Windows':
        return
    candidates = []
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        candidates.append(os.path.join(cuda_path, 'bin'))
    toolkit_root = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if os.path.isdir(toolkit_root):
        for name in sorted(os.listdir(toolkit_root), reverse=True):
            candidates.append(os.path.join(toolkit_root, name, 'bin'))
    for directory in candidates:
        if not os.path.isdir(directory):
            continue
        try:
            os.add_dll_directory(directory)
        except (AttributeError, OSError):
            pass
        path = os.environ.get('PATH', '')
        if directory not in path.split(os.pathsep):
            os.environ['PATH'] = directory + os.pathsep + path


def _has_robotics_pinocchio():
    try:
        import pinocchio  # noqa: F401
        from pinocchio import buildModelFromXML  # noqa: F401
        return True
    except ImportError:
        return False


def _check_control_mode(control_mode, sim_backend):
    if sim_backend != 'cpu' or control_mode not in _EE_CONTROL_MODES:
        return
    if _has_robotics_pinocchio():
        return
    raise ImportError(
        'ManiSkill control_mode=%r with sim_backend=cpu requires conda-forge pinocchio.'
        % control_mode
    )


def _resolve_sim_backend(sim_backend):
    requested = sim_backend
    if platform.system() == 'Windows' and sim_backend in ('auto', 'gpu'):
        if sim_backend == 'gpu':
            print(
                'WARNING: ManiSkill GPU physics unavailable on Windows; using cpu.',
                file=sys.stderr,
            )
        sim_backend = 'cpu'
    if sim_backend == 'auto':
        try:
            import torch
            sim_backend = 'gpu' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            sim_backend = 'cpu'
    return sim_backend


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value)
    return value


def _scalar(value, dtype=float):
    arr = _to_numpy(value)
    if arr is None:
        return dtype()
    if arr.ndim == 0:
        return dtype(arr.item())
    return dtype(arr.reshape(-1)[0])


def _bool_scalar(value):
    return bool(_scalar(value, dtype=np.float32))


def _extract_rgb(obs):
    if isinstance(obs, dict):
        if 'rgb' in obs:
            rgb = obs['rgb']
        elif 'sensor_data' in obs:
            sensor_data = obs['sensor_data']
            rgb = next(iter(sensor_data.values()))['rgb']
        else:
            raise KeyError('Expected rgb or sensor_data in obs, got %s' % list(obs.keys()))
    else:
        rgb = obs

    rgb = _to_numpy(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb = rgb.clip(0, 255).astype(np.uint8)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb


class ManiSkillPixelEnv:
    """ManiSkill wrapper with frame_skip=1 for dino_bsmpc offline/planning use."""

    def __init__(
        self,
        env_id='PickCube-v1',
        seed=1,
        obs_mode='rgb',
        control_mode='pd_ee_delta_pos',
        reward_mode='dense',
        image_size=128,
        frame_skip=1,
        sim_backend='auto',
        reconfiguration_freq=0,
        render_mode=None,
        max_episode_steps=50,
        show_goal=True,
        sensor_cam_eye_pos=None,
        sensor_cam_target_pos=None,
    ):
        self._seed = seed
        self._frame_skip = max(1, int(frame_skip))
        self._image_size = int(image_size)
        self._last_rgb_hwc = None
        self._max_episode_steps = max(1, int(max_episode_steps))
        self._episode_step = 0
        self._seeded = False
        self._show_goal = bool(show_goal)
        self._episode_seed = seed

        _ensure_windows_dll_paths()
        sim_backend = _resolve_sim_backend(sim_backend)
        _check_control_mode(control_mode, sim_backend)
        sensor_configs = dict(width=self._image_size, height=self._image_size)
        if sensor_cam_eye_pos is not None and sensor_cam_target_pos is not None:
            from mani_skill.utils import sapien_utils
            sensor_configs['pose'] = sapien_utils.look_at(
                list(sensor_cam_eye_pos), list(sensor_cam_target_pos)
            )
        env_kwargs = dict(
            obs_mode=obs_mode,
            control_mode=control_mode,
            reward_mode=reward_mode,
            render_mode=render_mode,
            sim_backend=sim_backend,
            sensor_configs=sensor_configs,
            max_episode_steps=self._max_episode_steps * self._frame_skip,
        )
        env = gymnasium.make(
            env_id,
            num_envs=1,
            reconfiguration_freq=reconfiguration_freq,
            **env_kwargs,
        )
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False)
        self._env = env
        self.env_id = env_id
        self.control_mode = control_mode

        sample = env.reset(seed=seed)[0]
        self._observation_shape = _extract_rgb(sample).shape

        single_action_space = self._resolve_single_action_space(env)
        self._action_low = _to_numpy(single_action_space.low).astype(np.float32)
        self._action_high = _to_numpy(single_action_space.high).astype(np.float32)
        self.action_dim = int(self._action_low.shape[0])

    @property
    def observation_space(self):
        import gym
        h, w, c = self._observation_shape
        return gym.spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)

    @property
    def action_space(self):
        import gym
        return gym.spaces.Box(
            low=self._action_low, high=self._action_high, dtype=np.float32
        )

    def seed(self, seed):
        self._seed = seed
        self._episode_seed = seed
        self._seeded = False

    @staticmethod
    def _resolve_single_action_space(env):
        space = getattr(env, 'single_action_space', None)
        if space is None:
            space = getattr(env.unwrapped, 'single_action_space', None)
        if space is not None:
            return space
        space = env.action_space
        low = _to_numpy(space.low)
        high = _to_numpy(space.high)
        if low.ndim > 1:
            low = low[0]
            high = high[0]
        import gym
        return gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

    def _process_obs(self, obs):
        rgb = _extract_rgb(obs)
        self._last_rgb_hwc = rgb
        return rgb

    def _process_info(self, info):
        out = {}
        if not isinstance(info, dict):
            return out
        for key in ('success', 'is_grasped', 'is_obj_placed', 'is_robot_static'):
            if key in info:
                out[key] = _bool_scalar(info[key])
        return out

    def _reveal_goal(self, obs):
        base = self._env.unwrapped
        hidden = getattr(base, '_hidden_objects', None)
        if not hidden:
            return obs
        for obj in list(hidden):
            try:
                obj.show_visual()
            except Exception:
                pass
        hidden.clear()
        try:
            return base.get_obs()
        except Exception:
            return obs

    def reset(self, seed=None):
        self._episode_step = 0
        if seed is not None:
            self._episode_seed = seed
            self._seeded = False
        if not self._seeded:
            obs, info = self._env.reset(seed=self._episode_seed)
            self._seeded = True
        else:
            obs, info = self._env.reset()
        if self._show_goal:
            obs = self._reveal_goal(obs)
        return self._process_obs(obs), info

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self._action_low,
            self._action_high,
        )
        self._episode_step += 1
        total_reward = 0.0
        num_substeps = 0
        ms_success_terminated = False
        ms_truncated = False
        info = {}

        for _ in range(self._frame_skip):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += _scalar(reward)
            num_substeps += 1
            if terminated:
                ms_success_terminated = True
            if truncated:
                ms_truncated = True
                break

        reward_out = total_reward / max(1, num_substeps)
        truncated = ms_truncated or self._episode_step >= self._max_episode_steps
        done = truncated

        processed_info = self._process_info(info)
        if ms_success_terminated:
            processed_info['success'] = True
        processed_info['ms_success_terminated'] = ms_success_terminated
        processed_info['terminated'] = ms_success_terminated
        processed_info['truncated'] = truncated
        return self._process_obs(obs), reward_out, done, processed_info

    def get_sim_state(self):
        base = self._env.unwrapped
        if hasattr(base, 'get_state'):
            state = base.get_state()
            if hasattr(state, 'cpu'):
                return state.cpu().numpy()
            return np.asarray(state)
        return None

    def set_sim_state(self, state):
        base = self._env.unwrapped
        if hasattr(base, 'set_state'):
            import torch
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state)
            base.set_state(state)
            return True
        return False

    def close(self):
        self._env.close()
