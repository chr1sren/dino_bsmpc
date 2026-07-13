"""Semantic state extraction for ManiSkill PickCube."""

import numpy as np


def _pose_p(pose):
    p = pose.p if hasattr(pose, 'p') else pose
    arr = np.asarray(p, dtype=np.float32).reshape(-1)
    return arr[:3]


def _pose_q(pose):
    if hasattr(pose, 'q'):
        q = pose.q
    else:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    arr = np.asarray(q, dtype=np.float32).reshape(-1)
    if arr.shape[0] == 4:
        return arr
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _gripper_opening(base):
    agent = getattr(base, 'agent', None)
    if agent is None:
        return np.array([0.0], dtype=np.float32)
    for attr in ('gripper_state', 'qpos'):
        if hasattr(agent, attr):
            val = getattr(agent, attr)
            arr = np.asarray(val, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return np.array([float(arr[-1])], dtype=np.float32)
    return np.array([0.0], dtype=np.float32)


def extract_pickcube_state(env_unwrapped):
    """
    Semantic state layout (14-dim):
      tcp_pos(3), tcp_quat(4), obj_pos(3), goal_pos(3), gripper_q(1)
    """
    base = env_unwrapped
    tcp_pose = base.agent.tcp.pose
    tcp_pos = _pose_p(tcp_pose)
    tcp_quat = _pose_q(tcp_pose)

    cube = getattr(base, 'cube', None) or getattr(base, 'obj', None)
    if cube is None:
        obj_pos = np.zeros(3, dtype=np.float32)
    else:
        obj_pos = _pose_p(cube.pose)

    goal_site = getattr(base, 'goal_site', None)
    if goal_site is not None:
        goal_pos = _pose_p(goal_site.pose)
    elif hasattr(base, 'goal_pos'):
        goal_pos = np.asarray(base.goal_pos, dtype=np.float32).reshape(-1)[:3]
    else:
        goal_pos = np.zeros(3, dtype=np.float32)

    gripper_q = _gripper_opening(base)
    state = np.concatenate([tcp_pos, tcp_quat, obj_pos, goal_pos, gripper_q], axis=0)
    return state.astype(np.float32)


def state_to_proprio(state):
    """Proprio for WM: tcp_pos + gripper opening."""
    state = np.asarray(state, dtype=np.float32)
    return np.concatenate([state[:3], state[-1:]], axis=0).astype(np.float32)


STATE_DIM = 14
PROPRIO_DIM = 4
