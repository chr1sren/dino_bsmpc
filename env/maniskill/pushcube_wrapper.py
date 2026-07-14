"""PushCube-v1 wrapper; reuses PickCube planning API / state layout."""

from env.maniskill.pickcube_wrapper import PickCubeWrapper


class PushCubeWrapper(PickCubeWrapper):
    """ManiSkill PushCube env for dino_bsmpc training/planning."""

    def __init__(self, env_id='PushCube-v1', **kwargs):
        kwargs.setdefault('max_episode_steps', 50)
        super().__init__(env_id=env_id, **kwargs)
