"""Render a single wall state with all available backgrounds and save as PNG."""

import torch
import matplotlib.pyplot as plt
from env.wall.wall_env_wrapper import WallEnvWrapper, WALL_BACKGROUND_CONFIGS, DEFAULT_CFG


def main():
    agent_position = torch.tensor([20.0, 15.0])
    for bg_name in WALL_BACKGROUND_CONFIGS:
        env = WallEnvWrapper(
            rng=42,
            wall_config=DEFAULT_CFG,
            fix_wall=True,
            cross_wall=False,
            fix_wall_location=32,
            fix_door_location=30,
            device='cpu',
            background=bg_name,
        )
        obs, state = env.reset(location=agent_position)
        visual = obs['visual']  # (3, 65, 65) float tensor
        img = visual.permute(1, 2, 0).byte().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img)
        ax.set_title(bg_name.replace('_', ' ').title(), fontsize=11)
        ax.axis('off')
        plt.tight_layout()
        out_path = f'wall_bg_{bg_name}.png'
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved {out_path}')
        plt.close()


if __name__ == '__main__':
    main()
