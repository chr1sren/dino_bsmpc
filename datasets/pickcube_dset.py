import pickle
import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional

from .traj_dset import TrajDataset, TrajSlicerDataset

try:
    import decord
    decord.bridge.set_bridge("torch")
except Exception:
    decord = None


class PickCubeDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        stats_path: Optional[str] = None,
    ):
        self.data_path = Path(data_path) / split
        self.transform = transform
        self.normalize_action = normalize_action

        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.proprios = torch.load(self.data_path / "proprios.pth").float()
        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        with open(self.data_path / "env_info.pkl", "rb") as f:
            self.env_infos = pickle.load(f)

        self.n_rollout = n_rollout
        n = self.n_rollout if self.n_rollout else len(self.seq_lengths)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.proprios = self.proprios[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.env_infos = self.env_infos[:n]
        print(f"Loaded PickCube {split}: {n} rollouts from {self.data_path}")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            if stats_path and Path(stats_path).exists():
                stats = torch.load(stats_path)
                self.action_mean = stats["action_mean"]
                self.action_std = stats["action_std"]
                self.state_mean = stats["state_mean"]
                self.state_std = stats["state_std"]
                self.proprio_mean = stats["proprio_mean"]
                self.proprio_std = stats["proprio_std"]
            else:
                self.action_mean, self.action_std = self._mean_std(self.actions, self.seq_lengths)
                self.state_mean, self.state_std = self._mean_std(self.states, self.seq_lengths)
                self.proprio_mean, self.proprio_std = self._mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    @staticmethod
    def _mean_std(data, traj_lengths):
        chunks = []
        for i, length in enumerate(traj_lengths):
            chunks.append(data[i, :length])
        stacked = torch.vstack(chunks)
        return torch.mean(stacked, dim=0), torch.std(stacked, dim=0).clamp(min=1e-6)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i, length in enumerate(self.seq_lengths):
            result.append(self.actions[i, :length])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        vid_path = self.data_path / "obses" / f"episode_{idx:03d}.mp4"
        pth_path = self.data_path / "obses" / f"episode_{idx:03d}.pth"
        if pth_path.exists():
            image = torch.load(pth_path).float()
            image = image[frames]
            image = image / 255.0
            image = rearrange(image, "T H W C -> T C H W")
        else:
            try:
                from decord import VideoReader
                reader = VideoReader(str(vid_path), num_threads=1)
                image = reader.get_batch(frames)
                image = image / 255.0
                image = rearrange(image, "T H W C -> T C H W")
            except Exception:
                import imageio.v2 as imageio
                video = imageio.mimread(str(vid_path))
                image = torch.from_numpy(np.stack([video[i] for i in frames])).float()
                image = image / 255.0
                image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {
            "visual": image,
            "proprio": self.proprios[idx, frames],
        }
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        return obs, act, state, self.env_infos[idx]

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_pickcube_slice_train_val(
    transform,
    data_path="data/pickcube_v1",
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    n_rollout=None,
):
    del split_ratio  # train/val dirs are pre-split during collection
    train_dset = PickCubeDataset(
        data_path=data_path,
        split="train",
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        stats_path=str(Path(data_path) / "train" / "stats.pth"),
    )
    val_dset = PickCubeDataset(
        data_path=data_path,
        split="val",
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        stats_path=str(Path(data_path) / "train" / "stats.pth"),
    )
    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)
    datasets = {"train": train_slices, "valid": val_slices}
    traj_dsets = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dsets
