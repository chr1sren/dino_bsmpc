#!/usr/bin/env python3
"""Unit tests for inverse-dynamics bisimulation integration."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.bisim import BisimModel, InverseDynamicsMLP
from models.visual_world_model import VWorldModel


class DummyEncoder(torch.nn.Module):
    name = "dummy"
    emb_dim = 384
    latent_ndim = 2
    patch_size = 14

    def forward(self, x):
        b = x.shape[0]
        return torch.randn(b, 196, 384, device=x.device)


class DummyProprioEncoder(torch.nn.Module):
    emb_dim = 10

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(4, 10)

    def forward(self, x):
        return self.net(x)


class DummyActionEncoder(torch.nn.Module):
    emb_dim = 10

    def __init__(self, in_chans=20):
        super().__init__()
        self.net = torch.nn.Linear(in_chans, 10)

    def forward(self, x):
        return self.net(x)


def test_inverse_dynamics_shapes():
    idm = InverseDynamicsMLP(latent_dim=64, action_dim=20)
    z = torch.randn(4, 3, 64)
    z_next = torch.randn(4, 3, 64)
    out = idm(z, z_next)
    assert out.shape == (4, 3, 20), out.shape
    print("PASS test_inverse_dynamics_shapes")


def test_bisim_id_target_detach():
    model = BisimModel(
        input_dim=196 * 384,
        latent_dim=64,
        action_dim=10,
        wm_action_dim=20,
        train_bisim_id_id=True,
        num_patches=196,
        patch_emb_dim=384,
    )
    b, t, p, d = 4, 3, 196, 64
    z = torch.randn(b, t, p, d, requires_grad=True)
    z2 = z.detach()
    nz = torch.randn(b, t, p, d)
    nz2 = nz.detach()
    reward = torch.randn(b, t, 1)
    loss, _, _, _, _, _, inv_l1 = model.calc_bisim_loss(
        z, z2, reward, reward, nz, nz2, epoch=0,
        train_w_reward_loss=False, id_lambda=0.1, use_id_target=True,
    )
    assert inv_l1.shape == (b, t)
    grad = torch.autograd.grad(loss.sum(), z, retain_graph=False)[0]
    assert grad is not None
    assert torch.isfinite(grad).all()
    print("PASS test_bisim_id_target_detach")


def test_id_supervision_grad_flow():
    bisim = BisimModel(
        input_dim=196 * 384,
        latent_dim=64,
        action_dim=10,
        wm_action_dim=20,
        train_bisim_id_id=True,
        num_patches=196,
        patch_emb_dim=384,
    )
    encoder = DummyEncoder()
    proprio_enc = DummyProprioEncoder()
    action_enc = DummyActionEncoder(in_chans=20)
    wm = VWorldModel(
        image_size=128,
        num_hist=3,
        num_pred=1,
        encoder=encoder,
        proprio_encoder=proprio_enc,
        action_encoder=action_enc,
        decoder=None,
        predictor=None,
        bisim_model=bisim,
        bisim_latent_dim=64,
        proprio_dim=4,
        action_dim=20,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        train_bisim_id_id=True,
        id_lambda=0.1,
        id_omega=0.1,
        train_w_reward_loss=False,
    )
    b, t, h, w = 2, 4, 128, 128
    obs = {
        "visual": torch.rand(b, t, 3, h, w),
        "proprio": torch.randn(b, t, 4),
    }
    act = torch.randn(b, t, 20)
    z_src = torch.randn(b, 3, 196, 64, requires_grad=True)
    z_tgt = torch.randn(b, 3, 196, 64, requires_grad=True)
    id_loss = wm.calc_id_supervision_loss(z_src, z_tgt, act[:, :3])
    assert id_loss.ndim == 0
    grad = torch.autograd.grad(id_loss, z_src, retain_graph=False)[0]
    assert grad is not None and torch.isfinite(grad).all()
    print("PASS test_id_supervision_grad_flow")


def test_old_checkpoint_compat():
    model = BisimModel(
        input_dim=196 * 384,
        latent_dim=64,
        action_dim=10,
        train_bisim_id_id=False,
        num_patches=196,
        patch_emb_dim=384,
    )
    assert model.inverse_dynamics is None
    state = model.state_dict()
    model2 = BisimModel(
        input_dim=196 * 384,
        latent_dim=64,
        action_dim=10,
        wm_action_dim=20,
        train_bisim_id_id=True,
        num_patches=196,
        patch_emb_dim=384,
    )
    missing, unexpected = model2.load_state_dict(state, strict=False)
    assert any('inverse_dynamics' in k for k in missing)
    print("PASS test_old_checkpoint_compat")


def main():
    test_inverse_dynamics_shapes()
    test_bisim_id_target_detach()
    test_id_supervision_grad_flow()
    test_old_checkpoint_compat()
    print("All ID-bisim unit tests passed.")


if __name__ == "__main__":
    main()
