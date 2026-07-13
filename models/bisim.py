import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np


def build_mlp(input_dim, hidden_dim, output_dim, num_hidden_layers):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ELU()]
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


class InverseDynamicsMLP(nn.Module):
    """Maps concat(z_t, z_{t+1}) -> action (BiSimBad bisim-id-id)."""

    def __init__(self, latent_dim, action_dim, hidden_dim=512):
        super().__init__()
        in_dim = latent_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z, z_next):
        return self.net(torch.cat([z, z_next], dim=-1))


def build_patch_encoder(input_dim, hidden_dim, output_dim, num_hidden_layers=1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class BisimModel(nn.Module):
    def __init__(
            self,
            input_dim,
            latent_dim,
            hidden_dim=256,
            num_hidden_layers=2,
            action_dim=10,
            wm_action_dim=None,
            train_bisim_id_id=False,
            bypass_dinov2=False,
            img_size=224,
            num_patches=196,  # number of output patches
            patch_emb_dim=384,  # encoder patch embedding dimension (384 for DINOv2, 768 for SimDINOv2)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.wm_action_dim = wm_action_dim if wm_action_dim is not None else action_dim
        self.train_bisim_id_id = train_bisim_id_id
        self.bypass_dinov2 = bypass_dinov2
        self.img_size = img_size
        self.num_patches = num_patches
        self.patch_dim = latent_dim
        self.patch_emb_dim = patch_emb_dim

        if bypass_dinov2:
            patch_size = 16  # DINOv2 patch size
            patch_pixel_dim = 3 * patch_size * patch_size  # 768

            middle_dim = 2 * self.patch_dim
            self.encoder = nn.Sequential(
                nn.Linear(patch_pixel_dim, middle_dim),  # 768 -> middle_dim
                ResBlock(middle_dim),
                nn.Linear(middle_dim, self.patch_dim),  # middle_dim -> patch_dim
            )

            # spatial positional embedding for output patches
            self.spatial_pos_emb = nn.Parameter(torch.randn(num_patches, self.patch_dim))

            # layer norm after projection
            self.proj_norm = nn.LayerNorm(self.patch_dim)
            self.patch_size = patch_size
        else:
            # patch_emb_dim -> middle_dim -> ResBlock -> patch_dim
            middle_dim = 2 * self.patch_dim
            self.encoder = nn.Sequential(
                nn.Linear(patch_emb_dim, middle_dim),
                ResBlock(middle_dim),
                nn.Linear(middle_dim, self.patch_dim),
            )

            # spatial positional embedding for output patches
            self.spatial_pos_emb = nn.Parameter(torch.randn(num_patches, self.patch_dim))

            # layer norm after projection
            self.proj_norm = nn.LayerNorm(self.patch_dim)

        reward_hidden_dim = (self.patch_dim + self.action_dim) * 2
        self.reward = build_mlp(self.patch_dim + self.action_dim, reward_hidden_dim, 1, num_hidden_layers=1)

        # aggregator: per-patch score -> softmax weights
        self.reward_aggregator = nn.Linear(self.patch_dim, 1)

        self.inverse_dynamics = None
        if self.train_bisim_id_id:
            self.inverse_dynamics = InverseDynamicsMLP(
                latent_dim=self.patch_dim,
                action_dim=self.wm_action_dim,
                hidden_dim=512,
            )

        self._initialize_weights()
        self.PCAMatrix = []
        self.PCA_Calced = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def log_bisim(self, data):
        """Log bisimulation model details"""
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.integer):
                serializable_data[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_data[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                serializable_data[key] = value.detach().cpu().numpy().tolist()
            elif isinstance(value, (list, tuple)):
                serializable_data[key] = []
                for item in value:
                    if isinstance(item, np.integer):
                        serializable_data[key].append(int(item))
                    elif isinstance(item, np.floating):
                        serializable_data[key].append(float(item))
                    elif isinstance(item, np.ndarray):
                        serializable_data[key].append(item.tolist())
                    elif isinstance(item, torch.Tensor):
                        serializable_data[key].append(item.detach().cpu().numpy().tolist())
                    else:
                        serializable_data[key].append(item)
            elif isinstance(value, dict):
                serializable_data[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.integer):
                        serializable_data[key][k] = int(v)
                    elif isinstance(v, np.floating):
                        serializable_data[key][k] = float(v)
                    elif isinstance(v, np.ndarray):
                        serializable_data[key][k] = v.tolist()
                    elif isinstance(v, torch.Tensor):
                        serializable_data[key][k] = v.detach().cpu().numpy().tolist()
                    else:
                        serializable_data[key][k] = v
            else:
                serializable_data[key] = value

        log_entry = {
            "timestamp": np.datetime64('now').astype(str),
            **serializable_data
        }
        with open("bisim_log.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def encode(self, input_data):
        """
        Maps input to bisimulation embeddings
        input:
        - If bypass_dinov2=False: z_dino: (b, t, p, d) - DinoV2 embeddings
        - If bypass_dinov2=True: obs: (b, t, 3, img_size, img_size) - Raw observations
        output: z_bisim: (b, t, num_patches, patch_dim)
        """
        if self.bypass_dinov2:
            b, t, c, h, w = input_data.shape

            patch_size = self.patch_size
            num_patches_h = h // patch_size  # 14
            num_patches_w = w // patch_size  # 14

            patches = input_data.reshape(b, t, c, num_patches_h, patch_size, num_patches_w, patch_size)
            patches = patches.permute(0, 1, 3, 5, 2, 4, 6)
            patches = patches.reshape(b, t, num_patches_h * num_patches_w, c, patch_size, patch_size)
            patches = patches.reshape(b, t, self.num_patches, c * patch_size * patch_size)

            z_bisim = self.encoder(patches)  # (b, t, num_patches, patch_dim)
            z_bisim = z_bisim + self.spatial_pos_emb.unsqueeze(0).unsqueeze(0)  # broadcast over batch and time
            z_bisim = self.proj_norm(z_bisim)
            event = "bisim_encode_direct_patches"

        else:
            # apply per-token encoding: patch_emb_dim -> patch_dim
            z_bisim = self.encoder(input_data)  # (b, t, 196, patch_dim)

            # add spatial positional embeddings
            z_bisim = z_bisim + self.spatial_pos_emb.unsqueeze(0).unsqueeze(0)  # broadcast over batch and time

            # apply layer norm
            z_bisim = self.proj_norm(z_bisim)

            event = "bisim_encode_dinov2_patches"

        self.log_bisim({
            "event": event,
            "input_shape": list(input_data.shape),
            "output_shape": list(z_bisim.shape),
            "bypass_dinov2": self.bypass_dinov2,
            "num_patches": self.num_patches,
            "patch_dim": self.patch_dim,
            "input_stats": {
                "mean": float(input_data.mean().item()),
                "std": float(input_data.std().item()),
                "min": float(input_data.min().item()),
                "max": float(input_data.max().item()),
            },
            "output_stats": {
                "mean": float(z_bisim.mean().item()),
                "std": float(z_bisim.std().item()),
                "min": float(z_bisim.min().item()),
                "max": float(z_bisim.max().item()),
            },
        })

        return z_bisim

    def predict_reward(self, z_bisim, action_emb):
        """
        Predicts reward from bisimulation state and action
        z_bisim: (b, p, d) or (b, t, p, d) where d == self.patch_dim
        action_emb: (b, a) or (b, t, a)
        returns: (b, 1) or (b, t, 1)
        """
        assert z_bisim.shape[-1] == self.patch_dim, \
            f"z_bisim last dim {z_bisim.shape[-1]} must equal patch_dim {self.patch_dim}"

        if z_bisim.dim() == 4:
            b, t, p, d = z_bisim.shape
            z2 = z_bisim.reshape(b * t, p, d)  # (bt, p, d)
            scores = self.reward_aggregator(z2)  # (bt, p, 1)
            weights = torch.softmax(scores, dim=1)  # (bt, p, 1)
            z_agg = (z2 * weights).sum(dim=1)  # (bt, d)
            if action_emb is None:
                a = torch.zeros(b, t, self.action_dim, device=z_bisim.device, dtype=z_bisim.dtype)
            else:
                a = action_emb
            if a.dim() == 2:  # (b, a) -> (b, t, a)
                a = a.unsqueeze(1).expand(b, t, -1)
            a = a.reshape(b * t, -1)  # (bt, a)
            x = torch.cat([z_agg, a], dim=-1)  # (bt, d+a)
            out = self.reward(x).reshape(b, t, 1)  # (b, t, 1)
            return out

        elif z_bisim.dim() == 3:
            b, p, d = z_bisim.shape
            scores = self.reward_aggregator(z_bisim)  # (b, p, 1)
            weights = torch.softmax(scores, dim=1)  # (b, p, 1)
            z_agg = (z_bisim * weights).sum(dim=1)  # (b, d)
            if action_emb is None:
                a = torch.zeros(b, self.action_dim, device=z_bisim.device, dtype=z_bisim.dtype)
            else:
                a = action_emb
            if a.dim() == 3:  # (b, t, a) -> (b, a) not allowed here
                raise ValueError("predict_reward got (b,p,d) states but (b,t,a) actions.")
            x = torch.cat([z_agg, a], dim=-1)  # (b, d+a)
            out = self.reward(x)  # (b, 1)
            return out

        else:
            raise ValueError(f"z_bisim must be (b,p,d) or (b,t,p,d), got {z_bisim.shape}")

    def compute_transition_distance(self, next_z_bisim, next_z_bisim2):
        """
        Per-sequence transition distance.
        next_z_bisim:  (b, t, p, d)
        next_z_bisim2: (b, t, p, d)
        Returns: (b,) transition distance
        """
        b, t, p, d = next_z_bisim.shape
        z1_pooled = next_z_bisim.mean(dim=2)   # (b, t, d)
        z2_pooled = next_z_bisim2.mean(dim=2)  # (b, t, d)

        diff = z1_pooled - z2_pooled  # (b, t, d)
        squared_diff = diff.pow(2).sum(dim=-1)  # (b, t)
        # average over time
        distances = squared_diff.mean(dim=-1)  # (b,)
        distances = torch.sqrt(distances + 1e-8)  # (b,)
        return distances

    def pool_latents(self, z_bisim):
        """Mean-pool patch latents: (b, t, p, d) -> (b, t, d)."""
        return z_bisim.mean(dim=2)

    def predict_inverse(self, z_bisim, next_z_bisim):
        """Predict WM actions from pooled bisim latents: (b, t, a_wm)."""
        if self.inverse_dynamics is None:
            raise RuntimeError("inverse_dynamics is not enabled on BisimModel")
        z_pool = self.pool_latents(z_bisim)
        next_z_pool = self.pool_latents(next_z_bisim)
        return self.inverse_dynamics(z_pool, next_z_pool)

    def compute_covariance_regularization(self, z_bisim, next_z_bisim,
                                          var_target: float = 1.0,
                                          eps: float = 1e-6):
        """
        Per-patch covariance regularization.
        Computes a (d x d) covariance matrix per patch and averages across patches.
        This is well-conditioned: N=2*b*t samples for d features (e.g., 240 samples for 32 features).

        Args:
            z_bisim:        (b, t, p, d)
            next_z_bisim:   (b, t, p, d)
            var_target:     target variance for diagonal
            eps:            numerical stability

        Returns:
            cov_reg: (b,) tensor broadcast per batch element
        """
        assert z_bisim.dim() == 4 and next_z_bisim.dim() == 4, \
            f"expected (b,t,p,d); got {z_bisim.shape} and {next_z_bisim.shape}"

        b, t, p, d = z_bisim.shape

        # (2b, t, p, d) -> (2bt, p, d)
        Z_all = torch.cat([z_bisim, next_z_bisim], dim=0)  # (2b, t, p, d)
        Z_flat = Z_all.reshape(-1, p, d)  # (N, p, d) where N = 2*b*t
        N = Z_flat.shape[0]

        # center per-patch: (N, p, d) - mean over N -> (1, p, d)
        Zc = Z_flat - Z_flat.mean(dim=0, keepdim=True)  # (N, p, d)

        # per-patch covariance: (p, d, d) via batched matmul
        # Zc transposed: (p, d, N) @ (p, N, d) -> (p, d, d)
        Zc_t = Zc.permute(1, 2, 0)  # (p, d, N)
        Zc_p = Zc.permute(1, 0, 2)  # (p, N, d)
        denom = max(N - 1, 1)
        C = torch.bmm(Zc_t, Zc_p) / denom  # (p, d, d)
        C = C + eps * torch.eye(d, device=C.device, dtype=C.dtype).unsqueeze(0)  # (p, d, d)

        # diagonal: (p, d)
        diag = torch.diagonal(C, dim1=1, dim2=2)  # (p, d)
        diag_loss = (diag - var_target).pow(2).mean()

        # off-diagonal: ||C||_F^2 - ||diag(C)||_2^2 per patch, averaged
        frob2 = (C * C).sum(dim=(1, 2))  # (p,)
        diag2 = (diag * diag).sum(dim=1)  # (p,)
        offdiag_sum = frob2 - diag2  # (p,)
        offdiag_norm = d * (d - 1)
        offdiag_loss = (offdiag_sum / max(offdiag_norm, 1)).mean()

        cov_reg = offdiag_loss + diag_loss

        return cov_reg.expand(b)

    def var_loss(self, z_bisim, var_target=0.1, epsilon=0):
        """
        Calculate variance loss per-patch.
        Computes variance across batch for each (patch, feature) independently.
        input: z_bisim: (b, t, num_patches, patch_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t,)
        """
        b, t, num_patches, patch_dim = z_bisim.shape

        # variance across batch dim for each (t, patch, feature)
        var = z_bisim.var(dim=0)  # (t, num_patches, patch_dim)
        std = torch.sqrt(var + epsilon)

        # if NaN appears, fallback to using var directly
        nan_mask = torch.isnan(std)
        if nan_mask.any():
            std = var
            print(f"WARNING: NaN or Inf in Variance computation")

        # compute max(0, var_target - std)
        loss = torch.relu(var_target - std)

        # average over patches and patch_dim -> (t,)
        return loss.mean(dim=(1, 2))

    def cal_pca(self, z_bisim):
        # (B, T, num_patches, patch_dim) -> (B*T, num_patches*patch_dim)
        B, T, n_patches, patch_dim = z_bisim.shape
        z_flat = z_bisim.reshape(B, T, n_patches * patch_dim)
        Z = z_flat.reshape(B * T, n_patches * patch_dim)

        Z_centered = Z - Z.mean(dim=0, keepdim=True)

        # PCA via SVD
        U, S, Vt = torch.linalg.svd(Z_centered, full_matrices=False)  # check  Vt or V
        V = Vt.T  # (n_patches*patch_dim, n_patches*patch_dim)

        self.PCAMatrix = V.detach()
        self.PCA_Calced = True

    def pca_var_loss(self, z_bisim, target_first=0.01, target_rest=2.0, num_pcs=10):
        """
        PCA variance loss:
        - 1st PC variance -> target_first
        - Next (num_pcs-1) PCs variance -> target_rest
        - Remaining PCs are unconstrained

        Args:
            z_bisim: Tensor (B, T, num_patches, patch_dim)
            target_first: variance target for the first PC
            target_rest: variance target for PCs 2..num_pcs
            num_pcs: number of PCs to regularize
        Returns:
            scalar loss
        """
        # (B, T, num_patches, patch_dim) -> (B*T, num_patches*patch_dim)
        B, T, n_patches, patch_dim = z_bisim.shape
        z_flat = z_bisim.reshape(B, T, n_patches * patch_dim)
        Z = z_flat.reshape(B * T, n_patches * patch_dim)

        Z_centered = Z - Z.mean(dim=0, keepdim=True)

        if not self.PCA_Calced:
            self.cal_pca(z_bisim)
        V = self.PCAMatrix
        num_pcs = min(num_pcs, V.shape[1])
        V_10 = V[:, :num_pcs]  # (n_patches*patch_dim, num_pcs)

        Z_proj = Z_centered @ V_10  # (B*T, num_pcs)
        var_V10 = torch.zeros(num_pcs, device=Z_proj.device)

        for i in range(num_pcs):
            var_V10[i] = Z_proj[:, i].var(unbiased=True)

        targets = torch.full_like(var_V10, target_rest)
        if num_pcs > 0:
            targets[0] = target_first

        # Loss = error between pc_var and targets
        loss = (torch.abs(var_V10 - targets)).mean()

        return loss

    def calc_var_loss(self, z_bisim, next_z_bisim, var_target=0.1, epsilon=0):

        """
        Calculate variance loss with memory buffer
        input: z_bisim: (b, t, num_patches, patch_dim)
        next_z_bisim: (b, t, num_patches, patch_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t)
        """

        T_Plus_1_z_bisim = torch.cat([z_bisim, next_z_bisim], dim=0)

        # calculate the loss
        loss = self.var_loss(T_Plus_1_z_bisim, var_target, epsilon)

        return loss  # dimension=(T+1), or memory sample+1

    def calc_PCAVar_loss(self, z_bisim, next_z_bisim, target_first=0.01, var_target=0.1, num_pcs=10):

        """
        Calculate PCA variance loss with memory buffer
        input: z_bisim: (b, t, num_patches, patch_dim)
        next_z_bisim: (b, t, num_patches, patch_dim)
        target_first: target variance for first PC
        var_target: target variance for other PCs
        num_pcs: number of principal components to regularize
        output: var_loss: (t)
        """

        T_Plus_1_z_bisim = torch.cat([z_bisim, next_z_bisim], dim=0)

        # calculate the loss
        loss = self.pca_var_loss(T_Plus_1_z_bisim, target_first, var_target, num_pcs)

        return loss  # dimension=(T+1), or memory sample+1

    def calc_bisim_loss(self, z_bisim, z_bisim2, reward, reward2, next_z_bisim, next_z_bisim2, epoch, discount=0.99,
                        train_w_reward_loss=True, var_loss_coef: float = 1.0, PCA1_loss_target: float = 0.01,
                        VC_target: float = 1.0,
                        num_pcs: int = 10, PCAloss_epoch: int = 50,
                        id_lambda: float = 0.0, use_id_target: bool = False):
        """
        Calculate bisimulation loss
        bisimulation metric: d(s1,s2) = |r(s1) - r(s2)| + γ · d(P(s1), P(s2)) + Variance Loss + Covariance Regularization
        input: z_bisim, z_bisim2: (b, t, num_patches, patch_dim)
               reward, reward2: (b, t, 1)
               next_z_bisim, next_z_bisim2: (b, t, num_patches, patch_dim)
        output: bisim_loss: (b, t)
        """
        # 1. compute distance per (b, t) — pool over patches, then distance in d-dim space
        b, t, p, d = z_bisim.shape
        z1_pooled = z_bisim.mean(dim=2)   # (b, t, d)
        z2_pooled = z_bisim2.mean(dim=2)  # (b, t, d)

        z_dist = F.smooth_l1_loss(z1_pooled, z2_pooled, reduction="none").sum(dim=-1)

        # 2. compute reward distance
        r_dist = torch.sum(F.smooth_l1_loss(reward, reward2, reduction="none"), dim=-1)

        # 3. compute transition distance between next states
        transition_dist = self.compute_transition_distance(next_z_bisim, next_z_bisim2)  # (b,)

        if torch.isnan(transition_dist).any() or torch.isinf(transition_dist).any():
            print("WARNING: NaN or Inf values detected in transition_dist!")
            print(f"transition_dist shape: {transition_dist.shape}")
            print(
                f"transition_dist stats: mean={transition_dist.mean().item():.6f}, std={transition_dist.std().item():.6f}, min={transition_dist.min().item():.6f}, max={transition_dist.max().item():.6f}")
            transition_dist = torch.ones_like(transition_dist)

        transition_dist = transition_dist.unsqueeze(1).expand(-1, r_dist.shape[1])  # (b, t)

        # 4. compute variance loss
        if epoch <= PCAloss_epoch:
            var_loss = self.calc_var_loss(z_bisim, next_z_bisim, VC_target, epsilon=1e-4)
        else:
            var_loss = self.calc_PCAVar_loss(z_bisim, next_z_bisim, PCA1_loss_target, VC_target, num_pcs)

        # 5. compute covariance regularization
        cov_reg = self.compute_covariance_regularization(z_bisim, next_z_bisim, var_target=VC_target)
        cov_reg = cov_reg.unsqueeze(1).expand(-1, r_dist.shape[1])  # (b, t)

        var_loss = var_loss * var_loss_coef
        cov_reg = cov_reg * var_loss_coef

        if train_w_reward_loss:
            target_bisimilarity = r_dist + discount * transition_dist
        else:
            target_bisimilarity = 0 * r_dist + discount * transition_dist

        inv_l1 = torch.zeros_like(z_dist)
        if (
            use_id_target
            and id_lambda > 0
            and self.inverse_dynamics is not None
        ):
            with torch.no_grad():
                z1 = z_bisim.detach()
                z2 = z_bisim2.detach()
                nz1 = next_z_bisim.detach()
                nz2 = next_z_bisim2.detach()
                inv_i = self.predict_inverse(z1, nz1)
                inv_j = self.predict_inverse(z2, nz2)
                inv_l1 = (inv_i - inv_j).abs().sum(dim=-1)
            target_bisimilarity = target_bisimilarity + id_lambda * inv_l1

        # 6. final bisim loss
        bisim_loss = (z_dist - target_bisimilarity).pow(2) + var_loss + cov_reg

        return bisim_loss, z_dist, r_dist, discount * transition_dist, var_loss, cov_reg, inv_l1
