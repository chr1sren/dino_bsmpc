import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
            self,
            horizon,
            topk,
            num_samples,
            var_scale,
            opt_steps,
            eval_every,
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            logging_prefix="plan_0",
            log_filename="logs.json",
            sigma_min=0.05,
            sigma_decay=0.95,
            action_clip=3.0,
            smooth_coef=0.0,
            **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.sigma_min = float(sigma_min)
        self.sigma_decay = float(sigma_decay)
        self.action_clip = float(action_clip)
        self.smooth_coef = float(smooth_coef)

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        if hasattr(self.wm, 'has_bisim') and self.wm.has_bisim:
            if not hasattr(self, '_bisim_goal_logged'):
                print(f"[CEM] Planning in bisimulation space (goal encoded, patch_dim={self.wm.bisim_patch_dim})")
                self._bisim_goal_logged = True
            if hasattr(self.wm, 'bypass_dinov2') and self.wm.bypass_dinov2:
                z_obs_g["visual"] = self.wm.encode_bisim(trans_obs_g)
            else:
                z_obs_g["visual"] = self.wm.encode_bisim(z_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        best_mu = mu.clone()
        best_loss = torch.full((n_evals,), float("inf"), device=self.device)

        # Reference: imagined objective of the warm-start / GT actions (if any).
        # If CEM later reports lower loss but worse real cube_disp → objective aliasing.
        if actions is not None:
            with torch.no_grad():
                ref_loss = self.objective_fn(
                    self.wm.rollout(obs_0=trans_obs_0, act=mu)[0],
                    z_obs_g,
                )
            ref_mean = float(ref_loss.mean().item())
            print(
                f"[CEM] {self.logging_prefix} imagined loss of init actions: "
                f"{ref_mean:.4f} (compare to CEM elite loss below)"
            )
            self.wandb_run.log(
                {f"{self.logging_prefix}/init_action_imagined_loss": ref_mean, "step": 0}
            )
            best_loss = ref_loss.detach().clone()
            # Tight exploration around a known-good init (e.g. dataset actions).
            sigma = torch.clamp(sigma * 0.25, min=self.sigma_min)

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                        torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                            self.device
                        )
                        * sigma[traj]
                        + mu[traj]
                )
                action[0] = mu[traj]  # keep current mean as a candidate
                if self.action_clip is not None and self.action_clip > 0:
                    action = torch.clamp(action, -self.action_clip, self.action_clip)
                with torch.no_grad():
                    rollout_result = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )
                    i_z_obses, i_zs = rollout_result

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                if self.smooth_coef > 0 and action.shape[1] > 1:
                    smooth = ((action[:, 1:] - action[:, :-1]) ** 2).mean(dim=(1, 2))
                    loss = loss + self.smooth_coef * smooth
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                elite_loss = loss[topk_idx[0]].item()
                losses.append(elite_loss)
                mu[traj] = topk_action.mean(dim=0)
                # std can be 0/NaN if elites collapse — keep a decaying exploration floor
                emp_std = topk_action.std(dim=0, unbiased=False)
                emp_std = torch.nan_to_num(emp_std, nan=self.var_scale)
                floor = max(
                    self.sigma_min,
                    self.var_scale * (self.sigma_decay ** (i + 1)),
                )
                sigma[traj] = torch.clamp(emp_std, min=floor)
                if elite_loss < best_loss[traj].item():
                    best_loss[traj] = elite_loss
                    best_mu[traj] = mu[traj].detach().clone()
            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    best_mu, filename=f"{self.logging_prefix}_output_{i + 1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return best_mu, np.full(n_evals, np.inf)  # all actions are valid
