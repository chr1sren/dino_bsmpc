import os
import json
import torch
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils


class PlanEvaluator:  # evaluator for planning
    def __init__(
            self,
            obs_0,
            obs_g,
            state_0,
            state_g,
            env,
            wm,
            frameskip,
            seed,
            preprocessor,
            n_plot_samples,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device
        self.plot_full = False  # plot all frames or frames after frameskip

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]):] = 0
        return result

    def eval_actions(
            self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        exec_actions = rearrange(
            actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics (+ WM vs env diagnostics)
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
            e_obses=e_obses,
            e_states=e_states,
            i_z_obses=i_z_obses,
            exec_actions=exec_actions,
        )

        if save_video:
            # Always dump real-env videos (no decoder required).
            self._save_env_only_video(
                e_visuals=e_visuals,
                successes=successes,
                filename=filename,
            )
            self._dump_diag_json(logs, filename=filename)

        # optional decoder compare plot (only if decoder exists)
        if self.wm.decoder is not None:
            if hasattr(self.wm, 'has_bisim') and self.wm.has_bisim:
                max_action_len = int(action_len.max()) if isinstance(action_len, np.ndarray) else int(action_len)
                b = e_visuals.shape[0]
                h, w, c = e_visuals.shape[2], e_visuals.shape[3], e_visuals.shape[4]
                i_visuals = torch.zeros(b, max_action_len + 1, h, w, c, device=self.device, dtype=torch.float32)
            else:
                i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
                i_visuals = self._mask_traj(
                    i_visuals, action_len + 1
                )
            e_vis_t = self.preprocessor.transform_obs_visual(e_visuals)
            e_vis_t = self._mask_traj(e_vis_t, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_vis_t,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes, e_obses, e_states

    def _encode_obs_np(self, obs_np):
        """Encode a numpy obs dict {'visual': (b,t,...), 'proprio': (b,t,...)} → latent dict on device."""
        obs_t = move_to_device(self.preprocessor.transform_obs(obs_np), self.device)
        with torch.no_grad():
            z = self.wm.encode_obs(obs_t)
            if hasattr(self.wm, "has_bisim") and self.wm.has_bisim:
                if hasattr(self.wm, "bypass_dinov2") and self.wm.bypass_dinov2:
                    z["visual"] = self.wm.encode_bisim(obs_t)
                else:
                    z["visual"] = self.wm.encode_bisim(z)
        return z

    def _compute_openloop_wm_errors(self, e_obses, i_z_obses):
        """
        Align env observations (every frameskip) with WM imagined latents and
        report MSE. High values ⇒ world-model open-loop fidelity problem.
        """
        T_env = e_obses["visual"].shape[1]
        idxs = list(range(0, T_env, self.frameskip))
        # WM rollout returns init frames + predicted steps; match lengths
        T_wm = i_z_obses["visual"].shape[1]
        n = min(len(idxs), T_wm)
        idxs = idxs[:n]

        obs_sub = {
            "visual": e_obses["visual"][:, idxs],
            "proprio": e_obses["proprio"][:, idxs],
        }
        z_real = self._encode_obs_np(obs_sub)

        vis_i = i_z_obses["visual"][:, :n]
        pro_i = i_z_obses["proprio"][:, :n]
        vis_e = z_real["visual"][:, :n]
        pro_e = z_real["proprio"][:, :n]

        # flatten non-batch dims
        vis_mse = torch.mean((vis_i - vis_e) ** 2).item()
        pro_mse = torch.mean((pro_i - pro_e) ** 2).item()
        vis_l2 = torch.norm((vis_i - vis_e).reshape(vis_i.shape[0], -1), dim=-1).mean().item()
        pro_l2 = torch.norm((pro_i - pro_e).reshape(pro_i.shape[0], -1), dim=-1).mean().item()
        return {
            "wm_openloop_visual_mse": vis_mse,
            "wm_openloop_proprio_mse": pro_mse,
            "wm_openloop_visual_l2": vis_l2,
            "wm_openloop_proprio_l2": pro_l2,
            "wm_openloop_aligned_T": float(n),
        }

    def _state_progress_metrics(self, e_states):
        """Cube / TCP movement over the env rollout (ManiSkill 14-d state layout)."""
        # e_states: (b, T, d)
        start = e_states[:, 0]
        end = e_states[:, -1]
        tcp0, tcp1 = start[:, :3], end[:, :3]
        obj0, obj1 = start[:, 7:10], end[:, 7:10]
        goal = end[:, 10:13]  # goal site in current episode

        place0 = np.linalg.norm(obj0 - start[:, 10:13], axis=-1)
        place1 = np.linalg.norm(obj1 - goal, axis=-1)
        cube_disp = np.linalg.norm(obj1 - obj0, axis=-1)
        tcp_disp = np.linalg.norm(tcp1 - tcp0, axis=-1)

        return {
            "place_err_start": float(np.mean(place0)),
            "place_err_end": float(np.mean(place1)),
            "place_err_delta": float(np.mean(place1 - place0)),  # negative = improved
            "cube_disp": float(np.mean(cube_disp)),
            "tcp_disp": float(np.mean(tcp_disp)),
        }

    def _diagnose(self, logs):
        """
        Heuristic label for where to look next.
        Returns a short string; also printed.
        """
        wm_vis = logs.get("wm_openloop_visual_mse", None)
        wm_pro = logs.get("wm_openloop_proprio_mse", None)
        place_d = logs.get("place_err_delta", 0.0)
        cube_d = logs.get("cube_disp", 0.0)
        tcp_d = logs.get("tcp_disp", 0.0)
        act = logs.get("action_l2_mean", 0.0)
        sr = logs.get("success_rate", 0.0)

        # thresholds are rough; meant as a triage, not a proof
        hints = []
        if sr >= 0.5:
            hints.append("OK_task_mostly_solved")
        if wm_pro is not None and wm_pro > 0.05:
            hints.append("WM_proprio_openloop_bad")
        if wm_vis is not None and wm_vis > 0.5:
            hints.append("WM_visual_openloop_bad")
        if act < 1e-3:
            hints.append("MPC_actions_near_zero")
        if tcp_d < 0.01 and act > 1e-3:
            hints.append("actions_not_moving_tcp_check_denorm_or_control")
        if tcp_d > 0.05 and cube_d < 0.01:
            hints.append("arm_moves_but_cube_static_contact_or_object_dynamics")
        if place_d > -0.01 and cube_d < 0.02:
            hints.append("no_task_progress")
        if (
            wm_pro is not None and wm_pro < 0.02
            and wm_vis is not None and wm_vis < 0.2
            and place_d > -0.01
        ):
            hints.append("WM_ok_but_MPC_objective_or_horizon_suspect")

        if not hints:
            hints.append("mixed_or_unclear_check_video")
        return "+".join(hints)

    def _compute_rollout_metrics(
        self,
        e_state,
        e_obs,
        i_z_obs,
        e_obses=None,
        e_states=None,
        i_z_obses=None,
        exec_actions=None,
    ):
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(
                value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        print(eval_results)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        e_obs_t = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs_t)
        if hasattr(self.wm, 'has_bisim') and self.wm.has_bisim:
            if hasattr(self.wm, 'bypass_dinov2') and self.wm.bypass_dinov2:
                e_z_obs["visual"] = self.wm.encode_bisim(e_obs_t)
            else:
                e_z_obs["visual"] = self.wm.encode_bisim(e_z_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })

        if exec_actions is not None:
            logs["action_l2_mean"] = float(np.mean(np.linalg.norm(exec_actions, axis=-1)))
            logs["action_abs_mean"] = float(np.mean(np.abs(exec_actions)))

        if e_states is not None:
            logs.update(self._state_progress_metrics(e_states))

        if e_obses is not None and i_z_obses is not None:
            try:
                logs.update(self._compute_openloop_wm_errors(e_obses, i_z_obses))
            except Exception as exc:
                print(f"[diag] open-loop WM metrics failed: {exc}")
                logs["wm_openloop_error"] = str(exc)

        logs["diag_hint"] = self._diagnose(logs)
        print(
            f"[diag] hint={logs['diag_hint']} | "
            f"wm_vis_mse={logs.get('wm_openloop_visual_mse', float('nan')):.4f} "
            f"wm_pro_mse={logs.get('wm_openloop_proprio_mse', float('nan')):.4f} | "
            f"place Δ={logs.get('place_err_delta', float('nan')):.4f} "
            f"cube_disp={logs.get('cube_disp', float('nan')):.4f} "
            f"tcp_disp={logs.get('tcp_disp', float('nan')):.4f} "
            f"act_l2={logs.get('action_l2_mean', float('nan')):.4f}"
        )

        return logs, successes

    def _to_uint8_hwc(self, frame):
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 + 1e-3:
                arr = (arr * 255.0).clip(0, 255)
            else:
                arr = arr.clip(0, 255)
            arr = arr.astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr

    def _save_env_only_video(self, e_visuals, successes, filename="output"):
        """
        Save real-env rollout videos (top: env, bottom/side: goal), no decoder needed.
        Files: {filename}_env_{idx}_{success|failure}.mp4
        """
        n = min(int(self.n_plot_samples), e_visuals.shape[0])
        goal = self.obs_g["visual"]
        for idx in range(n):
            success_tag = "success" if bool(np.asarray(successes)[idx]) else "failure"
            g = goal[idx]
            if g.ndim == 4:  # (1,H,W,C) or (T,H,W,C)
                g = g[0]
            g = self._to_uint8_hwc(g)
            path = f"{filename}_env_{idx}_{success_tag}.mp4"
            writer = imageio.get_writer(path, fps=12)
            T = e_visuals.shape[1]
            for t in range(T):
                frame = self._to_uint8_hwc(e_visuals[idx, t])
                # side-by-side: current | goal
                if frame.shape[0] == g.shape[0] and frame.shape[1] == g.shape[1]:
                    panel = np.concatenate([frame, g], axis=1)
                else:
                    panel = frame
                writer.append_data(panel)
            writer.close()
            print(f"[video] wrote {os.path.abspath(path)}")

    def _dump_diag_json(self, logs, filename="output"):
        path = f"{filename}_diag.json"
        serializable = {}
        for k, v in logs.items():
            if isinstance(v, (np.floating, np.integer)):
                serializable[k] = v.item()
            elif isinstance(v, (float, int, str, bool)) or v is None:
                serializable[k] = v
            else:
                try:
                    serializable[k] = float(v)
                except Exception:
                    serializable[k] = str(v)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"[diag] wrote {os.path.abspath(path)}")

    def _plot_rollout_compare(
            self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
                i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )
