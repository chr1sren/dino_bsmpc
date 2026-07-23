import os
import gym
import json
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
try:
    import submitit
except ImportError:  # local planning does not need SLURM/submitit
    submitit = None
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
    "bisim_model",
]


def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)


def launch_plan_jobs(
        epoch,
        cfg_dicts,
        plan_output_dir,
):
    if submitit is None:
        raise ImportError(
            "submitit is required for launch_plan_jobs (SLURM sweeps). "
            "For local planning, run: python plan.py --config-name plan_maniskill.yaml ..."
        )
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
        plan_cfg_path="",
        ckpt_base_path="",
        model_name="",
        model_epoch="final",
        planner=["gd", "cem"],
        goal_source=["dset"],
        goal_H=[1, 5, 10],
        alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
            self,
            cfg_dict: dict,
            wm: torch.nn.Module,
            dset,
            env: SubprocVectorEnv,
            env_name: str,
            frameskip: int,
            wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
            success_mode=self.cfg_dict.get("success_mode", "place_site"),
            success_thresh=float(self.cfg_dict.get("success_thresh", 0.025)),
        )

        if self.wandb_run is None or isinstance(
                self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []

        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env":  # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            # Use whatever the env actually restored (ManiSkill cannot teleport via 14-d state).
            self.state_0 = state_0
            self.state_g = state_g
            self.gt_actions = None
            self._log_plan_target_stats()
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            # CRITICAL: use the *actual* sim states after prepare/rollout, not the
            # dataset vectors (ManiSkill prepare used to lie by overwriting state).
            self.state_0 = rollout_states[:, 0]
            self.state_g = rollout_states[:, -1]
            self.gt_actions = wm_actions
            self._log_plan_target_stats()

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # ManiSkill env_info only stores episode-start sim_state. Mid-traj offsets
        # require replaying actions from t=0 (warmup). Default: start at offset 0.
        dset_start_only = bool(self.cfg_dict.get("dset_start_only", False))
        if self.env_name in ("pickcube", "pushcube"):
            dset_start_only = bool(self.cfg_dict.get("dset_start_only", True))

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            act = act.numpy() if hasattr(act, "numpy") else np.asarray(act)
            if dset_start_only:
                offset = 0
            else:
                offset = random.randint(0, max_offset)

            # Build env_info with optional warmup to reach mid-trajectory.
            e_info = dict(e_info) if e_info is not None else {}
            if offset > 0:
                # act is normalized in dset; denormalize for env.step
                warmup_norm = torch.as_tensor(act[:offset], dtype=torch.float32)
                warmup = self.data_preprocessor.denormalize_actions(warmup_norm)
                e_info["warmup_actions"] = warmup.numpy()
            else:
                e_info["warmup_actions"] = None

            obs = {
                key: arr[offset: offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset: offset + traj_len]
            act_seg = act[offset: offset + self.frameskip * self.goal_H]
            actions.append(torch.as_tensor(act_seg, dtype=torch.float32))
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def _log_plan_target_stats(self):
        """Print init/goal place_err so we catch empty goals / restore bugs early."""
        if self.state_0 is None or self.state_g is None:
            return
        if self.state_0.shape[-1] < 13:
            return
        pe0 = np.linalg.norm(self.state_0[:, 7:10] - self.state_0[:, 10:13], axis=-1)
        peg = np.linalg.norm(self.state_g[:, 7:10] - self.state_g[:, 10:13], axis=-1)
        cube = np.linalg.norm(self.state_g[:, 7:10] - self.state_0[:, 7:10], axis=-1)
        tcp = np.linalg.norm(self.state_g[:, :3] - self.state_0[:, :3], axis=-1)
        print(
            f"[plan targets] place_err init={float(pe0.mean()):.4f} "
            f"goal={float(peg.mean()):.4f} (Δ={float((peg - pe0).mean()):.4f}) | "
            f"cube_disp={float(cube.mean()):.4f} tcp_disp={float(tcp.mean()):.4f}"
        )
        if float(peg.mean()) > float(pe0.mean()) - 1e-3:
            print(
                "[plan targets] WARNING: goal segment does not reduce place_err. "
                "Increase goal_H or check demo quality / state restore."
            )
        thresh = float(self.cfg_dict.get("success_thresh", 0.025))
        mode = self.cfg_dict.get("success_mode", "place_site")
        if mode == "place_site" and float(peg.mean()) >= thresh:
            print(
                f"[plan targets] WARNING: goal place_err={float(peg.mean()):.4f} ≥ "
                f"success_thresh={thresh}. Even perfect CEM matching obs_g cannot "
                f"get place_site success. Use success_mode=goal_state or larger goal_H."
            )

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def _probe_gt_openloop(self):
        """
        Roll out dataset actions with the same WM/env interface used by CEM.
        If this WM error is tiny but CEM planning fails → planner/objective issue.
        If this WM error is already huge → action/obs interface bug (not 'val loss').
        """
        if self.gt_actions is None:
            print("[GT probe] skipped (no gt_actions; goal_source may be random_state)")
            return None
        print("[GT probe] evaluating dataset actions (no CEM) ...")
        actions = self.gt_actions.detach().to(self.device)
        action_len = np.full(actions.shape[0], np.inf)
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions,
            action_len,
            filename="gt_probe",
            save_video=True,
        )
        print(
            f"[GT probe] success={float(np.mean(np.asarray(successes).astype(float))):.3f} | "
            f"place_err={logs.get('mean_place_err', float('nan')):.4f} | "
            f"cube_disp={logs.get('cube_disp', float('nan')):.4f} | "
            f"wm_vis_mse={logs.get('wm_openloop_visual_mse', float('nan')):.4f} | "
            f"wm_pro_mse={logs.get('wm_openloop_proprio_mse', float('nan')):.4f} | "
            f"hint={logs.get('diag_hint', '?')}"
        )
        logs = {f"gt_probe/{k}": v for k, v in logs.items()}
        logs["step"] = 0
        self.wandb_run.log(logs)
        with open(self.log_filename, "a") as file:
            file.write(json.dumps({k: (float(v) if hasattr(v, "item") else v) for k, v in logs.items()}, default=str) + "\n")
        return logs

    def perform_planning(self):
        # Always separate "is the WM/env interface OK on demo actions?" from CEM.
        self._probe_gt_openloop()

        if self.debug_dset_init:
            # IMPORTANT: previously this only *initialized* CEM with GT; CEM still
            # optimized and overwrote the demo. True oracle = skip planner entirely.
            if self.gt_actions is None:
                raise ValueError("debug_dset_init=True requires goal_source=dset (gt_actions).")
            print(
                "[plan] debug_dset_init=True → GT-only oracle "
                "(skipping CEM/MPC entirely)"
            )
            actions = self.gt_actions.detach().to(self.device)
            action_len = np.full(actions.shape[0], np.inf)
        else:
            cem_init = str(self.cfg_dict.get("cem_init", "zero")).lower()
            actions_init = None
            if cem_init in ("gt", "demo", "warmstart") and self.gt_actions is not None:
                actions_init = self.gt_actions.detach().to(self.device)
                print(
                    f"[plan] cem_init={cem_init} → warm-start first CEM from dataset actions"
                )
            elif cem_init in ("gt", "demo", "warmstart"):
                print(f"[plan] cem_init={cem_init} requested but gt_actions is None; using zero init")
            actions, action_len = self.planner.plan(
                obs_0=self.obs_0,
                obs_g=self.obs_g,
                actions=actions_init,
            )

        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    has_bisim = train_cfg.get('has_bisim', False)
    if not has_bisim:
        result["bisim_model"] = None

    model_kwargs = {
        "encoder": result["encoder"],
        "proprio_encoder": result["proprio_encoder"],
        "action_encoder": result["action_encoder"],
        "predictor": result["predictor"],
        "decoder": result["decoder"],
        "proprio_dim": train_cfg.proprio_emb_dim,
        "action_dim": train_cfg.action_emb_dim,
        "concat_dim": train_cfg.concat_dim,
        "num_action_repeat": num_action_repeat,
        "num_proprio_repeat": train_cfg.num_proprio_repeat,
    }

    if has_bisim:
        model_kwargs.update({
            "bisim_model": result.get("bisim_model"),
            "bisim_latent_dim": train_cfg.get('bisim_latent_dim', 64),
            "bisim_hidden_dim": train_cfg.get('bisim_hidden_dim', 256),
            "bisim_coef": train_cfg.get('bisim_coef', 1.0),
            "var_loss_coef": train_cfg.get('var_loss_coef', 1.0),
            "PCA1_loss_target": train_cfg.get('PCA1_loss_target', 0.01),
            "VC_target": train_cfg.get('VC_target', 1.0),
            "num_pcs": train_cfg.get('num_pcs', 10),
            "PCAloss_epoch": train_cfg.get('PCAloss_epoch', 50),
            "train_bisim": train_cfg.model.get('train_bisim', True),
            "bypass_dinov2": train_cfg.model.get('bypass_dinov2', False),
            "bisim_memory_buffer_size": train_cfg.get('bisim_memory_buffer_size', 0),
            "bisim_comparison_size": train_cfg.get('bisim_comparison_size', 20),
            "train_bisim_id_id": train_cfg.model.get('train_bisim_id_id', False),
            "id_lambda": train_cfg.get('id_lambda', 0.0),
            "id_omega": train_cfg.get('id_omega', 0.0),
        })

    model = hydra.utils.instantiate(
        train_cfg.model,
        **model_kwargs
    )
    model.to(device)

    if has_bisim:
        print(f"[PLAN] Model loaded with bisimulation enabled (bisim_patch_dim={model.bisim_patch_dim})")

    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def _resolve_ckpt_base_path(ckpt_base_path):
    """
    Resolve checkpoint root. Hydra chdirs into plan_outputs/... so a relative
    ``ckpt_base_path=.`` must be interpreted against the *launch* cwd, not the
    run dir (otherwise ``./outputs/.../hydra.yaml`` is missing).
    """
    path = os.path.expanduser(str(ckpt_base_path))
    if os.path.isabs(path):
        return path
    try:
        orig = hydra.utils.get_original_cwd()
    except Exception:
        orig = os.getcwd()
    return os.path.abspath(os.path.join(orig, path))


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = _resolve_ckpt_base_path(cfg_dict["ckpt_base_path"])
    cfg_dict["ckpt_base_path"] = ckpt_base_path
    model_path = os.path.join(ckpt_base_path, "outputs", cfg_dict["model_name"])
    hydra_yaml = os.path.join(model_path, "hydra.yaml")
    if not os.path.isfile(hydra_yaml):
        raise FileNotFoundError(
            f"Missing training config: {hydra_yaml}\n"
            f"  ckpt_base_path={ckpt_base_path}\n"
            f"  Pass an absolute path, e.g. "
            f"ckpt_base_path=/home/medcvr/Documents/chris/dino_bsmpc"
        )
    with open(hydra_yaml, "r") as f:
        model_cfg = OmegaConf.load(f)

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
            Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    env_kwargs = dict(model_cfg.env.kwargs) if model_cfg.env.kwargs else {}
    if model_cfg.env.name == "point_maze" and "point_maze_env" in cfg_dict:
        background = cfg_dict.get("point_maze_env", {}).get("background")
        if background:
            env_kwargs["background"] = background

    # MPC can exceed the training episode length; avoid early truncation.
    if model_cfg.env.name in ("pickcube", "pushcube"):
        goal_H = int(cfg_dict.get("goal_H", 5))
        planner_cfg = cfg_dict.get("planner", {}) or {}
        max_iter = planner_cfg.get("max_iter", 10)
        max_iter = 10 if max_iter is None else int(max_iter)
        n_taken = int(planner_cfg.get("n_taken_actions", goal_H) or goal_H)
        frameskip = int(model_cfg.frameskip)
        need_steps = max_iter * n_taken * frameskip + 10
        env_kwargs["max_episode_steps"] = max(
            int(env_kwargs.get("max_episode_steps", 50)), need_steps
        )
        print(
            f"[plan] ManiSkill max_episode_steps={env_kwargs['max_episode_steps']} "
            f"(MPC budget ~{max_iter * n_taken * frameskip} ctrl steps)"
        )

    # use serial vector env for wall, deformable, and ManiSkill (SAPIEN/Vulkan)
    if model_cfg.env.name in ("wall", "deformable_env", "pickcube", "pushcube"):
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **env_kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **env_kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    # Honor WANDB_MODE=disabled for local debug runs.
    wandb_mode = os.environ.get("WANDB_MODE", "").lower()
    cfg_dict["wandb_logging"] = wandb_mode not in ("disabled", "offline_dry_run")
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
