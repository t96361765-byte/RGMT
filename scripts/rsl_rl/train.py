# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--motion-file",
    type=str,
    default=None,
    help="Override the RGMT motion source with a .pt/.pth file, a BeyondMimic .npz, or an NPZ directory.",
)
parser.add_argument(
    "--mastered-motion-file",
    type=str,
    default=None,
    help="Stage-II mastered/consolidation motion source (NPZ directory, one NPZ, or packed PT).",
)
parser.add_argument(
    "--stage1-checkpoint",
    type=str,
    default=None,
    help="Stage-I checkpoint used to initialize a fresh Extreme-RGMT Stage-II run.",
)
parser.add_argument(
    "--adaptive_sampler_state",
    type=str,
    default=None,
    help="Adaptive motion sampler sidecar to restore when resuming.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from types import MethodType

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import RGMT.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _reset_stage2_exploration_std(runner, std: float = 0.5) -> None:
    """Reset only trainable exploration after fresh Stage-I transfer, not resume."""
    distribution = runner.alg.actor.distribution
    if not 0.0 < std <= getattr(distribution, "max_std", float("inf")):
        raise ValueError(f"Invalid Stage-II exploration standard deviation: {std}")
    with torch.no_grad():
        if distribution.std_type == "scalar":
            distribution.std_param.fill_(std)
        elif distribution.std_type == "log":
            distribution.log_std_param.fill_(torch.log(torch.tensor(std)).item())
        else:
            raise ValueError(f"Unsupported standard deviation type: {distribution.std_type}")
    print(
        f"[INFO]: Fresh Stage-II trainable policy std reset to {std:g}; "
        "Stage-I mean policy and frozen reference preserved."
    )


def _disable_git_code_state_logging(runner) -> None:
    """Skip optional Git snapshots without disabling metrics or checkpoints."""

    def store_code_state(self):
        print("[INFO]: Git code snapshots disabled; metrics and checkpoints remain enabled.")
        return []

    runner.logger._store_code_state = MethodType(store_code_state, runner.logger)


def _adaptive_sidecar_path(checkpoint_path: str | os.PathLike[str]) -> Path:
    checkpoint = Path(checkpoint_path)
    suffix = checkpoint.stem.removeprefix("model_")
    return checkpoint.with_name(f"adaptive_sampler_{suffix}.pt")


def _install_adaptive_checkpoint_saver(runner) -> None:
    """Save adaptive sampler state beside every RSL-RL model checkpoint."""
    motion_dataset = getattr(runner.env.unwrapped, "_motion_dataset", None)
    if motion_dataset is None:
        return
    original_save = runner.save

    def save_with_adaptive_state(_runner, path: str, infos=None) -> None:
        original_save(path, infos)
        state = motion_dataset.adaptive_state_dict()
        state.update(
            {
                "source_checkpoint": Path(path).name,
                "captured_iteration": int(_runner.current_learning_iteration),
                "captured_unix_time": time.time(),
                "optimizer_learning_rate": float(
                    _runner.alg.optimizer.param_groups[0]["lr"]
                ),
                "ppo_learning_rate": float(_runner.alg.learning_rate),
            }
        )
        destination = _adaptive_sidecar_path(path)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, destination)

    runner.save = MethodType(save_with_adaptive_state, runner)


def _restore_adaptive_sampler(runner, state_path: Path) -> None:
    motion_dataset = getattr(runner.env.unwrapped, "_motion_dataset", None)
    if motion_dataset is None:
        raise RuntimeError(
            "An adaptive sampler state was requested, but this environment has no motion dataset."
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Adaptive sampler state must be a dictionary: {state_path}")
    motion_dataset.load_adaptive_state_dict(state)
    print(
        f"[INFO]: Restored {motion_dataset._bin_scores.numel()} adaptive motion bins "
        f"from: {state_path}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.motion_file is not None:
        if not hasattr(env_cfg, "motion_file"):
            raise ValueError(f"Task {args_cli.task} does not support --motion-file.")
        env_cfg.motion_file = os.path.abspath(os.path.expanduser(args_cli.motion_file))
    if args_cli.mastered_motion_file is not None:
        if not hasattr(env_cfg, "mastered_motion_file"):
            raise ValueError(f"Task {args_cli.task} does not support --mastered-motion-file.")
        env_cfg.mastered_motion_file = os.path.abspath(
            os.path.expanduser(args_cli.mastered_motion_file)
        )
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if args_cli.headless and not args_cli.video:
        from physics_only_usd import configure_headless_training_assets

        configure_headless_training_assets(env_cfg, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    _install_adaptive_checkpoint_saver(runner)
    _disable_git_code_state_logging(runner)
    is_stage2 = hasattr(runner.env.unwrapped, "get_stage2_transition_metadata")
    if is_stage2 and agent_cfg.resume and args_cli.stage1_checkpoint is not None:
        raise ValueError(
            "Use --stage1-checkpoint only for a fresh Stage-II run; use --resume for a Stage-II checkpoint."
        )
    if is_stage2 and not agent_cfg.resume:
        if args_cli.stage1_checkpoint is None:
            raise ValueError(
                "A fresh Extreme-RGMT Stage-II run requires --stage1-checkpoint so both "
                "the trainable policy and frozen reference policy start from Stage I."
            )
        stage1_checkpoint = Path(args_cli.stage1_checkpoint).expanduser().resolve()
        if not stage1_checkpoint.is_file():
            raise FileNotFoundError(f"Stage-I checkpoint not found: {stage1_checkpoint}")
        print(f"[INFO]: Initializing Stage II from Stage-I checkpoint: {stage1_checkpoint}")
        runner.load(
            str(stage1_checkpoint),
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "reference": False,
                "rnd": False,
            },
        )
        # The frozen reference has already been copied by PACEStarPPO.load().
        # Reset exploration before collecting any rollout; never do this on resume.
        _reset_stage2_exploration_std(runner)
        print(
            "[INFO]: Stage-I actor copied to frozen reference policy; "
            f"Stage-II optimizer learning rate: {runner.alg.learning_rate:.8g}"
        )
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        # RSL-RL restores the optimizer state but not its separate adaptive-LR
        # scalar.  Synchronize both before the first PPO update so a mature
        # policy does not jump back to the configured initial learning rate.
        if hasattr(runner.alg, "optimizer") and hasattr(runner.alg, "learning_rate"):
            restored_learning_rate = float(runner.alg.optimizer.param_groups[0]["lr"])
            runner.alg.learning_rate = restored_learning_rate
            print(f"[INFO]: Restored PPO learning rate: {restored_learning_rate:.8g}")

        if args_cli.adaptive_sampler_state is not None:
            adaptive_state_path = Path(args_cli.adaptive_sampler_state).expanduser().resolve()
            if not adaptive_state_path.is_file():
                raise FileNotFoundError(
                    f"Adaptive sampler state not found: {adaptive_state_path}"
                )
        else:
            adaptive_state_path = _adaptive_sidecar_path(resume_path)
        if adaptive_state_path.is_file():
            try:
                _restore_adaptive_sampler(runner, adaptive_state_path)
            except ValueError:
                # A policy can be incrementally trained on a new motion set,
                # but adaptive bin scores are dataset-specific. Keep strict
                # behavior for an explicitly requested sidecar; only the
                # automatically discovered old sidecar may be skipped.
                if args_cli.adaptive_sampler_state is not None:
                    raise
                logger.warning(
                    "Ignoring adaptive sampler sidecar %s because it does not match "
                    "the current motion dataset; new bin scores start from zero.",
                    adaptive_state_path,
                )
        else:
            logger.warning(
                "No adaptive sampler sidecar was found for %s; bin scores will start from zero.",
                resume_path,
            )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=not agent_cfg.resume,
    )

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
