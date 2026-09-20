# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--reference-robot",
    action="store_true",
    default=False,
    help="Show the Extreme-RGMT reference motion as a collision-free highlighted robot.",
)
parser.add_argument(
    "--reference-offset-y",
    type=float,
    default=None,
    help="Reference robot lateral offset in meters; use 0 for a transparent overlay.",
)
parser.add_argument(
    "--motion-file",
    type=str,
    default=None,
    help="Override the motion source with a .pt/.pth file, a BeyondMimic .npz, or an NPZ directory.",
)
parser.add_argument(
    "--motion-clip-id",
    type=int,
    default=None,
    help="Replay only this zero-based clip ID from the selected motion dataset.",
)
parser.add_argument(
    "--motion-start-time",
    type=float,
    default=None,
    help="Start time in seconds for --motion-clip-id (default: 0).",
)
parser.add_argument(
    "--full-motions",
    action="store_true",
    default=False,
    help=(
        "Play clips continuously from their first frame. Environments are assigned different clips, "
        "advance after a complete clip, and retry the same clip after an early failure."
    ),
)
parser.add_argument(
    "--disable-early-termination",
    action="store_true",
    default=False,
    help=(
        "Disable training-time tracking terminations during play (root orientation, root height, "
        "and key-body height errors). Motion-end and episode timeouts remain enabled."
    ),
)
parser.add_argument(
    "--record-torques",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Record 50 Hz unclipped PD target torque to CSV and stop at the first episode end. "
        "Enabled by default for single-environment Extreme-RGMT-Stage2-G1-TheShy-Mushroom play. "
        "Use --no-record-torques to retain continuous playback."
    ),
)
parser.add_argument(
    "--torque-output-dir",
    type=str,
    default=None,
    help="Torque CSV directory (default: this repository's logs/stage2).",
)
parser.add_argument(
    "--torque-duration",
    type=float,
    default=None,
    help="Optional recording duration in simulation seconds, e.g. a known full-circle period.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.torque_duration is not None and (
    not math.isfinite(args_cli.torque_duration) or args_cli.torque_duration <= 0
):
    parser.error("--torque-duration must be a finite positive number of seconds.")
if args_cli.record_torques is False and (
    args_cli.torque_duration is not None or args_cli.torque_output_dir is not None
):
    parser.error("Torque output/duration options cannot be combined with --no-record-torques.")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time
from pathlib import Path

from torque_recorder import TorqueCsvRecorder

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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RGMT.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    record_torques = args_cli.record_torques
    if record_torques is None:
        record_torques = (
            task_name == "Extreme-RGMT-Stage2-G1-TheShy-Mushroom" and env_cfg.scene.num_envs == 1
        ) or args_cli.torque_duration is not None or args_cli.torque_output_dir is not None
    if record_torques and (
        env_cfg.scene.num_envs != 1 or not isinstance(env_cfg, DirectRLEnvCfg)
        or not hasattr(env_cfg, "motion_file")
        or not math.isclose(env_cfg.sim.dt * env_cfg.decimation, 0.02, rel_tol=1e-6)
    ):
        raise ValueError("Torque recording requires a single RGMT robot with a 50 Hz policy step.")
    if env_cfg.scene.num_envs == 1 and hasattr(env_cfg, "allow_single_env_playback"):
        env_cfg.allow_single_env_playback = True
        print("[INFO]: Stage-II single-environment playback enabled (acquisition role).")
    if args_cli.reference_robot:
        if not hasattr(env_cfg, "visualize_reference_robot"):
            raise ValueError(f"Task {args_cli.task} does not support --reference-robot.")
        env_cfg.visualize_reference_robot = True
    if args_cli.reference_offset_y is not None:
        if not hasattr(env_cfg, "reference_robot_offset"):
            raise ValueError(f"Task {args_cli.task} does not support --reference-offset-y.")
        env_cfg.reference_robot_offset = (0.0, args_cli.reference_offset_y, 0.0)
    if args_cli.motion_file is not None:
        if not hasattr(env_cfg, "motion_file"):
            raise ValueError(f"Task {args_cli.task} does not support --motion-file.")
        env_cfg.motion_file = os.path.abspath(os.path.expanduser(args_cli.motion_file))
    if args_cli.motion_clip_id is not None:
        if not hasattr(env_cfg, "fixed_motion_clip_id"):
            raise ValueError(f"Task {args_cli.task} does not support --motion-clip-id.")
        if args_cli.motion_clip_id < 0:
            raise ValueError("--motion-clip-id must be non-negative.")
        env_cfg.fixed_motion_clip_id = args_cli.motion_clip_id
        env_cfg.fixed_motion_start_time_s = (
            0.0 if args_cli.motion_start_time is None else args_cli.motion_start_time
        )
    elif args_cli.motion_start_time is not None:
        raise ValueError("--motion-start-time requires --motion-clip-id.")
    if args_cli.motion_start_time is not None and args_cli.motion_start_time < 0.0:
        raise ValueError("--motion-start-time must be non-negative.")
    if args_cli.full_motions:
        if not hasattr(env_cfg, "replay_full_motions"):
            raise ValueError(f"Task {args_cli.task} does not support --full-motions.")
        env_cfg.replay_full_motions = True
        if record_torques:
            env_cfg.play_to_motion_end = True
        if args_cli.motion_clip_id is not None:
            env_cfg.fixed_motion_start_time_s = 0.0
    if args_cli.disable_early_termination:
        if not hasattr(env_cfg, "disable_early_termination"):
            raise ValueError(f"Task {args_cli.task} does not support --disable-early-termination.")
        env_cfg.disable_early_termination = True

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # convert pre-5.0 published checkpoints to the layout expected by rsl-rl >= 5.0 (no-op otherwise)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    recorder = None
    stop_reason = "simulation window closed"
    try:
        if record_torques:
            output_dir = args_cli.torque_output_dir or Path(__file__).resolve().parents[2] / "logs" / "stage2"
            recorder = TorqueCsvRecorder(env.unwrapped.robot, output_dir)
            print(f"[INFO]: Recording unclipped PD torque estimates (N m), 50 Hz: {recorder.path}")
            print("[INFO]: Sampling the first physics substep; stopping at the first episode end or Ctrl+C.")
            if args_cli.full_motions:
                print("[INFO]: Recording through the actual clip end; future reference queries clamp to its last frame.")
            print("[INFO]: One episode/clip is recorded; this does not detect individual full circles.")
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                if recorder is not None:
                    recorder.begin_step(
                        time_s=timestep * dt,
                        motion_clip_id=int(env.unwrapped._clip_ids[0].item()),
                        motion_time_s=float(env.unwrapped._motion_times()[0].item()),
                    )
                obs, _, dones, _ = env.step(actions)
                if recorder is not None:
                    recorder.end_step()
                    if bool(dones[0].item()):
                        stop_reason = "first episode ended (motion end, timeout, or termination)"
                        break
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)
            timestep += 1
            if recorder is not None and args_cli.torque_duration is not None:
                if timestep * dt >= args_cli.torque_duration - 1e-9:
                    stop_reason = "requested torque recording duration reached"
                    break
            if args_cli.video and timestep >= args_cli.video_length:
                stop_reason = "video length reached"
                break
            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"
        print("\n[INFO]: Playback interrupted; keeping recorded torque samples.")
    except BaseException:
        stop_reason = "playback error (partial recording)"
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
                print(f"[INFO]: Saved {recorder.rows_written} torque samples to {recorder.path} ({stop_reason}).")
        finally:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
