"""Finite Extreme-RGMT Stage-I environment and RSL-RL model smoke test."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument(
    "--task",
    choices=("Isaac-RGMT-G1-Debug-v0", "Isaac-Extreme-RGMT-Stage2-G1-Debug-v0"),
    default="Isaac-RGMT-G1-Debug-v0",
)
parser.add_argument(
    "--stage",
    choices=(
        "config",
        "env",
        "reset",
        "action_pipeline",
        "step",
        "actor_init",
        "actor_forward",
        "model",
    ),
    default="model",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback

import gymnasium as gym
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from isaaclab_tasks.utils import parse_env_cfg

import RGMT.tasks  # noqa: F401
from RGMT.tasks.direct.rgmt.agents.rgmt_models import ExtremeRGMTModel


def main() -> None:
    if args_cli.stage == "config":
        from RGMT.tasks.direct.rgmt.agents.rsl_rl_ppo_cfg import (
            ExtremeRGMTPPORunnerCfg,
            ExtremeRGMTStage2PPORunnerCfg,
        )

        cfg_type = (
            ExtremeRGMTStage2PPORunnerCfg
            if "Stage2" in args_cli.task
            else ExtremeRGMTPPORunnerCfg
        )
        runner_cfg = cfg_type().to_dict()
        assert runner_cfg["obs_groups"] == {"actor": ["policy"], "critic": ["critic"]}
        assert runner_cfg["actor"]["class_name"].endswith(":ExtremeRGMTModel")
        assert runner_cfg["critic"]["class_name"] == "MLPModel"
        assert runner_cfg["clip_actions"] is None
        if "Stage2" in args_cli.task:
            algorithm = runner_cfg["algorithm"]
            assert algorithm["class_name"].endswith(":PACEStarPPO")
            assert algorithm["pace_lambda_base"] == 0.3
            assert algorithm["pace_kappa"] == 5.0
            assert algorithm["pace_rho_ref"] == 0.6
            assert algorithm["pace_beta"] == 0.99
            assert algorithm["star_topk_ratio"] == 0.05
            assert algorithm["star_resample_ratio"] == 0.25
        print("[EXTREME-RGMT SMOKE TEST PASSED] RSL-RL config serialized", flush=True)
        return
    print("[EXTREME-RGMT SMOKE] creating environment", flush=True)
    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env = gym.make(args_cli.task, cfg=cfg)
    if args_cli.stage == "env":
        print("[EXTREME-RGMT SMOKE TEST PASSED] environment created", flush=True)
        env.close()
        return
    print("[EXTREME-RGMT SMOKE] resetting environment", flush=True)
    observations, _ = env.reset()
    if args_cli.stage == "reset":
        print("[EXTREME-RGMT SMOKE TEST PASSED] environment reset", flush=True)
        env.close()
        return
    if args_cli.stage == "action_pipeline":
        actions = torch.linspace(
            -1.5,
            1.5,
            env.unwrapped.cfg.action_space,
            device=env.unwrapped.device,
        ).repeat(args_cli.num_envs, 1)
        env.unwrapped._pre_physics_step(actions)
        expected_targets = (
            env.unwrapped._current_reference["joint_pos"]
            + env.unwrapped.cfg.action_scale * actions
            + env.unwrapped._motor_zero_offsets
        )
        assert torch.equal(env.unwrapped._actions, actions)
        assert torch.allclose(env.unwrapped._processed_actions, expected_targets)
        print(
            "[EXTREME-RGMT SMOKE TEST PASSED] "
            "unbounded PPO actions reach the joint-residual target unchanged",
            flush=True,
        )
        env.close()
        return
    for _ in range(args_cli.steps):
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        observations, rewards, _terminated, _truncated, _ = env.step(actions)

    assert observations["policy"].shape[-1] == 1728
    assert observations["critic"].shape[-1] == 261
    assert torch.isfinite(observations["policy"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(rewards).all()
    if args_cli.stage == "step":
        print(
            "[EXTREME-RGMT SMOKE TEST PASSED] "
            f"policy={tuple(observations['policy'].shape)}, critic={tuple(observations['critic'].shape)}",
            flush=True,
        )
        env.close()
        return

    tensor_observations = TensorDict(observations, batch_size=[args_cli.num_envs])
    actor = ExtremeRGMTModel(
        tensor_observations,
        {"actor": ["policy"], "critic": ["critic"]},
        "actor",
        29,
        hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": 0.6,
            "std_type": "scalar",
        },
    ).to(env.unwrapped.device)
    if args_cli.stage == "actor_init":
        print("[EXTREME-RGMT SMOKE TEST PASSED] actor initialized", flush=True)
        env.close()
        return
    actor_output = actor(tensor_observations)
    if args_cli.stage == "actor_forward":
        print(f"[EXTREME-RGMT SMOKE TEST PASSED] actions={tuple(actor_output.shape)}", flush=True)
        env.close()
        return
    critic = MLPModel(
        tensor_observations,
        {"actor": ["policy"], "critic": ["critic"]},
        "critic",
        1,
        hidden_dims=[1024, 1024, 512, 512],
        activation="elu",
        obs_normalization=False,
    ).to(env.unwrapped.device)

    critic_output = critic(tensor_observations)
    assert actor_output.shape == (args_cli.num_envs, 29)
    assert critic_output.shape == (args_cli.num_envs, 1)
    assert torch.isfinite(actor_output).all() and torch.isfinite(critic_output).all()

    print(
        "[EXTREME-RGMT SMOKE TEST PASSED] "
        f"policy={tuple(observations['policy'].shape)}, "
        f"critic={tuple(observations['critic'].shape)}, "
        f"actions={tuple(actor_output.shape)}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
