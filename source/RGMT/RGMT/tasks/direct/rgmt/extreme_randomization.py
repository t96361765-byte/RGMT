"""Extreme-RGMT Stage-I randomization helpers not provided by Isaac Lab."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def randomize_motor_strength(
    env,
    env_ids: torch.Tensor | None,
    strength_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Scale each environment's simulated joint effort limits.

    Table II specifies a motor-strength scale but does not prescribe
    per-joint sampling. A single scale is sampled per robot, preserving the
    relative torque limits across the G1 joints.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device)
    lower, upper = strength_range
    scales = torch.empty((len(env_ids), 1), device=asset.device).uniform_(lower, upper)
    nominal_limits = asset.data.joint_effort_limits[env_ids].clone()
    asset.write_joint_effort_limit_to_sim(nominal_limits * scales, env_ids=env_ids)
