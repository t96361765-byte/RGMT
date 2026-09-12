"""Apply Stage-I domain randomization only to Stage-II acquisition environments."""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp

from .extreme_randomization import randomize_motor_strength


def _acquisition_ids(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    allowed = env._stage2_acquisition_env_ids
    if env_ids is None:
        return allowed
    requested = torch.as_tensor(env_ids, dtype=torch.long, device=allowed.device)
    return requested[env._stage2_acquisition_mask[requested]]


class randomize_acquisition_rigid_body_material(mdp.randomize_rigid_body_material):
    def __call__(
        self,
        env,
        env_ids,
        static_friction_range,
        dynamic_friction_range,
        restitution_range,
        num_buckets,
        asset_cfg,
        make_consistent=False,
    ) -> None:
        ids = _acquisition_ids(env, env_ids)
        if ids.numel():
            super().__call__(
                env,
                ids,
                static_friction_range,
                dynamic_friction_range,
                restitution_range,
                num_buckets,
                asset_cfg,
                make_consistent,
            )


class randomize_acquisition_rigid_body_mass(mdp.randomize_rigid_body_mass):
    def __call__(
        self,
        env,
        env_ids,
        asset_cfg,
        mass_distribution_params,
        operation,
        distribution="uniform",
        recompute_inertia=True,
        min_mass=1.0e-6,
    ) -> None:
        ids = _acquisition_ids(env, env_ids)
        if ids.numel():
            super().__call__(
                env,
                ids,
                asset_cfg,
                mass_distribution_params,
                operation,
                distribution,
                recompute_inertia,
                min_mass,
            )


def randomize_rigid_body_com(env, env_ids, com_range, asset_cfg) -> None:
    ids = _acquisition_ids(env, env_ids)
    if ids.numel():
        mdp.randomize_rigid_body_com(env, ids, com_range, asset_cfg)


def randomize_acquisition_motor_strength(env, env_ids, strength_range, asset_cfg) -> None:
    ids = _acquisition_ids(env, env_ids)
    if ids.numel():
        randomize_motor_strength(env, ids, strength_range, asset_cfg)


class randomize_acquisition_actuator_gains(mdp.randomize_actuator_gains):
    def __call__(
        self,
        env,
        env_ids,
        asset_cfg,
        stiffness_distribution_params=None,
        damping_distribution_params=None,
        operation="abs",
        distribution="uniform",
    ) -> None:
        ids = _acquisition_ids(env, env_ids)
        if ids.numel():
            super().__call__(
                env,
                ids,
                asset_cfg,
                stiffness_distribution_params,
                damping_distribution_params,
                operation,
                distribution,
            )


class randomize_acquisition_joint_parameters(mdp.randomize_joint_parameters):
    def __call__(
        self,
        env,
        env_ids,
        asset_cfg,
        friction_distribution_params=None,
        armature_distribution_params=None,
        lower_limit_distribution_params=None,
        upper_limit_distribution_params=None,
        operation="abs",
        distribution="uniform",
    ) -> None:
        ids = _acquisition_ids(env, env_ids)
        if ids.numel():
            super().__call__(
                env,
                ids,
                asset_cfg,
                friction_distribution_params,
                armature_distribution_params,
                lower_limit_distribution_params,
                upper_limit_distribution_params,
                operation,
                distribution,
            )


def push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg) -> None:
    ids = _acquisition_ids(env, env_ids)
    if ids.numel():
        mdp.push_by_setting_velocity(env, ids, velocity_range, asset_cfg)
