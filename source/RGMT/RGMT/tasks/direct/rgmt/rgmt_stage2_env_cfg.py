"""Configuration for Extreme-RGMT Stage-II PACE/STAR training."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rgmt_env_cfg import ExtremeRGMTG1EnvCfg, ExtremeRGMTG1TheShyEnvCfg
from .stage2_randomization import (
    push_by_setting_velocity,
    randomize_acquisition_motor_strength,
    randomize_acquisition_actuator_gains,
    randomize_acquisition_joint_parameters,
    randomize_acquisition_rigid_body_mass,
    randomize_acquisition_rigid_body_material,
    randomize_rigid_body_com,
)


_THESHY_STAGE1_CFG = ExtremeRGMTG1TheShyEnvCfg()


@configclass
class ExtremeRGMTStage2EventCfg:
    """Table-II randomization restricted to acquisition environments."""

    ground_friction = EventTerm(
        func=randomize_acquisition_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.10, 1.75),
            "dynamic_friction_range": (0.10, 1.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    added_base_mass = EventTerm(
        func=randomize_acquisition_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-3.0, 6.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    base_com_offset = EventTerm(
        func=randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {
                "x": (-0.025, 0.025),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
        },
    )
    motor_strength = EventTerm(
        func=randomize_acquisition_motor_strength,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "strength_range": (0.8, 1.2),
        },
    )
    pd_gain_scale = EventTerm(
        func=randomize_acquisition_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_armature_scale = EventTerm(
        func=randomize_acquisition_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (1.0, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    external_push = EventTerm(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        },
    )


@configclass
class ExtremeRGMTStage2G1EnvCfg(ExtremeRGMTG1EnvCfg):
    """Stage-II environment split from Eq. (8) with acquisition fraction 0.8."""

    motion_file: str = r"D:\track_dataset\rgmt_stage2_npz"
    mastered_motion_file: str = r"D:\track_dataset\lafan1_npz"
    acquisition_fraction: float = 0.8
    # Set only by play.py for one-environment policy evaluation. Stage-II training keeps this
    # disabled because PACE/STAR requires simultaneous acquisition and consolidation roles.
    allow_single_env_playback: bool = False
    events: ExtremeRGMTStage2EventCfg = ExtremeRGMTStage2EventCfg()


@configclass
class ExtremeRGMTStage2G1TheShyEnvCfg(ExtremeRGMTStage2G1EnvCfg):
    """Stage-II TheShy task with mushroom only in acquisition environments."""

    robot = _THESHY_STAGE1_CFG.robot
    mushroom_spawn = _THESHY_STAGE1_CFG.mushroom_spawn
    mushroom_position = _THESHY_STAGE1_CFG.mushroom_position
    mushroom_physics_material = _THESHY_STAGE1_CFG.mushroom_physics_material
    mushroom_acquisition_only = True

    def __post_init__(self):
        # Mushroom contacts require more GPU contact patches than the default.
        self.sim.physx.gpu_max_rigid_patch_count = 2**19


@configclass
class ExtremeRGMTStage2G1DebugEnvCfg(ExtremeRGMTStage2G1EnvCfg):
    """Small Stage-II environment for integration smoke tests."""

    def __post_init__(self):
        self.scene.num_envs = 16
