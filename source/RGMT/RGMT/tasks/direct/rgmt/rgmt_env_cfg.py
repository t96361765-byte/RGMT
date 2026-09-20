"""Configuration for the Extreme-RGMT Stage-I G1 DirectRLEnv."""

from __future__ import annotations

from dataclasses import field

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from RGMT import RGMT_DATA_DIR
from RGMT.assets import G1_29DOF_CFG, G1_THESHY_CFG

from .extreme_randomization import randomize_motor_strength


@configclass
class ExtremeRGMTStage1EventCfg:
    """Table II dynamics randomization for Stage I."""

    ground_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
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
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-3.0, 6.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    base_com_offset = EventTerm(
        func=mdp.randomize_rigid_body_com,
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
        func=randomize_motor_strength,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "strength_range": (0.8, 1.2),
        },
    )
    pd_gain_scale = EventTerm(
        func=mdp.randomize_actuator_gains,
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
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (1.0, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    external_push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # The paper publishes the interval but not the push magnitude.
            # This conservative velocity impulse is therefore an explicit
            # implementation assumption rather than a claimed paper value.
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
            },
        },
    )


@configclass
class ExtremeRGMTG1EnvCfg(DirectRLEnvCfg):
    """Extreme-RGMT Stage-I task exposed through the original RGMT task ID.

    Observation layout is fixed and consumed by :class:`ExtremeRGMTModel`:
    ``21 x command(38) + 10 x proprioception(64) + 10 x action(29) = 1728``.
    """

    # control and rollout
    decimation = 4
    episode_length_s = 10.0
    action_space = 29
    observation_space = 1728
    # Critic state from Eq. (4):
    # current clean proprioception(64) + previous action(29) + current command(38)
    # + reference height(1) + 14 current link poses(14 * (position 3 + rotation 6))
    # + base linear velocity(3) = 261.
    state_space = 261

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,
    )
    robot: ArticulationCfg = G1_29DOF_CFG
    # Optional static scene prop. It is spawned in env_0 before environment
    # cloning so every environment receives an identical collision object.
    mushroom_spawn: sim_utils.UsdFileCfg | None = None
    mushroom_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mushroom_physics_material: sim_utils.RigidBodyMaterialCfg | None = None
    mushroom_acquisition_only = False
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )

    # Extreme-RGMT Stage-I temporal layout.
    proprio_dim = 64
    command_dim = 38
    history_steps = 10
    command_window_steps = 21
    action_history_steps = 10
    action_history_dim = 29
    action_scale = 1.0
    observation_noise = True
    gravity_noise_std = 0.05
    angular_velocity_noise_std = 0.20
    joint_position_noise_std = 0.01
    joint_velocity_noise_std = 0.50

    # Accepts the packed RGMT .pt/.pth schema, one BeyondMimic .npz, or a directory
    # containing multiple BeyondMimic G1 NPZ clips. All formats use the same RGMT sampler.
    motion_file: str = str(
        RGMT_DATA_DIR
        / "Motions"
        / "RGMT"
        / "Extreme_RGMT_motion_stage1_50Hz_4to1.pt"
    )
    # Optional deterministic replay controls. Leave the clip ID unset for training.
    fixed_motion_clip_id: int | None = None
    fixed_motion_start_time_s = 0.0
    # Play-only deterministic validation mode. Each environment begins at frame zero, advances
    # after reaching the clip end, and retries its current clip after an early termination.
    replay_full_motions = False
    # Enabled only by full-clip torque recording. The sampler clamps future
    # reference queries at the final frame instead of ending one window early.
    play_to_motion_end = False
    # Play-only diagnostic switch. Training leaves this false so failure-driven
    # termination and adaptive sampling behavior remain unchanged.
    disable_early_termination = False
    debug_static_reference = False
    adaptive_bin_duration_s = 1.0
    adaptive_ema_alpha = 0.001
    adaptive_score_clip = 1.0
    adaptive_uniform_ratio = 0.1
    motor_zero_offset_range = (-0.01, 0.01)
    command_linear_velocity_perturbation = 0.50
    command_angular_velocity_perturbation = 0.52
    command_gravity_perturbation = 0.05
    command_joint_position_perturbation = 0.10

    # Optional play-only visualization. The reference articulation is collision-free and is not
    # used by observations, rewards, terminations, or the policy.
    visualize_reference_robot = False
    reference_robot_offset: tuple[float, float, float] = (0.0, 1.0, 0.0)
    reference_robot_color: tuple[float, float, float] = (0.10, 0.65, 1.0)
    # Transparent articulated meshes can be depth-sorted behind the ground plane in Isaac Sim,
    # which makes the reference appear mirrored below the floor. Use an opaque highlight instead.
    reference_robot_opacity = 1.0

    # Fourteen-link tracking set used by the Extreme-RGMT reward.
    # Fixed joints are preserved by merge_fixed_joints=False.
    key_body_names: list[str] = field(
        default_factory=lambda: [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
    )
    termination_key_body_names: list[str] = field(
        default_factory=lambda: [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
    )
    undesired_contact_body_names: list[str] = field(
        default_factory=lambda: [
            (
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                r"(?!left_toe_link$)(?!right_toe_link$)"
                r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                r"(?!left_rubber_hand$)(?!right_rubber_hand$).+$"
            ),
        ]
    )
    # Table I reward coefficients.
    reward_scales: dict[str, float] = field(
        default_factory=lambda: {
            "global_anchor_orientation": 0.5,
            "relative_body_position": 1.0,
            "relative_body_orientation": 1.0,
            "global_body_linear_velocity": 1.0,
            "global_body_angular_velocity": 1.0,
            "action_rate": -0.1,
            "joint_limits": -10.0,
            "undesired_contact": -0.1,
            "feet_slip": -0.1,
        }
    )
    # Table I publishes reward weights but not exponential-kernel widths.
    tracking_sigmas: dict[str, float] = field(
        default_factory=lambda: {
            "global_anchor_orientation": 0.4,
            "relative_body_position": 0.3,
            "relative_body_orientation": 0.4,
            "global_body_linear_velocity": 1.0,
            "global_body_angular_velocity": 3.14,
        }
    )
    termination_root_height_tolerance = 0.25
    termination_root_orientation_error = 1.0
    termination_key_body_height_tolerance = 0.25
    undesired_contact_force_threshold = 1.0
    # Keep the existing foot-contact gate for the Extreme-RGMT feet-slip term.
    foot_contact_force_threshold = 80.0
    foot_body_names: list[str] = field(
        default_factory=lambda: [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
    )
    events: ExtremeRGMTStage1EventCfg = ExtremeRGMTStage1EventCfg()


@configclass
class ExtremeRGMTG1TheShyEnvCfg(ExtremeRGMTG1EnvCfg):
    """Stage-I task using the validated pan-hand model without its fixed hands."""

    robot: ArticulationCfg = G1_THESHY_CFG
    # Native bounds are centered on XY and z=[-0.276493043, 0.276493043].
    # After uniform 0.85 scaling, this translation keeps the base at z=0.
    mushroom_spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=str(RGMT_DATA_DIR / "Objects" / "mushroom" / "mushroom.usd"),
        scale=(0.85, 0.85, 0.85),
    )
    mushroom_position = (0.0, 0.0, 0.23501908655)
    # The converted USD has no authored physics material and otherwise inherits the
    # scene default of 1.0/1.0. Model the rubber-like apparatus explicitly at twice
    # those static and dynamic friction coefficients.
    mushroom_physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
    )


@configclass
class ExtremeRGMTG1DebugEnvCfg(ExtremeRGMTG1EnvCfg):
    """Asset and environment smoke-test configuration that does not require a motion dataset."""

    debug_static_reference = True
    observation_noise = False
    episode_length_s = 5.0

    def __post_init__(self):
        self.scene.num_envs = 16


@configclass
class ExtremeRGMTG1TheShyDebugEnvCfg(ExtremeRGMTG1TheShyEnvCfg):
    """Small asset smoke-test configuration for g1_theshy."""

    debug_static_reference = True
    observation_noise = False
    episode_length_s = 5.0

    def __post_init__(self):
        self.scene.num_envs = 16
