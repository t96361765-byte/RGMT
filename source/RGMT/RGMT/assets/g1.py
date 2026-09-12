"""Unitree G1 29-DoF articulation configurations."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from RGMT import RGMT_DATA_DIR


G1_USD_PATH = (
    RGMT_DATA_DIR
    / "Robots"
    / "G1"
    / "g1_29dof_fist_pan_usd"
    / "g1_29dof_fist_pan.usd"
)

G1_THESHY_USD_PATH = (
    RGMT_DATA_DIR / "Robots" / "G1" / "g1_theshy_usd" / "g1_theshy.usd"
)


G1_29DOF_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(G1_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.40,
            ".*_ankle_pitch_joint": -0.20,
            "left_shoulder_roll_joint": 0.30,
            "right_shoulder_roll_joint": -0.30,
            ".*_elbow_joint": 1.00,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # Per-joint torque limits are taken from the provided real G1
        # URDF/USD assets.  Keep these groups separate where the motor ratings
        # differ so Isaac PhysX applies the intended limit to every joint.
        "hip_pitch_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_yaw_joint"],
            effort_limit_sim=88.0,
            stiffness=100.0,
            damping=2.5,
            armature=0.01,
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint"],
            effort_limit_sim=139.0,
            stiffness=100.0,
            damping=2.5,
            armature=0.01,
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit_sim=139.0,
            stiffness=150.0,
            damping=4.0,
            armature=0.01,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*_joint"],
            effort_limit_sim=50.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            stiffness=150.0,
            damping=4.0,
            armature=0.01,
        ),
        "waist_roll_pitch": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            stiffness=150.0,
            damping=4.0,
            armature=0.01,
        ),
        "shoulders_elbows": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint"],
            effort_limit_sim=25.0,
            stiffness=40.0,
            damping=5.0,
            armature=0.01,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_roll_joint"],
            effort_limit_sim=25.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),
        "wrist_pitch_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim=5.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),
    },
)


# g1_theshy is deliberately the validated pan-hand articulation with only the
# two fixed rubber-hand links removed.  It keeps the original configuration
# except for the explicitly increased upper-body effort limits required by the
# mushroom motion.  Build a new actuator dictionary so G1_29DOF_CFG remains
# unchanged for the original fist-pan tasks.
G1_THESHY_CFG = G1_29DOF_CFG.replace(
    spawn=G1_29DOF_CFG.spawn.replace(usd_path=str(G1_THESHY_USD_PATH)),
    actuators={
        **G1_29DOF_CFG.actuators,
        "shoulders_elbows": G1_29DOF_CFG.actuators["shoulders_elbows"].replace(
            effort_limit_sim=50.0
        ),
        "wrist_roll": G1_29DOF_CFG.actuators["wrist_roll"].replace(
            effort_limit_sim=50.0
        ),
        "wrist_pitch_yaw": G1_29DOF_CFG.actuators["wrist_pitch_yaw"].replace(
            effort_limit_sim=30.0
        ),
    },
)
