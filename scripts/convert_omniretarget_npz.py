"""Convert a compact OmniRetarget G1 qpos NPZ to the 50 Hz motion schema used by RGMT play.

The source ``qpos`` is expected to contain root translation (xyz), root
quaternion (wxyz), and the 29 actuated G1 joints in GMR/MJCF kinematic order.
The output uses the breadth-first joint/body order expected by
``ExtremeRGMTMotionDataset._load_beyondmimic_npz``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from augment_extreme_rgmt_dataset import _forward_kinematics
from prepare_lafan1_gmr import (
    JOINT_NAMES,
    _angular_velocity_world,
    _linear_velocity,
    _parse_kinematic_tree,
    _resample_gmr_qpos,
)


OUTPUT_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

OUTPUT_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def convert(input_path: Path, output_path: Path, xml_path: Path, target_fps: float) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    xml_path = xml_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    if target_fps <= 0.0:
        raise ValueError("target_fps must be positive")

    with np.load(input_path, allow_pickle=False) as source:
        missing = [name for name in ("qpos", "fps") if name not in source]
        if missing:
            raise KeyError(f"{input_path} is missing OmniRetarget fields: {missing}")
        qpos = np.asarray(source["qpos"], dtype=np.float64)
        fps_values = np.asarray(source["fps"]).reshape(-1)
    if fps_values.size != 1:
        raise ValueError(f"{input_path} must contain exactly one FPS value")
    source_fps = float(fps_values[0])
    if not np.isfinite(qpos).all():
        raise ValueError(f"{input_path} contains non-finite qpos values")

    qpos = _resample_gmr_qpos(qpos, source_fps, target_fps)
    root_position = qpos[:, :3]
    root_quaternion_wxyz = qpos[:, 3:7]
    root_quaternion_xyzw = root_quaternion_wxyz[:, [1, 2, 3, 0]]
    joint_position_kinematic = qpos[:, 7:]

    bodies, xml_joint_names = _parse_kinematic_tree(xml_path)
    if xml_joint_names != JOINT_NAMES:
        raise ValueError(
            "G1 XML joint order differs from the OmniRetarget/GMR qpos order:\n"
            f"XML: {xml_joint_names}\nExpected: {JOINT_NAMES}"
        )
    available_bodies = {body["name"] for body in bodies}
    missing_bodies = [name for name in OUTPUT_BODY_NAMES if name not in available_bodies]
    if missing_bodies:
        raise ValueError(f"G1 XML is missing output bodies: {missing_bodies}")

    position_map, rotation_map = _forward_kinematics(
        root_position,
        root_quaternion_xyzw,
        joint_position_kinematic,
        bodies,
    )
    body_position = np.stack([position_map[name] for name in OUTPUT_BODY_NAMES], axis=1)
    body_quaternion_xyzw = np.stack([rotation_map[name] for name in OUTPUT_BODY_NAMES], axis=1)
    body_quaternion_wxyz = body_quaternion_xyzw[..., [3, 0, 1, 2]]

    joint_indices = [JOINT_NAMES.index(name) for name in OUTPUT_JOINT_NAMES]
    joint_position = joint_position_kinematic[:, joint_indices]
    joint_velocity = _linear_velocity(joint_position, target_fps)
    body_linear_velocity = _linear_velocity(body_position, target_fps)
    body_angular_velocity = np.stack(
        [
            _angular_velocity_world(body_quaternion_xyzw[:, index], target_fps)
            for index in range(len(OUTPUT_BODY_NAMES))
        ],
        axis=1,
    )

    arrays = {
        "fps": np.asarray([round(target_fps)], dtype=np.int32),
        "joint_pos": joint_position.astype(np.float32),
        "joint_vel": joint_velocity.astype(np.float32),
        "body_pos_w": body_position.astype(np.float32),
        "body_quat_w": body_quaternion_wxyz.astype(np.float32),
        "body_lin_vel_w": body_linear_velocity.astype(np.float32),
        "body_ang_vel_w": body_angular_velocity.astype(np.float32),
        "source_fps": np.asarray([source_fps], dtype=np.float32),
        "source_file": np.asarray(str(input_path)),
        "kinematic_model": np.asarray(str(xml_path)),
    }
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Converted field {name!r} contains non-finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    duration = (joint_position.shape[0] - 1) / target_fps
    print(
        f"saved={output_path} frames={joint_position.shape[0]} "
        f"fps={target_fps:g} duration={duration:.3f}s",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=50.0)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert(args.input, args.output, args.xml, args.target_fps)
