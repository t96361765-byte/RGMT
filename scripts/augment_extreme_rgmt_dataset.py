"""Add reference link orientations and angular velocities for Extreme-RGMT."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from prepare_lafan1_gmr import (
    _angular_velocity_world,
    _parse_kinematic_tree,
)


def _forward_kinematics(
    root_position: np.ndarray,
    root_quaternion_xyzw: np.ndarray,
    joint_position: np.ndarray,
    bodies: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame_count = root_position.shape[0]
    positions: list[np.ndarray] = [np.empty((0, 3))] * len(bodies)
    rotations: list[Rotation] = [Rotation.identity()] * len(bodies)
    positions[0] = root_position
    rotations[0] = Rotation.from_quat(root_quaternion_xyzw)

    for index, body in enumerate(bodies[1:], start=1):
        parent = body["parent"]
        parent_rotation = rotations[parent]
        local_position = np.broadcast_to(body["position"], (frame_count, 3))
        positions[index] = positions[parent] + parent_rotation.apply(local_position)
        local_rotation = Rotation.from_quat(
            np.broadcast_to(body["quaternion"], (frame_count, 4))
        )
        if body["joint_index"] is None:
            joint_rotation = Rotation.identity(frame_count)
        else:
            rotation_vectors = (
                body["joint_axis"][None, :]
                * joint_position[:, body["joint_index"], None]
            )
            joint_rotation = Rotation.from_rotvec(rotation_vectors)
        rotations[index] = parent_rotation * local_rotation * joint_rotation

    position_map = {
        body["name"]: positions[index] for index, body in enumerate(bodies)
    }
    rotation_map = {
        body["name"]: rotations[index].as_quat()
        for index, body in enumerate(bodies)
    }
    return position_map, rotation_map


def augment_dataset(args: argparse.Namespace) -> None:
    source = args.input.resolve()
    output = args.output.resolve()
    xml = args.xml.resolve()
    if source == output:
        raise ValueError("Refusing to overwrite the Extreme-RGMT source dataset.")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not xml.is_file():
        raise FileNotFoundError(xml)

    dataset = torch.load(source, map_location="cpu", weights_only=False)
    body_names = list(dataset["body_names"])
    joint_names = list(dataset["joint_names"])
    bodies, xml_joint_names = _parse_kinematic_tree(xml)
    if xml_joint_names != joint_names:
        raise ValueError("Dataset and G1 XML joint orders differ.")

    for index, motion in enumerate(dataset["motions"], start=1):
        root_position = np.asarray(motion["root_pos"], dtype=np.float64)
        root_quaternion_wxyz = np.asarray(motion["root_quat"], dtype=np.float64)
        root_quaternion_xyzw = root_quaternion_wxyz[:, [1, 2, 3, 0]]
        joint_position = np.asarray(motion["joint_pos"], dtype=np.float64)
        position_map, rotation_map = _forward_kinematics(
            root_position,
            root_quaternion_xyzw,
            joint_position,
            bodies,
        )
        computed_positions = np.stack(
            [position_map[name] for name in body_names], axis=1
        )
        stored_positions = np.asarray(motion["body_pos"], dtype=np.float64)
        max_position_error = float(
            np.max(np.abs(computed_positions - stored_positions))
        )
        if max_position_error > args.position_tolerance:
            raise ValueError(
                f"Motion {index - 1} FK mismatch: {max_position_error:.6g} m"
            )

        body_quaternion_xyzw = np.stack(
            [rotation_map[name] for name in body_names], axis=1
        )
        body_angular_velocity = np.stack(
            [
                _angular_velocity_world(
                    body_quaternion_xyzw[:, body_index], float(motion["fps"])
                )
                for body_index in range(len(body_names))
            ],
            axis=1,
        )
        body_quaternion_wxyz = body_quaternion_xyzw[..., [3, 0, 1, 2]]
        motion["body_quat"] = torch.as_tensor(
            body_quaternion_wxyz.astype(np.float32)
        )
        motion["body_ang_vel"] = torch.as_tensor(
            body_angular_velocity.astype(np.float32)
        )
        print(
            f"augmented={index}/{len(dataset['motions'])} "
            f"frames={joint_position.shape[0]} "
            f"fk_error={max_position_error:.3e}",
            flush=True,
        )

    metadata = dict(dataset.get("metadata", {}))
    metadata["name"] = "Extreme_RGMT_motion_stage1"
    metadata["extreme_rgmt_stage1"] = {
        "source": str(source),
        "xml": str(xml),
        "added_fields": ["body_quat", "body_ang_vel"],
        "quaternion_order": "wxyz",
        "angular_velocity_frame": "world",
    }
    dataset["metadata"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(dataset, temporary)
    os.replace(temporary, output)
    print(f"saved={output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--position-tolerance", type=float, default=1.0e-4)
    return parser


if __name__ == "__main__":
    augment_dataset(build_parser().parse_args())
