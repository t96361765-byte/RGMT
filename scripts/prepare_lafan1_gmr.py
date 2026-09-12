"""Prepare GMR-retargeted LAFAN1 source motions for Extreme-RGMT training.

The workflow intentionally separates the GMR environment from Isaac Lab:

1. ``retarget`` runs inside the GMR environment and writes the standard GMR
   pickle representation (NumPy, ``root_rot`` in xyzw order).
2. ``convert`` runs inside the Isaac Lab environment, performs forward
   kinematics, derives velocities, and writes one training-ready pickle per
   clip.
3. ``pack`` combines the converted pickles into the ``motions.pt`` file read
   for subsequent Extreme-RGMT dataset augmentation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

BODY_NAMES = [
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

MOTION_FIELDS = (
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "joint_pos",
    "joint_vel",
    "body_pos",
    "body_lin_vel",
)


class _Numpy2CompatUnpickler(pickle.Unpickler):
    """Load NumPy 2 pickles in the NumPy 1.26 Isaac Lab environment."""

    def find_class(self, module: str, name: str):
        if module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        elif module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        return super().find_class(module, name)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        try:
            return pickle.load(stream)
        except ModuleNotFoundError as error:
            if error.name != "numpy._core":
                raise
            stream.seek(0)
            return _Numpy2CompatUnpickler(stream).load()


def _atomic_pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=4)
    os.replace(temporary, path)


def _normalize_quaternions_xyzw(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Motion contains a zero-length root quaternion.")
    result /= norms
    for index in range(1, result.shape[0]):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _read_bvh_fps(path: Path) -> float:
    """Read the source sampling rate from a BVH ``Frame Time`` declaration."""
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.lower().startswith("frame time:"):
                frame_time = float(stripped.split(":", maxsplit=1)[1].strip())
                if not np.isfinite(frame_time) or frame_time <= 0.0:
                    raise ValueError(f"{path} has an invalid BVH frame time: {frame_time}")
                fps = 1.0 / frame_time
                # LAFAN1 commonly stores 1/30 s as the rounded decimal 0.033333.
                nearest_integer = round(fps)
                if abs(fps - nearest_integer) < 1.0e-3:
                    fps = float(nearest_integer)
                return fps
    raise ValueError(f"{path} does not contain a BVH 'Frame Time' declaration.")


def _resample_gmr_qpos(
    qpos: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """Resample standard GMR qpos without changing the retargeting solution."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] != 7 + len(JOINT_NAMES):
        raise ValueError(f"Invalid GMR qpos shape: {qpos.shape}")
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("Source and target FPS must be positive.")
    if np.isclose(source_fps, target_fps, rtol=0.0, atol=1.0e-9):
        return qpos.copy()

    source_times = np.arange(qpos.shape[0], dtype=np.float64) / source_fps
    source_duration = source_times[-1]
    # Keep a regular 1 / target_fps grid and never extrapolate beyond the
    # source clip. At most one target interval is omitted at the tail.
    target_frame_count = int(np.floor(source_duration * target_fps + 1.0e-9)) + 1
    if target_frame_count < 2:
        raise ValueError("Resampling would produce fewer than two frames.")
    target_times = np.arange(target_frame_count, dtype=np.float64) / target_fps

    result = np.empty((target_frame_count, qpos.shape[1]), dtype=np.float64)
    linear_columns = np.r_[0:3, 7:qpos.shape[1]]
    for column in linear_columns:
        result[:, column] = np.interp(target_times, source_times, qpos[:, column])

    root_quaternion_xyzw = _normalize_quaternions_xyzw(qpos[:, 3:7][:, [1, 2, 3, 0]])
    target_rotation = Slerp(
        source_times, Rotation.from_quat(root_quaternion_xyzw)
    )(target_times)
    result[:, 3:7] = target_rotation.as_quat()[:, [3, 0, 1, 2]]
    return result


def _linear_velocity(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    velocity = np.empty_like(values)
    velocity[0] = (values[1] - values[0]) * fps
    velocity[-1] = (values[-1] - values[-2]) * fps
    if values.shape[0] > 2:
        velocity[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return velocity


def _angular_velocity_world(quaternions_xyzw: np.ndarray, fps: float) -> np.ndarray:
    rotations = Rotation.from_quat(quaternions_xyzw)
    interval_velocity = (rotations[1:] * rotations[:-1].inv()).as_rotvec() * fps
    velocity = np.empty((quaternions_xyzw.shape[0], 3), dtype=np.float64)
    velocity[0] = interval_velocity[0]
    velocity[-1] = interval_velocity[-1]
    if quaternions_xyzw.shape[0] > 2:
        velocity[1:-1] = 0.5 * (interval_velocity[:-1] + interval_velocity[1:])
    return velocity


def _parse_kinematic_tree(xml_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    root = ET.parse(xml_path).getroot()
    world_body = root.find("worldbody")
    if world_body is None:
        raise ValueError(f"{xml_path} does not contain a worldbody element.")
    robot_root = world_body.find("body")
    if robot_root is None:
        raise ValueError(f"{xml_path} does not contain a robot root body.")

    bodies: list[dict[str, Any]] = []
    joint_names: list[str] = []

    def visit(node: ET.Element, parent_index: int) -> None:
        body_index = len(bodies)
        position = np.fromstring(node.attrib.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
        quaternion_wxyz = np.fromstring(
            node.attrib.get("quat", "1 0 0 0"), sep=" ", dtype=np.float64
        )
        quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
        joint_index: int | None = None
        joint_axis: np.ndarray | None = None

        if parent_index >= 0:
            movable_joints = [
                joint
                for joint in node.findall("joint")
                if joint.attrib.get("type", "hinge") not in {"free", "fixed"}
            ]
            if len(movable_joints) > 1:
                raise ValueError(f"Body {node.attrib.get('name')} has multiple movable joints.")
            if movable_joints:
                joint = movable_joints[0]
                joint_index = len(joint_names)
                joint_names.append(joint.attrib["name"])
                joint_axis = np.fromstring(
                    joint.attrib.get("axis", "0 0 1"), sep=" ", dtype=np.float64
                )

        bodies.append(
            {
                "name": node.attrib["name"],
                "parent": parent_index,
                "position": position,
                "quaternion": quaternion_xyzw,
                "joint_index": joint_index,
                "joint_axis": joint_axis,
            }
        )
        for child in node.findall("body"):
            visit(child, body_index)

    visit(robot_root, -1)
    return bodies, joint_names


def _forward_kinematics(
    root_position: np.ndarray,
    root_quaternion_xyzw: np.ndarray,
    joint_position: np.ndarray,
    bodies: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    frame_count = root_position.shape[0]
    body_positions: list[np.ndarray] = [np.empty((0, 3))] * len(bodies)
    body_rotations: list[Rotation] = [Rotation.identity()] * len(bodies)
    body_positions[0] = root_position
    body_rotations[0] = Rotation.from_quat(root_quaternion_xyzw)

    for index, body in enumerate(bodies[1:], start=1):
        parent_index = body["parent"]
        parent_rotation = body_rotations[parent_index]
        local_position = np.broadcast_to(body["position"], (frame_count, 3))
        body_positions[index] = body_positions[parent_index] + parent_rotation.apply(local_position)

        local_rotation = Rotation.from_quat(
            np.broadcast_to(body["quaternion"], (frame_count, 4))
        )
        if body["joint_index"] is None:
            joint_rotation = Rotation.identity(frame_count)
        else:
            rotation_vectors = (
                body["joint_axis"][None, :] * joint_position[:, body["joint_index"], None]
            )
            joint_rotation = Rotation.from_rotvec(rotation_vectors)
        body_rotations[index] = parent_rotation * local_rotation * joint_rotation

    return {body["name"]: body_positions[index] for index, body in enumerate(bodies)}


def _convert_motion(raw: dict[str, Any], xml_path: Path) -> dict[str, Any]:
    for key in ("fps", "root_pos", "root_rot", "dof_pos"):
        if key not in raw:
            raise KeyError(f"Raw GMR motion is missing '{key}'.")

    fps = float(raw["fps"])
    root_position = np.asarray(raw["root_pos"], dtype=np.float64)
    root_quaternion_xyzw = _normalize_quaternions_xyzw(raw["root_rot"])
    joint_position = np.asarray(raw["dof_pos"], dtype=np.float64)
    frame_count = joint_position.shape[0]
    if frame_count < 2:
        raise ValueError("Motion must contain at least two frames.")
    if root_position.shape != (frame_count, 3):
        raise ValueError(f"Invalid root_pos shape: {root_position.shape}")
    if root_quaternion_xyzw.shape != (frame_count, 4):
        raise ValueError(f"Invalid root_rot shape: {root_quaternion_xyzw.shape}")
    if joint_position.shape != (frame_count, len(JOINT_NAMES)):
        raise ValueError(f"Invalid dof_pos shape: {joint_position.shape}")
    if not all(
        np.isfinite(value).all()
        for value in (root_position, root_quaternion_xyzw, joint_position)
    ):
        raise ValueError("Motion contains NaN or infinite values.")

    bodies, xml_joint_names = _parse_kinematic_tree(xml_path)
    if xml_joint_names != JOINT_NAMES:
        raise ValueError(
            "GMR XML joint order does not match the Extreme-RGMT G1 order:\n"
            f"XML: {xml_joint_names}\nExtreme-RGMT: {JOINT_NAMES}"
        )
    all_body_positions = _forward_kinematics(
        root_position, root_quaternion_xyzw, joint_position, bodies
    )
    missing_bodies = [name for name in BODY_NAMES if name not in all_body_positions]
    if missing_bodies:
        raise ValueError(
            f"GMR XML is missing Extreme-RGMT tracking bodies: {missing_bodies}"
        )
    body_position = np.stack([all_body_positions[name] for name in BODY_NAMES], axis=1)

    root_quaternion_wxyz = root_quaternion_xyzw[:, [3, 0, 1, 2]]
    result = {
        "fps": fps,
        "joint_names": list(JOINT_NAMES),
        "body_names": list(BODY_NAMES),
        "root_pos": root_position.astype(np.float32),
        "root_quat": root_quaternion_wxyz.astype(np.float32),
        "root_lin_vel": _linear_velocity(root_position, fps).astype(np.float32),
        "root_ang_vel": _angular_velocity_world(root_quaternion_xyzw, fps).astype(np.float32),
        "joint_pos": joint_position.astype(np.float32),
        "joint_vel": _linear_velocity(joint_position, fps).astype(np.float32),
        "body_pos": body_position.astype(np.float32),
        "body_lin_vel": _linear_velocity(body_position, fps).astype(np.float32),
    }
    for metadata_key in ("source_fps", "source_frame_count", "source_file"):
        if metadata_key in raw:
            result[metadata_key] = raw[metadata_key]
    for field in MOTION_FIELDS:
        if not np.isfinite(result[field]).all():
            raise ValueError(f"Converted field '{field}' contains NaN or infinite values.")
    return result


def _iter_pickle_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pkl":
            raise ValueError(f"Expected a .pkl file, got {path}.")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.rglob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found under {path}.")
    return files


def _slice_converted_motion(
    motion: dict[str, Any],
    start: int,
    end: int,
    source_file: str,
) -> dict[str, Any]:
    """Copy a frame range and recompute velocities at its new boundaries."""
    fps = float(motion["fps"])
    result = {
        "fps": fps,
        "joint_names": list(motion["joint_names"]),
        "body_names": list(motion["body_names"]),
        "source_file": source_file,
        "source_frame_range": (start, end),
    }
    for field in MOTION_FIELDS:
        result[field] = np.asarray(motion[field][start:end], dtype=np.float32).copy()

    root_quaternion_xyzw = result["root_quat"][:, [1, 2, 3, 0]]
    result["root_lin_vel"] = _linear_velocity(result["root_pos"], fps).astype(np.float32)
    result["root_ang_vel"] = _angular_velocity_world(root_quaternion_xyzw, fps).astype(
        np.float32
    )
    result["joint_vel"] = _linear_velocity(result["joint_pos"], fps).astype(np.float32)
    result["body_lin_vel"] = _linear_velocity(result["body_pos"], fps).astype(np.float32)
    return result


def command_retarget(args: argparse.Namespace) -> None:
    gmr_root = args.gmr_root.resolve()
    sys.path.insert(0, str(gmr_root))
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    if args.bvh.is_file():
        source_files = [args.bvh]
        source_root = args.bvh.parent
        target_files = [args.output]
    elif args.bvh.is_dir():
        source_files = sorted(args.bvh.rglob("*.bvh"))
        if not source_files:
            raise FileNotFoundError(f"No .bvh files found under {args.bvh}.")
        source_root = args.bvh
        target_files = [
            args.output / source.relative_to(source_root).with_suffix(".pkl")
            for source in source_files
        ]
    else:
        raise FileNotFoundError(args.bvh)

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num-shards.")
    if args.num_shards > 1:
        source_files = source_files[args.shard_index :: args.num_shards]
        target_files = target_files[args.shard_index :: args.num_shards]
        print(
            f"shard={args.shard_index}/{args.num_shards} clips={len(source_files)}",
            flush=True,
        )

    for clip_index, (source, target) in enumerate(
        zip(source_files, target_files, strict=True), start=1
    ):
        if target.exists() and not args.overwrite:
            print(f"skipped={clip_index}/{len(source_files)} target={target}", flush=True)
            continue
        print(
            f"clip={clip_index}/{len(source_files)} source={source.name}",
            flush=True,
        )
        frames, human_height = load_bvh_file(str(source), format="lafan1")
        source_fps = (
            float(args.source_fps) if args.source_fps is not None else _read_bvh_fps(source)
        )
        retargeter = GeneralMotionRetargeting(
            src_human="bvh_lafan1",
            tgt_robot=args.robot,
            actual_human_height=human_height,
            verbose=False,
            use_velocity_limit=args.velocity_limit,
        )
        qpos = []
        for frame_index, frame in enumerate(frames):
            qpos.append(retargeter.retarget(frame).copy())
            if (frame_index + 1) % 1000 == 0 or frame_index + 1 == len(frames):
                print(
                    f"frames={frame_index + 1}/{len(frames)} clip={source.name}",
                    flush=True,
                )
        qpos_array = np.stack(qpos)
        source_frame_count = qpos_array.shape[0]
        qpos_array = _resample_gmr_qpos(qpos_array, source_fps, args.target_fps)
        raw = {
            "fps": float(args.target_fps),
            "source_fps": source_fps,
            "source_frame_count": source_frame_count,
            "source_file": str(source.relative_to(source_root)),
            "root_pos": qpos_array[:, :3],
            "root_rot": qpos_array[:, 3:7][:, [1, 2, 3, 0]],
            "dof_pos": qpos_array[:, 7:],
            "local_body_pos": None,
            "link_body_list": None,
        }
        _atomic_pickle_dump(raw, target)
        print(
            f"saved={target} source_fps={source_fps:g} target_fps={args.target_fps:g} "
            f"source_frames={source_frame_count} target_frames={qpos_array.shape[0]}",
            flush=True,
        )


def command_convert(args: argparse.Namespace) -> None:
    source_files = _iter_pickle_files(args.input)
    common_root = args.input if args.input.is_dir() else args.input.parent
    for index, source in enumerate(source_files, start=1):
        relative = source.relative_to(common_root)
        target = args.output_dir / relative
        if target.exists() and not args.overwrite:
            print(f"skipped={target}", flush=True)
            continue
        converted = _convert_motion(_load_pickle(source), args.xml)
        _atomic_pickle_dump(converted, target)
        print(
            f"converted={index}/{len(source_files)} file={source.name} "
            f"frames={converted['joint_pos'].shape[0]} target={target}",
            flush=True,
        )


def command_select(args: argparse.Namespace) -> None:
    if args.speed_limit <= 0.0:
        raise ValueError("--speed-limit must be positive.")
    if args.padding_seconds < 0.0:
        raise ValueError("--padding-seconds cannot be negative.")
    if args.min_duration_seconds <= 0.0:
        raise ValueError("--min-duration-seconds must be positive.")
    if args.min_consecutive_bad_intervals <= 0:
        raise ValueError("--min-consecutive-bad-intervals must be positive.")

    source_files = _iter_pickle_files(args.input_dir)
    excluded = set(args.exclude)
    known_names = {path.name for path in source_files}
    missing_exclusions = sorted(excluded - known_names)
    if missing_exclusions:
        raise ValueError(f"Excluded files were not found: {missing_exclusions}")

    existing = list(args.output_dir.glob("*.pkl")) if args.output_dir.is_dir() else []
    if existing:
        raise FileExistsError(
            f"{args.output_dir} already contains {len(existing)} pickle files. "
            "Use an empty output directory to avoid mixing selections."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "speed_limit_rad_s": args.speed_limit,
        "padding_seconds": args.padding_seconds,
        "minimum_duration_seconds": args.min_duration_seconds,
        "minimum_consecutive_bad_intervals": args.min_consecutive_bad_intervals,
        "excluded_files": sorted(excluded),
        "source_file_count": len(source_files),
        "segments": [],
    }
    total_source_frames = 0
    total_selected_frames = 0
    selected_source_count = 0

    for source in source_files:
        if source.name in excluded:
            print(f"excluded={source.name}", flush=True)
            continue
        motion = _load_pickle(source)
        if motion.get("joint_names") != JOINT_NAMES:
            raise ValueError(f"{source} has an unexpected joint order.")
        if motion.get("body_names") != BODY_NAMES:
            raise ValueError(f"{source} has an unexpected body order.")

        fps = float(motion["fps"])
        joint_position = np.asarray(motion["joint_pos"], dtype=np.float64)
        frame_count = joint_position.shape[0]
        total_source_frames += frame_count
        interval_speed = np.abs(np.diff(joint_position, axis=0) * fps)
        bad_interval_mask = np.any(interval_speed > args.speed_limit, axis=1)
        bad_transitions = np.diff(
            np.r_[False, bad_interval_mask, False].astype(np.int8)
        )
        bad_starts = np.flatnonzero(bad_transitions == 1)
        bad_ends = np.flatnonzero(bad_transitions == -1)
        bad_runs = [
            (int(start), int(end))
            for start, end in zip(bad_starts, bad_ends, strict=True)
        ]
        rejected_runs = [
            (start, end)
            for start, end in bad_runs
            if end - start >= args.min_consecutive_bad_intervals
        ]

        valid = np.ones(frame_count, dtype=bool)
        padding_frames = int(round(args.padding_seconds * fps))
        for start, end in rejected_runs:
            # Intervals [start, end) connect frames [start, end]. Include both
            # endpoint frames, plus the requested temporal guard band.
            invalid_start = max(0, start - padding_frames)
            invalid_end = min(frame_count, end + 1 + padding_frames)
            valid[invalid_start:invalid_end] = False

        transitions = np.diff(np.r_[False, valid, False].astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        minimum_frames = max(2, int(np.ceil(args.min_duration_seconds * fps)) + 1)
        ranges = [
            (int(start), int(end))
            for start, end in zip(starts, ends, strict=True)
            if end - start >= minimum_frames
        ]
        if ranges:
            selected_source_count += 1

        for segment_index, (start, end) in enumerate(ranges, start=1):
            target = args.output_dir / f"{source.stem}__segment_{segment_index:04d}.pkl"
            segment = _slice_converted_motion(motion, start, end, source.name)
            _atomic_pickle_dump(segment, target)
            total_selected_frames += end - start
            manifest["segments"].append(
                {
                    "file": target.name,
                    "source_file": source.name,
                    "source_frame_range": [start, end],
                    "frames": end - start,
                    "duration_seconds": (end - start - 1) / fps,
                }
            )
        print(
            f"selected={source.name} bad_intervals={int(bad_interval_mask.sum())} "
            f"bad_runs={len(bad_runs)} rejected_runs={len(rejected_runs)} "
            f"segments={len(ranges)}",
            flush=True,
        )

    manifest.update(
        {
            "selected_source_count": selected_source_count,
            "selected_segment_count": len(manifest["segments"]),
            "source_frames_after_whole_clip_exclusion": total_source_frames,
            "selected_frames": total_selected_frames,
        }
    )
    manifest_path = args.output_dir / "selection_manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, manifest_path)
    print(
        f"saved={args.output_dir} segments={len(manifest['segments'])} "
        f"selected_sources={selected_source_count} frames={total_selected_frames}",
        flush=True,
    )


def command_pack(args: argparse.Namespace) -> None:
    import torch

    source_files = _iter_pickle_files(args.input_dir)
    common_root = args.input_dir if args.input_dir.is_dir() else args.input_dir.parent
    motions = []
    source_names = []
    for index, source in enumerate(source_files, start=1):
        converted = _load_pickle(source)
        if converted.get("joint_names") != JOINT_NAMES:
            raise ValueError(f"{source} has an unexpected joint order.")
        if converted.get("body_names") != BODY_NAMES:
            raise ValueError(f"{source} has an unexpected body order.")
        motion = {"fps": float(converted["fps"])}
        for field in MOTION_FIELDS:
            motion[field] = torch.as_tensor(converted[field], dtype=torch.float32)
        motions.append(motion)
        source_names.append(source.relative_to(common_root).as_posix())
        print(f"packed={index}/{len(source_files)} file={source.name}", flush=True)

    dataset = {
        "joint_names": list(JOINT_NAMES),
        "body_names": list(BODY_NAMES),
        "motions": motions,
        "source_files": source_names,
    }
    manifest_path = common_root / "selection_manifest.json"
    if manifest_path.is_file():
        dataset["metadata"] = {
            "selection_manifest": json.loads(manifest_path.read_text(encoding="utf-8"))
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    torch.save(dataset, temporary)
    os.replace(temporary, args.output)
    print(f"saved={args.output} motions={len(motions)}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    retarget = subparsers.add_parser("retarget", help="Retarget one LAFAN1 BVH with GMR.")
    retarget.add_argument("--gmr-root", type=Path, required=True)
    retarget.add_argument("--bvh", type=Path, required=True)
    retarget.add_argument("--output", type=Path, required=True)
    retarget.add_argument("--robot", default="unitree_g1_fist")
    retarget.add_argument(
        "--target-fps",
        "--fps",
        dest="target_fps",
        type=float,
        default=50.0,
        help="Output sampling rate after GMR retargeting (default: 50 Hz).",
    )
    retarget.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Override the rate read from each BVH Frame Time declaration.",
    )
    retarget.add_argument("--velocity-limit", action="store_true")
    retarget.add_argument("--overwrite", action="store_true")
    retarget.add_argument("--num-shards", type=int, default=1)
    retarget.add_argument("--shard-index", type=int, default=0)
    retarget.set_defaults(func=command_retarget)

    convert = subparsers.add_parser("convert", help="Convert raw GMR pickle files.")
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.add_argument("--xml", type=Path, required=True)
    convert.add_argument("--overwrite", action="store_true")
    convert.set_defaults(func=command_convert)

    select = subparsers.add_parser(
        "select", help="Remove high-speed intervals and retain clean motion segments."
    )
    select.add_argument("--input-dir", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--speed-limit", type=float, default=3.0 * np.pi)
    select.add_argument("--padding-seconds", type=float, default=0.2)
    select.add_argument("--min-duration-seconds", type=float, default=2.0)
    select.add_argument(
        "--min-consecutive-bad-intervals",
        type=int,
        default=1,
        help="Ignore shorter isolated runs of over-limit joint-speed intervals.",
    )
    select.add_argument("--exclude", action="append", default=[])
    select.set_defaults(func=command_select)

    pack = subparsers.add_parser("pack", help="Pack converted pickles into motions.pt.")
    pack.add_argument("--input-dir", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.set_defaults(func=command_pack)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
