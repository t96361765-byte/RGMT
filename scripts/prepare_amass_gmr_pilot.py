"""Build a conservative Extreme-RGMT source pilot from legacy GMR AMASS pickles.

The local ``amass-dev-g1`` corpus uses the legacy 23-actuated-DoF G1 layout:

* 12 leg joints
* waist yaw
* five joints per arm (shoulder pitch/roll/yaw, elbow, wrist roll)

Extreme-RGMT controls the 29-actuated-DoF G1. This script preserves every retargeted
legacy joint and inserts neutral values for waist roll/pitch and wrist
pitch/yaw.  It then runs the same conversion and forward-kinematics path used
for the LAFAN1 data, applies conservative whole-clip quality gates, and writes
one training-ready ``.pt`` file plus a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prepare_lafan1_gmr import (
    BODY_NAMES,
    JOINT_NAMES,
    MOTION_FIELDS,
    _convert_motion,
    _load_pickle,
)


ACCAD_FILES = [
    "A1__Stand.pkl",
    "A2__Sway.pkl",
    "A3__Swing_arms.pkl",
    "A4__Look_Around.pkl",
    "A7__Crouch.pkl",
    "B1___stand_to_walk.pkl",
    "B2___walk_to_stand.pkl",
    "B3___walk1.pkl",
    "B4___stand_to_walk_back.pkl",
    "B5___walk_backwards.pkl",
    "B6___walk_backwards_to_stand.pkl",
    "B7___walk_backwards_turn_forwards.pkl",
    "B9___walk_turn_left__90_.pkl",
    "B12___walk_turn_right__90_.pkl",
    "B22___side_step_left.pkl",
    "B23___side_step_right.pkl",
    "B24___walk_to_crouch.pkl",
    "B25___crouch_to_walk1.pkl",
    "D3___Conversation_Gestures.pkl",
]

SFU_FILES = [
    "0005_BackwardsWalk001.pkl",
    "0005_Walking001.pkl",
    "0007_Balance001.pkl",
    "0007_Walking001.pkl",
    "0008_Walking001.pkl",
    "0018_TipToe001.pkl",
    "0018_Walking001.pkl",
]

SSM_FILES = [
    "ankles.pkl",
    "ATUSquat_sync.pkl",
    "knees.pkl",
    "shake_shoulders.pkl",
    "shoulders.pkl",
    "walking.pkl",
    "walking_01.pkl",
    "wrists_up_down.pkl",
]

TOTALCAPTURE_FILES = [
    "s1_acting1.pkl",
    "s1_freestyle1.pkl",
    "s1_walking1.pkl",
    "s1_walking2.pkl",
    "s2_walking1.pkl",
    "s3_walking1.pkl",
]

TRANSITION_FILES = [
    "crouchwalk_stand.pkl",
    "crouchwalk_walk.pkl",
    "dance_stand.pkl",
    "dance_walk.pkl",
    "run_stand.pkl",
    "run_walk.pkl",
    "turntwist_stand.pkl",
    "turntwist_walk.pkl",
    "walkbackwards_stand.pkl",
    "walksideways_stand.pkl",
    "walksideways_walk.pkl",
]


def _add_existing(
    candidates: list[Path],
    dataset_root: Path,
    subset: str,
    filenames: list[str],
) -> None:
    for filename in filenames:
        path = dataset_root / subset / filename
        if not path.is_file():
            raise FileNotFoundError(f"Pilot candidate is missing: {path}")
        candidates.append(path)


def _add_first_matches(
    candidates: list[Path],
    dataset_root: Path,
    subset: str,
    patterns: list[str],
) -> None:
    directory = dataset_root / subset
    filenames = sorted(path.name for path in directory.glob("*.pkl"))
    for pattern in patterns:
        matches = [name for name in filenames if re.search(pattern, name, re.IGNORECASE)]
        if not matches:
            raise FileNotFoundError(f"No '{subset}' file matches pilot pattern: {pattern}")
        candidates.append(directory / matches[0])


def build_candidate_list(dataset_root: Path) -> list[Path]:
    """Return a bounded, filename-driven set of varied basic motions."""
    candidates: list[Path] = []
    _add_existing(candidates, dataset_root, "accad", ACCAD_FILES)

    for subject in ("rub001", "rub002"):
        _add_first_matches(
            candidates,
            dataset_root,
            "biomotionlab_ntroje",
            [
                rf"^{subject}_.*treadmill_norm",
                rf"^{subject}_.*treadmill_slow",
                rf"^{subject}_.*normal_walk1",
                rf"^{subject}_.*normal_walk2",
                rf"^{subject}_.*circle_walk",
                rf"^{subject}_.*lifting_light1",
                rf"^{subject}_.*knocking1",
            ],
        )

    _add_first_matches(
        candidates,
        dataset_root,
        "dfaust",
        [r"_chicken_wings", r"_hips", r"_knees", r"_shake_arms", r"_shake_shoulders"],
    )
    _add_first_matches(
        candidates,
        dataset_root,
        "eyes_japan",
        [
            r"audience_01_standing",
            r"gesture_etc_05_touch_nose",
            r"gesture_etc_06_jiggle_knee",
            r"gesture_etc_08_stretch",
            r"gesture_etc_11_hold_arms",
            r"gesture_etc_21_clapping",
        ],
    )
    _add_first_matches(
        candidates,
        dataset_root,
        "mpi_mosh",
        [r"simple_crouch", r"_stretches", r"_ankles", r"_neck", r"_shoulders", r"_army_poses"],
    )
    _add_existing(candidates, dataset_root, "sfu", SFU_FILES)
    _add_existing(candidates, dataset_root, "ssm_synced", SSM_FILES)
    _add_existing(candidates, dataset_root, "totalcapture", TOTALCAPTURE_FILES)
    _add_existing(candidates, dataset_root, "transitions", TRANSITION_FILES)

    # Preserve the declared order while avoiding accidental duplicate matches.
    return list(dict.fromkeys(candidates))


def expand_legacy_g1_23dof(joint_position: np.ndarray) -> np.ndarray:
    """Map the legacy G1 23-DoF order to Extreme-RGMT's 29-DoF order."""
    if joint_position.ndim != 2 or joint_position.shape[1] != 23:
        raise ValueError(f"Expected legacy G1 dof_pos with shape [T, 23], got {joint_position.shape}.")

    expanded = np.zeros((joint_position.shape[0], 29), dtype=np.float32)
    expanded[:, 0:12] = joint_position[:, 0:12]
    expanded[:, 12] = joint_position[:, 12]
    expanded[:, 15:20] = joint_position[:, 13:18]
    expanded[:, 22:27] = joint_position[:, 18:23]
    return expanded


def _prepare_raw_motion(raw: dict[str, Any]) -> tuple[dict[str, Any], str]:
    joint_position = np.asarray(raw["dof_pos"], dtype=np.float32)
    if joint_position.ndim != 2:
        raise ValueError(f"Invalid dof_pos shape: {joint_position.shape}")
    if joint_position.shape[1] == 23:
        prepared = dict(raw)
        prepared["dof_pos"] = expand_legacy_g1_23dof(joint_position)
        return prepared, "legacy_g1_23dof_neutral_fill"
    if joint_position.shape[1] == 29:
        return raw, "native_g1_29dof"
    raise ValueError(f"Unsupported G1 dof_pos shape: {joint_position.shape}")


def _quality_metrics(motion: dict[str, Any]) -> dict[str, float]:
    joint_speed = np.max(np.abs(motion["joint_vel"]), axis=1)
    root_horizontal_speed = np.linalg.norm(motion["root_lin_vel"][:, :2], axis=1)
    root_angular_speed = np.linalg.norm(motion["root_ang_vel"], axis=1)
    duration = (motion["joint_pos"].shape[0] - 1) / float(motion["fps"])
    return {
        "duration_seconds": float(duration),
        "max_joint_speed_rad_s": float(joint_speed.max()),
        "p95_root_horizontal_speed_m_s": float(np.quantile(root_horizontal_speed, 0.95)),
        "p95_root_angular_speed_rad_s": float(np.quantile(root_angular_speed, 0.95)),
        "minimum_root_height_m": float(motion["root_pos"][:, 2].min()),
    }


def _rejection_reasons(metrics: dict[str, float], args: argparse.Namespace) -> list[str]:
    reasons = []
    if metrics["duration_seconds"] < args.minimum_duration:
        reasons.append("duration")
    if metrics["max_joint_speed_rad_s"] > args.maximum_joint_speed:
        reasons.append("joint_speed")
    if metrics["p95_root_horizontal_speed_m_s"] > args.maximum_p95_root_speed:
        reasons.append("root_speed")
    if metrics["p95_root_angular_speed_rad_s"] > args.maximum_p95_root_angular_speed:
        reasons.append("root_angular_speed")
    if metrics["minimum_root_height_m"] < args.minimum_root_height:
        reasons.append("root_height")
    return reasons


def build_dataset(args: argparse.Namespace) -> None:
    dataset_root = args.input.resolve()
    xml_path = args.xml.resolve()
    output_path = args.output.resolve()
    candidates = build_candidate_list(dataset_root)

    motions = []
    source_files = []
    manifest_entries = []
    for index, source in enumerate(candidates, start=1):
        relative_source = source.relative_to(dataset_root).as_posix()
        entry: dict[str, Any] = {"source_file": relative_source}
        try:
            raw, mapping = _prepare_raw_motion(_load_pickle(source))
            motion = _convert_motion(raw, xml_path)
            metrics = _quality_metrics(motion)
            reasons = _rejection_reasons(metrics, args)
            entry.update({"mapping": mapping, "metrics": metrics, "rejection_reasons": reasons})
            if not reasons:
                packed_motion = {"fps": float(motion["fps"])}
                for field in MOTION_FIELDS:
                    packed_motion[field] = torch.as_tensor(motion[field], dtype=torch.float32)
                motions.append(packed_motion)
                source_files.append(relative_source)
                entry["status"] = "accepted"
            else:
                entry["status"] = "rejected"
        except Exception as error:
            entry.update(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        manifest_entries.append(entry)
        print(
            f"checked={index}/{len(candidates)} status={entry['status']} file={relative_source}",
            flush=True,
        )

    errors = [entry for entry in manifest_entries if entry["status"] == "error"]
    if errors:
        raise RuntimeError(f"{len(errors)} pilot candidates failed conversion. First error: {errors[0]}")
    if not motions:
        raise RuntimeError("All AMASS pilot candidates were rejected.")

    total_duration = sum(
        entry["metrics"]["duration_seconds"]
        for entry in manifest_entries
        if entry["status"] == "accepted"
    )
    dataset = {
        "joint_names": list(JOINT_NAMES),
        "body_names": list(BODY_NAMES),
        "motions": motions,
        "source_files": source_files,
        "metadata": {
            "name": "amass_basic_pilot",
            "source_root": str(dataset_root),
            "candidate_count": len(candidates),
            "accepted_count": len(motions),
            "duration_seconds": total_duration,
            "legacy_mapping": {
                "source_dof": 23,
                "target_dof": 29,
                "neutral_target_joints": [
                    "waist_roll_joint",
                    "waist_pitch_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(dataset, temporary_output)
    os.replace(temporary_output, output_path)

    manifest = {
        "input": str(dataset_root),
        "output": str(output_path),
        "xml": str(xml_path),
        "quality_thresholds": {
            "minimum_duration_seconds": args.minimum_duration,
            "maximum_joint_speed_rad_s": args.maximum_joint_speed,
            "maximum_p95_root_horizontal_speed_m_s": args.maximum_p95_root_speed,
            "maximum_p95_root_angular_speed_rad_s": args.maximum_p95_root_angular_speed,
            "minimum_root_height_m": args.minimum_root_height,
        },
        "candidate_count": len(candidates),
        "accepted_count": len(motions),
        "rejected_count": len(candidates) - len(motions),
        "duration_seconds": total_duration,
        "motions": manifest_entries,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    print(
        f"saved={output_path} motions={len(motions)} "
        f"duration_minutes={total_duration / 60.0:.2f} manifest={manifest_path}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Root of the retargeted AMASS G1 pickle tree.")
    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="Current 29-DoF G1 fist MJCF used for Extreme-RGMT FK.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination Extreme-RGMT source .pt dataset.",
    )
    parser.add_argument("--minimum-duration", type=float, default=2.0)
    parser.add_argument("--maximum-joint-speed", type=float, default=8.0)
    parser.add_argument("--maximum-p95-root-speed", type=float, default=1.5)
    parser.add_argument("--maximum-p95-root-angular-speed", type=float, default=2.5)
    parser.add_argument("--minimum-root-height", type=float, default=0.3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.maximum_joint_speed > 3.0 * math.pi:
        raise ValueError("--maximum-joint-speed must not exceed the established 3*pi safety limit.")
    build_dataset(args)


if __name__ == "__main__":
    main()
