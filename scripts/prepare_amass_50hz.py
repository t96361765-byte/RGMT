"""Select and convert legacy GMR AMASS motions for Extreme-RGMT at 50 Hz."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from prepare_amass_gmr_pilot import expand_legacy_g1_23dof
from prepare_lafan1_gmr import (
    BODY_NAMES,
    JOINT_NAMES,
    _atomic_pickle_dump,
    _convert_motion,
    _load_pickle,
    _resample_gmr_qpos,
    _slice_converted_motion,
)


def _candidate_files(root: Path) -> list[tuple[str, Path]]:
    """Return basic/diverse candidates, with ACCAD first and other sets as fallback."""
    candidates: list[tuple[str, Path]] = []

    accad_exclude = re.compile(
        r"(^[EG]\d)|cartwheel|calib|blank|extended|form_|ericcamper|walkdog",
        re.IGNORECASE,
    )
    for path in sorted((root / "accad").glob("*.pkl")):
        if not accad_exclude.search(path.stem):
            candidates.append(("accad", path))

    totalcapture_order = ("walking", "acting")
    totalcapture = sorted((root / "totalcapture").glob("*.pkl"))
    for keyword in totalcapture_order:
        candidates.extend(
            ("totalcapture", path)
            for path in totalcapture
            if keyword in path.stem.lower()
        )

    sfu_basic = re.compile(
        r"walk|balance|crawl|jog|stomp|slowtrot|sideskip|yoga|tiptoe|catwalk|moonwalk",
        re.IGNORECASE,
    )
    candidates.extend(
        ("sfu", path)
        for path in sorted((root / "sfu").glob("*.pkl"))
        if sfu_basic.search(path.stem)
    )

    ssm_dynamic = re.compile(
        r"kick|punch|jump|hop|dribble|special|dance", re.IGNORECASE
    )
    candidates.extend(
        ("ssm_synced", path)
        for path in sorted((root / "ssm_synced").glob("*.pkl"))
        if not ssm_dynamic.search(path.stem)
    )

    transitions_dynamic = re.compile(
        r"airkick|kick|jump|punch|devish|dance|longjump", re.IGNORECASE
    )
    candidates.extend(
        ("transitions", path)
        for path in sorted((root / "transitions").glob("*.pkl"))
        if not transitions_dynamic.search(path.stem)
    )

    # Large corpora are true fallbacks. Only walking-like files are considered,
    # and conversion stops as soon as the requested duration is reached.
    fallback_pattern = re.compile(r"walk|treadmill", re.IGNORECASE)
    for subset in ("biomotionlab_ntroje", "cmu", "kit"):
        candidates.extend(
            (subset, path)
            for path in sorted((root / subset).glob("*.pkl"))
            if fallback_pattern.search(path.stem)
        )
    return candidates


def _prepare_50hz_raw(
    raw: dict[str, Any],
    source_file: str,
    target_fps: float,
) -> tuple[dict[str, Any], str]:
    for key in ("fps", "root_pos", "root_rot", "dof_pos"):
        if key not in raw:
            raise KeyError(f"AMASS motion is missing '{key}'.")
    source_fps = float(raw["fps"])
    root_position = np.asarray(raw["root_pos"], dtype=np.float64)
    root_quaternion_xyzw = np.asarray(raw["root_rot"], dtype=np.float64)
    joint_position = np.asarray(raw["dof_pos"], dtype=np.float64)
    if joint_position.ndim != 2:
        raise ValueError(f"Invalid AMASS dof_pos shape: {joint_position.shape}")
    if joint_position.shape[1] == 23:
        joint_position = expand_legacy_g1_23dof(joint_position).astype(np.float64)
        mapping = "legacy_g1_23dof_neutral_fill"
    elif joint_position.shape[1] == 29:
        mapping = "native_g1_29dof"
    else:
        raise ValueError(f"Unsupported AMASS G1 dof_pos shape: {joint_position.shape}")
    frame_count = joint_position.shape[0]
    if root_position.shape != (frame_count, 3):
        raise ValueError(f"Invalid AMASS root_pos shape: {root_position.shape}")
    if root_quaternion_xyzw.shape != (frame_count, 4):
        raise ValueError(f"Invalid AMASS root_rot shape: {root_quaternion_xyzw.shape}")
    if not all(
        np.isfinite(value).all()
        for value in (root_position, root_quaternion_xyzw, joint_position)
    ):
        raise ValueError("AMASS motion contains NaN or infinite values.")

    qpos_wxyz = np.concatenate(
        (
            root_position,
            root_quaternion_xyzw[:, [3, 0, 1, 2]],
            joint_position,
        ),
        axis=1,
    )
    qpos_wxyz = _resample_gmr_qpos(qpos_wxyz, source_fps, target_fps)
    prepared = {
        "fps": float(target_fps),
        "source_fps": source_fps,
        "source_frame_count": frame_count,
        "source_file": source_file,
        "root_pos": qpos_wxyz[:, :3],
        "root_rot": qpos_wxyz[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos_wxyz[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    return prepared, mapping


def _valid_ranges(
    motion: dict[str, Any],
    speed_limit: float,
    minimum_bad_run: int,
    minimum_duration: float,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    fps = float(motion["fps"])
    joint_position = np.asarray(motion["joint_pos"], dtype=np.float64)
    frame_count = joint_position.shape[0]
    bad = np.any(np.abs(np.diff(joint_position, axis=0) * fps) > speed_limit, axis=1)
    transitions = np.diff(np.r_[False, bad, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    bad_runs = [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    ]
    rejected_runs = [
        (start, end)
        for start, end in bad_runs
        if end - start >= minimum_bad_run
    ]

    valid = np.ones(frame_count, dtype=bool)
    for start, end in rejected_runs:
        valid[start : end + 1] = False
    valid_transitions = np.diff(np.r_[False, valid, False].astype(np.int8))
    valid_starts = np.flatnonzero(valid_transitions == 1)
    valid_ends = np.flatnonzero(valid_transitions == -1)
    minimum_frames = max(2, int(math.ceil(minimum_duration * fps)) + 1)
    ranges = [
        (int(start), int(end))
        for start, end in zip(valid_starts, valid_ends, strict=True)
        if end - start >= minimum_frames
    ]
    return ranges, {
        "bad_intervals": int(bad.sum()),
        "bad_runs": len(bad_runs),
        "rejected_runs": len(rejected_runs),
    }


def build_dataset(args: argparse.Namespace) -> None:
    root = args.input.resolve()
    output = args.output_dir.resolve()
    xml = args.xml.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    existing = list(output.glob("*.pkl")) if output.is_dir() else []
    if existing and not args.overwrite:
        raise FileExistsError(f"{output} already contains {len(existing)} pickle files.")
    output.mkdir(parents=True, exist_ok=True)

    target_seconds = float(args.target_hours) * 3600.0
    selected_seconds = 0.0
    selected_frames = 0
    selected_sources: set[str] = set()
    subset_seconds: dict[str, float] = {}
    entries: list[dict[str, Any]] = []
    output_count = 0
    written_names: set[str] = set()

    candidates = _candidate_files(root)
    for candidate_index, (subset, source) in enumerate(candidates, start=1):
        if target_seconds - selected_seconds < args.minimum_duration:
            break
        relative = source.relative_to(root).as_posix()
        entry: dict[str, Any] = {"source_file": relative, "subset": subset}
        try:
            prepared, mapping = _prepare_50hz_raw(
                _load_pickle(source), relative, args.target_fps
            )
            motion = _convert_motion(prepared, xml)
            ranges, quality = _valid_ranges(
                motion,
                args.speed_limit,
                args.minimum_consecutive_bad_intervals,
                args.minimum_duration,
            )
            entry.update(
                {
                    "mapping": mapping,
                    "source_fps": float(prepared["source_fps"]),
                    "target_fps": float(args.target_fps),
                    "quality": quality,
                    "candidate_segments": len(ranges),
                    "selected_segments": [],
                }
            )
            for segment_index, (start, end) in enumerate(ranges, start=1):
                remaining = target_seconds - selected_seconds
                if remaining < args.minimum_duration:
                    break
                duration = (end - start - 1) / args.target_fps
                if duration > remaining:
                    end = start + int(math.floor(remaining * args.target_fps)) + 1
                    duration = (end - start - 1) / args.target_fps
                if duration < args.minimum_duration:
                    continue
                filename = (
                    f"{subset}__{source.stem}__segment_{segment_index:04d}.pkl"
                )
                segment = _slice_converted_motion(motion, start, end, relative)
                _atomic_pickle_dump(segment, output / filename)
                written_names.add(filename)
                output_count += 1
                selected_seconds += duration
                selected_frames += end - start
                selected_sources.add(relative)
                subset_seconds[subset] = subset_seconds.get(subset, 0.0) + duration
                entry["selected_segments"].append(
                    {
                        "file": filename,
                        "frame_range_50hz": [start, end],
                        "frames": end - start,
                        "duration_seconds": duration,
                    }
                )
            entry["status"] = "selected" if entry["selected_segments"] else "not_selected"
        except Exception as error:
            entry.update(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        entries.append(entry)
        print(
            f"checked={candidate_index}/{len(candidates)} subset={subset} "
            f"status={entry['status']} selected_minutes={selected_seconds / 60.0:.2f}",
            flush=True,
        )

    if output_count == 0:
        raise RuntimeError("No AMASS motion segments were selected.")
    if selected_seconds < target_seconds - args.minimum_duration:
        raise RuntimeError(
            f"Candidate pool only provided {selected_seconds / 3600.0:.3f} h, "
            f"below target {args.target_hours:.3f} h."
        )

    if args.overwrite:
        for stale in existing:
            if stale.name not in written_names:
                stale.unlink()

    manifest = {
        "input": str(root),
        "output": str(output),
        "xml": str(xml),
        "target_duration_hours": args.target_hours,
        "selected_duration_hours": selected_seconds / 3600.0,
        "selected_frames": selected_frames,
        "selected_segment_count": output_count,
        "selected_source_count": len(selected_sources),
        "subset_duration_hours": {
            key: value / 3600.0 for key, value in subset_seconds.items()
        },
        "conversion": {
            "target_fps": args.target_fps,
            "speed_limit_rad_s": args.speed_limit,
            "minimum_consecutive_bad_intervals": args.minimum_consecutive_bad_intervals,
            "minimum_segment_duration_seconds": args.minimum_duration,
            "legacy_neutral_fill_joints": [
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
        },
        "motions": entries,
    }
    manifest_path = output / "selection_manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(
        f"saved={output} segments={output_count} sources={len(selected_sources)} "
        f"duration_hours={selected_seconds / 3600.0:.6f}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--target-hours", type=float, required=True)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--speed-limit", type=float, default=3.0 * math.pi)
    parser.add_argument("--minimum-consecutive-bad-intervals", type=int, default=2)
    parser.add_argument("--minimum-duration", type=float, default=1.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite deterministic output files from an interrupted identical run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.target_hours <= 0.0:
        raise ValueError("--target-hours must be positive.")
    if args.target_fps <= 0.0:
        raise ValueError("--target-fps must be positive.")
    if args.speed_limit <= 0.0:
        raise ValueError("--speed-limit must be positive.")
    if args.minimum_consecutive_bad_intervals <= 0:
        raise ValueError("--minimum-consecutive-bad-intervals must be positive.")
    if args.minimum_duration <= 0.0:
        raise ValueError("--minimum-duration must be positive.")
    build_dataset(args)


if __name__ == "__main__":
    main()
