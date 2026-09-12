"""Build the source-aware intermediate mix for Extreme-RGMT Stage I."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch


REQUIRED_FIELDS = (
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "joint_pos",
    "joint_vel",
    "body_pos",
    "body_lin_vel",
)


def _load_dataset(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(dataset, dict):
        raise TypeError(f"{path} must contain a dictionary.")
    for key in ("joint_names", "body_names", "motions"):
        if key not in dataset:
            raise KeyError(f"{path} is missing '{key}'.")
    if not dataset["motions"]:
        raise ValueError(f"{name} dataset contains no motions.")
    return dataset


def _duration_seconds(motion: dict[str, Any], source: str) -> float:
    if "fps" not in motion:
        raise KeyError(f"{source} is missing 'fps'.")
    fps = float(motion["fps"])
    if fps <= 0.0:
        raise ValueError(f"{source} has invalid fps={fps}.")
    for field in REQUIRED_FIELDS:
        if field not in motion:
            raise KeyError(f"{source} is missing '{field}'.")
    frame_count = int(motion["joint_pos"].shape[0])
    if frame_count < 2:
        raise ValueError(f"{source} must contain at least two frames.")
    if any(int(motion[field].shape[0]) != frame_count for field in REQUIRED_FIELDS):
        raise ValueError(f"{source} fields do not share the same frame count.")
    return (frame_count - 1) / fps


def _source_files(dataset: dict[str, Any], dataset_path: Path, label: str) -> list[str]:
    motion_count = len(dataset["motions"])
    stored_sources = dataset.get("source_files")
    if stored_sources is not None:
        if len(stored_sources) != motion_count:
            raise ValueError(f"{dataset_path} source_files length does not match its motions.")
        return [f"{label}/{source}" for source in stored_sources]

    # Recover the retained LAFAN1 segment names when the adjacent selection
    # manifest is available. Fall back to stable clip indices otherwise.
    lafan_manifest = dataset_path.parent / "lafan1_pkl_select" / "selection_manifest.json"
    if label == "lafan1" and lafan_manifest.is_file():
        manifest = json.loads(lafan_manifest.read_text(encoding="utf-8"))
        segments = manifest.get("segments", [])
        if len(segments) == motion_count:
            return [f"{label}/{segment['file']}" for segment in segments]

    return [f"{label}/clip_{index:05d}" for index in range(motion_count)]


def _validate_shared_schema(
    amass: dict[str, Any],
    lafan1: dict[str, Any],
    amass_path: Path,
    lafan1_path: Path,
) -> None:
    if list(amass["joint_names"]) != list(lafan1["joint_names"]):
        raise ValueError(f"Joint order differs between {amass_path} and {lafan1_path}.")
    if list(amass["body_names"]) != list(lafan1["body_names"]):
        raise ValueError(f"Body order differs between {amass_path} and {lafan1_path}.")


def _group_weights(durations: list[float], group_probability: float) -> list[float]:
    total_duration = sum(durations)
    if total_duration <= 0.0:
        raise ValueError("A source group has non-positive total duration.")
    return [group_probability * duration / total_duration for duration in durations]


def mix_datasets(args: argparse.Namespace) -> None:
    amass_path = args.amass.resolve()
    lafan1_path = args.lafan1.resolve()
    output_path = args.output.resolve()
    if args.amass_weight <= 0.0 or args.lafan1_weight <= 0.0:
        raise ValueError("Source sampling weights must be positive.")

    amass = _load_dataset(amass_path, "AMASS")
    lafan1 = _load_dataset(lafan1_path, "LAFAN1")
    _validate_shared_schema(amass, lafan1, amass_path, lafan1_path)

    amass_sources = _source_files(amass, amass_path, "amass")
    lafan1_sources = _source_files(lafan1, lafan1_path, "lafan1")
    amass_durations = [
        _duration_seconds(motion, source)
        for motion, source in zip(amass["motions"], amass_sources, strict=True)
    ]
    lafan1_durations = [
        _duration_seconds(motion, source)
        for motion, source in zip(lafan1["motions"], lafan1_sources, strict=True)
    ]

    weight_sum = args.amass_weight + args.lafan1_weight
    amass_probability = args.amass_weight / weight_sum
    lafan1_probability = args.lafan1_weight / weight_sum
    sampling_weights = _group_weights(amass_durations, amass_probability)
    sampling_weights.extend(_group_weights(lafan1_durations, lafan1_probability))

    motions = list(amass["motions"]) + list(lafan1["motions"])
    source_files = amass_sources + lafan1_sources
    source_labels = ["amass"] * len(amass["motions"]) + ["lafan1"] * len(lafan1["motions"])
    sampling_groups = [0] * len(amass["motions"]) + [1] * len(lafan1["motions"])
    output = {
        "joint_names": list(amass["joint_names"]),
        "body_names": list(amass["body_names"]),
        "motions": motions,
        "sampling_weights": torch.tensor(sampling_weights, dtype=torch.float32),
        "sampling_groups": torch.tensor(sampling_groups, dtype=torch.long),
        "sampling_group_names": ["amass", "lafan1"],
        "source_labels": source_labels,
        "source_files": source_files,
        "metadata": {
            "name": "Extreme_RGMT_stage1_source_mix",
            "sampling_policy": "source_ratio_then_eligible_duration_within_source",
            "sources": {
                "amass": {
                    "path": str(amass_path),
                    "motion_count": len(amass["motions"]),
                    "duration_seconds": sum(amass_durations),
                    "sampling_probability": amass_probability,
                },
                "lafan1": {
                    "path": str(lafan1_path),
                    "motion_count": len(lafan1["motions"]),
                    "duration_seconds": sum(lafan1_durations),
                    "sampling_probability": lafan1_probability,
                },
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(output, temporary_output)
    os.replace(temporary_output, output_path)

    manifest = {
        "output": str(output_path),
        "sampling_policy": output["metadata"]["sampling_policy"],
        "requested_ratio": {
            "amass": args.amass_weight,
            "lafan1": args.lafan1_weight,
        },
        "normalized_sampling_probability": {
            "amass": amass_probability,
            "lafan1": lafan1_probability,
        },
        "sources": output["metadata"]["sources"],
        "total_motion_count": len(motions),
        "total_duration_seconds": sum(amass_durations) + sum(lafan1_durations),
        "sampling_weight_sum": float(output["sampling_weights"].sum()),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)

    print(
        f"saved={output_path} motions={len(motions)} "
        f"amass_probability={amass_probability:.6f} "
        f"lafan1_probability={lafan1_probability:.6f} "
        f"manifest={manifest_path}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass", type=Path, required=True)
    parser.add_argument("--lafan1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amass-weight", type=float, default=1.0)
    parser.add_argument("--lafan1-weight", type=float, default=2.0)
    return parser


def main() -> None:
    mix_datasets(build_parser().parse_args())


if __name__ == "__main__":
    main()
