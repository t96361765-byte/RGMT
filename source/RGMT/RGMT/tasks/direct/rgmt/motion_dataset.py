"""Validated motion dataset for the Extreme-RGMT Stage-I environment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch


_REQUIRED_FIELDS = (
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "joint_pos",
    "joint_vel",
    "body_pos",
    "body_lin_vel",
    "body_quat",
    "body_ang_vel",
)

# Isaac/PhysX articulation ordering produced by the BeyondMimic G1 model used to
# create ``D:\track_dataset\lafan1_npz``. NPZ does not store names, so keeping
# this explicit is essential: its breadth-first articulation order differs from
# the kinematic order used by the original RGMT PT dataset.
_BEYONDMIMIC_G1_JOINT_NAMES = (
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

_BEYONDMIMIC_G1_BODY_NAMES = (
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

_BEYONDMIMIC_NPZ_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class ExtremeRGMTMotionDataset:
    """Load strictly compatible Extreme-RGMT motion clips."""

    def __init__(
        self,
        motion_file: str,
        robot_joint_names: list[str],
        requested_body_names: list[str],
        device: str | torch.device,
        adaptive_bin_duration_s: float = 1.0,
        adaptive_ema_alpha: float = 0.05,
        adaptive_score_clip: float = 1.0,
        adaptive_uniform_ratio: float = 0.2,
    ):
        path = Path(motion_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Extreme-RGMT motion dataset not found: {path}. "
                "Add the processed dataset or use task 'Isaac-RGMT-G1-Debug-v0'."
            )
        if path.is_dir() or path.suffix.lower() == ".npz":
            raw = self._load_beyondmimic_npz(path, robot_joint_names, requested_body_names)
        elif path.is_file() and path.suffix.lower() in {".pt", ".pth"}:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        else:
            raise ValueError(
                f"Unsupported Extreme-RGMT dataset '{path}'. Expected a .pt/.pth file, "
                "a BeyondMimic .npz file, or a directory containing .npz files."
            )

        self.device = torch.device(device)
        self.required_fields = _REQUIRED_FIELDS
        self._validate_root(raw, path)

        source_files = raw.get("source_files")
        if source_files is None:
            self.source_files = [f"clip_{index:05d}" for index in range(len(raw["motions"]))]
        else:
            if len(source_files) != len(raw["motions"]):
                raise ValueError(
                    f"{path} has {len(source_files)} source names for {len(raw['motions'])} motion clips."
                )
            self.source_files = [str(name) for name in source_files]

        dataset_joint_names = list(raw["joint_names"])
        dataset_body_names = list(raw["body_names"])
        self.joint_indices = self._resolve_names(robot_joint_names, dataset_joint_names, "joint")
        self.body_indices = self._resolve_names(requested_body_names, dataset_body_names, "body")

        self.clips: list[dict[str, torch.Tensor]] = []
        durations = []
        for clip_index, clip_raw in enumerate(raw["motions"]):
            clip = self._validate_and_prepare_clip(clip_raw, clip_index)
            self.clips.append(clip)
            durations.append((clip["joint_pos"].shape[0] - 1) / float(clip["fps"].item()))

        self.durations = torch.tensor(durations, dtype=torch.float32, device=self.device)
        if "sampling_weights" in raw:
            sampling_weights = torch.as_tensor(
                raw["sampling_weights"], dtype=torch.float32, device=self.device
            )
            if sampling_weights.shape != self.durations.shape:
                raise ValueError(
                    f"{path} has {sampling_weights.numel()} sampling weights for "
                    f"{self.durations.numel()} motion clips."
                )
            if not torch.isfinite(sampling_weights).all():
                raise ValueError(f"{path} contains non-finite sampling weights.")
            if torch.any(sampling_weights < 0.0):
                raise ValueError(f"{path} contains negative sampling weights.")
            if sampling_weights.sum() <= 0.0:
                raise ValueError(f"{path} sampling weights must have a positive sum.")
            self.sample_weights = sampling_weights / sampling_weights.sum()
        else:
            self.sample_weights = self.durations / self.durations.sum()
        self.sampling_groups: torch.Tensor | None = None
        self._sampling_group_ids: list[int] = []
        self._sampling_group_probabilities: dict[int, torch.Tensor] = {}
        if "sampling_groups" in raw:
            sampling_groups = torch.as_tensor(
                raw["sampling_groups"], dtype=torch.long, device=self.device
            )
            if sampling_groups.shape != self.durations.shape:
                raise ValueError(
                    f"{path} has {sampling_groups.numel()} sampling groups for "
                    f"{self.durations.numel()} motion clips."
                )
            if torch.any(sampling_groups < 0):
                raise ValueError(f"{path} contains negative sampling group IDs.")
            self.sampling_groups = sampling_groups
            self._sampling_group_ids = [int(group_id) for group_id in torch.unique(sampling_groups).tolist()]
            for group_id in self._sampling_group_ids:
                group_mask = sampling_groups == group_id
                probability = self.sample_weights[group_mask].sum()
                if probability <= 0.0:
                    raise ValueError(f"{path} sampling group {group_id} has no positive weight.")
                self._sampling_group_probabilities[group_id] = probability
        frame_counts = [clip["joint_pos"].shape[0] for clip in self.clips]
        self._frame_counts = torch.tensor(frame_counts, dtype=torch.long, device=self.device)
        self._frame_starts = torch.cumsum(
            torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=self.device),
                    self._frame_counts[:-1],
                )
            ),
            dim=0,
        )
        self._fps = torch.stack([clip["fps"] for clip in self.clips])
        self._packed = {
            key: torch.cat([clip[key] for clip in self.clips], dim=0)
            for key in self.required_fields
        }

        # Preserve the existing ``clips`` interface with views into the packed
        # tensors, avoiding a second copy of the complete dataset.
        frame_start = 0
        for clip, frame_count in zip(self.clips, frame_counts, strict=True):
            frame_end = frame_start + frame_count
            for key in self.required_fields:
                clip[key] = self._packed[key][frame_start:frame_end]
            frame_start = frame_end

        self.adaptive_bin_duration_s = float(adaptive_bin_duration_s)
        self.adaptive_ema_alpha = float(adaptive_ema_alpha)
        self.adaptive_score_clip = float(adaptive_score_clip)
        self.adaptive_uniform_ratio = float(adaptive_uniform_ratio)
        if self.adaptive_bin_duration_s <= 0.0:
            raise ValueError("Adaptive sampling bin duration must be positive.")
        if not 0.0 < self.adaptive_ema_alpha <= 1.0:
            raise ValueError("Adaptive sampling EMA alpha must be in (0, 1].")
        if self.adaptive_score_clip <= 0.0:
            raise ValueError("Adaptive sampling score clip must be positive.")
        if self.adaptive_uniform_ratio <= 0.0:
            raise ValueError("Adaptive sampling uniform ratio must be positive.")
        self._initialize_adaptive_bins()

    @classmethod
    def _load_beyondmimic_npz(
        cls,
        path: Path,
        robot_joint_names: list[str],
        requested_body_names: list[str],
    ) -> dict[str, Any]:
        """Load one or more BeyondMimic NPZ clips into RGMT's internal schema."""
        files = sorted(path.rglob("*.npz")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No .npz motion files found under {path}.")

        joint_indices = cls._resolve_names(
            robot_joint_names, list(_BEYONDMIMIC_G1_JOINT_NAMES), "BeyondMimic joint"
        )
        body_indices = cls._resolve_names(
            requested_body_names, list(_BEYONDMIMIC_G1_BODY_NAMES), "BeyondMimic body"
        )
        pelvis_index = _BEYONDMIMIC_G1_BODY_NAMES.index("pelvis")
        motions: list[dict[str, Any]] = []
        source_files: list[str] = []

        for file_index, file in enumerate(files):
            with np.load(file, allow_pickle=False) as data:
                missing = [name for name in _BEYONDMIMIC_NPZ_FIELDS if name not in data]
                if missing:
                    raise KeyError(f"{file} is missing BeyondMimic fields: {missing}")
                fps_values = np.asarray(data["fps"]).reshape(-1)
                if fps_values.size != 1:
                    raise ValueError(f"{file} must contain exactly one FPS value.")
                fps = float(fps_values[0])
                if not np.isclose(fps, 50.0, rtol=0.0, atol=1.0e-6):
                    raise ValueError(f"{file} has fps={fps}; this RGMT dataset requires 50 Hz.")

                joint_pos = np.asarray(data["joint_pos"])
                joint_vel = np.asarray(data["joint_vel"])
                body_pos = np.asarray(data["body_pos_w"])
                body_quat = np.asarray(data["body_quat_w"])
                body_lin_vel = np.asarray(data["body_lin_vel_w"])
                body_ang_vel = np.asarray(data["body_ang_vel_w"])
                if joint_pos.ndim != 2 or joint_pos.shape[1] != len(_BEYONDMIMIC_G1_JOINT_NAMES):
                    raise ValueError(
                        f"{file} has joint_pos shape {joint_pos.shape}; expected [T, 29]."
                    )
                if joint_vel.shape != joint_pos.shape:
                    raise ValueError(f"{file} joint_vel shape does not match joint_pos.")
                expected_body_prefix = (joint_pos.shape[0], len(_BEYONDMIMIC_G1_BODY_NAMES))
                expected_shapes = {
                    "body_pos_w": expected_body_prefix + (3,),
                    "body_quat_w": expected_body_prefix + (4,),
                    "body_lin_vel_w": expected_body_prefix + (3,),
                    "body_ang_vel_w": expected_body_prefix + (3,),
                }
                arrays = {
                    "body_pos_w": body_pos,
                    "body_quat_w": body_quat,
                    "body_lin_vel_w": body_lin_vel,
                    "body_ang_vel_w": body_ang_vel,
                }
                for name, expected_shape in expected_shapes.items():
                    if arrays[name].shape != expected_shape:
                        raise ValueError(
                            f"{file} has {name} shape {arrays[name].shape}; expected {expected_shape}."
                        )
                for name, value in (("joint_pos", joint_pos), ("joint_vel", joint_vel), *arrays.items()):
                    if not np.isfinite(value).all():
                        raise ValueError(f"{file} contains non-finite values in {name}.")

                # Copy only the state RGMT consumes. This drops 16 untracked bodies before upload to GPU.
                motion = {
                    "fps": fps,
                    "root_pos": body_pos[:, pelvis_index].copy(),
                    "root_quat": body_quat[:, pelvis_index].copy(),
                    "root_lin_vel": body_lin_vel[:, pelvis_index].copy(),
                    "root_ang_vel": body_ang_vel[:, pelvis_index].copy(),
                    "joint_pos": joint_pos[:, joint_indices].copy(),
                    "joint_vel": joint_vel[:, joint_indices].copy(),
                    "body_pos": body_pos[:, body_indices].copy(),
                    "body_quat": body_quat[:, body_indices].copy(),
                    "body_lin_vel": body_lin_vel[:, body_indices].copy(),
                    "body_ang_vel": body_ang_vel[:, body_indices].copy(),
                }
                motions.append(motion)
                source_files.append(
                    file.relative_to(path).as_posix() if path.is_dir() else str(file)
                )
                print(
                    f"[INFO]: Loaded BeyondMimic NPZ {file_index + 1}/{len(files)}: "
                    f"{file.name} ({joint_pos.shape[0]} frames, {fps:g} Hz)",
                    flush=True,
                )

        return {
            "joint_names": list(robot_joint_names),
            "body_names": list(requested_body_names),
            "motions": motions,
            "source_files": source_files,
            "metadata": {
                "format": "beyondmimic_npz",
                "source": str(path.resolve()),
                "clip_count": len(motions),
            },
        }

    @staticmethod
    def _validate_root(raw: Any, path: Path) -> None:
        if not isinstance(raw, dict):
            raise TypeError(f"{path} must contain a dictionary.")
        for key in ("joint_names", "body_names", "motions"):
            if key not in raw:
                raise KeyError(f"{path} is missing top-level key '{key}'.")
        if not raw["motions"]:
            raise ValueError(f"{path} contains no motion clips.")

    @staticmethod
    def _resolve_names(requested: list[str], available: list[str], kind: str) -> list[int]:
        missing = [name for name in requested if name not in available]
        if missing:
            raise ValueError(f"Motion dataset is missing {kind} names: {missing}")
        return [available.index(name) for name in requested]

    def _validate_and_prepare_clip(self, raw: dict[str, Any], clip_index: int) -> dict[str, torch.Tensor]:
        if "fps" not in raw:
            raise KeyError(f"Motion clip {clip_index} is missing 'fps'.")
        fps = float(raw["fps"])
        if fps <= 0.0:
            raise ValueError(f"Motion clip {clip_index} has invalid fps={fps}.")
        for key in self.required_fields:
            if key not in raw:
                raise KeyError(f"Motion clip {clip_index} is missing '{key}'.")

        clip = {
            key: torch.as_tensor(raw[key], dtype=torch.float32, device=self.device).contiguous()
            for key in self.required_fields
        }
        num_frames = clip["joint_pos"].shape[0]
        if num_frames < 2:
            raise ValueError(f"Motion clip {clip_index} needs at least two frames.")
        if any(value.shape[0] != num_frames for value in clip.values()):
            raise ValueError(f"All tensors in motion clip {clip_index} must have the same frame count.")
        if clip["root_pos"].shape[1:] != (3,) or clip["root_quat"].shape[1:] != (4,):
            raise ValueError(f"Motion clip {clip_index} has invalid root pose shapes.")
        if clip["joint_pos"].shape[1] <= max(self.joint_indices):
            raise ValueError(f"Motion clip {clip_index} does not contain all mapped joints.")
        if clip["body_pos"].shape[1] <= max(self.body_indices):
            raise ValueError(f"Motion clip {clip_index} does not contain all mapped bodies.")
        if clip["body_quat"].ndim != 3 or clip["body_quat"].shape[2] != 4:
            raise ValueError(f"Motion clip {clip_index} has invalid body_quat shape.")
        if clip["body_ang_vel"].ndim != 3 or clip["body_ang_vel"].shape[2] != 3:
            raise ValueError(f"Motion clip {clip_index} has invalid body_ang_vel shape.")

        # Reorder once at load time so environment tensors always follow Isaac Lab's ordering.
        clip["joint_pos"] = clip["joint_pos"][:, self.joint_indices]
        clip["joint_vel"] = clip["joint_vel"][:, self.joint_indices]
        clip["body_pos"] = clip["body_pos"][:, self.body_indices]
        clip["body_lin_vel"] = clip["body_lin_vel"][:, self.body_indices]
        clip["body_quat"] = torch.nn.functional.normalize(
            clip["body_quat"][:, self.body_indices], dim=-1
        )
        clip["body_ang_vel"] = clip["body_ang_vel"][:, self.body_indices]
        clip["root_quat"] = torch.nn.functional.normalize(clip["root_quat"], dim=-1)
        clip["fps"] = torch.tensor(fps, dtype=torch.float32, device=self.device)
        return clip

    def _initialize_adaptive_bins(self) -> None:
        """Create fixed temporal bins and their EMA difficulty scores."""
        counts = torch.ceil(self.durations / self.adaptive_bin_duration_s).long()
        counts = torch.clamp(counts, min=1)
        self._bin_counts = counts
        self._bin_offsets = torch.cumsum(
            torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=self.device),
                    counts[:-1],
                )
            ),
            dim=0,
        )
        self._bin_clip_ids = torch.repeat_interleave(
            torch.arange(len(self.clips), device=self.device), counts
        )
        flat_ids = torch.arange(self._bin_clip_ids.numel(), device=self.device)
        self._bin_local_ids = flat_ids - self._bin_offsets[self._bin_clip_ids]
        self._bin_start_times = (
            self._bin_local_ids.float() * self.adaptive_bin_duration_s
        )
        self._bin_scores = torch.zeros_like(self._bin_start_times)
        self._bin_base_weights = (
            self.sample_weights[self._bin_clip_ids]
            / self._bin_counts[self._bin_clip_ids].float()
        )

    def _adaptive_bin_weights(self, eligible: torch.Tensor) -> torch.Tensor:
        """Equation (13), while preserving configured source-group priors."""
        score = torch.clamp(
            self._bin_scores / self.adaptive_score_clip, min=0.0, max=1.0
        )
        weights = torch.zeros_like(score)
        if self.sampling_groups is None:
            group_ids = [None]
        else:
            group_ids = self._sampling_group_ids

        for group_id in group_ids:
            if group_id is None:
                group_mask = eligible
                group_probability = torch.tensor(1.0, device=self.device)
            else:
                group_mask = (
                    eligible
                    & (self.sampling_groups[self._bin_clip_ids] == group_id)
                )
                group_probability = self._sampling_group_probabilities[group_id]
            if not torch.any(group_mask):
                continue
            base = self._bin_base_weights[group_mask]
            base = base / base.sum()
            difficulty = score[group_mask]
            if difficulty.sum() > 0.0:
                difficulty = difficulty / difficulty.sum()
            else:
                difficulty = torch.zeros_like(difficulty)
            group_weights = difficulty + self.adaptive_uniform_ratio * base
            group_weights = group_weights / group_weights.sum()
            weights[group_mask] = group_probability * group_weights
        return weights

    def sample_starts(self, count: int, future_horizon_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample clip IDs and valid start times that leave room for the future command window."""
        max_start = self.durations[self._bin_clip_ids] - future_horizon_s
        bin_end = torch.minimum(
            self._bin_start_times + self.adaptive_bin_duration_s,
            max_start,
        )
        eligible = bin_end >= self._bin_start_times
        if not torch.any(eligible):
            raise ValueError(
                f"No adaptive motion bin is long enough for a {future_horizon_s:.3f} s horizon."
            )
        weights = self._adaptive_bin_weights(eligible)
        if weights.sum() <= 0.0:
            raise RuntimeError("Adaptive motion sampling produced zero probability.")
        sampled_bins = torch.multinomial(
            weights / weights.sum(), count, replacement=True
        )
        clip_ids = self._bin_clip_ids[sampled_bins]
        starts = self._bin_start_times[sampled_bins]
        ends = torch.minimum(
            starts + self.adaptive_bin_duration_s,
            self.durations[clip_ids] - future_horizon_s,
        )
        start_times = starts + torch.rand(count, device=self.device) * (
            ends - starts
        )
        return clip_ids, start_times

    def sample_uniform_starts(
        self, count: int, future_horizon_s: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample valid time bins, without adaptive difficulty scores.

        Stage II uses this path for the mastered/consolidation environments.
        The base bin weights retain the dataset's clip/group proportions while
        removing the failure-driven acquisition bias.
        """
        max_start = self.durations[self._bin_clip_ids] - future_horizon_s
        bin_end = torch.minimum(
            self._bin_start_times + self.adaptive_bin_duration_s,
            max_start,
        )
        eligible = bin_end >= self._bin_start_times
        if not torch.any(eligible):
            raise ValueError(
                f"No motion bin is long enough for a {future_horizon_s:.3f} s horizon."
            )
        weights = torch.where(eligible, self._bin_base_weights, torch.zeros_like(self._bin_base_weights))
        sampled_bins = torch.multinomial(weights / weights.sum(), count, replacement=True)
        clip_ids = self._bin_clip_ids[sampled_bins]
        starts = self._bin_start_times[sampled_bins]
        ends = torch.minimum(
            starts + self.adaptive_bin_duration_s,
            self.durations[clip_ids] - future_horizon_s,
        )
        start_times = starts + torch.rand(count, device=self.device) * (ends - starts)
        return clip_ids, start_times

    def flat_bin_ids(self, clip_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Map clip/time pairs to flattened temporal-bin identifiers."""
        local_bins = torch.floor(times / self.adaptive_bin_duration_s).long()
        local_bins = torch.clamp(local_bins, min=0)
        local_bins = torch.minimum(local_bins, self._bin_counts[clip_ids] - 1)
        return self._bin_offsets[clip_ids] + local_bins

    def adaptive_probabilities(self) -> torch.Tensor:
        """Return the current normalized acquisition probability of every bin."""
        eligible = torch.ones_like(self._bin_scores, dtype=torch.bool)
        weights = self._adaptive_bin_weights(eligible)
        return weights / torch.clamp(weights.sum(), min=torch.finfo(weights.dtype).eps)

    def update_adaptive_scores(
        self,
        clip_ids: torch.Tensor,
        start_times: torch.Tensor,
        failures: torch.Tensor,
    ) -> None:
        """Update temporal-bin EMA scores from completed rollout outcomes."""
        if clip_ids.numel() == 0:
            return
        local_bins = torch.floor(
            start_times / self.adaptive_bin_duration_s
        ).long()
        local_bins = torch.minimum(local_bins, self._bin_counts[clip_ids] - 1)
        flat_bins = self._bin_offsets[clip_ids] + local_bins
        unique_bins, inverse = torch.unique(flat_bins, return_inverse=True)
        sums = torch.zeros(len(unique_bins), device=self.device)
        counts = torch.zeros(len(unique_bins), device=self.device)
        sums.scatter_add_(0, inverse, failures.float())
        counts.scatter_add_(0, inverse, torch.ones_like(failures, dtype=torch.float32))
        observed = sums / torch.clamp(counts, min=1.0)
        alpha = self.adaptive_ema_alpha
        self._bin_scores[unique_bins] = (
            (1.0 - alpha) * self._bin_scores[unique_bins] + alpha * observed
        )

    def adaptive_fingerprint(self) -> str:
        """Return a stable fingerprint for validating adaptive sampler state."""
        digest = hashlib.sha256()
        tensors = (self._bin_counts, self.durations, self.sample_weights)
        for tensor in tensors:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        if self.sampling_groups is not None:
            digest.update(
                self.sampling_groups.detach().cpu().contiguous().numpy().tobytes()
            )
        return digest.hexdigest()

    def adaptive_state_dict(self) -> dict[str, Any]:
        """Return a portable CPU snapshot of the adaptive motion sampler."""
        return {
            "format_version": 1,
            "bin_scores": self._bin_scores.detach().cpu(),
            "bin_counts": self._bin_counts.detach().cpu(),
            "bin_offsets": self._bin_offsets.detach().cpu(),
            "bin_duration_s": self.adaptive_bin_duration_s,
            "ema_alpha": self.adaptive_ema_alpha,
            "score_clip": self.adaptive_score_clip,
            "uniform_ratio": self.adaptive_uniform_ratio,
            "durations": self.durations.detach().cpu(),
            "sample_weights": self.sample_weights.detach().cpu(),
            "sampling_groups": (
                None
                if self.sampling_groups is None
                else self.sampling_groups.detach().cpu()
            ),
            "dataset_fingerprint": self.adaptive_fingerprint(),
        }

    def load_adaptive_state_dict(self, state: dict[str, Any]) -> None:
        """Restore EMA bin scores after strict dataset/configuration checks."""
        if int(state.get("format_version", -1)) != 1:
            raise ValueError(
                f"Unsupported adaptive sampler format_version={state.get('format_version')}."
            )
        expected_fingerprint = self.adaptive_fingerprint()
        if state.get("dataset_fingerprint") != expected_fingerprint:
            raise ValueError(
                "Adaptive sampler state belongs to a different motion dataset: "
                f"expected {expected_fingerprint}, got {state.get('dataset_fingerprint')}."
            )
        scalar_checks = {
            "bin_duration_s": self.adaptive_bin_duration_s,
            "ema_alpha": self.adaptive_ema_alpha,
            "score_clip": self.adaptive_score_clip,
            "uniform_ratio": self.adaptive_uniform_ratio,
        }
        for name, expected in scalar_checks.items():
            actual = float(state.get(name, float("nan")))
            if not torch.isclose(torch.tensor(actual), torch.tensor(expected), rtol=1.0e-6, atol=1.0e-8):
                raise ValueError(
                    f"Adaptive sampler {name} mismatch: expected {expected}, got {actual}."
                )
        scores = torch.as_tensor(state["bin_scores"], dtype=self._bin_scores.dtype)
        if scores.shape != self._bin_scores.shape:
            raise ValueError(
                f"Adaptive bin score shape mismatch: expected {tuple(self._bin_scores.shape)}, "
                f"got {tuple(scores.shape)}."
            )
        if not torch.isfinite(scores).all():
            raise ValueError("Adaptive sampler state contains non-finite bin scores.")
        if torch.any(scores < 0.0) or torch.any(scores > self.adaptive_score_clip):
            raise ValueError(
                f"Adaptive bin scores must be in [0, {self.adaptive_score_clip}]."
            )
        counts = torch.as_tensor(state["bin_counts"], dtype=self._bin_counts.dtype)
        offsets = torch.as_tensor(state["bin_offsets"], dtype=self._bin_offsets.dtype)
        if not torch.equal(counts.cpu(), self._bin_counts.detach().cpu()):
            raise ValueError("Adaptive sampler bin counts do not match the current dataset.")
        if not torch.equal(offsets.cpu(), self._bin_offsets.detach().cpu()):
            raise ValueError("Adaptive sampler bin offsets do not match the current dataset.")
        self._bin_scores.copy_(scores.to(self.device))

    def sample(self, clip_ids: torch.Tensor, times: torch.Tensor) -> dict[str, torch.Tensor]:
        """Linearly interpolate motion state at ``times``.

        ``clip_ids`` has shape ``[N]`` and ``times`` may have shape ``[N]`` or
        ``[N, K]``. Returned tensors preserve those leading dimensions.
        Quaternions use sign-corrected normalized interpolation.
        """
        if times.shape[0] != clip_ids.shape[0]:
            raise ValueError("The first dimension of times must match clip_ids.")
        original_shape = times.shape
        samples_per_clip = times[0].numel() if times.ndim > 1 else 1
        flat_times = times.reshape(-1)
        flat_ids = clip_ids[:, None].expand(-1, samples_per_clip).reshape(-1)
        output: dict[str, torch.Tensor] = {}

        query = torch.minimum(torch.clamp(flat_times, min=0.0), self.durations[flat_ids])
        frame = query * self._fps[flat_ids]
        i0 = torch.floor(frame).long()
        i1 = torch.minimum(i0 + 1, self._frame_counts[flat_ids] - 1)
        alpha = frame - i0.float()
        global_i0 = self._frame_starts[flat_ids] + i0
        global_i1 = self._frame_starts[flat_ids] + i1

        for key in self.required_fields:
            value0 = self._packed[key][global_i0]
            value1 = self._packed[key][global_i1]
            blend = alpha.view((-1,) + (1,) * (value0.ndim - 1))
            if key in {"root_quat", "body_quat"}:
                sign = torch.where(
                    torch.sum(value0 * value1, dim=-1, keepdim=True) < 0.0,
                    -torch.ones_like(value1[..., :1]),
                    torch.ones_like(value1[..., :1]),
                )
                output[key] = torch.nn.functional.normalize(
                    value0 + blend * (value1 * sign - value0), dim=-1
                )
            else:
                output[key] = value0 + blend * (value1 - value0)

        leading_shape = original_shape
        return {key: value.reshape(leading_shape + value.shape[1:]) for key, value in output.items()}
