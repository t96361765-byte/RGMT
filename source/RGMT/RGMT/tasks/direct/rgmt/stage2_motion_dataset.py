"""Motion routing for Extreme-RGMT Stage II acquisition and consolidation."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from .motion_dataset import ExtremeRGMTMotionDataset


class ExtremeRGMTStage2MotionDataset:
    """Expose challenge and mastered datasets through one clip-ID namespace.

    Acquisition clips occupy ``[0, num_acquisition_clips)`` and use the
    failure-adaptive sampler. Mastered clips follow them and are sampled
    uniformly for PACE consolidation.
    """

    def __init__(
        self,
        acquisition_motion_file: str,
        mastered_motion_file: str,
        robot_joint_names: list[str],
        requested_body_names: list[str],
        device: str | torch.device,
        adaptive_bin_duration_s: float,
        adaptive_ema_alpha: float,
        adaptive_score_clip: float,
        adaptive_uniform_ratio: float,
    ) -> None:
        common = dict(
            robot_joint_names=robot_joint_names,
            requested_body_names=requested_body_names,
            device=device,
            adaptive_bin_duration_s=adaptive_bin_duration_s,
            adaptive_ema_alpha=adaptive_ema_alpha,
            adaptive_score_clip=adaptive_score_clip,
            adaptive_uniform_ratio=adaptive_uniform_ratio,
        )
        self.acquisition = ExtremeRGMTMotionDataset(
            motion_file=acquisition_motion_file, **common
        )
        self.mastered = ExtremeRGMTMotionDataset(
            motion_file=mastered_motion_file, **common
        )
        self.device = torch.device(device)
        self.num_acquisition_clips = len(self.acquisition.clips)
        self.clips = self.acquisition.clips + self.mastered.clips
        self.durations = torch.cat((self.acquisition.durations, self.mastered.durations))
        self.source_files = [
            f"acquisition:{path}" for path in self.acquisition.source_files
        ] + [f"mastered:{path}" for path in self.mastered.source_files]

        # Preserve the interface used by the existing adaptive checkpoint saver.
        self._bin_scores = self.acquisition._bin_scores

    def sample_acquisition_starts(
        self, count: int, future_horizon_s: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.acquisition.sample_starts(count, future_horizon_s)

    def sample_consolidation_starts(
        self, count: int, future_horizon_s: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip_ids, times = self.mastered.sample_uniform_starts(count, future_horizon_s)
        return clip_ids + self.num_acquisition_clips, times

    def sample(self, clip_ids: torch.Tensor, times: torch.Tensor) -> dict[str, torch.Tensor]:
        if times.shape[0] != clip_ids.shape[0]:
            raise ValueError("The first dimension of times must match clip_ids.")
        acquisition_mask = clip_ids < self.num_acquisition_clips
        output: dict[str, torch.Tensor] = {}
        for mask, dataset, offset in (
            (acquisition_mask, self.acquisition, 0),
            (~acquisition_mask, self.mastered, self.num_acquisition_clips),
        ):
            if not torch.any(mask):
                continue
            sampled = dataset.sample(clip_ids[mask] - offset, times[mask])
            for key, values in sampled.items():
                if key not in output:
                    output[key] = torch.empty(
                        times.shape + values.shape[len(times.shape) :],
                        dtype=values.dtype,
                        device=values.device,
                    )
                output[key][mask] = values
        return output

    def update_acquisition_scores(
        self, clip_ids: torch.Tensor, times: torch.Tensor, failures: torch.Tensor
    ) -> None:
        mask = clip_ids < self.num_acquisition_clips
        self.acquisition.update_adaptive_scores(
            clip_ids[mask], times[mask], failures[mask]
        )

    def acquisition_transition_metadata(
        self, clip_ids: torch.Tensor, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return STAR bin IDs and ``w_t = B p_b`` for acquisition samples."""
        bins = torch.full_like(clip_ids, -1)
        difficulty = torch.zeros_like(times, dtype=torch.float32)
        mask = clip_ids < self.num_acquisition_clips
        if torch.any(mask):
            local_bins = self.acquisition.flat_bin_ids(clip_ids[mask], times[mask])
            probabilities = self.acquisition.adaptive_probabilities()
            bins[mask] = local_bins
            difficulty[mask] = probabilities[local_bins] * probabilities.numel()
        return bins, difficulty

    def adaptive_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.acquisition.adaptive_fingerprint().encode())
        digest.update(self.mastered.adaptive_fingerprint().encode())
        return digest.hexdigest()

    def adaptive_state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "stage": 2,
            "dataset_fingerprint": self.adaptive_fingerprint(),
            "acquisition": self.acquisition.adaptive_state_dict(),
            "mastered_fingerprint": self.mastered.adaptive_fingerprint(),
            # Convenient top-level copies retained for diagnostics.
            "bin_scores": self.acquisition._bin_scores.detach().cpu(),
            "bin_counts": self.acquisition._bin_counts.detach().cpu(),
            "bin_duration_s": self.acquisition.adaptive_bin_duration_s,
            "ema_alpha": self.acquisition.adaptive_ema_alpha,
            "uniform_ratio": self.acquisition.adaptive_uniform_ratio,
        }

    def load_adaptive_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("format_version", -1)) != 2 or int(state.get("stage", -1)) != 2:
            raise ValueError("A Stage-II adaptive sampler requires a version-2 Stage-II sidecar.")
        if state.get("dataset_fingerprint") != self.adaptive_fingerprint():
            raise ValueError("Stage-II adaptive sampler state belongs to different motion datasets.")
        if state.get("mastered_fingerprint") != self.mastered.adaptive_fingerprint():
            raise ValueError("Stage-II mastered dataset fingerprint mismatch.")
        self.acquisition.load_adaptive_state_dict(state["acquisition"])

