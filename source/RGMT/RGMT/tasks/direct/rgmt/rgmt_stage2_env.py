"""Extreme-RGMT Stage-II environment with PACE acquisition/consolidation roles."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .rgmt_env import ExtremeRGMTG1Env
from .rgmt_stage2_env_cfg import ExtremeRGMTStage2G1EnvCfg
from .stage2_motion_dataset import ExtremeRGMTStage2MotionDataset


class ExtremeRGMTStage2G1Env(ExtremeRGMTG1Env):
    """Run challenge acquisition and mastered consolidation in one simulation."""

    cfg: ExtremeRGMTStage2G1EnvCfg

    def __init__(
        self, cfg: ExtremeRGMTStage2G1EnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        if not 0.0 < cfg.acquisition_fraction < 1.0:
            raise ValueError("Stage-II acquisition_fraction must lie strictly between 0 and 1.")
        if cfg.scene.num_envs < 2 and not cfg.allow_single_env_playback:
            raise ValueError(
                "Stage-II PACE training requires at least two environments. "
                "Use play.py --num_envs 1 for explicit single-environment playback."
            )
        if cfg.scene.num_envs == 1:
            acquisition_count = 1
        else:
            acquisition_count = min(
                max(int(round(cfg.scene.num_envs * cfg.acquisition_fraction)), 1),
                cfg.scene.num_envs - 1,
            )
        role_device = torch.device(cfg.sim.device)
        self._stage2_acquisition_mask = torch.zeros(
            cfg.scene.num_envs, dtype=torch.bool, device=role_device
        )
        self._stage2_acquisition_mask[:acquisition_count] = True
        self._stage2_acquisition_env_ids = torch.arange(
            acquisition_count, dtype=torch.long, device=role_device
        )
        self._stage2_consolidation_env_ids = torch.arange(
            acquisition_count, cfg.scene.num_envs, dtype=torch.long, device=role_device
        )
        super().__init__(cfg, render_mode, **kwargs)
        print(
            "[INFO]: Extreme-RGMT Stage II roles: "
            f"{len(self._stage2_acquisition_env_ids)} acquisition / "
            f"{len(self._stage2_consolidation_env_ids)} consolidation environments."
        )

    def _create_motion_dataset(self) -> ExtremeRGMTStage2MotionDataset:
        return ExtremeRGMTStage2MotionDataset(
            acquisition_motion_file=self.cfg.motion_file,
            mastered_motion_file=self.cfg.mastered_motion_file,
            robot_joint_names=self.robot.joint_names,
            requested_body_names=self.cfg.key_body_names,
            device=self.device,
            adaptive_bin_duration_s=self.cfg.adaptive_bin_duration_s,
            adaptive_ema_alpha=self.cfg.adaptive_ema_alpha,
            adaptive_score_clip=self.cfg.adaptive_score_clip,
            adaptive_uniform_ratio=self.cfg.adaptive_uniform_ratio,
        )

    @property
    def stage2_motion_dataset(self) -> ExtremeRGMTStage2MotionDataset:
        assert isinstance(self._motion_dataset, ExtremeRGMTStage2MotionDataset)
        return self._motion_dataset

    def _update_motion_sampling_scores(self, env_ids: torch.Tensor) -> None:
        acquisition = self._stage2_acquisition_mask[env_ids]
        selected = env_ids[acquisition]
        if selected.numel() == 0:
            return
        # The failure is assigned to the temporal bin reached when the episode
        # ended, rather than only to its initial bin.
        self.stage2_motion_dataset.update_acquisition_scores(
            self._clip_ids[selected],
            self._motion_times()[selected],
            self.reset_terminated[selected].float(),
        )

    def _sample_motion_starts(
        self, env_ids: torch.Tensor, future_horizon_s: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip_ids = torch.empty(len(env_ids), dtype=torch.long, device=self.device)
        start_times = torch.empty(len(env_ids), dtype=torch.float32, device=self.device)
        acquisition = self._stage2_acquisition_mask[env_ids]
        acquisition_count = int(acquisition.sum().item())
        consolidation_count = len(env_ids) - acquisition_count
        if acquisition_count:
            clips, times = self.stage2_motion_dataset.sample_acquisition_starts(
                acquisition_count, future_horizon_s
            )
            clip_ids[acquisition], start_times[acquisition] = clips, times
        if consolidation_count:
            clips, times = self.stage2_motion_dataset.sample_consolidation_starts(
                consolidation_count, future_horizon_s
            )
            clip_ids[~acquisition], start_times[~acquisition] = clips, times
        return clip_ids, start_times

    def _build_proprioception(self, noisy: bool) -> torch.Tensor:
        proprioception = super()._build_proprioception(noisy=noisy)
        if noisy and self._stage2_consolidation_env_ids.numel():
            clean = super()._build_proprioception(noisy=False)
            proprioception[self._stage2_consolidation_env_ids] = clean[
                self._stage2_consolidation_env_ids
            ]
        return proprioception

    def _perturb_command_window(self, command_window: torch.Tensor) -> torch.Tensor:
        perturbed = super()._perturb_command_window(command_window)
        perturbed[self._stage2_consolidation_env_ids] = command_window[
            self._stage2_consolidation_env_ids
        ]
        return perturbed

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            requested = self.robot._ALL_INDICES
        else:
            requested = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(requested)
        consolidation = requested[~self._stage2_acquisition_mask[requested]]
        if consolidation.numel():
            self._motor_zero_offsets[consolidation] = 0.0
            self._command_linear_velocity_offsets[consolidation] = 0.0
            self._command_angular_velocity_offsets[consolidation] = 0.0
            self._command_gravity_offsets[consolidation] = 0.0
            self._command_joint_position_offsets[consolidation] = 0.0

    def get_stage2_transition_metadata(self) -> dict[str, torch.Tensor]:
        """Metadata captured by PACE/STAR before each simulator step."""
        bin_ids, difficulty = self.stage2_motion_dataset.acquisition_transition_metadata(
            self._clip_ids, self._motion_times()
        )
        return {
            "acquisition_mask": self._stage2_acquisition_mask,
            "bin_ids": bin_ids,
            "difficulty": difficulty,
        }
