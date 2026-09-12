# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Agent configuration and model for Extreme-RGMT Stage I."""

from .rgmt_models import ExtremeRGMTModel
from .rsl_rl_ppo_cfg import ExtremeRGMTPPORunnerCfg, ExtremeRGMTStage2PPORunnerCfg

__all__ = [
    "ExtremeRGMTModel",
    "ExtremeRGMTPPORunnerCfg",
    "ExtremeRGMTStage2PPORunnerCfg",
]
