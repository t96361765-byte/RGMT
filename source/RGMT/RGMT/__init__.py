# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extreme-RGMT Stage-I motion-tracking reinforcement learning for Unitree G1."""

from pathlib import Path

RGMT_EXT_DIR = Path(__file__).resolve().parents[1]
RGMT_DATA_DIR = RGMT_EXT_DIR / "data"

# Register Gym environments.
from .tasks import *
