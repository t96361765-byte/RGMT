"""Gym registrations for the in-place Extreme-RGMT Stage-I G1 tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-RGMT-G1-v0",
    entry_point=f"{__name__}.rgmt_env:ExtremeRGMTG1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgmt_env_cfg:ExtremeRGMTG1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ExtremeRGMTPPORunnerCfg",
    },
)

gym.register(
    id="Extreme-RGMT-Stage1-G1-TheShy-Mushroom",
    entry_point=f"{__name__}.rgmt_env:ExtremeRGMTG1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgmt_env_cfg:ExtremeRGMTG1TheShyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ExtremeRGMTPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Extreme-RGMT-Stage2-G1-v0",
    entry_point=f"{__name__}.rgmt_stage2_env:ExtremeRGMTStage2G1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgmt_stage2_env_cfg:ExtremeRGMTStage2G1EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ExtremeRGMTStage2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Extreme-RGMT-Stage2-G1-TheShy-Mushroom",
    entry_point=f"{__name__}.rgmt_stage2_env:ExtremeRGMTStage2G1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgmt_stage2_env_cfg:ExtremeRGMTStage2G1TheShyEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ExtremeRGMTStage2PPORunnerCfg"
        ),
    },
)

