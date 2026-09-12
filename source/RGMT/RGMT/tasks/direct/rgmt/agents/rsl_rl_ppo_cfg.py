"""RSL-RL 5 PPO configuration for Extreme-RGMT Stage I."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BoundedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Configuration for the state-independent Gaussian with a hard std ceiling."""

    max_std: float = 1.0


@configclass
class ExtremeRGMTPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Extreme-RGMT Stage-I Table III training configuration."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 5000
    experiment_name = "extreme_rgmt_stage1_g1"
    run_name = ""
    logger = "tensorboard"
    # Keep the PPO action and the action executed by the environment identical.
    # The policy output is already a joint-position residual in radians (Eq. 3),
    # so the RSL-RL wrapper must not clip it before stepping the environment.
    clip_actions = None
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="RGMT.tasks.direct.rgmt.agents.rgmt_models:ExtremeRGMTModel",
        hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=BoundedGaussianDistributionCfg(
            class_name=(
                "RGMT.tasks.direct.rgmt.agents.rgmt_models:"
                "BoundedGaussianDistribution"
            ),
            init_std=0.5,
            std_type="scalar",
            max_std=1.0,
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[1024, 1024, 512, 512],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PACEStarPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Published Stage-II PACE/STAR coefficients."""

    class_name: str = (
        "RGMT.tasks.direct.rgmt.agents.pace_star_ppo:PACEStarPPO"
    )
    pace_lambda_base: float = 0.3
    pace_kappa: float = 5.0
    pace_rho_ref: float = 0.6
    pace_beta: float = 0.99
    star_topk_ratio: float = 0.05
    star_resample_ratio: float = 0.25
    # Numerical epsilon in Eq. (23); the paper does not publish its value.
    star_epsilon: float = 1.0e-8


@configclass
class ExtremeRGMTStage2PPORunnerCfg(ExtremeRGMTPPORunnerCfg):
    """Extreme-RGMT Stage-II Table-III PPO with PACE and STAR."""

    experiment_name = "extreme_rgmt_stage2_g1"
    max_iterations = 100_000
    algorithm = PACEStarPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
