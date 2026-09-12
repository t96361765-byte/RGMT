"""RSL-RL 5 model implementing the Extreme-RGMT Stage-I policy."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution


class BoundedGaussianDistribution(GaussianDistribution):
    """State-independent Gaussian whose learned standard deviation is capped.

    The parameter is projected before each distribution update. Projection is
    performed without autograd so gradients at the ceiling can still move the
    parameter back toward a smaller standard deviation.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "scalar",
        max_std: float = 1.0,
    ) -> None:
        if max_std <= 0.0:
            raise ValueError("max_std must be positive.")
        if init_std > max_std:
            raise ValueError(
                f"init_std ({init_std}) must not exceed max_std ({max_std})."
            )
        self.max_std = float(max_std)
        super().__init__(output_dim=output_dim, init_std=init_std, std_type=std_type)

    def update(self, mlp_output: torch.Tensor) -> None:
        min_positive_std = torch.finfo(mlp_output.dtype).eps
        with torch.no_grad():
            if self.std_type == "scalar":
                self.std_param.clamp_(min=min_positive_std, max=self.max_std)
            else:
                self.log_std_param.clamp_(
                    min=math.log(min_positive_std), max=math.log(self.max_std)
                )
        super().update(mlp_output)


def _sinusoidal_position_encoding(sequence_length: int, embedding_dim: int) -> torch.Tensor:
    """Build the fixed sinusoidal encoding used by both Stage-I encoders."""
    position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / embedding_dim)
    )
    positional_encoding = torch.zeros(sequence_length, embedding_dim)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    return positional_encoding


class _MultiHeadAttention(nn.Module):
    """Small explicit MHA implementation avoiding backend-specific fused kernels."""

    def __init__(self, embedding_dim: int, num_heads: int = 4):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = value.shape
        return value.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_heads = self._split_heads(self.query_projection(query))
        key_heads = self._split_heads(self.key_projection(key))
        value_heads = self._split_heads(self.value_projection(value))
        scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[None, None], -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value_heads).transpose(1, 2).contiguous()
        attended = attended.view(query.shape[0], query.shape[1], -1)
        return self.output_projection(attended)


class _FiniteScalarQuantizer(nn.Module):
    """Element-wise FSQ with a straight-through rounding estimator.

    Extreme-RGMT specifies two 32-D FSQ tokens but does not publish the number
    of scalar levels. The level count is therefore explicit and configurable;
    the Stage-I configuration uses eight levels as a conservative default.
    """

    def __init__(self, levels: int = 8):
        super().__init__()
        if levels < 2:
            raise ValueError("FSQ needs at least two scalar levels.")
        self.levels = levels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Bound every scalar before quantization. The normalized result remains
        # in [-1, 1], independent of the selected number of levels.
        bounded = torch.tanh(value)
        scaled = (bounded + 1.0) * 0.5 * float(self.levels - 1)
        quantized = scaled + (torch.round(scaled) - scaled).detach()
        return 2.0 * quantized / float(self.levels - 1) - 1.0


class _ExtremeHistoryEncoder(nn.Module):
    """Encode interleaved previous-action and proprioceptive tokens."""

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        history_steps: int,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ELU(),
            nn.Linear(128, embedding_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ELU(),
            nn.Linear(64, embedding_dim),
        )
        self.state_norm = nn.LayerNorm(embedding_dim)
        self.action_norm = nn.LayerNorm(embedding_dim)
        sequence_length = 2 * history_steps
        self.register_buffer(
            "positional_encoding",
            _sinusoidal_position_encoding(sequence_length, embedding_dim),
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(sequence_length, sequence_length, dtype=torch.bool),
                diagonal=1,
            ),
        )
        self.attention = _MultiHeadAttention(embedding_dim, num_heads=4)
        self.attention_input_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_input_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ELU(),
            nn.Linear(128, embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, proprio_history: torch.Tensor, action_history: torch.Tensor
    ) -> torch.Tensor:
        state_tokens = self.state_norm(self.state_encoder(proprio_history))
        action_tokens = self.action_norm(self.action_encoder(action_history))
        # Equation (6): [a_{t-H-1}, o_{t-H}, ..., a_{t-1}, o_t].
        tokens = torch.stack((action_tokens, state_tokens), dim=2).flatten(1, 2)
        tokens = tokens + self.positional_encoding.unsqueeze(0)
        normalized = self.attention_input_norm(tokens)
        tokens = tokens + self.attention(
            normalized,
            normalized,
            normalized,
            attention_mask=self.causal_mask,
        )
        tokens = tokens + self.feed_forward(self.feed_forward_input_norm(tokens))
        tokens = self.output_norm(tokens)
        # Aggregate the causal history into the Stage-I dynamics embedding.
        return torch.max(tokens, dim=1).values


class _ExtremeCommandEncoder(nn.Module):
    """Aggregate the reference window with the history embedding as query."""

    def __init__(
        self,
        command_dim: int,
        command_steps: int,
        embedding_dim: int = 64,
        fsq_levels: int = 8,
    ):
        super().__init__()
        self.command_encoder = nn.Sequential(
            nn.Linear(command_dim, 128),
            nn.ELU(),
            nn.Linear(128, embedding_dim),
        )
        self.command_norm = nn.LayerNorm(embedding_dim)
        self.register_buffer(
            "positional_encoding",
            _sinusoidal_position_encoding(command_steps, embedding_dim),
        )
        self.cross_attention = _MultiHeadAttention(embedding_dim, num_heads=4)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.fsq = _FiniteScalarQuantizer(fsq_levels)

    def forward(self, history: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        command_tokens = self.command_norm(self.command_encoder(commands))
        command_tokens = command_tokens + self.positional_encoding.unsqueeze(0)
        aggregated = self.cross_attention(
            history.unsqueeze(1),
            command_tokens,
            command_tokens,
        ).squeeze(1)
        aggregated = self.output_norm(aggregated)
        # Equation (10): two independently quantized 32-D tokens.
        token_pair = aggregated.view(aggregated.shape[0], 2, 32)
        return self.fsq(token_pair).flatten(1)


class ExtremeRGMTModel(MLPModel):
    """Extreme-RGMT Stage-I actor."""

    command_dim = 38
    proprio_dim = 64
    action_dim = 29
    command_steps = 21
    history_steps = 10
    embedding_dim = 64
    fsq_levels = 8
    actor_observation_dim = (
        command_dim * command_steps
        + proprio_dim * history_steps
        + action_dim * history_steps
    )

    def __init__(self, *args, **kwargs):
        obs = args[0] if args else kwargs["obs"]
        obs_groups = args[1] if len(args) > 1 else kwargs["obs_groups"]
        obs_set = args[2] if len(args) > 2 else kwargs["obs_set"]
        active_groups = obs_groups[obs_set]
        flat_dim = sum(obs[group].shape[-1] for group in active_groups)
        if flat_dim != self.actor_observation_dim:
            raise ValueError(
                "ExtremeRGMTModel needs exactly "
                f"{self.actor_observation_dim} actor inputs, got {flat_dim} "
                f"for '{obs_set}'."
            )
        super().__init__(*args, **kwargs)
        self.history_encoder = _ExtremeHistoryEncoder(
            proprio_dim=self.proprio_dim,
            action_dim=self.action_dim,
            history_steps=self.history_steps,
            embedding_dim=self.embedding_dim,
        )
        self.command_encoder = _ExtremeCommandEncoder(
            command_dim=self.command_dim,
            command_steps=self.command_steps,
            embedding_dim=self.embedding_dim,
            fsq_levels=self.fsq_levels,
        )

    def _get_latent_dim(self) -> int:
        # Equation (11): current o_prop, previous action, and quantized command.
        return self.proprio_dim + self.action_dim + self.embedding_dim

    def _split_observation(
        self, flat_observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        command_size = self.command_steps * self.command_dim
        proprio_size = self.history_steps * self.proprio_dim
        action_size = self.history_steps * self.action_dim
        commands = flat_observation[:, :command_size].view(
            -1, self.command_steps, self.command_dim
        )
        proprio_history = flat_observation[
            :, command_size : command_size + proprio_size
        ].view(-1, self.history_steps, self.proprio_dim)
        action_history = flat_observation[
            :, command_size + proprio_size : command_size + proprio_size + action_size
        ].view(-1, self.history_steps, self.action_dim)
        return commands, proprio_history, action_history

    def _encode_flat_observation(self, flat_observation: torch.Tensor) -> torch.Tensor:
        commands, proprio_history, action_history = self._split_observation(
            flat_observation
        )
        history = self.history_encoder(proprio_history, action_history)
        quantized_command = self.command_encoder(history, commands)
        return torch.cat(
            (proprio_history[:, -1], action_history[:, -1], quantized_command),
            dim=-1,
        )

    def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
        flat_observation = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        flat_observation = self.obs_normalizer(flat_observation)
        return self._encode_flat_observation(flat_observation)

    def as_jit(self) -> nn.Module:
        return _ExportExtremeRGMTModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _ExportExtremeRGMTModel(self, verbose=verbose)


class _ExportExtremeRGMTModel(nn.Module):
    """Deterministic export wrapper for the Extreme-RGMT Stage-I actor."""

    is_recurrent: bool = False

    def __init__(self, model: ExtremeRGMTModel, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.input_size = model.obs_dim
        self.command_dim = model.command_dim
        self.proprio_dim = model.proprio_dim
        self.action_dim = model.action_dim
        self.command_steps = model.command_steps
        self.history_steps = model.history_steps
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.command_encoder = copy.deepcopy(model.command_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            copy.deepcopy(model.distribution.as_deterministic_output_module())
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        observation = self.obs_normalizer(observation)
        command_size = self.command_steps * self.command_dim
        proprio_size = self.history_steps * self.proprio_dim
        commands = observation[:, :command_size].view(
            -1, self.command_steps, self.command_dim
        )
        proprio_history = observation[
            :, command_size : command_size + proprio_size
        ].view(-1, self.history_steps, self.proprio_dim)
        action_history = observation[:, command_size + proprio_size :].view(
            -1, self.history_steps, self.action_dim
        )
        history = self.history_encoder(proprio_history, action_history)
        quantized_command = self.command_encoder(history, commands)
        latent = torch.cat(
            (proprio_history[:, -1], action_history[:, -1], quantized_command),
            dim=-1,
        )
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        pass

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
