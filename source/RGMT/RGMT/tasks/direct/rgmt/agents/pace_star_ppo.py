"""PACE and STAR optimization for Extreme-RGMT Stage II."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class PACEStarRolloutStorage(RolloutStorage):
    """Rollout storage augmented with Stage-II role and STAR metadata."""

    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            self.acquisition_mask: torch.Tensor | None = None
            self.bin_ids: torch.Tensor | None = None
            self.difficulty: torch.Tensor | None = None
            self.terminated: torch.Tensor | None = None
            self.timeouts: torch.Tensor | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        shape = (self.num_transitions_per_env, self.num_envs)
        self.acquisition_masks = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.bin_ids = torch.full(shape, -1, dtype=torch.long, device=self.device)
        self.difficulty = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.terminated = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.timeouts = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.raw_advantages = torch.zeros(shape + (1,), dtype=torch.float32, device=self.device)

    def add_transition(self, transition: Transition) -> None:
        step = self.step
        if any(
            value is None
            for value in (
                transition.acquisition_mask,
                transition.bin_ids,
                transition.difficulty,
                transition.terminated,
                transition.timeouts,
            )
        ):
            raise RuntimeError("PACE/STAR transition metadata was not populated.")
        self.acquisition_masks[step].copy_(transition.acquisition_mask)
        self.bin_ids[step].copy_(transition.bin_ids)
        self.difficulty[step].copy_(transition.difficulty)
        self.terminated[step].copy_(transition.terminated)
        self.timeouts[step].copy_(transition.timeouts)
        super().add_transition(transition)


class PACEStarPPO(PPO):
    """Clipped PPO acquisition plus PACE consolidation and STAR resampling."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: PACEStarRolloutStorage,
        env,
        pace_lambda_base: float = 0.3,
        pace_kappa: float = 5.0,
        pace_rho_ref: float = 0.6,
        pace_beta: float = 0.99,
        star_topk_ratio: float = 0.05,
        star_resample_ratio: float = 0.25,
        star_epsilon: float = 1.0e-8,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        if self.rnd or self.symmetry:
            raise ValueError("PACEStarPPO currently requires RND and symmetry augmentation to be disabled.")
        self.env = env
        self.reference_actor = copy.deepcopy(self.actor).to(self.device)
        self._freeze_reference_actor()
        self.transition = PACEStarRolloutStorage.Transition()
        self.pace_lambda_base = float(pace_lambda_base)
        self.pace_kappa = float(pace_kappa)
        self.pace_rho_ref = float(pace_rho_ref)
        self.pace_beta = float(pace_beta)
        self.star_topk_ratio = float(star_topk_ratio)
        self.star_resample_ratio = float(star_resample_ratio)
        self.star_epsilon = float(star_epsilon)
        self.pace_rho_ema = self.pace_rho_ref
        self.pace_lambda_con = self.pace_lambda_base
        self._last_rho = self.pace_rho_ref
        self._last_valid_acquisition = 0
        self._last_valid_consolidation = 0
        self._last_star_pool_fraction = 0.0
        self._last_star_high_fraction = 0.0

    def _freeze_reference_actor(self) -> None:
        self.reference_actor.eval()
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)

    def initialize_reference_from_actor(self) -> None:
        """Synchronize the frozen policy after loading the Stage-I base actor."""
        self.reference_actor.load_state_dict(self.actor.state_dict())
        self._freeze_reference_actor()

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        metadata = self.env.get_stage2_transition_metadata()
        self.transition.acquisition_mask = metadata["acquisition_mask"].to(self.device).clone()
        self.transition.bin_ids = metadata["bin_ids"].to(self.device).clone()
        self.transition.difficulty = metadata["difficulty"].to(self.device).clone()
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        timeouts = extras.get("time_outs")
        if timeouts is None:
            timeouts = torch.zeros_like(dones, dtype=torch.bool)
        else:
            timeouts = timeouts.to(self.device).bool()
        self.transition.timeouts = timeouts
        self.transition.terminated = dones.to(self.device).bool() & ~timeouts
        super().process_env_step(obs, rewards, dones, extras)

    def _fragment_ids(self) -> torch.Tensor:
        """Assign a unique ID to every maximal per-environment rollout fragment."""
        st = self.storage
        dones = st.dones.squeeze(-1).bool()
        local = torch.zeros_like(dones, dtype=torch.long)
        if st.num_transitions_per_env > 1:
            local[1:] = torch.cumsum(dones[:-1].long(), dim=0)
        env_offsets = torch.arange(st.num_envs, device=self.device)[None, :] * (
            st.num_transitions_per_env + 1
        )
        return local + env_offsets

    def _valid_fragment_mask(self, fragment_ids: torch.Tensor) -> torch.Tensor:
        """Mark all transitions in an early-terminated fragment as invalid.

        The paper does not further formalize ``valid sample``. This operational
        definition follows its motivation: samples from an attempt that ends in
        early failure are excluded, whereas timeout-completed and open fragments
        remain valid.
        """
        flat_fragments = fragment_ids.flatten()
        size = int(flat_fragments.max().item()) + 1
        failed = torch.zeros(size, dtype=torch.long, device=self.device)
        failed.scatter_reduce_(
            0,
            flat_fragments,
            self.storage.terminated.flatten().long(),
            reduce="amax",
            include_self=True,
        )
        return ~failed[fragment_ids].bool()

    @staticmethod
    def _normalize_group(values: torch.Tensor, epsilon: float) -> torch.Tensor:
        if values.numel() == 0:
            return values
        return (values - values.mean()) / (values.std(unbiased=False) + epsilon)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        st.raw_advantages.copy_(st.returns - st.values)
        st.advantages.zero_()
        acquisition = st.acquisition_masks
        high = acquisition & (st.difficulty > 1.0)
        easy = acquisition & ~high
        raw = st.raw_advantages.squeeze(-1)
        st.advantages.squeeze(-1)[high] = self._normalize_group(raw[high], self.star_epsilon)
        st.advantages.squeeze(-1)[easy] = self._normalize_group(raw[easy], self.star_epsilon)

        fragment_ids = self._fragment_ids()
        valid = self._valid_fragment_mask(fragment_ids)
        valid_acquisition = int(torch.count_nonzero(valid & acquisition).item())
        valid_consolidation = int(torch.count_nonzero(valid & ~acquisition).item())
        denominator = valid_acquisition + valid_consolidation
        rho = valid_acquisition / denominator if denominator else self.pace_rho_ref
        self.pace_rho_ema = (
            self.pace_beta * self.pace_rho_ema + (1.0 - self.pace_beta) * rho
        )
        self.pace_lambda_con = min(
            1.0,
            self.pace_lambda_base
            + self.pace_kappa * max(0.0, self.pace_rho_ema - self.pace_rho_ref),
        )
        self._last_rho = rho
        self._last_valid_acquisition = valid_acquisition
        self._last_valid_consolidation = valid_consolidation
        self._last_star_high_fraction = float(high.sum().item()) / max(int(acquisition.sum().item()), 1)

    def _build_star_pool(self, fragment_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        st = self.storage
        acquisition = st.acquisition_masks.flatten()
        high = acquisition & (st.difficulty.flatten() > 1.0)
        if not torch.any(high):
            return torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, device=self.device)

        flat_fragments = fragment_ids.flatten()
        flat_bins = st.bin_ids.flatten()
        flat_raw = st.raw_advantages.squeeze(-1).flatten()
        high_bins = flat_bins[high]
        high_fragments = flat_fragments[high]
        fragment_stride = int(flat_fragments.max().item()) + 1
        pair_keys = high_bins * fragment_stride + high_fragments
        unique_pairs, inverse = torch.unique(pair_keys, return_inverse=True)
        sums = torch.zeros(unique_pairs.numel(), device=self.device)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, inverse, flat_raw[high])
        counts.scatter_add_(0, inverse, torch.ones_like(flat_raw[high]))
        scores = sums / torch.clamp(counts, min=1.0)
        pair_bins = torch.div(unique_pairs, fragment_stride, rounding_mode="floor")
        pair_fragments = unique_pairs.remainder(fragment_stride)

        # Stable two-key sort: bin ascending, then raw-advantage score descending.
        score_order = torch.argsort(scores, descending=True, stable=True)
        bin_order = torch.argsort(pair_bins[score_order], stable=True)
        order = score_order[bin_order]
        sorted_bins = pair_bins[order]
        _, group_counts = torch.unique_consecutive(sorted_bins, return_counts=True)
        group_starts = torch.cumsum(group_counts, 0) - group_counts
        ranks = torch.arange(order.numel(), device=self.device) - torch.repeat_interleave(
            group_starts, group_counts
        )
        retained = torch.clamp(
            torch.ceil(group_counts.float() * self.star_topk_ratio).long(), min=1
        )
        keep = ranks < torch.repeat_interleave(retained, group_counts)
        selected_fragments = torch.unique(pair_fragments[order[keep]])
        pool_mask = acquisition & torch.isin(flat_fragments, selected_fragments)
        pool_indices = torch.nonzero(pool_mask, as_tuple=False).squeeze(-1)
        if pool_indices.numel() == 0:
            return pool_indices, torch.empty(0, device=self.device)

        fragment_count = fragment_stride
        eta_sums = torch.zeros(fragment_count, device=self.device)
        eta_counts = torch.zeros(fragment_count, device=self.device)
        acquisition_fragments = flat_fragments[acquisition]
        eta_sums.scatter_add_(0, acquisition_fragments, st.difficulty.flatten()[acquisition])
        eta_counts.scatter_add_(0, acquisition_fragments, torch.ones_like(st.difficulty.flatten()[acquisition]))
        eta = eta_sums / torch.clamp(eta_counts, min=1.0)
        weights = eta[flat_fragments[pool_indices]]
        if weights.sum() <= 0.0:
            weights = torch.ones_like(weights)
        return pool_indices, weights / weights.sum()

    def _update_learning_rate(self, old_params, new_params) -> None:
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl_mean = torch.mean(self.actor.get_kl_divergence(old_params, new_params))
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                lr = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr, src=0)
                self.learning_rate = lr.item()
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate

    def update(self) -> dict[str, float]:
        st = self.storage
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("PACEStarPPO currently supports the feed-forward ExtremeRGMTModel only.")
        fragment_ids = self._fragment_ids()
        pool_indices, pool_weights = self._build_star_pool(fragment_ids)
        acquisition_indices = torch.nonzero(st.acquisition_masks.flatten(), as_tuple=False).squeeze(-1)
        consolidation_indices = torch.nonzero(~st.acquisition_masks.flatten(), as_tuple=False).squeeze(-1)
        self._last_star_pool_fraction = float(pool_indices.numel()) / max(acquisition_indices.numel(), 1)

        observations = st.observations.flatten(0, 1)
        actions = st.actions.flatten(0, 1)
        values_old = st.values.flatten(0, 1)
        returns = st.returns.flatten(0, 1)
        old_log_prob = st.actions_log_prob.flatten(0, 1)
        advantages = st.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in st.distribution_params)
        acquisition_batch_size = max(acquisition_indices.numel() // self.num_mini_batches, 1)
        consolidation_batch_size = max(consolidation_indices.numel() // self.num_mini_batches, 1)

        totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "consolidation": 0.0}
        updates = 0
        for _ in range(self.num_learning_epochs):
            acquisition_perm = acquisition_indices[
                torch.randperm(acquisition_indices.numel(), device=self.device)
            ]
            consolidation_perm = consolidation_indices[
                torch.randperm(consolidation_indices.numel(), device=self.device)
            ]
            for batch_index in range(self.num_mini_batches):
                a_start = batch_index * acquisition_batch_size
                a_stop = min(a_start + acquisition_batch_size, acquisition_perm.numel())
                base_indices = acquisition_perm[a_start:a_stop]
                if base_indices.numel() == 0:
                    continue
                star_count = 0
                if pool_indices.numel():
                    star_count = min(
                        base_indices.numel(), max(int(math.floor(self.star_resample_ratio * base_indices.numel())), 1)
                    )
                if star_count:
                    standard_count = base_indices.numel() - star_count
                    sampled_star = pool_indices[
                        torch.multinomial(pool_weights, star_count, replacement=True)
                    ]
                    batch_ids = torch.cat((base_indices[:standard_count], sampled_star))
                else:
                    batch_ids = base_indices

                batch_obs = observations[batch_ids]
                self.actor(batch_obs, stochastic_output=True)
                actions_log_prob = self.actor.get_output_log_prob(actions[batch_ids])
                distribution_params = tuple(p for p in self.actor.output_distribution_params)
                entropy = self.actor.output_entropy
                predicted_values = self.critic(batch_obs)
                old_params = tuple(p[batch_ids] for p in old_distribution_params)
                self._update_learning_rate(old_params, distribution_params)

                ratio = torch.exp(actions_log_prob - old_log_prob[batch_ids].squeeze(-1))
                batch_advantages = advantages[batch_ids].squeeze(-1)
                surrogate = -batch_advantages * ratio
                surrogate_clipped = -batch_advantages * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                if self.use_clipped_value_loss:
                    value_clipped = values_old[batch_ids] + (
                        predicted_values - values_old[batch_ids]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_loss = torch.max(
                        (predicted_values - returns[batch_ids]).pow(2),
                        (value_clipped - returns[batch_ids]).pow(2),
                    ).mean()
                else:
                    value_loss = (returns[batch_ids] - predicted_values).pow(2).mean()

                c_start = batch_index * consolidation_batch_size
                c_stop = min(c_start + consolidation_batch_size, consolidation_perm.numel())
                consolidation_ids = consolidation_perm[c_start:c_stop]
                if consolidation_ids.numel():
                    consolidation_obs = observations[consolidation_ids]
                    current_actions = self.actor(consolidation_obs)
                    with torch.inference_mode():
                        reference_actions = self.reference_actor(consolidation_obs)
                    consolidation_loss = torch.sum(
                        (current_actions - reference_actions) ** 2, dim=-1
                    ).mean()
                else:
                    consolidation_loss = torch.zeros((), device=self.device)

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                    + self.pace_lambda_con * consolidation_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                totals["value"] += value_loss.item()
                totals["surrogate"] += surrogate_loss.item()
                totals["entropy"] += entropy.mean().item()
                totals["consolidation"] += consolidation_loss.item()
                updates += 1

        st.clear()
        divisor = max(updates, 1)
        return {
            "value": totals["value"] / divisor,
            "surrogate": totals["surrogate"] / divisor,
            "entropy": totals["entropy"] / divisor,
            "pace_consolidation": totals["consolidation"] / divisor,
            "pace_lambda": self.pace_lambda_con,
            "pace_rho": self._last_rho,
            "pace_rho_ema": self.pace_rho_ema,
            "pace_valid_acquisition": float(self._last_valid_acquisition),
            "pace_valid_consolidation": float(self._last_valid_consolidation),
            "star_pool_fraction": self._last_star_pool_fraction,
            "star_high_fraction": self._last_star_high_fraction,
        }

    def train_mode(self) -> None:
        super().train_mode()
        self.reference_actor.eval()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.reference_actor.eval()

    def save(self) -> dict:
        state = super().save()
        state.update(
            {
                "reference_actor_state_dict": self.reference_actor.state_dict(),
                "pace_rho_ema": self.pace_rho_ema,
                "pace_lambda_con": self.pace_lambda_con,
            }
        )
        return state

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        load_reference = load_cfg is None or load_cfg.get("reference", True)
        if load_reference and "reference_actor_state_dict" in loaded_dict:
            self.reference_actor.load_state_dict(
                loaded_dict["reference_actor_state_dict"], strict=strict
            )
        elif load_cfg is None or load_cfg.get("actor", False):
            self.initialize_reference_from_actor()
        if load_cfg is None:
            self.pace_rho_ema = float(loaded_dict.get("pace_rho_ema", self.pace_rho_ref))
            self.pace_lambda_con = float(
                loaded_dict.get("pace_lambda_con", self.pace_lambda_base)
            )
        self._freeze_reference_actor()
        return load_iteration

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        reference = [self.reference_actor.state_dict()]
        torch.distributed.broadcast_object_list(reference, src=0)
        self.reference_actor.load_state_dict(reference[0])

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> "PACEStarPPO":
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")
        storage = PACEStarRolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        return alg_class(
            actor,
            critic,
            storage,
            env=env.unwrapped,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
