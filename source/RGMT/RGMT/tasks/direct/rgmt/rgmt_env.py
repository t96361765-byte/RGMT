"""Direct Isaac Lab implementation of the Extreme-RGMT Stage-I pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from pxr import Gf, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)

from .motion_dataset import ExtremeRGMTMotionDataset
from .rgmt_env_cfg import ExtremeRGMTG1EnvCfg


class ExtremeRGMTG1Env(DirectRLEnv):
    """G1 motion tracking with the Extreme-RGMT Stage-I policy interface."""

    cfg: ExtremeRGMTG1EnvCfg

    def __init__(
        self, cfg: ExtremeRGMTG1EnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        if self.robot.num_joints != self.cfg.action_space:
            raise RuntimeError(
                f"Extreme-RGMT expects {self.cfg.action_space} actuated joints, but the imported G1 has "
                f"{self.robot.num_joints}: {self.robot.joint_names}"
            )
        if self.reference_robot is not None and self.reference_robot.joint_names != self.robot.joint_names:
            raise RuntimeError(
                "Reference G1 joint order differs from the controlled G1, so the reference "
                "motion cannot be visualized safely."
            )
        expected_actor_dim = (
            self.cfg.command_window_steps * self.cfg.command_dim
            + self.cfg.history_steps * self.cfg.proprio_dim
            + self.cfg.action_history_steps * self.cfg.action_history_dim
        )
        if self.cfg.observation_space != expected_actor_dim:
            raise ValueError("Extreme-RGMT observation-space dimensions are inconsistent.")
        expected_state_dim = (
            self.cfg.proprio_dim
            + self.cfg.action_history_dim
            + self.cfg.command_dim
            + 1
            + len(self.cfg.key_body_names) * 9
            + 3
        )
        if self.cfg.state_space != expected_state_dim:
            raise ValueError(
                f"Extreme-RGMT critic-state dimension must be {expected_state_dim}, got {self.cfg.state_space}."
            )

        self._key_body_ids, key_body_names = self.robot.find_bodies(self.cfg.key_body_names, preserve_order=True)
        if key_body_names != self.cfg.key_body_names:
            raise RuntimeError(
                f"Could not resolve Extreme-RGMT tracking bodies in the requested order: {key_body_names}"
            )
        self._termination_body_ids, termination_body_names = self.robot.find_bodies(
            self.cfg.termination_key_body_names, preserve_order=True
        )
        if termination_body_names != self.cfg.termination_key_body_names:
            raise RuntimeError(
                "Could not resolve Extreme-RGMT termination bodies in the requested order: "
                f"{termination_body_names}"
            )
        self._termination_reference_ids = [
            self.cfg.key_body_names.index(name) for name in self.cfg.termination_key_body_names
        ]
        self._undesired_sensor_ids, _ = self.contact_sensor.find_bodies(
            self.cfg.undesired_contact_body_names, preserve_order=True
        )
        self._foot_body_ids, foot_body_names = self.robot.find_bodies(
            self.cfg.foot_body_names, preserve_order=True
        )
        self._foot_sensor_ids, foot_sensor_names = self.contact_sensor.find_bodies(
            self.cfg.foot_body_names, preserve_order=True
        )
        if (
            foot_body_names != self.cfg.foot_body_names
            or foot_sensor_names != self.cfg.foot_body_names
        ):
            raise RuntimeError(
                "Could not resolve Extreme-RGMT foot bodies in the requested order."
            )

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)
        self._proprio_history = torch.zeros(
            (self.num_envs, self.cfg.history_steps, self.cfg.proprio_dim), device=self.device
        )
        self._action_history = torch.zeros(
            (
                self.num_envs,
                self.cfg.action_history_steps,
                self.cfg.action_history_dim,
            ),
            device=self.device,
        )
        self._clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._motion_start_times = torch.zeros(self.num_envs, device=self.device)
        self._reference_robot_offset = torch.tensor(
            self.cfg.reference_robot_offset, dtype=torch.float32, device=self.device
        )
        self._motor_zero_offsets = torch.zeros_like(self._actions)
        self._command_linear_velocity_offsets = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._command_angular_velocity_offsets = torch.zeros_like(
            self._command_linear_velocity_offsets
        )
        self._command_gravity_offsets = torch.zeros_like(
            self._command_linear_velocity_offsets
        )
        self._command_joint_position_offsets = torch.zeros_like(self._actions)

        self._motion_dataset: ExtremeRGMTMotionDataset | None = None
        if not self.cfg.debug_static_reference:
            self._motion_dataset = self._create_motion_dataset()
            if self.cfg.replay_full_motions:
                # The normal 10 s training horizon must not truncate long validation clips.
                longest_clip = float(self._motion_dataset.durations.max().item())
                future_horizon = (self.cfg.command_window_steps // 2) * self.step_dt
                self.cfg.episode_length_s = longest_clip + future_horizon + self.step_dt
                print("[INFO]: Full-motion validation replay enabled.")
                for clip_id, (source, duration) in enumerate(
                    zip(
                        self._motion_dataset.source_files,
                        self._motion_dataset.durations.tolist(),
                        strict=True,
                    )
                ):
                    print(f"[INFO]:   clip {clip_id}: {source} ({duration:.2f} s)")

        self._current_reference = self._sample_reference(torch.zeros(self.num_envs, device=self.device))
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self.cfg.reward_scales
        }

    def _create_motion_dataset(self) -> ExtremeRGMTMotionDataset:
        """Create the Stage-I dataset; Stage II overrides this routing hook."""
        return ExtremeRGMTMotionDataset(
            motion_file=self.cfg.motion_file,
            robot_joint_names=self.robot.joint_names,
            requested_body_names=self.cfg.key_body_names,
            device=self.device,
            adaptive_bin_duration_s=self.cfg.adaptive_bin_duration_s,
            adaptive_ema_alpha=self.cfg.adaptive_ema_alpha,
            adaptive_score_clip=self.cfg.adaptive_score_clip,
            adaptive_uniform_ratio=self.cfg.adaptive_uniform_ratio,
        )

    def _update_motion_sampling_scores(self, env_ids: torch.Tensor) -> None:
        """Update Stage-I failure EMA before resetting completed environments."""
        assert self._motion_dataset is not None
        self._motion_dataset.update_adaptive_scores(
            self._clip_ids[env_ids],
            self._motion_start_times[env_ids],
            self.reset_terminated[env_ids].float(),
        )

    def _sample_motion_starts(
        self, env_ids: torch.Tensor, future_horizon_s: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Stage-I motion starts; Stage II splits by environment role."""
        assert self._motion_dataset is not None
        return self._motion_dataset.sample_starts(len(env_ids), future_horizon_s)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)
        self.reference_robot: Articulation | None = None
        if self.cfg.visualize_reference_robot:
            reference_spawn = self.cfg.robot.spawn.replace(
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=self.cfg.reference_robot_color,
                    opacity=self.cfg.reference_robot_opacity,
                ),
            )
            reference_cfg = self.cfg.robot.replace(
                prim_path="/World/envs/env_.*/ReferenceRobot",
                spawn=reference_spawn,
            )
            self.reference_robot = Articulation(reference_cfg)

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.articulations["robot"] = self.robot
        if self.reference_robot is not None:
            self.scene.articulations["reference_robot"] = self.reference_robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            ),
        )
        if self.cfg.mushroom_spawn is not None:
            self.cfg.mushroom_spawn.func(
                "/World/envs/env_0/Mushroom",
                self.cfg.mushroom_spawn,
                translation=self.cfg.mushroom_position,
                orientation=(1.0, 0.0, 0.0, 0.0),
            )
            if self.cfg.mushroom_physics_material is not None:
                mushroom_material_path = "/World/envs/env_0/Mushroom/physicsMaterial"
                self.cfg.mushroom_physics_material.func(
                    mushroom_material_path,
                    self.cfg.mushroom_physics_material,
                )
                # Bind on the authored asset root so the material is inherited by
                # the collision mesh inside the instanceable geometry prototype.
                # Authoring directly on an instance proxy is not allowed by USD.
                stage = sim_utils.get_current_stage()
                mushroom_prim = stage.GetPrimAtPath("/World/envs/env_0/Mushroom")
                material_binding_api = UsdShade.MaterialBindingAPI.Apply(mushroom_prim)
                material_binding_api.Bind(
                    UsdShade.Material(stage.GetPrimAtPath(mushroom_material_path)),
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
        self.scene.clone_environments(copy_from_source=False)
        if (
            self.cfg.mushroom_spawn is not None
            and self.cfg.mushroom_acquisition_only
            and hasattr(self, "_stage2_consolidation_env_ids")
        ):
            # Stage-II consolidation replays ordinary mastered motions. Keep
            # identical cloned topology for PhysX replication, but park the
            # apparatus far below those environments so it cannot affect the
            # contacts or retention objective.
            stage = sim_utils.get_current_stage()
            for env_id in self._stage2_consolidation_env_ids.cpu().tolist():
                mushroom_prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{int(env_id)}/Mushroom"
                )
                mushroom_prim.GetAttribute("xformOp:translate").Set(
                    Gf.Vec3d(0.0, 0.0, -100.0)
                )
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _motion_times(self) -> torch.Tensor:
        return self._motion_start_times + self.episode_length_buf.float() * self.step_dt

    def _sample_reference(self, times: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample either the real dataset or the explicit debug-only neutral reference."""
        if self._motion_dataset is not None:
            return self._motion_dataset.sample(self._clip_ids, times)

        leading_shape = times.shape
        num_windows = times[0].numel() if times.ndim > 1 else 1
        root_default = self.robot.data.default_root_state[:, :7]
        joint_default = self.robot.data.default_joint_pos
        body_local = self.robot.data.body_pos_w[:, self._key_body_ids] - self.scene.env_origins[:, None, :]
        body_quat = self.robot.data.body_quat_w[:, self._key_body_ids]

        def expand(value: torch.Tensor) -> torch.Tensor:
            if times.ndim == 1:
                return value
            return value[:, None].expand((-1, num_windows) + value.shape[1:])

        return {
            "root_pos": expand(root_default[:, :3]),
            "root_quat": expand(root_default[:, 3:7]),
            "root_lin_vel": torch.zeros(leading_shape + (3,), device=self.device),
            "root_ang_vel": torch.zeros(leading_shape + (3,), device=self.device),
            "joint_pos": expand(joint_default),
            "joint_vel": torch.zeros(leading_shape + (self.cfg.action_space,), device=self.device),
            "body_pos": expand(body_local),
            "body_lin_vel": torch.zeros(
                leading_shape + (len(self._key_body_ids), 3), device=self.device
            ),
            "body_quat": expand(body_quat),
            "body_ang_vel": torch.zeros(
                leading_shape + (len(self._key_body_ids), 3), device=self.device
            ),
        }

    def _set_current_reference(self) -> None:
        self._current_reference = self._sample_reference(self._motion_times())

    def _update_reference_robot(self, env_ids: Sequence[int] | None = None) -> None:
        """Write the current kinematic reference into the optional play-only G1."""
        if self.reference_robot is None:
            return
        if env_ids is None:
            env_ids = self.reference_robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        reference = self._current_reference
        root_position = (
            reference["root_pos"][env_ids]
            + self.scene.env_origins[env_ids]
            + self._reference_robot_offset
        )
        root_pose = torch.cat((root_position, reference["root_quat"][env_ids]), dim=-1)
        root_velocity = torch.cat(
            (reference["root_lin_vel"][env_ids], reference["root_ang_vel"][env_ids]), dim=-1
        )
        joint_position = reference["joint_pos"][env_ids]
        joint_velocity = reference["joint_vel"][env_ids]

        self.reference_robot.write_root_pose_to_sim(root_pose, env_ids)
        self.reference_robot.write_root_velocity_to_sim(root_velocity, env_ids)
        self.reference_robot.write_joint_state_to_sim(
            joint_position, joint_velocity, None, env_ids
        )
        self.reference_robot.set_joint_position_target(joint_position, env_ids=env_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        # Preserve the exact Gaussian sample stored by PPO. Clipping here would
        # make PPO optimize a different action from the one executed by the
        # robot and observed by the action history/action-rate reward.
        self._actions.copy_(actions)
        self._set_current_reference()
        self._processed_actions = (
            self._current_reference["joint_pos"]
            + self.cfg.action_scale * self._actions
            + self._motor_zero_offsets
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._processed_actions)
        self._update_reference_robot()

    def _build_proprioception(self, noisy: bool) -> torch.Tensor:
        gravity = self.robot.data.projected_gravity_b
        angular_velocity = self.robot.data.root_ang_vel_b
        joint_error = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_velocity = self.robot.data.joint_vel
        if noisy and self.cfg.observation_noise:
            gravity = gravity + torch.randn_like(gravity) * self.cfg.gravity_noise_std
            angular_velocity = (
                angular_velocity + torch.randn_like(angular_velocity) * self.cfg.angular_velocity_noise_std
            )
            joint_error = joint_error + torch.randn_like(joint_error) * self.cfg.joint_position_noise_std
            joint_velocity = joint_velocity + torch.randn_like(joint_velocity) * self.cfg.joint_velocity_noise_std
        return torch.cat(
            (gravity, angular_velocity, joint_error, joint_velocity), dim=-1
        )

    def _build_command_window(self) -> torch.Tensor:
        offsets = torch.arange(
            -(self.cfg.command_window_steps // 2),
            self.cfg.command_window_steps // 2 + 1,
            device=self.device,
            dtype=torch.float32,
        )
        query_times = self._motion_times()[:, None] + offsets[None, :] * self.step_dt
        if self._motion_dataset is not None:
            max_times = self._motion_dataset.durations[self._clip_ids, None]
            query_times = torch.clamp(query_times, min=0.0)
            query_times = torch.minimum(query_times, max_times)
        else:
            query_times = torch.clamp(query_times, min=0.0)
        reference = self._sample_reference(query_times)

        flat_quat = reference["root_quat"].reshape(-1, 4)
        linear_velocity_b = quat_apply_inverse(
            flat_quat, reference["root_lin_vel"].reshape(-1, 3)
        ).view(self.num_envs, self.cfg.command_window_steps, 3)
        angular_velocity_b = quat_apply_inverse(
            flat_quat, reference["root_ang_vel"].reshape(-1, 3)
        ).view(self.num_envs, self.cfg.command_window_steps, 3)
        gravity_w = torch.zeros_like(reference["root_lin_vel"])
        gravity_w[..., 2] = -1.0
        gravity_b = quat_apply_inverse(flat_quat, gravity_w.reshape(-1, 3)).view(
            self.num_envs, self.cfg.command_window_steps, 3
        )
        return torch.cat(
            (linear_velocity_b, angular_velocity_b, gravity_b, reference["joint_pos"]), dim=-1
        )

    def _perturb_command_window(self, command_window: torch.Tensor) -> torch.Tensor:
        """Apply the Table II per-environment command perturbations."""
        perturbed = command_window.clone()
        perturbed[:, :, 0:3] += self._command_linear_velocity_offsets[:, None, :]
        perturbed[:, :, 3:6] += self._command_angular_velocity_offsets[:, None, :]
        perturbed[:, :, 6:9] += self._command_gravity_offsets[:, None, :]
        perturbed[:, :, 9:] += self._command_joint_position_offsets[:, None, :]
        return perturbed

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._set_current_reference()
        clean_proprio = self._build_proprioception(noisy=False)
        noisy_proprio = self._build_proprioception(noisy=True)
        self._proprio_history[:, :-1] = self._proprio_history[:, 1:].clone()
        self._proprio_history[:, -1] = noisy_proprio
        self._action_history[:, :-1] = self._action_history[:, 1:].clone()
        self._action_history[:, -1] = self._actions

        clean_command_window = self._build_command_window()
        actor_command_window = self._perturb_command_window(clean_command_window)
        actor_obs = torch.cat(
            (
                actor_command_window.flatten(1),
                self._proprio_history.flatten(1),
                self._action_history.flatten(1),
            ),
            dim=-1,
        )
        current_command = clean_command_window[:, self.cfg.command_window_steps // 2]
        link_positions = self.robot.data.body_pos_w[:, self._key_body_ids] - self.scene.env_origins[:, None, :]
        link_rotation_matrices = matrix_from_quat(self.robot.data.body_quat_w[:, self._key_body_ids])
        link_orientations = link_rotation_matrices[..., :2].reshape(self.num_envs, len(self._key_body_ids), 6)
        link_poses = torch.cat((link_positions, link_orientations), dim=-1).flatten(1)
        privileged = torch.cat(
            (
                self._current_reference["root_pos"][:, 2:3],
                link_poses,
                self.robot.data.root_lin_vel_b,
            ),
            dim=-1,
        )
        critic_obs = torch.cat(
            (clean_proprio, self._actions, current_command, privileged), dim=-1
        )
        return {"policy": actor_obs, "critic": critic_obs}

    @staticmethod
    def _exp_tracking(error: torch.Tensor, sigma: float) -> torch.Tensor:
        return torch.exp(-error / (sigma * sigma))

    def _get_rewards(self) -> torch.Tensor:
        """Table I motion-tracking reward for the Extreme-RGMT base policy."""
        ref = self._current_reference
        current_pos = self.robot.data.body_pos_w[:, self._key_body_ids]
        current_quat = self.robot.data.body_quat_w[:, self._key_body_ids]
        current_lin_vel = self.robot.data.body_lin_vel_w[:, self._key_body_ids]
        current_ang_vel = self.robot.data.body_ang_vel_w[:, self._key_body_ids]
        ref_pos = ref["body_pos"] + self.scene.env_origins[:, None, :]
        ref_quat = ref["body_quat"]

        current_relative_pos = current_pos - current_pos[:, :1]
        reference_relative_pos = ref_pos - ref_pos[:, :1]
        current_anchor_inv = quat_inv(current_quat[:, 0])[:, None, :].expand_as(
            current_quat
        )
        reference_anchor_inv = quat_inv(ref_quat[:, 0])[:, None, :].expand_as(
            ref_quat
        )
        current_relative_quat = quat_mul(
            current_anchor_inv.reshape(-1, 4), current_quat.reshape(-1, 4)
        ).view_as(current_quat)
        reference_relative_quat = quat_mul(
            reference_anchor_inv.reshape(-1, 4), ref_quat.reshape(-1, 4)
        ).view_as(ref_quat)

        errors = {
            "global_anchor_orientation": quat_error_magnitude(
                self.robot.data.root_quat_w, ref["root_quat"]
            )
            ** 2,
            "relative_body_position": torch.mean(
                (current_relative_pos - reference_relative_pos) ** 2,
                dim=(-2, -1),
            ),
            "relative_body_orientation": torch.mean(
                quat_error_magnitude(
                    current_relative_quat.reshape(-1, 4),
                    reference_relative_quat.reshape(-1, 4),
                ).view(self.num_envs, -1)
                ** 2,
                dim=-1,
            ),
            "global_body_linear_velocity": torch.mean(
                (current_lin_vel - ref["body_lin_vel"]) ** 2,
                dim=(-2, -1),
            ),
            "global_body_angular_velocity": torch.mean(
                (current_ang_vel - ref["body_ang_vel"]) ** 2,
                dim=(-2, -1),
            ),
        }
        rewards = {
            name: self._exp_tracking(errors[name], self.cfg.tracking_sigmas[name])
            * self.cfg.reward_scales[name]
            for name in errors
        }

        force_history = self.contact_sensor.data.net_forces_w_history
        undesired_contact = (
            torch.max(
                torch.norm(
                    force_history[:, :, self._undesired_sensor_ids], dim=-1
                ),
                dim=1,
            ).values
            > self.cfg.undesired_contact_force_threshold
        ).float().sum(dim=-1)
        foot_contact = (
            torch.max(
                torch.norm(force_history[:, :, self._foot_sensor_ids], dim=-1),
                dim=1,
            ).values
            > self.cfg.foot_contact_force_threshold
        )
        foot_speed = torch.norm(
            self.robot.data.body_lin_vel_w[:, self._foot_body_ids, :2], dim=-1
        )
        feet_slip = torch.sum(foot_contact.float() * foot_speed**2, dim=-1)
        limits = self.robot.data.soft_joint_pos_limits
        joint_limit_violation = (
            torch.relu(limits[:, :, 0] - self.robot.data.joint_pos)
            + torch.relu(self.robot.data.joint_pos - limits[:, :, 1])
        ).sum(dim=-1)
        rewards.update(
            {
                "action_rate": torch.sum(
                    (self._actions - self._previous_actions) ** 2, dim=-1
                )
                * self.cfg.reward_scales["action_rate"],
                "joint_limits": joint_limit_violation
                * self.cfg.reward_scales["joint_limits"],
                "undesired_contact": undesired_contact
                * self.cfg.reward_scales["undesired_contact"],
                "feet_slip": feet_slip * self.cfg.reward_scales["feet_slip"],
            }
        )
        for name, value in rewards.items():
            self._episode_sums[name] += value
        return torch.stack(list(rewards.values()), dim=0).sum(dim=0) * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._set_current_reference()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self._motion_dataset is not None:
            future_horizon = (self.cfg.command_window_steps // 2) * self.step_dt
            motion_end = self._motion_times() + future_horizon >= self._motion_dataset.durations[self._clip_ids]
            time_out |= motion_end

        if self.cfg.disable_early_termination:
            return torch.zeros_like(time_out), time_out

        current_key_body_heights = (
            self.robot.data.body_pos_w[:, self._termination_body_ids, 2]
            - self.scene.env_origins[:, None, 2]
        )
        reference_key_body_heights = self._current_reference["body_pos"][
            :, self._termination_reference_ids, 2
        ]
        low_key_body = torch.any(
            current_key_body_heights
            < reference_key_body_heights - self.cfg.termination_key_body_height_tolerance,
            dim=-1,
        )
        current_root_height = (
            self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        )
        reference_root_height = self._current_reference["root_pos"][:, 2]
        root_too_low = (
            current_root_height
            < reference_root_height - self.cfg.termination_root_height_tolerance
        )
        root_orientation_error = quat_error_magnitude(
            self.robot.data.root_quat_w, self._current_reference["root_quat"]
        )
        root_orientation_mismatch = (
            root_orientation_error > self.cfg.termination_root_orientation_error
        )
        terminated = low_key_body | root_too_low | root_orientation_mismatch
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if (
            self._motion_dataset is not None
            and self.common_step_counter > 0
        ):
            self._update_motion_sampling_scores(env_ids)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        count = len(env_ids)
        if self._motion_dataset is not None:
            horizon = (self.cfg.command_window_steps // 2) * self.step_dt
            if self.cfg.fixed_motion_clip_id is not None:
                fixed_clip_id = int(self.cfg.fixed_motion_clip_id)
                if not 0 <= fixed_clip_id < len(self._motion_dataset.clips):
                    raise ValueError(
                        f"fixed_motion_clip_id={fixed_clip_id} is outside the dataset range "
                        f"[0, {len(self._motion_dataset.clips) - 1}]."
                    )
                max_start = max(
                    float(self._motion_dataset.durations[fixed_clip_id].item()) - horizon,
                    0.0,
                )
                fixed_start = min(max(float(self.cfg.fixed_motion_start_time_s), 0.0), max_start)
                clip_ids = torch.full(
                    (count,), fixed_clip_id, dtype=torch.long, device=self.device
                )
                start_times = torch.full(
                    (count,), fixed_start, dtype=torch.float32, device=self.device
                )
            elif self.cfg.replay_full_motions:
                clip_count = len(self._motion_dataset.clips)
                if self.common_step_counter == 0:
                    clip_ids = env_ids % clip_count
                else:
                    clip_ids = self._clip_ids[env_ids].clone()
                    completed = self.reset_time_outs[env_ids]
                    clip_ids[completed] = (
                        clip_ids[completed] + self.num_envs
                    ) % clip_count
                start_times = torch.zeros(count, dtype=torch.float32, device=self.device)
            else:
                clip_ids, start_times = self._sample_motion_starts(env_ids, horizon)
            self._clip_ids[env_ids] = clip_ids
            self._motion_start_times[env_ids] = start_times
            reference = self._motion_dataset.sample(clip_ids, start_times)
        else:
            self._clip_ids[env_ids] = 0
            self._motion_start_times[env_ids] = 0.0
            reference = {key: value[env_ids] for key, value in self._sample_reference(self._motion_times()).items()}

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = reference["root_pos"] + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = reference["root_quat"]
        root_state[:, 7:10] = reference["root_lin_vel"]
        root_state[:, 10:13] = reference["root_ang_vel"]
        joint_pos = reference["joint_pos"].clone()
        joint_vel = reference["joint_vel"].clone()

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = joint_pos
        self._action_history[env_ids] = 0.0
        self._motor_zero_offsets[env_ids] = torch.empty(
            (count, self.cfg.action_space), device=self.device
        ).uniform_(*self.cfg.motor_zero_offset_range)
        self._command_linear_velocity_offsets[env_ids] = torch.empty(
            (count, 3), device=self.device
        ).uniform_(
            -self.cfg.command_linear_velocity_perturbation,
            self.cfg.command_linear_velocity_perturbation,
        )
        self._command_angular_velocity_offsets[env_ids] = torch.empty(
            (count, 3), device=self.device
        ).uniform_(
            -self.cfg.command_angular_velocity_perturbation,
            self.cfg.command_angular_velocity_perturbation,
        )
        self._command_gravity_offsets[env_ids] = torch.empty(
            (count, 3), device=self.device
        ).uniform_(
            -self.cfg.command_gravity_perturbation,
            self.cfg.command_gravity_perturbation,
        )
        self._command_joint_position_offsets[env_ids] = torch.empty(
            (count, self.cfg.action_space), device=self.device
        ).uniform_(
            -self.cfg.command_joint_position_perturbation,
            self.cfg.command_joint_position_perturbation,
        )
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._current_reference = self._sample_reference(self._motion_times())
        if self.reference_robot is not None:
            self.reference_robot.reset(env_ids)
            self._update_reference_robot(env_ids)
        initial_proprio = self._build_proprioception(noisy=False)
        self._proprio_history[env_ids] = initial_proprio[env_ids, None, :]

        self.extras["log"] = {}
        for name, episode_sum in self._episode_sums.items():
            self.extras["log"][f"Episode_Reward/{name}"] = (
                torch.mean(episode_sum[env_ids]) / self.max_episode_length_s
            )
            episode_sum[env_ids] = 0.0
        self.extras["log"]["Episode_Termination/terminated"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
