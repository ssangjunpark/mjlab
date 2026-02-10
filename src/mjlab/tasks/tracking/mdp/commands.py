from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]


class MultiMotionLoader:
  """Loader for multiple motion trajectories with per-trajectory indexing."""

  def __init__(
    self, motion_files: tuple[str, ...] | list[str], body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    self.device = device
    self.num_motions = len(motion_files)
    self.motions: list[MotionLoader] = []
    self.trajectory_lengths: list[int] = []

    for motion_file in motion_files:
      motion = MotionLoader(motion_file, body_indexes, device)
      self.motions.append(motion)
      self.trajectory_lengths.append(motion.time_step_total)

    self.trajectory_lengths_tensor = torch.tensor(
      self.trajectory_lengths, dtype=torch.long, device=device
    )
    self.max_trajectory_length = max(self.trajectory_lengths)

  def get_joint_pos(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
    """Get joint positions for given motion IDs and time steps."""
    batch_size = motion_ids.shape[0]
    joint_dim = self.motions[0].joint_pos.shape[1]
    result = torch.zeros(batch_size, joint_dim, device=self.device)
    for i in range(self.num_motions):
      mask = motion_ids == i
      if mask.any():
        clamped_time = torch.clamp(time_steps[mask], 0, self.trajectory_lengths[i] - 1)
        result[mask] = self.motions[i].joint_pos[clamped_time]
    return result

  def get_joint_vel(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
    """Get joint velocities for given motion IDs and time steps."""
    batch_size = motion_ids.shape[0]
    joint_dim = self.motions[0].joint_vel.shape[1]
    result = torch.zeros(batch_size, joint_dim, device=self.device)
    for i in range(self.num_motions):
      mask = motion_ids == i
      if mask.any():
        clamped_time = torch.clamp(time_steps[mask], 0, self.trajectory_lengths[i] - 1)
        result[mask] = self.motions[i].joint_vel[clamped_time]
    return result

  def get_body_pos_w(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
    """Get body positions for given motion IDs and time steps."""
    batch_size = motion_ids.shape[0]
    num_bodies = self.motions[0].body_pos_w.shape[1]
    result = torch.zeros(batch_size, num_bodies, 3, device=self.device)
    for i in range(self.num_motions):
      mask = motion_ids == i
      if mask.any():
        clamped_time = torch.clamp(time_steps[mask], 0, self.trajectory_lengths[i] - 1)
        result[mask] = self.motions[i].body_pos_w[clamped_time]
    return result

  def get_body_quat_w(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
    """Get body quaternions for given motion IDs and time steps."""
    batch_size = motion_ids.shape[0]
    num_bodies = self.motions[0].body_quat_w.shape[1]
    result = torch.zeros(batch_size, num_bodies, 4, device=self.device)
    result[:, :, 0] = 1.0  # Identity quaternion default
    for i in range(self.num_motions):
      mask = motion_ids == i
      if mask.any():
        clamped_time = torch.clamp(time_steps[mask], 0, self.trajectory_lengths[i] - 1)
        result[mask] = self.motions[i].body_quat_w[clamped_time]
    return result

  def get_body_lin_vel_w(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
    """Get body linear velocities for given motion IDs and time steps."""
    batch_size = motion_ids.shape[0]
    num_bodies = self.motions[0].body_lin_vel_w.shape[1]
    result = torch.zeros(batch_size, num_bodies, 3, device=self.device)
    for i in range(self.num_motions):
      mask = motion_ids == i
      if mask.any():
        clamped_time = torch.clamp(time_steps[mask], 0, self.trajectory_lengths[i] - 1)
        result[mask] = self.motions[i].body_lin_vel_w[clamped_time]
    return result

  def get_body_ang_vel_w(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
    """Get body angular velocities for given motion IDs and time steps."""
    batch_size = motion_ids.shape[0]
    num_bodies = self.motions[0].body_ang_vel_w.shape[1]
    result = torch.zeros(batch_size, num_bodies, 3, device=self.device)
    for i in range(self.num_motions):
      mask = motion_ids == i
      if mask.any():
        clamped_time = torch.clamp(time_steps[mask], 0, self.trajectory_lengths[i] - 1)
        result[mask] = self.motions[i].body_ang_vel_w[clamped_time]
    return result


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    # Support both single motion file and multiple motion files
    self._use_multi_motion = len(cfg.motion_files) > 0

    if self._use_multi_motion:
      print("[INFO] using multiple motion, activating multiple action sampling mode")
    
    self.multi_motion: MultiMotionLoader | None = None
    self.motion_ids: torch.Tensor | None = None
    if self._use_multi_motion:
      self.multi_motion = MultiMotionLoader(
        cfg.motion_files, self.body_indexes, device=self.device
      )
      # Use first motion as reference for single-motion API compatibility
      self.motion = self.multi_motion.motions[0]
      # Track which motion each env is currently following
      self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    else:
      self.motion = MotionLoader(
        self.cfg.motion_file, self.body_indexes, device=self.device
      )

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # For multi-motion: per-trajectory bin counts for adaptive sampling
    if self._use_multi_motion:
      assert self.multi_motion is not None  # Type narrowing
      self.bin_count_per_traj: list[int] = [
        int(m.time_step_total // (1 / env.step_dt)) + 1
        for m in self.multi_motion.motions
      ]
      self.max_bin_count = max(self.bin_count_per_traj)
      self.bin_count = self.max_bin_count  # For compatibility
      # Per-trajectory failure counts: (num_motions, max_bin_count)
      self.bin_failed_count_per_traj = torch.zeros(
        self.multi_motion.num_motions, self.max_bin_count, dtype=torch.float, device=self.device
      )
      self._current_bin_failed_per_traj = torch.zeros(
        self.multi_motion.num_motions, self.max_bin_count, dtype=torch.float, device=self.device
      )
      # Trajectory-level failure counts for sampling which trajectory
      self.traj_failed_count = torch.zeros(
        self.multi_motion.num_motions, dtype=torch.float, device=self.device
      )
      self._current_traj_failed = torch.zeros(
        self.multi_motion.num_motions, dtype=torch.float, device=self.device
      )
    else:
      self.bin_count_per_traj = []
      self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1

    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
    # Multi-motion specific metrics
    self.metrics["sampling_traj_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_traj_id"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      return self.multi_motion.get_joint_pos(self.motion_ids, self.time_steps)
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      return self.multi_motion.get_joint_vel(self.motion_ids, self.time_steps)
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      return (
        self.multi_motion.get_body_pos_w(self.motion_ids, self.time_steps)
        + self._env.scene.env_origins[:, None, :]
      )
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      return self.multi_motion.get_body_quat_w(self.motion_ids, self.time_steps)
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      return self.multi_motion.get_body_lin_vel_w(self.motion_ids, self.time_steps)
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      return self.multi_motion.get_body_ang_vel_w(self.motion_ids, self.time_steps)
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      body_pos = self.multi_motion.get_body_pos_w(self.motion_ids, self.time_steps)
      return body_pos[:, self.motion_anchor_body_index] + self._env.scene.env_origins
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      body_quat = self.multi_motion.get_body_quat_w(self.motion_ids, self.time_steps)
      return body_quat[:, self.motion_anchor_body_index]
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      body_lin_vel = self.multi_motion.get_body_lin_vel_w(self.motion_ids, self.time_steps)
      return body_lin_vel[:, self.motion_anchor_body_index]
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      body_ang_vel = self.multi_motion.get_body_ang_vel_w(self.motion_ids, self.time_steps)
      return body_ang_vel[:, self.motion_anchor_body_index]
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample.
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count)
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      # Uniformly sample trajectory and time step
      num_motions = self.multi_motion.num_motions
      sampled_traj_ids = torch.randint(
        0, num_motions, (len(env_ids),), device=self.device
      )
      self.motion_ids[env_ids] = sampled_traj_ids

      # Sample time steps within each trajectory's valid range
      for i in range(num_motions):
        traj_mask = sampled_traj_ids == i
        if traj_mask.any():
          traj_length = self.multi_motion.trajectory_lengths[i]
          traj_env_ids = env_ids[traj_mask]
          self.time_steps[traj_env_ids] = torch.randint(
            0, traj_length, (len(traj_env_ids),), device=self.device
          )

      self.metrics["sampling_traj_entropy"][:] = 1.0
      self.metrics["sampling_traj_id"][env_ids] = sampled_traj_ids.float()
    else:
      self.time_steps[env_ids] = torch.randint(
        0, self.motion.time_step_total, (len(env_ids),), device=self.device
      )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

  def _sample_from_multiple_traj(self, env_ids: torch.Tensor):
    """Sample from multiple trajectories with hierarchical adaptive sampling.

    First samples which trajectory to use (based on trajectory-level failures),
    then samples a time step within that trajectory (based on per-trajectory
    bin failures). This ensures no discontinuities across trajectory boundaries.
    """
    if not self._use_multi_motion:
      raise RuntimeError("_sample_from_multiple_traj called without multi-motion setup")

    assert self.multi_motion is not None and self.motion_ids is not None
    num_motions = self.multi_motion.num_motions
    episode_failed = self._env.termination_manager.terminated[env_ids]

    # Update failure statistics for trajectories and bins
    if torch.any(episode_failed):
      failed_env_ids = env_ids[episode_failed]
      failed_motion_ids = self.motion_ids[failed_env_ids]
      failed_time_steps = self.time_steps[failed_env_ids]

      # Update trajectory-level failures
      self._current_traj_failed[:] = torch.bincount(
        failed_motion_ids, minlength=num_motions
      ).float()

      # Update per-trajectory bin failures
      for i in range(num_motions):
        traj_mask = failed_motion_ids == i
        if traj_mask.any():
          traj_length = self.multi_motion.trajectory_lengths[i]
          bin_count = self.bin_count_per_traj[i]
          bin_indices = torch.clamp(
            (failed_time_steps[traj_mask] * bin_count) // max(traj_length, 1),
            0,
            bin_count - 1,
          )
          self._current_bin_failed_per_traj[i, :bin_count] = torch.bincount(
            bin_indices, minlength=bin_count
          ).float()

    # Step 1: Sample which trajectory to use for each env
    # Compute trajectory sampling probabilities based on failures
    traj_probs = self.traj_failed_count + self.cfg.adaptive_uniform_ratio / float(num_motions)
    traj_probs = traj_probs / traj_probs.sum()

    sampled_traj_ids = torch.multinomial(traj_probs, len(env_ids), replacement=True)
    self.motion_ids[env_ids] = sampled_traj_ids

    # Compute trajectory-level entropy for metrics
    H_traj = -(traj_probs * (traj_probs + 1e-12).log()).sum()
    H_traj_norm = H_traj / math.log(num_motions) if num_motions > 1 else 1.0
    self.metrics["sampling_traj_entropy"][:] = H_traj_norm
    self.metrics["sampling_traj_id"][env_ids] = sampled_traj_ids.float()

    # Step 2: For each trajectory, sample time steps using adaptive sampling
    for i in range(num_motions):
      traj_mask = sampled_traj_ids == i
      if not traj_mask.any():
        continue

      traj_env_ids = env_ids[traj_mask]
      traj_length = self.multi_motion.trajectory_lengths[i]
      bin_count = self.bin_count_per_traj[i]

      # Get per-trajectory bin failure counts
      bin_failed = self.bin_failed_count_per_traj[i, :bin_count]

      # Compute sampling probabilities with smoothing kernel
      sampling_probs = bin_failed + self.cfg.adaptive_uniform_ratio / float(bin_count)
      sampling_probs = torch.nn.functional.pad(
        sampling_probs.unsqueeze(0).unsqueeze(0),
        (0, self.cfg.adaptive_kernel_size - 1),
        mode="replicate",
      )
      # Create kernel of appropriate size
      kernel_size = min(self.cfg.adaptive_kernel_size, bin_count)
      kernel = torch.tensor(
        [self.cfg.adaptive_lambda**j for j in range(kernel_size)],
        device=self.device,
      )
      kernel = kernel / kernel.sum()
      sampling_probs = torch.nn.functional.conv1d(
        sampling_probs, kernel.view(1, 1, -1)
      ).view(-1)[:bin_count]
      sampling_probs = sampling_probs / sampling_probs.sum()

      # Sample bins and convert to time steps
      sampled_bins = torch.multinomial(sampling_probs, len(traj_env_ids), replacement=True)
      self.time_steps[traj_env_ids] = (
        (sampled_bins + sample_uniform(0.0, 1.0, (len(traj_env_ids),), device=self.device))
        / bin_count
        * (traj_length - 1)
      ).long()

    # Compute overall sampling entropy (average across trajectories)
    total_bins = sum(self.bin_count_per_traj)
    all_probs = []
    for i in range(num_motions):
      bin_count = self.bin_count_per_traj[i]
      bin_failed = self.bin_failed_count_per_traj[i, :bin_count]
      probs = bin_failed + self.cfg.adaptive_uniform_ratio / float(bin_count)
      probs = probs / probs.sum() * traj_probs[i]
      all_probs.append(probs)
    all_probs = torch.cat(all_probs)
    all_probs = all_probs / all_probs.sum()
    H = -(all_probs * (all_probs + 1e-12).log()).sum()
    H_norm = H / math.log(total_bins) if total_bins > 1 else 1.0
    pmax, imax = all_probs.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / total_bins


  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
      if self._use_multi_motion:
        assert self.multi_motion is not None and self.motion_ids is not None
        # For start mode with multi-motion, randomly assign trajectories
        self.motion_ids[env_ids] = torch.randint(
          0, self.multi_motion.num_motions, (len(env_ids),), device=self.device
        )
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    elif self.cfg.sampling_mode == "multiple":
      if not self._use_multi_motion:
        raise RuntimeError(
          "sampling_mode='multiple' requires motion_files to be set with multiple trajectories"
        )
      self._sample_from_multiple_traj(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      if self._use_multi_motion:
        # Use multi-trajectory adaptive sampling when multiple files are provided
        self._sample_from_multiple_traj(env_ids)
      else:
        self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self):
    self.time_steps += 1

    # Check which envs have exceeded their trajectory length
    if self._use_multi_motion:
      assert self.multi_motion is not None and self.motion_ids is not None
      # Get per-env trajectory lengths based on assigned motion_ids
      traj_lengths = self.multi_motion.trajectory_lengths_tensor[self.motion_ids]
      env_ids = torch.where(self.time_steps >= traj_lengths)[0]
    else:
      env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]

    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    # Update failure statistics for adaptive sampling
    if self.cfg.sampling_mode == "adaptive" or self.cfg.sampling_mode == "multiple":
      if self._use_multi_motion:
        # Update per-trajectory bin failure counts
        self.bin_failed_count_per_traj = (
          self.cfg.adaptive_alpha * self._current_bin_failed_per_traj
          + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count_per_traj
        )
        self._current_bin_failed_per_traj.zero_()
        # Update trajectory-level failure counts
        self.traj_failed_count = (
          self.cfg.adaptive_alpha * self._current_traj_failed
          + (1 - self.cfg.adaptive_alpha) * self.traj_failed_count
        )
        self._current_traj_failed.zero_()
      else:
        self.bin_failed_count = (
          self.cfg.adaptive_alpha * self._current_bin_failed
          + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str = ""  # Single motion file (for backward compatibility)
  motion_files: tuple[str, ...] = ()  # Multiple motion files for multi-trajectory mode
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start", "multiple"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
