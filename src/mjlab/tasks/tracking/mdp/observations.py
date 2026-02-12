from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def _get_future_anchor(
  command: MotionCommand, steps_ahead: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """Get anchor pos/quat at a future timestep, clamped to trajectory bounds."""
  future_steps = command.time_steps + steps_ahead
  if command._use_multi_motion:
    assert command.multi_motion is not None and command.motion_ids is not None
    traj_lengths = command.multi_motion.trajectory_lengths_tensor[command.motion_ids]
    future_steps = torch.clamp(future_steps, max=traj_lengths - 1)
    body_pos = command.multi_motion.get_body_pos_w(command.motion_ids, future_steps)
    body_quat = command.multi_motion.get_body_quat_w(command.motion_ids, future_steps)
    anchor_pos = (
      body_pos[:, command.motion_anchor_body_index] + command._env.scene.env_origins
    )
    anchor_quat = body_quat[:, command.motion_anchor_body_index]
  else:
    future_steps = torch.clamp(future_steps, max=command.motion.time_step_total - 1)
    anchor_pos = (
      command.motion.body_pos_w[future_steps, command.motion_anchor_body_index]
      + command._env.scene.env_origins
    )
    anchor_quat = command.motion.body_quat_w[
      future_steps, command.motion_anchor_body_index
    ]
  return anchor_pos, anchor_quat


def motion_anchor_pos_b_lookahead(
  env: ManagerBasedRlEnv,
  command_name: str,
  num_future_frames: int = 3,
  frame_interval: int = 5,
) -> torch.Tensor:
  """Future reference anchor positions in base frame.

  Args:
    num_future_frames: Number of future frames to include.
    frame_interval: Interval between future frames (in sim steps).

  Returns:
    (num_envs, num_future_frames * 3) tensor of future anchor positions in base frame.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  parts = []
  for i in range(1, num_future_frames + 1):
    future_pos, future_quat = _get_future_anchor(command, i * frame_interval)
    pos_b, _ = subtract_frame_transforms(
      command.robot_anchor_pos_w,
      command.robot_anchor_quat_w,
      future_pos,
      future_quat,
    )
    parts.append(pos_b)
  return torch.cat(parts, dim=-1)


def motion_anchor_ori_b_lookahead(
  env: ManagerBasedRlEnv,
  command_name: str,
  num_future_frames: int = 3,
  frame_interval: int = 5,
) -> torch.Tensor:
  """Future reference anchor orientations in base frame.

  Args:
    num_future_frames: Number of future frames to include.
    frame_interval: Interval between future frames (in sim steps).

  Returns:
    (num_envs, num_future_frames * 6) tensor of future anchor orientations
    (2-col rotation matrix) in base frame.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  parts = []
  for i in range(1, num_future_frames + 1):
    future_pos, future_quat = _get_future_anchor(command, i * frame_interval)
    _, ori_b = subtract_frame_transforms(
      command.robot_anchor_pos_w,
      command.robot_anchor_quat_w,
      future_pos,
      future_quat,
    )
    mat = matrix_from_quat(ori_b)
    parts.append(mat[..., :2].reshape(mat.shape[0], -1))
  return torch.cat(parts, dim=-1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)
