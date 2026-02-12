"""RL configuration for Unitree G1 tracking task."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=0.1,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(1024, 1024, 1024, 1024, 1024, 1024),
      critic_hidden_dims=(1024, 1024, 1024, 1024),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=1,
      num_mini_batches=4,
      learning_rate=2e-5,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=50.0,
    ),
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=32,
    max_iterations=30_000,
  )
