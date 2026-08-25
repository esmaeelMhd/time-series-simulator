"""Gymnasium environment adapter for world models.

This module provides a Gymnasium-compatible interface for trained world models,
enabling their use with RL algorithms and MPC controllers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ..data.dataset import GroupedTimeSeriesDataset
from ..models.base import WorldModelBase


class WorldModelEnv(gym.Env):
    """Gymnasium environment wrapper for a trained world model.
    
    This environment uses a trained world model as the dynamics simulator.
    It can be used for:
    1. Model-based RL (train a policy using the world model)
    2. MPC (plan actions using the world model)
    3. Testing and evaluation
    
    The environment operates by:
    - Selecting a historical exogenous sequence as background "scenario"
    - Using the world model to simulate system response to actions
    - Computing rewards based on predicted outputs
    
    Parameters
    ----------
    world_model : WorldModelBase
        Trained world model to use for simulation.
    dataset : GroupedTimeSeriesDataset
        Dataset containing historical data for exogenous inputs.
    warmup_len : int
        Length of warmup sequence for state initialization.
    episode_len : int
        Maximum episode length.
    control_dim : int
        Dimension of control inputs (action space).
    exo_dim : int
        Dimension of exogenous inputs.
    output_dim : int
        Dimension of outputs (observation space).
    reward_fn : callable, optional
        Function to compute reward from predicted outputs.
        Signature: reward_fn(outputs: np.ndarray) -> float
        If None, uses negative MSE from a target (requires target_output).
    target_output : np.ndarray, optional
        Target output for default reward function.
    control_bounds : tuple of np.ndarray, optional
        (low, high) bounds for control inputs. If None, uses [-inf, inf].
    device : torch.device or str, default "cpu"
        Device for model inference.
    
    Notes
    -----
    This is a minimal stub implementation. For production use, you may want to:
    - Add more sophisticated reward shaping
    - Support multi-objective rewards
    - Add action penalties (e.g., control effort)
    - Support domain-specific constraints
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        world_model: WorldModelBase,
        dataset: GroupedTimeSeriesDataset,
        warmup_len: int,
        episode_len: int,
        control_dim: int,
        exo_dim: int,
        output_dim: int,
        reward_fn: Optional[callable] = None,
        target_output: Optional[np.ndarray] = None,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__()

        self.world_model = world_model
        self.world_model.eval()
        self.dataset = dataset
        self.warmup_len = warmup_len
        self.episode_len = episode_len
        self.control_dim = control_dim
        self.exo_dim = exo_dim
        self.output_dim = output_dim
        self.device = torch.device(device)

        # Reward function
        if reward_fn is not None:
            self.reward_fn = reward_fn
        elif target_output is not None:
            # Default: negative MSE from target
            self.target_output = target_output
            self.reward_fn = lambda obs: -np.mean((obs - self.target_output) ** 2)
        else:
            raise ValueError("Either reward_fn or target_output must be provided")

        # Action space (controls)
        if control_bounds is not None:
            low, high = control_bounds
        else:
            low = np.full(control_dim, -np.inf, dtype=np.float32)
            high = np.full(control_dim, np.inf, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Observation space (outputs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(output_dim,), dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.model_state = None
        self.current_obs = None
        self.exo_sequence = None
        self.scenario_start_idx = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to a new episode.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional options (e.g., specific scenario index).
        
        Returns
        -------
        observation : np.ndarray
            Initial observation.
        info : dict
            Additional information.
        """
        super().reset(seed=seed)

        # Sample a random scenario (starting point in dataset)
        if options is not None and "scenario_idx" in options:
            start_idx = options["scenario_idx"]
        else:
            max_start = len(self.dataset.values) - (self.warmup_len + self.episode_len)
            start_idx = self.np_random.integers(self.warmup_len, max_start)

        self.scenario_start_idx = start_idx

        # Get warmup data and initialize model state
        warmup_data = self.dataset.get_warmup_window(start_idx, self.warmup_len)
        warmup_inputs = torch.tensor(
            warmup_data["inputs"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            self.model_state = self.world_model.init_state(warmup_inputs)

        # Get exogenous sequence for this episode
        rollout_data = self.dataset.get_rollout_slice(start_idx, self.episode_len)
        self.exo_sequence = rollout_data["inputs"][:, self.control_dim:self.control_dim+self.exo_dim]

        # Initial observation (last output from warmup)
        self.current_obs = warmup_data["outputs"][-1]
        self.current_step = 0

        info = {
            "scenario_idx": start_idx,
        }

        return self.current_obs.astype(np.float32), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Parameters
        ----------
        action : np.ndarray
            Control inputs (actions) to apply.
        
        Returns
        -------
        observation : np.ndarray
            Next observation (predicted output).
        reward : float
            Reward for this step.
        terminated : bool
            Whether the episode is terminated (reached goal).
        truncated : bool
            Whether the episode is truncated (max steps).
        info : dict
            Additional information.
        """
        if self.model_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Prepare inputs
        control_t = torch.tensor(
            action, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        exo_t = torch.tensor(
            self.exo_sequence[self.current_step],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        prev_output_t = torch.tensor(
            self.current_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Simulate one step
        with torch.no_grad():
            self.model_state, pred_t = self.world_model.step(
                self.model_state, control_t, exo_t, prev_output_t
            )

        # Update observation
        self.current_obs = pred_t.cpu().numpy().squeeze(0)

        # Compute reward
        reward = self.reward_fn(self.current_obs)

        # Check termination
        self.current_step += 1
        truncated = self.current_step >= self.episode_len
        terminated = False  # Could add goal-reaching logic here

        info = {
            "step": self.current_step,
        }

        return self.current_obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        """Render the environment (stub implementation)."""
        if self.current_obs is not None:
            print(f"Step {self.current_step}: obs={self.current_obs}")

    def close(self):
        """Clean up resources."""
        pass


# Example usage (commented out):
"""
# Load trained model
model = LSTMWorldModel(input_dim=10, output_dim=3)
model.load_state_dict(torch.load("checkpoint.pth"))

# Create environment
env = WorldModelEnv(
    world_model=model,
    dataset=dataset,
    warmup_len=24,
    episode_len=100,
    control_dim=2,
    exo_dim=5,
    output_dim=3,
    target_output=np.array([0.5, 0.3, 0.2]),
)

# Use with RL
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # or use a policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
"""

