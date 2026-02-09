"""Abstract base class for world models (simulators) for control.

HOT PATH: Model rollout is called every training step.
Optimizations in subclasses should follow:
- Preallocate output tensors (no list.append in rollout loops)
- Minimize tensor allocations in step()
- Use direct tensor indexing

A world model learns the dynamics of a system:
    s_{t+1} = f(s_t, u_t, e_t)

where:
    s_t: state/observation at time t
    u_t: control input (action) at time t
    e_t: exogenous input (disturbance, context) at time t
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal

import torch
import torch.nn as nn


class WorldModelBase(nn.Module, ABC):
    """Abstract base class for recurrent world models.
    
    A world model must be able to:
    1. Initialize its internal state from a warmup sequence
    2. Perform single-step predictions
    3. Perform multi-step autoregressive rollouts
    
    This interface is designed to be:
    - Domain-agnostic (works for any time-series control problem)
    - Compatible with RL/MPC frameworks
    - Easy to wrap into Gymnasium environments
    """
    
    @abstractmethod
    def init_state(self, warmup_seq: torch.Tensor) -> Any:
        """Initialize the model's hidden state from a warmup sequence.
        
        Parameters
        ----------
        warmup_seq : torch.Tensor
            Warmup sequence of shape (batch_size, warmup_len, input_dim).
            This should contain controls, exogenous inputs, and previous outputs
            concatenated along the feature dimension.
        
        Returns
        -------
        state : Any
            The initialized hidden state. For LSTMs this is typically (h, c).
            For transformers it might be a cache or None.
        """
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self,
        state: Any,
        control_t: torch.Tensor,
        exo_t: torch.Tensor,
        prev_output_t: Optional[torch.Tensor] = None,
    ) -> tuple[Any, torch.Tensor]:
        """Perform a single prediction step.
        
        Parameters
        ----------
        state : Any
            Current hidden state from previous step or init_state().
        control_t : torch.Tensor
            Control inputs at time t, shape (batch_size, control_dim).
        exo_t : torch.Tensor
            Exogenous inputs at time t, shape (batch_size, exo_dim).
        prev_output_t : torch.Tensor, optional
            Previous output/observation at time t, shape (batch_size, output_dim).
            If None, assumes this is the first step after warmup.
        
        Returns
        -------
        new_state : Any
            Updated hidden state.
        prediction : torch.Tensor
            Predicted output at time t+1, shape (batch_size, output_dim).
        """
        raise NotImplementedError
    
    def rollout(
        self,
        warmup_seq: Dict[str, torch.Tensor],
        rollout_inputs: Dict[str, torch.Tensor],
        horizon: int,
        feedback: Literal["model", "teacher", "mixed"] = "model",
        teacher_forcing_ratio: float = 0.0,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Perform multi-step autoregressive rollout.
        
        HOT PATH: This is called every training step.
        Optimizations:
        - Preallocated predictions tensor (no list.append)
        - Direct tensor assignment
        
        This is the key method for training world models with multi-step losses.
        
        Parameters
        ----------
        warmup_seq : dict
            Dictionary containing warmup sequences:
            - "inputs": (batch_size, warmup_len, input_dim) - full input features
            OR separately:
            - "controls": (batch_size, warmup_len, control_dim)
            - "exogenous": (batch_size, warmup_len, exo_dim)
            - "outputs": (batch_size, warmup_len, output_dim)
        rollout_inputs : dict
            Dictionary containing inputs for the rollout horizon:
            - "controls": (batch_size, horizon, control_dim)
            - "exogenous": (batch_size, horizon, exo_dim)
        horizon : int
            Number of steps to roll out.
        feedback : {"model", "teacher", "mixed"}
            - "model": Use model's own predictions (pure autoregressive)
            - "teacher": Use ground truth targets (teacher forcing)
            - "mixed": Stochastically mix model and teacher based on ratio
        teacher_forcing_ratio : float, default 0.0
            Probability of using teacher forcing at each step when feedback="mixed".
            Ignored for "model" and "teacher" modes.
        targets : torch.Tensor, optional
            Ground truth targets of shape (batch_size, horizon, output_dim).
            Required when feedback="teacher" or feedback="mixed".
        
        Returns
        -------
        dict
            Dictionary containing:
            - "predictions": (batch_size, horizon, output_dim)
            - "states": list of hidden states at each step (optional)
        
        Notes
        -----
        This default implementation provides a general rollout loop.
        Subclasses can override for efficiency or custom behavior.
        """
        # Validate inputs
        if feedback in ["teacher", "mixed"] and targets is None:
            raise ValueError(f"targets required when feedback='{feedback}'")
        
        # Initialize state from warmup
        if "inputs" in warmup_seq:
            warmup_inputs = warmup_seq["inputs"]
        else:
            # Concatenate controls, exogenous, and outputs
            warmup_inputs = torch.cat([
                warmup_seq["controls"],
                warmup_seq["exogenous"],
                warmup_seq["outputs"]
            ], dim=-1)
        
        state = self.init_state(warmup_inputs)
        
        # Extract rollout inputs
        controls = rollout_inputs["controls"]  # (B, H, C)
        exogenous = rollout_inputs["exogenous"]  # (B, H, E)
        batch_size = controls.shape[0]
        device = controls.device
        
        # Get last output from warmup as initial prev_output
        if "outputs" in warmup_seq:
            prev_output = warmup_seq["outputs"][:, -1, :]  # (B, O)
            output_dim = prev_output.shape[-1]
        else:
            # Extract from concatenated inputs (assume outputs are last)
            # Try model attribute first, then targets
            output_dim = getattr(self, 'output_dim', None)
            if output_dim is None:
                output_dim = targets.shape[-1] if targets is not None else None
            if output_dim is None:
                raise ValueError(
                    "Cannot infer output_dim. Provide warmup 'outputs', "
                    "set model.output_dim, or pass targets."
                )
            prev_output = warmup_inputs[:, -1, -output_dim:]
        
        # HOT PATH: Preallocate predictions tensor (Rule 5: no allocations in loop)
        predictions = torch.empty(
            batch_size, horizon, output_dim,
            dtype=torch.float32, device=device
        )
        states = []
        
        # Rollout loop - recurrence is unavoidable, but minimize overhead
        for t in range(horizon):
            control_t = controls[:, t, :]  # (B, C)
            exo_t = exogenous[:, t, :]  # (B, E)
            
            # Predict next step
            state, pred_t = self.step(state, control_t, exo_t, prev_output)
            
            # HOT PATH: Direct assignment to preallocated tensor
            predictions[:, t, :] = pred_t
            states.append(state)
            
            # Determine what to use as prev_output for next step
            if feedback == "model":
                prev_output = pred_t
            elif feedback == "teacher":
                prev_output = targets[:, t, :]
            elif feedback == "mixed":
                # Scheduled sampling: randomly choose between model and teacher
                use_teacher = torch.rand(batch_size, 1, device=device) < teacher_forcing_ratio
                prev_output = torch.where(use_teacher, targets[:, t, :], pred_t)
        
        return {
            "predictions": predictions,
            "states": states,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for compatibility with existing training code.
        
        This method provides backward compatibility with the old SimpleLSTM interface.
        It assumes x contains the full sequence (warmup + horizon) and returns
        predictions for the last pred_len steps.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).
        
        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, pred_len, output_dim).
        
        Notes
        -----
        Subclasses should implement this for backward compatibility, or override
        to provide a more efficient implementation.
        """
        raise NotImplementedError(
            "Subclasses must implement forward() for backward compatibility, "
            "or use rollout() directly for world model training."
        )

