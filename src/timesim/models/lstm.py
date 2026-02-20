"""LSTM-based world model for time-series simulation.

HOT PATH: Model forward and rollout are called every training step.
Optimizations:
- Preallocated output tensors in rollout (no list.append)
- Efficient tensor slicing in step()
- Minimized Python overhead in inner loops
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Literal, Tuple

import torch
import torch.nn as nn

from .base import WorldModelBase


class LSTMWorldModel(WorldModelBase):
    """LSTM-based world model for control and simulation.
    
    HOT PATH: forward() and rollout() are called every training step.
    
    This model learns to predict the next state/observation given:
    - Current hidden state (LSTM memory)
    - Control inputs (actions)
    - Exogenous inputs (disturbances, context)
    - Previous outputs (for autoregressive feedback)
    
    The model can be used for:
    1. Multi-step prediction with teacher forcing (training)
    2. Autoregressive rollouts (simulation/evaluation)
    3. Model-based RL or MPC
    
    Parameters
    ----------
    input_dim : int
        Total input dimension (controls + exogenous + outputs).
    output_dim : int
        Output/target dimension.
    hidden_dim : int, default 64
        LSTM hidden state dimension.
    num_layers : int, default 2
        Number of LSTM layers.
    dropout : float, default 0.0
        Dropout probability between LSTM layers.
    pred_len : int, default 1
        Prediction horizon for forward() compatibility.
        For world model training, use rollout() instead.
    
    Notes
    -----
    For world model training, the input should be structured as:
        [controls, exogenous, previous_outputs]
    concatenated along the feature dimension.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        pred_len: int = 1,
        control_dim: Optional[int] = None,
        exo_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        # For step() method: if control_dim and exo_dim are provided,
        # the LSTM expects control + exo + output concatenated
        # Otherwise, input_dim should already include everything
        self.control_dim = control_dim
        self.exo_dim = exo_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.fc = nn.Linear(hidden_dim, self.output_dim)
    
    def init_state(self, warmup_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state from warmup sequence.
        
        Parameters
        ----------
        warmup_seq : torch.Tensor
            Warmup sequence of shape (batch_size, warmup_len, input_dim).
        
        Returns
        -------
        h : torch.Tensor
            Hidden state of shape (num_layers, batch_size, hidden_dim).
        c : torch.Tensor
            Cell state of shape (num_layers, batch_size, hidden_dim).
        """
        # Run LSTM over warmup sequence to get final hidden state
        _, (h, c) = self.lstm(warmup_seq)
        return h, c
    
    def step(
        self,
        state: Tuple[torch.Tensor, torch.Tensor],
        control_t: torch.Tensor,
        exo_t: torch.Tensor,
        prev_output_t: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Perform single prediction step.
        
        Parameters
        ----------
        state : tuple of torch.Tensor
            Current LSTM state (h, c).
        control_t : torch.Tensor
            Control inputs at time t, shape (batch_size, control_dim).
        exo_t : torch.Tensor
            Exogenous inputs at time t, shape (batch_size, exo_dim).
        prev_output_t : torch.Tensor, optional
            Previous output at time t, shape (batch_size, output_dim).
        
        Returns
        -------
        new_state : tuple of torch.Tensor
            Updated LSTM state (h, c).
        prediction : torch.Tensor
            Predicted output at time t+1, shape (batch_size, output_dim).
        """
        h, c = state
        
        # Concatenate inputs based on what's provided
        # The input to LSTM should match input_dim from __init__
        if prev_output_t is not None:
            # Autoregressive mode: concatenate control + exo + previous output
            input_t = torch.cat([control_t, exo_t, prev_output_t], dim=-1)
        else:
            # Initial step or no feedback: just control + exo
            # Pad with zeros for output dimension to match input_dim
            batch_size = control_t.shape[0]
            device = control_t.device
            zero_output = torch.zeros(batch_size, self.output_dim, device=device)
            input_t = torch.cat([control_t, exo_t, zero_output], dim=-1)
        
        # Verify dimension matches
        if input_t.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, "
                f"got {input_t.shape[-1]}. "
                f"Make sure input_dim = control_dim + exo_dim + output_dim"
            )
        
        # Add time dimension for LSTM
        input_t = input_t.unsqueeze(1)  # (B, 1, F)
        
        # LSTM step
        out, (h_new, c_new) = self.lstm(input_t, (h, c))
        
        # Predict next output
        pred = self.fc(out.squeeze(1))  # (B, output_dim)
        
        return (h_new, c_new), pred
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for backward compatibility.
        
        This method maintains compatibility with existing training code
        that uses the SimpleLSTM interface.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).
        
        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, pred_len, output_dim).
        """
        # Run LSTM over input sequence
        out, _ = self.lstm(x)  # (B, T, hidden_dim)
        
        # Use last hidden state for prediction
        last = out[:, -1, :]  # (B, hidden_dim)
        pred = self.fc(last)  # (B, output_dim)
        
        # Repeat for pred_len steps (simple baseline)
        pred_seq = pred.unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return pred_seq
    
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
        - Direct tensor indexing for output (no intermediate copies)
        - Minimized Python overhead
        
        Parameters
        ----------
        warmup_seq : dict
            Dictionary with "inputs" key containing warmup sequence
            of shape (batch_size, warmup_len, input_dim).
        rollout_inputs : dict
            Dictionary containing:
            - "controls": (batch_size, horizon, control_dim)
            - "exogenous": (batch_size, horizon, exo_dim)
        horizon : int
            Number of steps to roll out.
        feedback : {"model", "teacher", "mixed"}
            Feedback mode for rollout.
        teacher_forcing_ratio : float
            Ratio for mixed feedback mode.
        targets : torch.Tensor, optional
            Ground truth targets for teacher forcing.
        
        Returns
        -------
        dict
            Dictionary with "predictions" and "states" keys.
        """
        # Validate inputs
        if feedback in ["teacher", "mixed"] and targets is None:
            raise ValueError(f"targets required when feedback='{feedback}'")
        
        # Initialize state
        warmup_inputs = warmup_seq["inputs"]
        state = self.init_state(warmup_inputs)
        
        # Extract rollout inputs
        controls = rollout_inputs["controls"]  # (B, H, C)
        exogenous = rollout_inputs["exogenous"]  # (B, H, E)
        batch_size = controls.shape[0]
        device = controls.device
        
        # Get initial previous output from warmup
        # Assume outputs are the last output_dim features of warmup
        prev_output = warmup_inputs[:, -1, -(self.output_dim):]  # (B, O)
        
        # HOT PATH: Preallocate predictions tensor (Rule 5: no allocations in loop)
        predictions = torch.empty(
            batch_size, horizon, self.output_dim,
            dtype=torch.float32, device=device
        )
        
        # States list is needed for potential downstream use, but we only
        # store references, not copies
        states = []
        
        # Rollout loop - unavoidable due to recurrence, but minimized overhead
        for t in range(horizon):
            # Direct indexing (no intermediate tensors)
            control_t = controls[:, t, :]  # (B, C)
            exo_t = exogenous[:, t, :]  # (B, E)
            
            # Predict next step
            state, pred_t = self.step(state, control_t, exo_t, prev_output)
            
            # HOT PATH: Direct assignment to preallocated tensor
            predictions[:, t, :] = pred_t
            states.append(state)
            
            # Determine feedback for next step
            if feedback == "model":
                prev_output = pred_t
            elif feedback == "teacher":
                prev_output = targets[:, t, :]
            elif feedback == "mixed":
                # Scheduled sampling - single random generation per step
                use_teacher = torch.rand(batch_size, 1, device=device) < teacher_forcing_ratio
                prev_output = torch.where(use_teacher, targets[:, t, :], pred_t)
        
        return {
            "predictions": predictions,
            "states": states,
        }


# Backward compatibility alias
SimpleLSTM = LSTMWorldModel
