"""NLinear: Normalization-Linear model for time series forecasting.

Based on: "Are Transformers Effective for Time Series Forecasting?"
https://arxiv.org/abs/2205.13504

NLinear subtracts the last value before linear projection, then adds it back.
This simple normalization makes the model robust to distribution shift.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

from .base import WorldModelBase


class NLinear(nn.Module):
    """Normalization-Linear model for time series forecasting.
    
    Subtracts the last value of the input sequence before applying
    a linear projection, then adds it back. This handles non-stationary
    data effectively.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    seq_len : int
        Input sequence length.
    pred_len : int
        Prediction horizon length.
    output_dim : int, optional
        Number of output features. Defaults to input_dim.
    individual : bool, default False
        If True, use separate linear layers per channel.
        If False, share weights across channels.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        output_dim: Optional[int] = None,
        individual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual

        # Linear projections
        if individual:
            # Separate linear layer per channel
            self.linear = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(input_dim)
            ])
        else:
            # Shared linear layer
            self.linear = nn.Linear(seq_len, pred_len)

        # Output projection if dimensions differ
        if self.output_dim != self.input_dim:
            self.output_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.output_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, input_dim).
        
        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, pred_len, output_dim).
        """
        # Get last value and normalize (detach to not backprop through it)
        seq_last = x[:, -1:, :].detach()
        x_norm = x - seq_last

        if self.individual:
            # Apply per-channel linear layers
            batch_size = x.shape[0]
            out = torch.zeros(
                batch_size, self.pred_len, self.input_dim,
                dtype=x.dtype, device=x.device
            )
            for i in range(self.input_dim):
                # x_norm: (B, T, F) -> need (B, T) for linear
                out[:, :, i] = self.linear[i](x_norm[:, :, i])
        else:
            # Shared linear: (B, T, F) -> (B, F, T) -> linear -> (B, F, pred_len) -> (B, pred_len, F)
            out = self.linear(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        # Add back the last value (for all prediction steps)
        out = out + seq_last

        # Project to output dimension if needed
        if self.output_proj is not None:
            out = self.output_proj(out)

        return out


class NLinearWorldModel(WorldModelBase):
    """NLinear adapted for world model interface with autoregressive rollout.
    
    This wraps NLinear to support the WorldModelBase interface for
    multi-step training and simulation.
    
    Parameters
    ----------
    input_dim : int
        Total input dimension (controls + exogenous + outputs).
    output_dim : int
        Output/target dimension.
    seq_len : int
        Input sequence length (warmup length).
    pred_len : int, default 1
        Prediction horizon per step.
    individual : bool, default False
        Use per-channel linear layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        seq_len: int = 24,
        pred_len: int = 1,
        individual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.model = NLinear(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            output_dim=self.output_dim,
            individual=individual,
        )

    def init_state(self, warmup_seq: torch.Tensor) -> torch.Tensor:
        """Initialize state from warmup sequence.
        
        For NLinear, the "state" is the recent history window.
        
        Parameters
        ----------
        warmup_seq : torch.Tensor
            Warmup sequence of shape (batch, warmup_len, input_dim).
        
        Returns
        -------
        torch.Tensor
            State tensor (the last seq_len steps).
        """
        return warmup_seq[:, -self.seq_len:, :]

    def step(
        self,
        state: torch.Tensor,
        control_t: torch.Tensor,
        exo_t: torch.Tensor,
        prev_output_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform single prediction step.
        
        Parameters
        ----------
        state : torch.Tensor
            Current state (history window).
        control_t : torch.Tensor
            Control inputs at time t.
        exo_t : torch.Tensor
            Exogenous inputs at time t.
        prev_output_t : torch.Tensor, optional
            Previous output.
        
        Returns
        -------
        new_state : torch.Tensor
            Updated state.
        prediction : torch.Tensor
            Predicted output.
        """
        # Predict from current state
        pred = self.model(state)[:, 0, :]  # Take first prediction step

        # Sliding-window update uses current prediction as next output features.
        # Teacher/mixed corrections are applied in rollout().
        next_input = torch.cat([control_t, exo_t, pred], dim=-1)

        # Update state by sliding window
        new_state = torch.cat([state[:, 1:, :], next_input.unsqueeze(1)], dim=1)

        return new_state, pred

    def rollout(
        self,
        warmup_seq: Dict[str, torch.Tensor],
        rollout_inputs: Dict[str, torch.Tensor],
        horizon: int,
        feedback: Literal["model", "teacher", "mixed"] = "model",
        teacher_forcing_ratio: float = 0.0,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Windowed rollout with aligned state updates."""
        return self._rollout_windowed_feedback(
            warmup_seq=warmup_seq,
            rollout_inputs=rollout_inputs,
            horizon=horizon,
            feedback=feedback,
            teacher_forcing_ratio=teacher_forcing_ratio,
            targets=targets,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for compatibility.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, input_dim).
        
        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, pred_len, output_dim).
        """
        return self.model(x)
