"""DLinear: Decomposition-Linear model for time series forecasting.

Based on: "Are Transformers Effective for Time Series Forecasting?"
https://arxiv.org/abs/2205.13504

DLinear decomposes time series into trend and seasonal components,
then applies separate linear layers to each.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Literal, Tuple

import torch
import torch.nn as nn

from .base import WorldModelBase


class MovingAverage(nn.Module):
    """Moving average block to extract trend from time series.
    
    HOT PATH: Used in every forward pass for decomposition.
    Uses nn.AvgPool1d for efficient computation.
    """
    
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply moving average.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, features).
        
        Returns
        -------
        torch.Tensor
            Smoothed output of shape (batch, seq_len, features).
        """
        # Pad on both ends to maintain sequence length
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, :1, :].expand(-1, pad_len, -1)
        end = x[:, -1:, :].expand(-1, pad_len, -1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # Apply pooling (requires channels-first format)
        x_permuted = x_padded.permute(0, 2, 1)  # (B, F, T)
        out = self.avg(x_permuted)
        return out.permute(0, 2, 1)  # (B, T, F)


class SeriesDecomposition(nn.Module):
    """Decompose time series into trend and seasonal components."""
    
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose into seasonal and trend.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, features).
        
        Returns
        -------
        seasonal : torch.Tensor
            Seasonal component (residual).
        trend : torch.Tensor
            Trend component (moving average).
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """Decomposition-Linear model for time series forecasting.
    
    Decomposes input into trend and seasonal components, applies
    separate linear projections, then combines them.
    
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
    kernel_size : int, default 25
        Kernel size for moving average decomposition.
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
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        
        # Decomposition layer
        self.decomposition = SeriesDecomposition(kernel_size)
        
        # Linear projections
        if individual:
            # Separate linear layer per channel
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(input_dim)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(input_dim)
            ])
        else:
            # Shared linear layer across channels
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)
        
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
        # Decompose into seasonal and trend
        seasonal, trend = self.decomposition(x)
        
        # Transpose for linear: (B, T, F) -> (B, F, T)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        
        if self.individual:
            # Apply per-channel linear layers
            batch_size = x.shape[0]
            seasonal_out = torch.zeros(
                batch_size, self.input_dim, self.pred_len,
                dtype=x.dtype, device=x.device
            )
            trend_out = torch.zeros_like(seasonal_out)
            
            for i in range(self.input_dim):
                seasonal_out[:, i, :] = self.linear_seasonal[i](seasonal[:, i, :])
                trend_out[:, i, :] = self.linear_trend[i](trend[:, i, :])
        else:
            # Shared linear across channels
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)
        
        # Combine and transpose back: (B, F, T) -> (B, T, F)
        out = (seasonal_out + trend_out).permute(0, 2, 1)
        
        # Project to output dimension if needed
        if self.output_proj is not None:
            out = self.output_proj(out)
        
        return out


class DLinearWorldModel(WorldModelBase):
    """DLinear adapted for world model interface with autoregressive rollout.
    
    This wraps DLinear to support the WorldModelBase interface for
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
    kernel_size : int, default 25
        Kernel size for decomposition.
    individual : bool, default False
        Use per-channel linear layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        seq_len: int = 24,
        pred_len: int = 1,
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.model = DLinear(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            output_dim=self.output_dim,
            kernel_size=kernel_size,
            individual=individual,
        )
    
    def init_state(self, warmup_seq: torch.Tensor) -> torch.Tensor:
        """Initialize state from warmup sequence.
        
        For DLinear, the "state" is the recent history window.
        
        Parameters
        ----------
        warmup_seq : torch.Tensor
            Warmup sequence of shape (batch, warmup_len, input_dim).
        
        Returns
        -------
        torch.Tensor
            State tensor (the last seq_len steps).
        """
        # Keep the last seq_len steps as state
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
            Current state (history window), shape (batch, seq_len, input_dim).
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
        
        # Build next input from control + exo + prediction
        if prev_output_t is not None:
            next_input = torch.cat([control_t, exo_t, prev_output_t], dim=-1)
        else:
            next_input = torch.cat([control_t, exo_t, pred], dim=-1)
        
        # Update state by sliding window
        new_state = torch.cat([state[:, 1:, :], next_input.unsqueeze(1)], dim=1)
        
        return new_state, pred
    
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

