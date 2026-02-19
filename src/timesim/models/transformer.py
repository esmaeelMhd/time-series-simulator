"""Transformer-based models for time series forecasting.

Includes:
- SimpleTransformer: Basic transformer encoder for sequence modeling
- TransformerWorldModel: WorldModelBase wrapper for autoregressive rollout
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Literal, Tuple

import torch
import torch.nn as nn

from .base import WorldModelBase


class SimpleTransformer(nn.Module):
    """A minimal Transformer encoder-decoder that predicts the next *pred_len* steps.

    The design goal is **simplicity**, not SOTA accuracy.
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int = 64,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 pred_len: int = 1,
                 out_dim: int | None = None):
        super().__init__()
        self.pred_len = pred_len
        self.out_dim = out_dim or input_dim  # default: same as input

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        # Project to model dimension and add positional encoding
        x_emb = self.input_proj(x)
        x_emb = self.pos_encoding(x_emb)
        enc_out = self.encoder(x_emb)  # (B, T, d_model)
        last = enc_out[:, -1, :]  # use last timestep representation
        pred_step = self.fc_out(last)  # (B, F)
        pred_seq = pred_step.unsqueeze(1).repeat(1, self.pred_len, 1)  # repeat
        return pred_seq


class PositionalEncoding(nn.Module):
    """Classic sine/cosine positional embeddings taken from PyTorch tutorial."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerWorldModel(WorldModelBase):
    """Transformer adapted for world model interface with autoregressive rollout.

    Wraps SimpleTransformer for the WorldModelBase interface using a sliding
    window as state, following the same pattern as DLinear/NLinear/TFT
    world models.

    Parameters
    ----------
    input_dim : int
        Total input dimension (controls + exogenous + outputs).
    output_dim : int, optional
        Output/target dimension. Defaults to input_dim.
    seq_len : int, default 24
        Input sequence length (warmup / sliding window length).
    pred_len : int, default 1
        Prediction horizon per step.
    d_model : int, default 64
        Transformer model dimension.
    nhead : int, default 4
        Number of attention heads.
    num_layers : int, default 2
        Number of transformer encoder layers.
    dim_feedforward : int, default 128
        Feed-forward hidden dimension.
    dropout : float, default 0.1
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        seq_len: int = 24,
        pred_len: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.model = SimpleTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pred_len=pred_len,
            out_dim=self.output_dim,
        )

    def init_state(self, warmup_seq: torch.Tensor) -> torch.Tensor:
        """Initialize state from warmup sequence (sliding window).

        Parameters
        ----------
        warmup_seq : torch.Tensor
            Warmup sequence of shape (batch, warmup_len, input_dim).

        Returns
        -------
        torch.Tensor
            State tensor (last seq_len steps).
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
            Current state (history window), shape (batch, seq_len, input_dim).
        control_t : torch.Tensor
            Control inputs at time t.
        exo_t : torch.Tensor
            Exogenous inputs at time t.
        prev_output_t : torch.Tensor, optional
            Previous output at time t.

        Returns
        -------
        new_state : torch.Tensor
            Updated state (slid window).
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
