"""Temporal Fusion Transformer (TFT) for interpretable time series forecasting.

Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
https://arxiv.org/abs/1912.09363

TFT combines:
- Variable selection networks for feature importance
- Gated Residual Networks (GRN) for non-linear processing
- LSTM encoder-decoder for temporal patterns
- Multi-head attention for long-range dependencies
- Interpretable attention weights
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import WorldModelBase


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) for controlled information flow."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        return out[..., :self.output_dim] * torch.sigmoid(out[..., self.output_dim:])


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) for non-linear feature processing.
    
    GRN allows the model to learn complex non-linear relationships while
    maintaining gradient flow through skip connections.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    output_dim : int, optional
        Output dimension. Defaults to input_dim.
    context_dim : int, optional
        Context vector dimension for conditional processing.
    dropout : float, default 0.0
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim or input_dim

        # Dense layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Optional context projection
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_proj = None

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # GLU for output gating
        self.glu = GatedLinearUnit(hidden_dim, self.output_dim)

        # Skip connection (with projection if dimensions differ)
        if input_dim != self.output_dim:
            self.skip_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.skip_proj = None

        # Layer norm
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        context : torch.Tensor, optional
            Context vector for conditional processing.
        
        Returns
        -------
        torch.Tensor
            Processed output.
        """
        # First dense layer with ELU activation
        hidden = F.elu(self.fc1(x))

        # Add context if provided
        if context is not None and self.context_proj is not None:
            hidden = hidden + self.context_proj(context)

        # Second dense layer
        hidden = F.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)

        # GLU gating
        gated = self.glu(hidden)

        # Skip connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x

        # Residual connection + layer norm
        return self.layer_norm(skip + gated)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature importance learning.
    
    Learns which input features are most relevant for the prediction task.
    Outputs both transformed features and importance weights.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Hidden dimension for GRN.
    num_features : int
        Number of individual features to select from.
    dropout : float, default 0.0
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_features: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Feature-wise GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim // num_features,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_features)
        ])

        # Softmax weights GRN
        self.weights_grn = GatedResidualNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            context_dim=context_dim,
            dropout=dropout,
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (..., input_dim).
        context : torch.Tensor, optional
            Context vector.
        
        Returns
        -------
        transformed : torch.Tensor
            Selected and transformed features.
        weights : torch.Tensor
            Feature importance weights.
        """
        # Split input into individual features
        feature_size = x.shape[-1] // self.num_features
        features = torch.split(x, feature_size, dim=-1)

        # Process each feature through its GRN
        processed = []
        for i, feat in enumerate(features):
            processed.append(self.feature_grns[i](feat))
        processed = torch.stack(processed, dim=-2)  # (..., num_features, hidden_dim)

        # Compute selection weights
        flat_x = x.reshape(*x.shape[:-1], -1)
        weights = self.weights_grn(flat_x, context)
        weights = self.softmax(weights)  # (..., num_features)

        # Apply weights
        weights_expanded = weights.unsqueeze(-1)  # (..., num_features, 1)
        selected = (processed * weights_expanded).sum(dim=-2)  # (..., hidden_dim)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretable attention weights.
    
    Modified attention that provides interpretable weights for each head.
    
    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch, seq_len, d_model).
        key : torch.Tensor
            Key tensor.
        value : torch.Tensor
            Value tensor.
        mask : torch.Tensor, optional
            Attention mask.
        
        Returns
        -------
        output : torch.Tensor
            Attention output.
        attn_weights : torch.Tensor
            Attention weights (interpretable).
        """
        batch_size, seq_len, _ = query.shape

        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, v)

        # Reshape and project
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)

        # Average attention weights across heads for interpretability
        avg_attn_weights = attn_weights.mean(dim=1)

        return output, avg_attn_weights


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for interpretable forecasting.
    
    Combines variable selection, gated residual networks, LSTM encoding,
    and multi-head attention for accurate and interpretable predictions.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output features.
    seq_len : int
        Input sequence length.
    pred_len : int
        Prediction horizon.
    hidden_dim : int, default 64
        Hidden dimension for all components.
    n_heads : int, default 4
        Number of attention heads.
    num_lstm_layers : int, default 2
        Number of LSTM layers.
    dropout : float, default 0.1
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        seq_len: int = 24,
        pred_len: int = 1,
        hidden_dim: int = 64,
        n_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        # Variable selection for historical inputs
        self.historical_vsn = VariableSelectionNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_features=1,  # Simplified: treat as single entity
            dropout=dropout,
        )

        # LSTM encoder
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        # Gated skip connection
        self.post_lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)

        # Static enrichment GRN
        self.static_enrichment = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Temporal self-attention
        self.self_attention = InterpretableMultiHeadAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.post_attn_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)

        # Position-wise feed-forward
        self.positionwise_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.output_dim * pred_len)

        # Store attention weights for interpretability
        self._attention_weights = None
        self._feature_weights = None

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
        batch_size = x.shape[0]

        # Embed inputs
        embedded = self.input_embedding(x)  # (B, T, hidden)

        # Variable selection
        selected, feat_weights = self.historical_vsn(embedded)
        self._feature_weights = feat_weights

        # LSTM encoding
        lstm_out, _ = self.encoder_lstm(selected.unsqueeze(-2) if selected.dim() == 2 else selected)

        # Gated skip connection
        gated = self.post_lstm_gate(lstm_out)
        enriched = self.post_lstm_norm(embedded + gated)

        # Static enrichment
        enriched = self.static_enrichment(enriched)

        # Self-attention (causal mask for decoder)
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=x.device), diagonal=1).bool()
        mask = ~mask  # Invert: True where attention is allowed

        attn_out, attn_weights = self.self_attention(
            query=enriched,
            key=enriched,
            value=enriched,
            mask=mask.unsqueeze(0).unsqueeze(0),
        )
        self._attention_weights = attn_weights

        # Post-attention gating
        gated_attn = self.post_attn_gate(attn_out)
        temporal = self.post_attn_norm(enriched + gated_attn)

        # Position-wise processing
        output = self.positionwise_grn(temporal)

        # Take last position and project to predictions
        last = output[:, -1, :]  # (B, hidden)
        pred = self.output_proj(last)  # (B, pred_len * output_dim)
        pred = pred.view(batch_size, self.pred_len, self.output_dim)

        return pred

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass for interpretability."""
        return self._attention_weights

    def get_feature_weights(self) -> Optional[torch.Tensor]:
        """Get feature selection weights from last forward pass."""
        return self._feature_weights


class TFTWorldModel(WorldModelBase):
    """TFT adapted for world model interface with autoregressive rollout.
    
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
    hidden_dim : int, default 64
        Hidden dimension.
    n_heads : int, default 4
        Number of attention heads.
    num_lstm_layers : int, default 2
        Number of LSTM layers.
    dropout : float, default 0.1
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        seq_len: int = 24,
        pred_len: int = 1,
        hidden_dim: int = 64,
        n_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.model = TemporalFusionTransformer(
            input_dim=input_dim,
            output_dim=self.output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
        )

    def init_state(self, warmup_seq: torch.Tensor) -> torch.Tensor:
        """Initialize state from warmup sequence."""
        return warmup_seq[:, -self.seq_len:, :]

    def step(
        self,
        state: torch.Tensor,
        control_t: torch.Tensor,
        exo_t: torch.Tensor,
        prev_output_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform single prediction step."""
        # Predict from current state
        pred = self.model(state)[:, 0, :]

        # Sliding-window update uses current prediction as next output features.
        # Teacher/mixed corrections are applied in rollout().
        next_input = torch.cat([control_t, exo_t, pred], dim=-1)

        # Update state
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
        """Standard forward pass."""
        return self.model(x)

    def get_interpretability(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get interpretability information from last forward pass."""
        return {
            "attention_weights": self.model.get_attention_weights(),
            "feature_weights": self.model.get_feature_weights(),
        }
