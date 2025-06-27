from __future__ import annotations

import math

import torch
import torch.nn as nn


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
                 pred_len: int = 1):
        super().__init__()
        self.pred_len = pred_len
        self.out_dim = input_dim  # predict same dimensionality

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