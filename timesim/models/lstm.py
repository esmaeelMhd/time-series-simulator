import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """A minimal LSTM encoder that predicts the next *pred_len* steps."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 pred_len: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.pred_len = pred_len
        self.out_dim = input_dim  # predict same number of features

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        out, _ = self.lstm(x)
        # use last hidden state for prediction base
        last = out[:, -1, :]  # (B, hidden)
        pred = self.fc(last)  # (B, F)
        # repeat along pred_len dimension
        pred_seq = pred.unsqueeze(1).repeat(1, self.pred_len, 1)
        return pred_seq 