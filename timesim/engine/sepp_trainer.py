from __future__ import annotations

from typing import List, Dict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from ..data.sepp_dataset import SEPPWindowDataset
from .rollout import rollout_autoregressive
from ..utils.dilate import dilate_loss


class SEPPTrainer:
    """Trainer that implements See-Every-Possible-Past (vectorised)."""

    def __init__(self,
                 model: torch.nn.Module,
                 df,
                 groups: Dict[str, List[str]],
                 input_groups: List[str],
                 output_groups: List[str],
                 seq_len: int,
                 pred_len: int,
                 h_max: int,
                 stride: int = 1,
                 horizons: List[int] | None = None,
                 batch_size: int = 64,
                 device: str | torch.device = "cpu",
                 val_loader: DataLoader | None = None,
                 patience: int = 5,
                 ckpt_path: str | Path | None = None,
                 on_improve=None):

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.output_cols = sum((groups[g] for g in output_groups), [])
        self.dataset = SEPPWindowDataset(df, groups, input_groups, output_groups, seq_len, h_max, stride)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.h_max = h_max
        self.horizons = horizons or self._geom_horizons(pred_len, h_max)
        self.loss_fn = torch.nn.MSELoss()
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.val_loader = val_loader
        self.patience = patience
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.best_loss: float | None = None
        self.no_improve = 0
        self.on_improve = on_improve

    @staticmethod
    def _geom_horizons(pred_len: int, h_max: int):
        hs = []
        h = pred_len
        while h <= h_max:
            hs.append(h)
            h *= 2
        return hs

    def _loss_on_window(self, window: torch.Tensor):
        # window: (B, seq_len+h_max, F)
        in_idx = torch.tensor(self.dataset.in_idx, dtype=torch.long)
        out_idx = torch.tensor(self.dataset.out_idx, dtype=torch.long)

        window = window.to(self.device)
        x0_full = window[:, :self.dataset.seq_len, :]
        x0 = x0_full[:, :, in_idx]
        targets_full = window[:, self.dataset.seq_len:, :][:, :, out_idx]

        preds = self._rollout_with_insert(x0, window[:, :, :], in_idx, out_idx)
        loss = 0.0
        for h in self.horizons:
            t = targets_full[:, h-1:h, :]
            p = preds[:, h-1:h, :]
            loss += self.loss_fn(p, t)
        loss = loss / len(self.horizons)
        return loss

    def _rollout_with_insert(self, x0, window_full, in_idx, out_idx):
        """Autoregressive rollout replacing overlapping input cols with preds."""
        device = self.device
        B = x0.shape[0]
        preds = []
        x = x0.clone().to(device)
        for _ in range(self.h_max):
            y = self.model(x)  # (B, pred_len, F_out)
            step = y[:, -1, :]  # (B, F_out)
            preds.append(step.cpu())
            # prepare next input row (full dim)
            last_row = window_full[:, self.dataset.seq_len-1, :].clone()  # (B, F_full)
            # insert predictions into columns that overlap inputs
            intersect = [j for j, col_idx in enumerate(out_idx) if col_idx in in_idx]
            if intersect:
                for j in intersect:
                    full_col = out_idx[j]
                    last_row[:, full_col] = step[:, j]
            new_row_in = last_row[:, in_idx]
            x = torch.cat([x, new_row_in.unsqueeze(1)], dim=1)[:, 1:, :]
        return torch.stack(preds, dim=1)  # (B, h_max, F_out)

    def _validate(self):
        if self.val_loader is None:
            return None
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self._loss_on_window(batch.to(torch.float32))
                losses.append(loss.item())
        if len(losses) == 0:
            return None
        return sum(losses)/len(losses)

    def fit(self, epochs: int = 3):
        for epoch in range(1, epochs+1):
            self.current_epoch = epoch
            epoch_losses = []
            for batch in self.loader:
                self.opt.zero_grad()
                loss = self._loss_on_window(batch)
                loss.backward()
                self.opt.step()
                epoch_losses.append(loss.item())
            train_loss = sum(epoch_losses)/len(epoch_losses)
            val_loss = self._validate()
            val_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"

            print(f"[SEPP Epoch {epoch}] train={train_loss:.4f} | val={val_str}")

            if self.best_loss is None or val_loss < self.best_loss - 1e-6:
                self.best_loss = val_loss
                self.no_improve = 0
                if self.ckpt_path is not None:
                    torch.save(self.model.state_dict(), self.ckpt_path)
                if self.on_improve is not None:
                    self.on_improve(self.model, epoch)
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print("Early stopping triggered (patience).")
                    break 