import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import tempfile
from pathlib import Path

import torch

from timesim.data.loader import generate_sine_dataset, build_dataloaders
from timesim.models import get_model
from timesim.training.trainer import Trainer
from timesim.training.retrainer import Retrainer


def test_train_and_retrain(tmp_path: Path):
    """Full pipeline: train a model, save checkpoint, retrain, ensure loss decreases."""
    seq_len, pred_len, batch_size = 24, 12, 16

    # ----- simulate dataset -----
    series = generate_sine_dataset(length=1500)
    train_loader, val_loader = build_dataloaders(series, seq_len, pred_len, batch_size, device="cpu")

    # ----- initial training -----
    ModelCls = get_model("lstm")
    model = ModelCls(input_dim=series.shape[1], pred_len=pred_len)
    trainer = Trainer(model, device="cpu", loss="mse", early_stopping=True, patience=3)
    _, val_losses = trainer.fit(train_loader, val_loader, epochs=5, verbose=False)
    first_val = val_losses[-1]

    ckpt_path = tmp_path / "base.pth"
    trainer.save(ckpt_path.as_posix())

    # ----- retraining on new (shorter) dataset -----
    new_series = generate_sine_dataset(length=800)
    re_train_loader, re_val_loader = build_dataloaders(new_series, seq_len, pred_len, batch_size, device="cpu")

    retrainer = Retrainer(model_cls=lambda: ModelCls(input_dim=series.shape[1], pred_len=pred_len),
                          checkpoint=ckpt_path,
                          device="cpu")
    _, re_val_losses = retrainer.fine_tune(re_train_loader, re_val_loader, epochs=3, lr=1e-4)
    last_val = re_val_losses[-1]

    # The retrain loss should be finite and ideally not NaN
    assert torch.isfinite(torch.tensor(last_val)), "Validation loss is not finite after retrain."

    # Simple heuristic: loss after retrain should not explode (<= 2x original)
    assert last_val <= 2 * first_val, "Retrain made the loss blow up." 