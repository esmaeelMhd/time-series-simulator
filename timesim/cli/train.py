import argparse
from pathlib import Path

import torch

from timesim.data.loader import generate_sine_dataset, build_dataloaders
from timesim.models import get_model
from timesim.engine.trainer import Trainer
from timesim.utils.config import load_config
from timesim.utils.logger import create_run_dir, init_logging
from timesim.utils.plotting import save_loss_plot


def parse_args():
    p = argparse.ArgumentParser(description="Train a time-series model")
    p.add_argument("--config", type=str, default="configs/base.yml",
                   help="Path to YAML config with defaults")

    # Override-able params
    p.add_argument("--epochs", type=int)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--pred-len", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--model", type=str)
    p.add_argument("--device", type=str)
    p.add_argument("--ckpt", type=str)

    return p.parse_args()


def main():
    cli_args = parse_args()

    # -------------------------------------------------------
    # 1) Load config and merge CLI overrides
    # -------------------------------------------------------
    cfg = load_config(cli_args.config, cli_args)

    # -------------------------------------------------------
    # 2) Create run directory & logging utilities
    # -------------------------------------------------------
    if isinstance(cfg.get("model"), str):
        _model_name = cfg["model"]
    else:
        _model_name = cfg.get("model", {}).get("type", "model")
    run_dir = create_run_dir(name_parts={"model": _model_name})
    logger, tb_writer = init_logging(run_dir)

    # Save the final config
    import yaml
    with open(Path(run_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # -------------------------------------------------------
    # 3) Build dataset & loaders (placeholder: sine waves for now)
    # -------------------------------------------------------
    series = generate_sine_dataset(length=2000)
    seq_len = cfg.get("seq_len") or cfg.get("dataset", {}).get("seq_len", 24)
    pred_len = cfg.get("pred_len") or cfg.get("dataset", {}).get("pred_len", 12)
    batch_size = cfg.get("batch_size") or cfg.get("dataset", {}).get("batch_size", 32)
    device = cfg.get("device") or cfg.get("misc", {}).get("device", "cpu")

    train_loader, val_loader = build_dataloaders(series,
                                                seq_len=seq_len,
                                                pred_len=pred_len,
                                                batch_size=batch_size,
                                                device=device)

    # -------------------------------------------------------
    # 4) Create model & trainer
    # -------------------------------------------------------
    ModelCls = get_model(_model_name)
    model = ModelCls(input_dim=series.shape[1], pred_len=pred_len)

    trainer = Trainer(model,
                      device=device,
                      run_dir=run_dir,
                      writer=tb_writer)

    train_losses, val_losses = trainer.fit(train_loader, val_loader, epochs=cfg.get("epochs", cfg.get("training", {}).get("epochs", 5)))

    # Plot loss
    save_loss_plot(train_losses, val_losses, Path(run_dir)/"figs"/"loss.png")

    # -------------------------------------------------------
    # 5) Save checkpoint to run dir (or custom path)
    # -------------------------------------------------------
    ckpt_path = cli_args.ckpt or (Path(run_dir) / "checkpoint.pth").as_posix()
    trainer.save(ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main() 