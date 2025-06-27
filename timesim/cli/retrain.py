import argparse
from pathlib import Path

from timesim.data.loader import generate_sine_dataset, build_dataloaders
from timesim.models import get_model
from timesim.engine.retrainer import Retrainer
from timesim.utils.config import load_config
from timesim.utils.logger import create_run_dir, init_logging
from timesim.utils.plotting import save_loss_plot


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a checkpoint on new data")
    p.add_argument("--config", type=str, default="configs/base.yml",
                   help="YAML config with defaults")

    # Overrides
    p.add_argument("--ckpt", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--pred-len", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--model", type=str)
    return p.parse_args()


def main():
    cli_args = parse_args()

    cfg = load_config(cli_args.config, cli_args)

    # ----------------------------------------------------
    run_dir = create_run_dir(name_parts={"retrain": Path(cfg["ckpt"]).stem})
    logger, tb_writer = init_logging(run_dir)

    # Save merged config
    import yaml
    with open(Path(run_dir)/"config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # ----------------------------------------------------
    # Data preparation (synthetic example)
    series = generate_sine_dataset(length=1000)
    seq_len = cfg.get("seq_len", 24)
    pred_len = cfg.get("pred_len", 12)
    batch_size = cfg.get("batch_size", 32)
    device = cfg.get("device", "cpu")

    train_loader, val_loader = build_dataloaders(series,
                                                seq_len=seq_len,
                                                pred_len=pred_len,
                                                batch_size=batch_size,
                                                device=device)

    # ----------------------------------------------------
    # Determine model name string (handle nested dict in YAML)
    if isinstance(cfg.get("model"), str):
        _model_name = cfg["model"]
    else:
        _model_name = cfg.get("model", {}).get("type", "lstm")

    ModelCls = get_model(_model_name)
    retrainer = Retrainer(model_cls=lambda: ModelCls(input_dim=series.shape[1], pred_len=pred_len),
                          checkpoint=cfg["ckpt"],
                          device=device)

    train_losses, val_losses = retrainer.fine_tune(train_loader, val_loader, epochs=cfg.get("epochs", 3))

    # Metrics & plots
    save_loss_plot(train_losses, val_losses, Path(run_dir)/"figs"/"retrain_loss.png", title="Retrain loss")

    # Save new checkpoint
    new_ckpt = Path(run_dir)/"checkpoint_retrained.pth"
    import torch
    retrainer.model.cpu()
    torch.save(retrainer.model.state_dict(), new_ckpt)
    logger.info(f"Saved fine-tuned checkpoint to {new_ckpt}")


if __name__ == "__main__":
    main() 