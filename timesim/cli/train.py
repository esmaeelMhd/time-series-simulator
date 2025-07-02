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
    dataset_name = cfg.get("dataset", {}).get("name", "sine")
    run_dir = create_run_dir(dataset=dataset_name, model=_model_name)
    logger, tb_writer = init_logging(run_dir)

    # Save the final config
    import yaml
    with open(Path(run_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # -------------------------------------------------------
    # 3) Build dataset & loaders (placeholder: sine waves for now)
    # -------------------------------------------------------
    dataset_cfg = cfg.get("dataset", {})
    seq_len = cfg.get("seq_len") or dataset_cfg.get("seq_len", 24)
    pred_len = cfg.get("pred_len") or dataset_cfg.get("pred_len", 12)
    batch_size = cfg.get("batch_size") or dataset_cfg.get("batch_size", 32)
    device = cfg.get("device") or cfg.get("misc", {}).get("device", "cpu")

    input_groups = ["control"]
    output_groups = ["objective"]

    if dataset_cfg.get("name", "sine") == "sine":
        # Synthetic fallback
        series = generate_sine_dataset(length=2000)
        train_loader, val_loader = build_dataloaders(series,
                                                    seq_len=seq_len,
                                                    pred_len=pred_len,
                                                    batch_size=batch_size,
                                                    device=device)
        groups = {"control": ["sine"], "exogenous": [], "objective": ["sine"]}
        df_raw = None
    else:
        from timesim.data.loader import load_csv_dataset, build_grouped_dataloaders
        df_raw = load_csv_dataset(dataset_cfg["csv"],
                                  index_col=dataset_cfg.get("index_col", "date"),
                                  slice_cfg=dataset_cfg.get("slice"))
        groups = dataset_cfg["variables"]

        io_cfg = cfg.get("model_io", {})
        input_groups = io_cfg.get("input_groups", ["control"])
        output_groups = io_cfg.get("output_groups", ["objective"])

        train_loader, val_loader, scaler = build_grouped_dataloaders(df_raw,
                                                             groups,
                                                             input_groups,
                                                             output_groups,
                                                             seq_len=seq_len,
                                                             pred_len=pred_len,
                                                             batch_size=batch_size,
                                                             device=device)

        # Save fitted scaler for future use
        from joblib import dump
        dump(scaler, run_dir/"scaler.pkl")

    # -------------------------------------------------------
    # 4) Create model & trainer
    # -------------------------------------------------------
    ModelCls = get_model(_model_name)

    if df_raw is None:
        input_dim = series.shape[1]
    else:
        input_dim = len(sum((groups[g] for g in input_groups), []))

    out_dim = len(output_groups) if df_raw is not None else input_dim
    model_kwargs = dict(input_dim=input_dim, pred_len=pred_len)
    # Pass out_dim if model supports it
    if 'out_dim' in ModelCls.__init__.__code__.co_varnames:
        model_kwargs['out_dim'] = out_dim

    model = ModelCls(**model_kwargs)

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
    ckpt_path = cli_args.ckpt or (run_dir / "checkpoint.pth").as_posix()
    trainer.save(ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # -------------------------------------------------------
    # 6) Simulation evaluation (10x pred_len)
    # -------------------------------------------------------
    if df_raw is not None:
        from timesim.utils.simulation import simulate_autoregressive
        horizon = 10 * pred_len
        simulate_autoregressive(model,
                                df_raw,
                                groups,
                                input_groups=input_groups,
                                output_groups=output_groups,
                                seq_len=seq_len,
                                horizon=horizon,
                                device=device,
                                out_fig=Path(run_dir)/"figs"/f"simulation_{horizon}.png",
                                run_dir=run_dir)


if __name__ == "__main__":
    main() 