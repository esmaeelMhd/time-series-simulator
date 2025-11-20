"""Example: Train a world model for wastewater treatment plant simulation.

This example demonstrates how to use the refactored library to train a
time-series world model that can predict system behavior under different
control strategies.
"""

import argparse
from pathlib import Path
import yaml

import torch
import numpy as np

from timesim.data.loader import load_csv_dataset, build_grouped_dataloaders
from timesim.data.sampling import (
    RandomStartRandomHorizon,
    RandomStartFixedHorizon,
    DailyFixedHorizon,
    GeometricHorizonSampling,
)
from timesim.models import get_model
from timesim.training import WorldModelTrainer
from timesim.utils.logger import create_run_dir, init_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a world model for wastewater treatment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/wastewater/config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--device", type=str, help="Override device")
    return parser.parse_args()


def create_sampling_strategy(config):
    """Create sampling strategy from config."""
    sampling_cfg = config["training"].get("sampling", {})
    strategy_name = sampling_cfg.get("strategy", "random_fixed")
    
    if strategy_name == "random_random":
        return RandomStartRandomHorizon(
            h_min=sampling_cfg.get("h_min", 12),
            h_max=sampling_cfg.get("h_max", 48),
        )
    elif strategy_name == "random_fixed":
        return RandomStartFixedHorizon(
            horizon=sampling_cfg.get("horizon", 24),
        )
    elif strategy_name == "daily_fixed":
        return DailyFixedHorizon(
            start_hour=sampling_cfg.get("start_hour", 0),
            horizon=sampling_cfg.get("horizon", 24),
            samples_per_hour=sampling_cfg.get("samples_per_hour", 1),
        )
    elif strategy_name == "geometric":
        return GeometricHorizonSampling(
            pred_len=config["dataset"]["pred_len"],
            h_max=sampling_cfg.get("h_max", 64),
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy_name}")


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override from CLI
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["misc"]["device"] = args.device
    
    # Set seed
    seed = config["misc"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create run directory
    run_dir = create_run_dir(
        dataset=config["dataset"]["name"],
        model=config["model"]["type"],
    )
    logger, tb_writer = init_logging(run_dir)
    logger.info(f"Starting training run in {run_dir}")
    
    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    
    # Load dataset
    logger.info("Loading dataset...")
    df = load_csv_dataset(
        config["dataset"]["csv"],
        index_col=config["dataset"]["index_col"],
    )
    
    groups = config["dataset"]["variables"]
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]
    
    # Build dataloaders
    train_loader, val_loader, scaler = build_grouped_dataloaders(
        df,
        groups,
        input_groups,
        output_groups,
        seq_len=config["dataset"]["seq_len"],
        pred_len=config["dataset"]["pred_len"],
        batch_size=config["dataset"]["batch_size"],
        train_split=config["dataset"].get("train_split", 0.8),
        device=config["misc"]["device"],
    )
    
    # Save scaler
    from joblib import dump
    dump(scaler, run_dir / "scaler.pkl")
    logger.info(f"Saved scaler to {run_dir / 'scaler.pkl'}")
    
    # Get datasets for world model trainer
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # Create model
    logger.info("Creating model...")
    ModelCls = get_model(config["model"]["type"])
    
    input_dim = len(sum((groups[g] for g in input_groups), []))
    output_dim = len(sum((groups[g] for g in output_groups), []))
    
    model = ModelCls(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.0),
        pred_len=config["dataset"]["pred_len"],
    )
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Input dim: {input_dim}, Output dim: {output_dim}")
    
    # Create sampling strategy
    sampling_strategy = create_sampling_strategy(config)
    logger.info(f"Sampling strategy: {sampling_strategy.__class__.__name__}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = WorldModelTrainer(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        sampling_strategy=sampling_strategy,
        warmup_len=config["training"].get("warmup_len", 24),
        batch_size=config["dataset"]["batch_size"],
        loss_type=config["training"].get("loss_type", "mse"),
        training_mode=config["training"].get("mode", "multi_step"),
        feedback=config["training"].get("feedback", "model"),
        teacher_forcing_ratio=config["training"].get("teacher_forcing_ratio", 0.0),
        one_step_weight=config["training"].get("one_step_weight", 0.5),
        optimizer=torch.optim.Adam(
            model.parameters(),
            lr=config["training"].get("learning_rate", 1e-3),
        ),
        device=config["misc"]["device"],
        early_stopping=config["training"].get("early_stopping", False),
        patience=config["training"].get("patience", 5),
        run_dir=run_dir,
        writer=tb_writer,
    )
    
    # Train
    logger.info("Starting training...")
    train_losses, val_losses = trainer.fit(
        epochs=config["training"]["epochs"],
        verbose=True,
    )
    
    # Save checkpoint
    checkpoint_path = run_dir / "checkpoint.pth"
    trainer.save(checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Plot losses
    from timesim.utils.plotting import save_loss_plot
    save_loss_plot(train_losses, val_losses, run_dir / "figs" / "loss.png")
    
    logger.info("Training complete!")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses[-1] is not None:
        logger.info(f"Final val loss: {val_losses[-1]:.4f}")


if __name__ == "__main__":
    main()

