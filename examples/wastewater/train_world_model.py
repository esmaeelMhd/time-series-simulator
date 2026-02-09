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
    
    # Compute dimensions based on what's actually in the dataset
    # input_cols: columns that form the input features
    # output_cols: columns that are prediction targets
    input_cols = sum((groups[g] for g in input_groups), [])
    output_cols = sum((groups[g] for g in output_groups), [])
    
    # For world models, the LSTM sees: input_cols + output_cols (for autoregressive feedback)
    # The output_cols are appended to enable feeding predictions back as inputs
    input_dim = len(input_cols) + len(output_cols)
    output_dim = len(output_cols)
    
    # Compute control and exogenous dimensions from input_groups
    control_dim = len([c for c in input_cols if c in groups.get("control", [])])
    exo_dim = len([c for c in input_cols if c in groups.get("exogenous", [])])
    
    model = ModelCls(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.0),
        pred_len=config["dataset"]["pred_len"],
        control_dim=control_dim,
        exo_dim=exo_dim,
    )
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Input cols: {input_cols}")
    logger.info(f"Output cols: {output_cols}")
    logger.info(f"Input dim: {input_dim} (input_cols={len(input_cols)} + output_cols={len(output_cols)})")
    logger.info(f"Output dim: {output_dim}, Control dim: {control_dim}, Exo dim: {exo_dim}")
    
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
        use_gpu=config["misc"].get("use_gpu", False),
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
    
    # -------------------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------------------
    from timesim.utils.plotting import (
        save_loss_plot,
        save_forecast_plot,
        save_multi_forecast_plot,
        save_training_summary,
    )
    
    figs_dir = run_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    
    # 1. Loss curve
    logger.info("Generating loss plot...")
    save_loss_plot(
        train_losses, val_losses,
        figs_dir / "loss.png",
        title=f"{model.__class__.__name__} Training Progress",
        show_stats=True,
    )
    
    # 2. Recursive forecast evaluation on validation data
    logger.info("Generating recursive forecast plots...")
    
    # Get horizon from config (use pred_len as the forecast horizon)
    pred_len = config["dataset"]["pred_len"]
    warmup_len = config["training"].get("warmup_len", 24)
    
    # Run recursive forecast on multiple windows from validation set
    model.eval()
    device = config["misc"]["device"]
    
    # Select multiple starting points from validation data
    n_forecast_windows = config["training"].get("n_forecast_windows", 4)
    val_data = val_dataset.values
    val_len = len(val_data)
    
    # Ensure we have enough data for warmup + horizon
    min_required = warmup_len + pred_len
    if val_len >= min_required:
        # Sample evenly spaced starting points
        max_start = val_len - min_required
        if max_start > 0:
            step = max(1, max_start // n_forecast_windows)
            start_indices = list(range(0, max_start, step))[:n_forecast_windows]
        else:
            start_indices = [0]
        
        gt_list = []
        pred_list = []
        
        with torch.no_grad():
            for start_idx in start_indices:
                # Extract warmup and horizon data
                warmup_end = start_idx + warmup_len
                horizon_end = warmup_end + pred_len
                
                # Get full sequence data
                warmup_data = val_data[start_idx:warmup_end]
                horizon_data = val_data[warmup_end:horizon_end]
                
                # Prepare inputs
                warmup_inputs = warmup_data[:, val_dataset.in_idx]  # (warmup_len, n_inputs)
                warmup_outputs = warmup_data[:, val_dataset.out_idx]  # (warmup_len, n_outputs)
                horizon_inputs = horizon_data[:, val_dataset.in_idx]  # (pred_len, n_inputs)
                horizon_outputs = horizon_data[:, val_dataset.out_idx]  # (pred_len, n_outputs)
                
                # Concatenate for model input (inputs + outputs for autoregressive)
                warmup_full = np.concatenate([warmup_inputs, warmup_outputs], axis=-1)
                horizon_inputs_full = np.concatenate([
                    horizon_inputs, 
                    np.zeros_like(horizon_outputs)  # placeholder for outputs
                ], axis=-1)
                
                # Convert to tensors
                warmup_tensor = torch.from_numpy(warmup_full).float().unsqueeze(0).to(device)
                controls_tensor = torch.from_numpy(horizon_inputs[:, :control_dim]).float().unsqueeze(0).to(device)
                exogenous_tensor = torch.from_numpy(horizon_inputs[:, control_dim:]).float().unsqueeze(0).to(device)
                
                # Prepare inputs as expected by the model
                warmup_seq = {"inputs": warmup_tensor}
                rollout_inputs = {
                    "controls": controls_tensor,
                    "exogenous": exogenous_tensor,
                }
                
                # Run rollout
                result = model.rollout(
                    warmup_seq=warmup_seq,
                    rollout_inputs=rollout_inputs,
                    horizon=pred_len,
                )
                
                predictions = result["predictions"].squeeze(0).cpu().numpy()  # (pred_len, n_outputs)
                
                gt_list.append(horizon_outputs)
                pred_list.append(predictions)
        
        # Single window forecast plot (first window)
        save_forecast_plot(
            ground_truth=gt_list[0],
            predictions=pred_list[0],
            column_names=output_cols,
            out_path=figs_dir / "forecast_single.png",
            title=f"Recursive Forecast (horizon={pred_len})",
            show_metrics=True,
        )
        
        # Multi-window forecast plot
        if len(gt_list) > 1:
            save_multi_forecast_plot(
                ground_truth_list=gt_list,
                predictions_list=pred_list,
                column_names=output_cols,
                out_path=figs_dir / "forecast_multi.png",
                start_indices=start_indices,
                title=f"Multi-Window Forecast Evaluation (horizon={pred_len})",
            )
        
        # Training summary (combines loss + forecast)
        save_training_summary(
            train_losses=train_losses,
            val_losses=val_losses,
            ground_truth=gt_list[0],
            predictions=pred_list[0],
            column_names=output_cols,
            out_path=figs_dir / "training_summary.png",
            model_name=model.__class__.__name__,
        )
        
        logger.info(f"Saved forecast plots to {figs_dir}")
    else:
        logger.warning(f"Validation data too short for forecast plots (need {min_required}, have {val_len})")
    
    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    logger.info("Training complete!")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses[-1] is not None:
        logger.info(f"Final val loss: {val_losses[-1]:.4f}")
    logger.info(f"Figures saved to: {figs_dir}")


if __name__ == "__main__":
    main()

