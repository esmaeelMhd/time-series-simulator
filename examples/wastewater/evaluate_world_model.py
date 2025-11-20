"""Example: Evaluate a trained world model.

This script loads a trained world model and evaluates its performance on
test data using various metrics and visualizations.
"""

import argparse
from pathlib import Path
import yaml

import torch
import numpy as np
import matplotlib.pyplot as plt

from timesim.data.loader import load_csv_dataset
from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.models import get_model
from timesim.utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained world model")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=120,
        help="Evaluation horizon (number of steps)",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=10,
        help="Number of test scenarios to evaluate",
    )
    return parser.parse_args()


def evaluate_rollout(model, dataset, start_idx, horizon, warmup_len, device):
    """Evaluate a single rollout."""
    model.eval()
    
    # Get warmup
    warmup_data = dataset.get_warmup_window(start_idx, warmup_len)
    warmup_inputs = torch.tensor(
        warmup_data["inputs"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    
    # Get rollout data
    rollout_data = dataset.get_rollout_slice(start_idx, horizon)
    rollout_inputs_np = rollout_data["inputs"]
    targets_np = rollout_data["targets"]
    
    # Split inputs (assuming control + exogenous)
    control_dim = len([c for c in dataset.input_cols if c in dataset.groups.get("control", [])])
    exo_dim = len([c for c in dataset.input_cols if c in dataset.groups.get("exogenous", [])])
    
    if control_dim + exo_dim == 0:
        control_dim = rollout_inputs_np.shape[-1]
        exo_dim = 0
    
    controls = torch.tensor(
        rollout_inputs_np[:, :control_dim], dtype=torch.float32, device=device
    ).unsqueeze(0)
    
    if exo_dim > 0:
        exogenous = torch.tensor(
            rollout_inputs_np[:, control_dim:control_dim+exo_dim],
            dtype=torch.float32, device=device
        ).unsqueeze(0)
    else:
        exogenous = torch.zeros(1, horizon, 0, device=device)
    
    # Rollout
    with torch.no_grad():
        result = model.rollout(
            warmup_seq={"inputs": warmup_inputs},
            rollout_inputs={"controls": controls, "exogenous": exogenous},
            horizon=horizon,
            feedback="model",
        )
    
    predictions = result["predictions"].cpu().numpy().squeeze(0)
    
    return predictions, targets_np


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    
    # Load config
    with open(run_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Evaluating model from {run_dir}")
    
    # Load dataset
    print("Loading dataset...")
    df = load_csv_dataset(
        config["dataset"]["csv"],
        index_col=config["dataset"]["index_col"],
    )
    
    groups = config["dataset"]["variables"]
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]
    
    # Create test dataset (use last 20% of data)
    train_split = config["dataset"].get("train_split", 0.8)
    n_train = int(len(df) * train_split)
    test_df = df.iloc[n_train:]
    
    # Load scaler
    from joblib import load
    scaler = load(run_dir / "scaler.pkl")
    
    test_dataset = GroupedTimeSeriesDataset(
        test_df,
        groups,
        input_groups,
        output_groups,
        seq_len=config["dataset"]["seq_len"],
        pred_len=config["dataset"]["pred_len"],
        scale=True,
        scaler=scaler,
    )
    
    # Load model
    print("Loading model...")
    ModelCls = get_model(config["model"]["type"])
    
    control_dim = len(groups.get("control", []))
    exo_dim = len(groups.get("exogenous", []))
    output_dim = len(sum((groups[g] for g in output_groups), []))
    
    # For world models, input_dim = control + exogenous + output (for autoregressive feedback)
    input_dim = control_dim + exo_dim + output_dim
    
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
    
    device = torch.device(config["misc"]["device"])
    model.load_state_dict(torch.load(run_dir / "checkpoint.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # Evaluate on multiple scenarios
    print(f"Evaluating on {args.num_scenarios} scenarios...")
    warmup_len = config["training"].get("warmup_len", 24)
    horizon = args.horizon
    
    all_predictions = []
    all_targets = []
    
    # Sample random starting points
    max_start = len(test_dataset.values) - (warmup_len + horizon)
    start_indices = np.linspace(warmup_len, max_start, args.num_scenarios, dtype=int)
    
    for i, start_idx in enumerate(start_indices):
        print(f"  Scenario {i+1}/{args.num_scenarios}...", end="\r")
        predictions, targets = evaluate_rollout(
            model, test_dataset, start_idx, horizon, warmup_len, device
        )
        all_predictions.append(predictions)
        all_targets.append(targets)
    
    print()
    
    # Compute metrics
    print("\nComputing metrics...")
    all_predictions = np.array(all_predictions)  # (num_scenarios, horizon, output_dim)
    all_targets = np.array(all_targets)
    
    # Aggregate metrics
    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(mse)
    
    # Per-step metrics
    step_mse = np.mean((all_predictions - all_targets) ** 2, axis=(0, 2))  # (horizon,)
    step_mae = np.mean(np.abs(all_predictions - all_targets), axis=(0, 2))
    
    print(f"\nOverall Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    # Plot results
    print("\nGenerating plots...")
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Per-step error
    plt.figure(figsize=(10, 5))
    plt.plot(step_mse, label="MSE", marker="o", markersize=3)
    plt.plot(step_mae, label="MAE", marker="s", markersize=3)
    plt.xlabel("Step")
    plt.ylabel("Error")
    plt.title("Per-Step Prediction Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "per_step_error.png", dpi=150)
    print(f"  Saved {output_dir / 'per_step_error.png'}")
    
    # Plot 2: Example rollouts
    fig, axes = plt.subplots(min(3, args.num_scenarios), 1, figsize=(12, 8))
    if args.num_scenarios == 1:
        axes = [axes]
    
    for i in range(min(3, args.num_scenarios)):
        ax = axes[i]
        for j in range(all_targets.shape[-1]):  # For each output dimension
            ax.plot(all_targets[i, :, j], label=f"True (dim {j})", linestyle="-", alpha=0.7)
            ax.plot(all_predictions[i, :, j], label=f"Pred (dim {j})", linestyle="--", alpha=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value (scaled)")
        ax.set_title(f"Scenario {i+1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "example_rollouts.png", dpi=150)
    print(f"  Saved {output_dir / 'example_rollouts.png'}")
    
    # Save metrics to file
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Evaluation Metrics (horizon={horizon}, scenarios={args.num_scenarios})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"MSE:  {mse:.6f}\n")
        f.write(f"MAE:  {mae:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
    
    print(f"  Saved {metrics_file}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

