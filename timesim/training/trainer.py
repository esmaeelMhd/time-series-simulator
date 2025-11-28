"""Unified trainer for world models with multi-step rollout training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Callable
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import GroupedTimeSeriesDataset
from ..data.sampling import SamplingStrategy, RandomStartFixedHorizon
from ..models.base import WorldModelBase
from ..utils.early_stop import EarlyStopping
from .losses import OneStepLoss, MultiStepLoss, CombinedLoss
from .rollout import batch_rollout_padded


class WorldModelTrainer:
    """Unified trainer for world models with multi-step rollout training.
    
    This trainer implements the "see every possible path" approach by:
    1. Sampling diverse (start_index, horizon) pairs using a SamplingStrategy
    2. Performing multi-environment rollouts in parallel
    3. Computing multi-step losses to reduce compounding errors
    
    Parameters
    ----------
    model : WorldModelBase
        World model to train.
    dataset : GroupedTimeSeriesDataset
        Training dataset.
    val_dataset : GroupedTimeSeriesDataset, optional
        Validation dataset.
    sampling_strategy : SamplingStrategy
        Strategy for sampling rollout starting points and horizons.
    warmup_len : int
        Length of warmup sequence for state initialization.
    batch_size : int, default 32
        Number of rollouts per training batch.
    loss_type : {"mse", "mae", "huber"}, default "mse"
        Base loss function type.
    training_mode : {"multi_step", "one_step", "combined"}, default "multi_step"
        Training mode:
        - "multi_step": Pure autoregressive rollout loss
        - "one_step": Teacher-forced one-step loss
        - "combined": Weighted combination of both
    feedback : {"model", "teacher", "mixed"}, default "model"
        Feedback mode for multi-step rollouts.
    teacher_forcing_ratio : float, default 0.0
        Ratio for mixed feedback mode.
    one_step_weight : float, default 0.5
        Weight for one-step loss in combined mode.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to use. If None, creates Adam with lr=1e-3.
    device : torch.device or str, default "cpu"
        Device for training. Ignored if use_gpu=True.
    use_gpu : bool, default False
        If True, automatically use GPU if available, otherwise use CPU.
        Takes precedence over device parameter.
    early_stopping : bool, default False
        Whether to use early stopping.
    patience : int, default 5
        Patience for early stopping.
    run_dir : str or Path, optional
        Directory for saving outputs and logs.
    writer : SummaryWriter, optional
        TensorBoard writer for logging.
    """
    
    def __init__(
        self,
        model: WorldModelBase,
        dataset: GroupedTimeSeriesDataset,
        val_dataset: Optional[GroupedTimeSeriesDataset] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        warmup_len: int = 24,
        batch_size: int = 32,
        loss_type: Literal["mse", "mae", "huber"] = "mse",
        training_mode: Literal["multi_step", "one_step", "combined"] = "multi_step",
        feedback: Literal["model", "teacher", "mixed"] = "model",
        teacher_forcing_ratio: float = 0.0,
        one_step_weight: float = 0.5,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device | str = "cpu",
        use_gpu: bool = False,
        early_stopping: bool = False,
        patience: int = 5,
        run_dir: Optional[str | Path] = None,
        writer: Optional["SummaryWriter"] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.warmup_len = warmup_len
        self.batch_size = batch_size
        self.training_mode = training_mode
        self.feedback = feedback
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Device setup
        if use_gpu:
            # Automatically use GPU if available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("Warning: use_gpu=True but CUDA is not available. Using CPU instead.")
        else:
            # Use the specified device
            self.device = torch.device(device)
        self.model.to(self.device)
        
        # Sampling strategy
        if sampling_strategy is None:
            # Default: random start with fixed horizon
            self.sampling_strategy = RandomStartFixedHorizon(horizon=dataset.pred_len)
        else:
            self.sampling_strategy = sampling_strategy
        
        # Loss function
        if training_mode == "one_step":
            self.loss_fn = OneStepLoss(loss_type=loss_type)
        elif training_mode == "multi_step":
            self.loss_fn = MultiStepLoss(loss_type=loss_type)
        elif training_mode == "combined":
            self.loss_fn = CombinedLoss(
                one_step_weight=one_step_weight,
                multi_step_weight=1.0 - one_step_weight,
                loss_type=loss_type,
            )
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience) if early_stopping else None
        
        # Logging
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            if writer is not None:
                self.writer = writer
            else:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.writer = SummaryWriter(log_dir=self.run_dir)
                except ModuleNotFoundError:
                    self.writer = None
            
            # CSV metrics file
            self.metrics_path = self.run_dir / "metrics.csv"
            if not self.metrics_path.exists():
                with open(self.metrics_path, "w", encoding="utf-8") as f:
                    f.write("epoch,train_loss,val_loss\n")
        else:
            self.writer = writer
            self.metrics_path = None
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng()
    
    def _train_step(self) -> float:
        """Perform one training step (one batch of rollouts).
        
        Returns
        -------
        float
            Training loss for this batch.
        """
        self.model.train()
        
        # Sample rollout starting points and horizons
        start_indices, horizons = self.sampling_strategy.sample(
            dataset_length=len(self.dataset.values),
            batch_size=self.batch_size,
            warmup_len=self.warmup_len,
            rng=self.rng,
        )
        
        # Perform batched rollouts
        if self.training_mode == "combined":
            # Need both teacher-forced and model-feedback rollouts
            result_teacher = batch_rollout_padded(
                self.model, self.dataset, start_indices, horizons,
                self.warmup_len, feedback="teacher", device=self.device
            )
            result_model = batch_rollout_padded(
                self.model, self.dataset, start_indices, horizons,
                self.warmup_len, feedback="model", device=self.device
            )
            
            predictions_teacher = result_teacher["predictions"]
            predictions_model = result_model["predictions"]
            targets = result_teacher["targets"]
            mask = result_teacher["mask"]
            
            # Compute loss
            self.optimizer.zero_grad()
            loss, info = self.loss_fn(predictions_teacher, predictions_model, targets)
            
        else:
            # Single rollout mode
            result = batch_rollout_padded(
                self.model, self.dataset, start_indices, horizons,
                self.warmup_len, feedback=self.feedback,
                teacher_forcing_ratio=self.teacher_forcing_ratio,
                device=self.device
            )
            
            predictions = result["predictions"]
            targets = result["targets"]
            mask = result["mask"]
            
            # Mask out padded values for loss computation
            # Expand mask to match output dimensions
            mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
            predictions_masked = predictions * mask_expanded
            targets_masked = targets * mask_expanded
            
            # Compute loss
            self.optimizer.zero_grad()
            loss = self.loss_fn(predictions_masked, targets_masked)
        
        # Guard against NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN/Inf in training loss. Check data and model stability.")
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def _validate(self) -> Optional[float]:
        """Validate on validation dataset.
        
        Returns
        -------
        float or None
            Validation loss, or None if no validation dataset.
        """
        if self.val_dataset is None:
            return None
        
        self.model.eval()
        
        # Sample validation rollouts
        val_batches = max(1, len(self.val_dataset) // (self.batch_size * self.warmup_len))
        val_losses = []
        
        for _ in range(val_batches):
            start_indices, horizons = self.sampling_strategy.sample(
                dataset_length=len(self.val_dataset.values),
                batch_size=self.batch_size,
                warmup_len=self.warmup_len,
                rng=self.rng,
            )
            
            result = batch_rollout_padded(
                self.model, self.val_dataset, start_indices, horizons,
                self.warmup_len, feedback="model", device=self.device
            )
            
            predictions = result["predictions"]
            targets = result["targets"]
            mask = result["mask"]
            
            # Mask out padded values
            mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
            predictions_masked = predictions * mask_expanded
            targets_masked = targets * mask_expanded
            
            # Compute loss
            if self.training_mode == "combined":
                # For validation, just use multi-step loss
                loss_fn = MultiStepLoss(loss_type="mse")
                loss = loss_fn(predictions_masked, targets_masked)
            else:
                loss = self.loss_fn(predictions_masked, targets_masked)
            
            val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def fit(
        self,
        epochs: int = 10,
        steps_per_epoch: Optional[int] = None,
        verbose: bool = True,
    ) -> tuple[list[float], list[Optional[float]]]:
        """Train the world model.
        
        Parameters
        ----------
        epochs : int, default 10
            Number of training epochs.
        steps_per_epoch : int, optional
            Number of training steps per epoch. If None, uses a heuristic
            based on dataset size.
        verbose : bool, default True
            Whether to print progress.
        
        Returns
        -------
        train_losses : list of float
            Training loss per epoch.
        val_losses : list of float or None
            Validation loss per epoch.
        """
        if steps_per_epoch is None:
            # Heuristic: aim to see each starting point ~once per epoch
            dataset_len = len(self.dataset.values) - self.warmup_len
            steps_per_epoch = max(1, dataset_len // self.batch_size)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting training: {epochs} epochs, {steps_per_epoch} steps/epoch")
            print(f"Batch size: {self.batch_size}, Training mode: {self.training_mode}")
            print(f"{'='*70}\n")
        
        train_losses = []
        val_losses = []
        start_time = time.time()
        epoch_times = []
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Training
            self.model.train()
            epoch_losses = []
            
            if verbose:
                # Create progress bar for training steps
                pbar = tqdm(
                    range(steps_per_epoch),
                    desc=f"Epoch {epoch}/{epochs} [Train]",
                    unit="step",
                    leave=False,
                    ncols=100,
                )
            else:
                pbar = range(steps_per_epoch)
            
            for step in pbar:
                batch_loss = self._train_step()
                epoch_losses.append(batch_loss)
                
                if verbose:
                    # Update progress bar with current loss
                    current_avg_loss = np.mean(epoch_losses)
                    pbar.set_postfix({"loss": f"{current_avg_loss:.4f}"})
            
            train_loss = np.mean(epoch_losses)
            train_losses.append(train_loss)
            
            # Validation
            if verbose:
                print(f"  Validating...", end=" ", flush=True)
            val_start_time = time.time()
            val_loss = self._validate()
            val_losses.append(val_loss)
            val_time = time.time() - val_start_time
            
            if verbose:
                print(f"done ({val_time:.2f}s)")
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            elapsed_time = time.time() - start_time
            avg_epoch_time = np.mean(epoch_times)
            remaining_epochs = epochs - epoch
            eta = avg_epoch_time * remaining_epochs if remaining_epochs > 0 else 0
            
            # Logging - now print every epoch when verbose
            if verbose:
                msg = f"[Epoch {epoch}/{epochs}] "
                msg += f"train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.6f}"
                msg += f" | time={epoch_time:.2f}s"
                if epoch > 1:
                    msg += f" | ETA={eta:.1f}s"
                print(msg)
            
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Time/epoch", epoch_time, epoch)
            
            if self.metrics_path is not None:
                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch},{train_loss},{val_loss if val_loss is not None else ''}\n")
            
            # Early stopping
            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    if verbose:
                        print(f"\n{'='*70}")
                        print("Early stopping triggered.")
                        if self.early_stopping.best_loss is not None:
                            print(f"Best validation loss: {self.early_stopping.best_loss:.6f}")
                        print(f"Stopped at epoch {epoch}/{epochs}")
                        print(f"{'='*70}\n")
                    break
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training completed!")
            print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            print(f"Average epoch time: {np.mean(epoch_times):.2f}s")
            if len(train_losses) > 0:
                print(f"Final train loss: {train_losses[-1]:.6f}")
                if val_losses[-1] is not None:
                    print(f"Final val loss: {val_losses[-1]:.6f}")
            print(f"{'='*70}\n")
        
        return train_losses, val_losses
    
    def save(self, path: str | Path):
        """Save model checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Path to save checkpoint.
        """
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str | Path):
        """Load model checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Path to load checkpoint from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()


# Backward compatibility: alias to old Trainer
class Trainer(nn.Module):
    """Legacy trainer for backward compatibility.
    
    This maintains the old Trainer interface for existing code.
    For new code, use WorldModelTrainer instead.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss: str = "mse",
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device | str = "cpu",
        early_stopping: bool = False,
        patience: int = 5,
        run_dir: Optional[str] = None,
        writer: Optional["SummaryWriter"] = None,
    ):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "dilate":
            self.loss_fn = None  # Will use dilate_loss function
        else:
            raise ValueError("loss must be mse or dilate")
        
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_type = loss
        self.early_stopping = EarlyStopping(patience=patience) if early_stopping else None
        
        # Logging
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            if writer is not None:
                self.writer = writer
            else:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.writer = SummaryWriter(log_dir=self.run_dir)
                except ModuleNotFoundError:
                    self.writer = None
            
            self.metrics_path = self.run_dir / "metrics.csv"
            if not self.metrics_path.exists():
                with open(self.metrics_path, "w", encoding="utf-8") as f:
                    f.write("epoch,train_loss,val_loss\n")
        else:
            self.writer = writer
            self.metrics_path = None
    
    def _step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(x)
        
        if self.loss_type == "mse":
            loss = self.loss_fn(pred, y)
        else:
            from ..utils.dilate import dilate_loss
            loss, _, _ = dilate_loss(y, pred, device=self.device)
        
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN/Inf in training loss.")
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def _validate(self, loader: DataLoader):
        self.model.eval()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            
            if self.loss_type == "mse":
                batch_loss = self.loss_fn(pred, y)
            else:
                from ..utils.dilate import dilate_loss
                batch_loss, _, _ = dilate_loss(y, pred, device=self.device)
            
            total += batch_loss.item() * len(x)
            n += len(x)
        self.model.train()
        return total / max(n, 1)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
    ):
        train_losses, val_losses = [], []
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for batch in train_loader:
                batch_loss = self._step(batch)
                epoch_losses.append(batch_loss)
            
            train_loss = sum(epoch_losses) / len(epoch_losses)
            val_loss = self._validate(val_loader) if val_loader else None
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if verbose and (epoch == 1 or epoch == epochs or epoch % 5 == 0):
                msg = f"[Epoch {epoch}/{epochs}] train={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | val={val_loss:.4f}"
                print(msg)
            
            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    if verbose:
                        print("Early stopping triggered.")
                    break
            
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
            
            if self.metrics_path is not None:
                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch},{train_loss},{val_loss if val_loss is not None else ''}\n")
        
        return train_losses, val_losses
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

