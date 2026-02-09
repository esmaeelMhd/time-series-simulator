"""XGBoost wrapper for time series forecasting.

XGBoost is a tree-based gradient boosting model that operates differently
from neural networks. It uses a fit/predict paradigm rather than forward().

This module provides a wrapper that makes XGBoost compatible with the
time series forecasting workflow.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Literal, Tuple, Union
import warnings

import numpy as np

# Optional XGBoost import
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None


class XGBoostForecaster:
    """XGBoost model for time series forecasting.
    
    This wraps XGBoost to provide a consistent interface for time series
    prediction. Unlike neural networks, XGBoost:
    - Uses fit() instead of gradient-based training
    - Flattens the input window into features
    - Can use recursive or direct multi-step strategies
    
    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.
    seq_len : int
        Input sequence length (lookback window).
    pred_len : int
        Prediction horizon length.
    output_dim : int, optional
        Number of output features. Defaults to input_dim.
    strategy : {"recursive", "direct"}, default "recursive"
        Multi-step prediction strategy:
        - "recursive": Predict one step, feed back, repeat
        - "direct": Train separate model for each horizon step
    n_estimators : int, default 100
        Number of boosting rounds.
    max_depth : int, default 6
        Maximum tree depth.
    learning_rate : float, default 0.1
        Boosting learning rate.
    **xgb_params
        Additional parameters passed to XGBRegressor.
    
    Examples
    --------
    >>> model = XGBoostForecaster(input_dim=5, seq_len=24, pred_len=12)
    >>> model.fit(X_train, y_train)  # X: (N, seq_len, features), y: (N, pred_len, out_features)
    >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        output_dim: Optional[int] = None,
        strategy: Literal["recursive", "direct"] = "recursive",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **xgb_params,
    ):
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )
        
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.strategy = strategy
        
        self.xgb_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            **xgb_params,
        }
        
        # Models (one per output dim, or one per (output_dim, horizon) for direct)
        self.models_: Optional[list] = None
        self._fitted = False
    
    def _flatten_input(self, X: np.ndarray) -> np.ndarray:
        """Flatten 3D input to 2D for XGBoost.
        
        Parameters
        ----------
        X : np.ndarray
            Input of shape (n_samples, seq_len, input_dim).
        
        Returns
        -------
        np.ndarray
            Flattened input of shape (n_samples, seq_len * input_dim).
        """
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = False,
    ) -> "XGBoostForecaster":
        """Fit the XGBoost model(s).
        
        Parameters
        ----------
        X : np.ndarray
            Training inputs of shape (n_samples, seq_len, input_dim).
        y : np.ndarray
            Training targets of shape (n_samples, pred_len, output_dim)
            or (n_samples, output_dim) for single-step.
        eval_set : tuple, optional
            Validation set (X_val, y_val) for early stopping.
        verbose : bool, default False
            Print training progress.
        
        Returns
        -------
        self
            Fitted model.
        """
        # Flatten input
        X_flat = self._flatten_input(X)
        
        # Handle target shape
        if y.ndim == 2:
            # (n_samples, output_dim) -> assume pred_len=1
            y = y.reshape(y.shape[0], 1, -1)
        
        n_samples, pred_len, output_dim = y.shape
        
        # Prepare eval set if provided
        eval_flat = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_flat = self._flatten_input(X_val)
            if y_val.ndim == 2:
                y_val = y_val.reshape(y_val.shape[0], 1, -1)
        
        if self.strategy == "recursive":
            # Train one model per output dimension (single-step prediction)
            self.models_ = []
            for out_idx in range(output_dim):
                # Use only first step of target for training
                y_out = y[:, 0, out_idx]
                
                # early_stopping_rounds moved to constructor in xgboost >= 2.1
                model_params = dict(self.xgb_params)
                if eval_set is not None:
                    model_params["early_stopping_rounds"] = 10
                model = xgb.XGBRegressor(**model_params)
                
                fit_params = {"verbose": verbose}
                if eval_set is not None:
                    y_val_out = y_val[:, 0, out_idx]
                    fit_params["eval_set"] = [(X_val_flat, y_val_out)]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_flat, y_out, **fit_params)
                
                self.models_.append(model)
        
        elif self.strategy == "direct":
            # Train separate model for each (horizon, output) pair
            self.models_ = []
            for h in range(pred_len):
                horizon_models = []
                for out_idx in range(output_dim):
                    y_out = y[:, h, out_idx]
                    
                    model_params = dict(self.xgb_params)
                    if eval_set is not None:
                        model_params["early_stopping_rounds"] = 10
                    model = xgb.XGBRegressor(**model_params)
                    
                    fit_params = {"verbose": verbose}
                    if eval_set is not None:
                        y_val_out = y_val[:, h, out_idx]
                        fit_params["eval_set"] = [(X_val_flat, y_val_out)]
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_flat, y_out, **fit_params)
                    
                    horizon_models.append(model)
                self.models_.append(horizon_models)
        
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input of shape (n_samples, seq_len, input_dim).
        
        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples, pred_len, output_dim).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.pred_len, self.output_dim))
        
        if self.strategy == "recursive":
            # Recursive prediction: predict one step, update input, repeat
            current_input = X.copy()
            
            for h in range(self.pred_len):
                X_flat = self._flatten_input(current_input)
                
                # Predict all output dimensions
                step_pred = np.zeros((n_samples, self.output_dim))
                for out_idx, model in enumerate(self.models_):
                    step_pred[:, out_idx] = model.predict(X_flat)
                
                predictions[:, h, :] = step_pred
                
                # Update input for next step (slide window)
                if h < self.pred_len - 1:
                    # Shift window and append prediction
                    # This assumes last input_dim features are the targets
                    new_step = current_input[:, -1, :].copy()
                    new_step[:, -self.output_dim:] = step_pred
                    current_input = np.concatenate([
                        current_input[:, 1:, :],
                        new_step[:, np.newaxis, :]
                    ], axis=1)
        
        elif self.strategy == "direct":
            # Direct prediction: each horizon has its own model
            X_flat = self._flatten_input(X)
            
            for h in range(self.pred_len):
                for out_idx in range(self.output_dim):
                    predictions[:, h, out_idx] = self.models_[h][out_idx].predict(X_flat)
        
        return predictions
    
    def save(self, path: str):
        """Save model to file.
        
        Parameters
        ----------
        path : str
            Path to save model.
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models_,
                "params": {
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "seq_len": self.seq_len,
                    "pred_len": self.pred_len,
                    "strategy": self.strategy,
                    "xgb_params": self.xgb_params,
                },
                "fitted": self._fitted,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "XGBoostForecaster":
        """Load model from file.
        
        Parameters
        ----------
        path : str
            Path to load model from.
        
        Returns
        -------
        XGBoostForecaster
            Loaded model.
        """
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        model = cls(**data["params"])
        model.models_ = data["models"]
        model._fitted = data["fitted"]
        return model


class XGBoostEnsemble:
    """Ensemble of XGBoost models for uncertainty estimation.
    
    Trains multiple XGBoost models with different random seeds
    to provide prediction intervals.
    
    Parameters
    ----------
    n_models : int, default 5
        Number of models in ensemble.
    **kwargs
        Parameters passed to XGBoostForecaster.
    """
    
    def __init__(self, n_models: int = 5, **kwargs):
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )
        
        self.n_models = n_models
        self.kwargs = kwargs
        self.models_: list = []
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = False,
    ) -> "XGBoostEnsemble":
        """Fit ensemble models."""
        self.models_ = []
        for i in range(self.n_models):
            model = XGBoostForecaster(
                random_state=i * 42,
                **self.kwargs,
            )
            model.fit(X, y, eval_set=eval_set, verbose=verbose)
            self.models_.append(model)
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate predictions with optional uncertainty.
        
        Parameters
        ----------
        X : np.ndarray
            Input array.
        return_std : bool, default False
            If True, also return standard deviation.
        
        Returns
        -------
        mean : np.ndarray
            Mean predictions.
        std : np.ndarray, optional
            Standard deviation (if return_std=True).
        """
        all_preds = np.stack([m.predict(X) for m in self.models_], axis=0)
        mean = all_preds.mean(axis=0)
        
        if return_std:
            std = all_preds.std(axis=0)
            return mean, std
        return mean

