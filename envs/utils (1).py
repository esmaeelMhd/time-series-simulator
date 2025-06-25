import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
import joblib
from copy import deepcopy
from typing import Any

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Local Module Imports
from models import NLinear, Informer, Transformer, Autoformer, DLinear
from models.LSTM import LSTMModel
from exp.exp_main_env import Exp_Main
from exp.exp_lstm_env import ExpLSTM

def load_dataset(data_root_path, data_name):
    """Loads the dataset used in training the model."""
    dataset_path = os.path.join(data_root_path, data_name)
    df = pd.read_csv(dataset_path, index_col=["date"], parse_dates=["date"], infer_datetime_format=True)
    df.sort_index(inplace=True)
    df = df.astype('float32').fillna(method='ffill')
        
def load_args(args_path, model_name):
    """Loads the args file of the model."""
    file_path = os.path.join(args_path, model_name)
    try:
        with open(os.path.join(file_path, 'args.pkl'), 'rb') as file:
            args = pickle.load(file)
    except OSError as e:
        print(f"Unable to open {file_path}: {e}", file=sys.stderr)
        sys.exit(1)  # Exit if configuration cannot be loaded
    
    return args

def args_mapping(args, map_dict):
    # Assign variables based on the mapping
    for possible_name, standardized_name in map_dict.items():
        if hasattr(args, possible_name):
            # Assign to a local variable dynamically
            setattr(args, standardized_name, getattr(args, possible_name))
    
    return args

class ModelBuilder():
    def __init__(self, **kwargs):
        super().__init__()
        self.custom_model = kwargs.get('custom_model', False)
        self.model = kwargs.get('model', None)
        self.args_path = kwargs.get('args_path', './args')
        self.checkpoint_path = kwargs.get('checkpoint_path', './checkpoints')
        self.model_type = kwargs.get('model_type', 'LSTM')
        self.model_name = kwargs.get('model_name', 'LSTM')
        self.checkpoint = kwargs.get('checkpoint', 'checkpoint.pth')
        self.use_gpu = kwargs.get('use_gpu', True)
        self.use_multi_gpu = kwargs.get('use_multi_gpu', False)
        self.device_ids = kwargs.get('device_ids', '0,1')
        
        self.device = 'cuda' if self.use_gpu else 'cpu'
        
        # Conditional requirements  
        if self.custom_model and self.model is None:
            raise ValueError("model should be provided if custom_model is True")
 
    def load_model(self):
        """Loads the model from a checkpoint and sets it up for inference or further training."""
        self.model = self._build_model().to(self.device)
        self._load_model_checkpoint()
        self.model.eval()
        
        return self.model
    
    def _build_model(self):
        """Constructs the model based on specifications in model_args."""
        # Define a dictionary mapping model types to their respective classes.
        model_dict = {
            'LSTM': LSTMModel,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear
        }
    
        # Check if the specified model type is supported
        if not self.custom_model and self.model_type not in model_dict:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
        # Initialize the model; assuming all models have a similar constructor interface
        if not self.custom_model and self.model_type in model_dict:
            model_class = model_dict[self.model_type]
            self.model_args = load_args(self.args_path, self.model_name)
            model = model_class(self.model_args).float()
        elif self.custom_model:
            model = self.model
    
        # Apply Data Parallel if using multiple GPUs
        if self.use_multi_gpu and self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.device_ids)
    
        return model.to(self.device)  # Ensure the model is on the correct device

    def _load_model_checkpoint(self):
        """Loads the model state from a checkpoint file."""
        model_path = os.path.join(self.checkpoint_path, self.model_name)
        checkpoint_file = os.path.join(model_path, self.checkpoint)
        try:
            self.model.load_state_dict(torch.load(checkpoint_file))
        except FileNotFoundError:
            print(f"Checkpoint file not found: {checkpoint_file}", file=sys.stderr)
            raise
        except Exception as e:
            print(e)

    def initialize_model_exp(self):
        """Initializes the experiment based on the model type."""
        if not self.custom_model:
            Exp = ExpLSTM if self.model_type == 'LSTM' else Exp_Main
            self.exp = Exp(self.model_args, self.model)
        else:
            self.exp = self.model

class ScalerHandler():
    def __init__(self, **kwargs):
        super().__init__()
        self.scaler_path = kwargs.get('scaler_path', './scalers')
        self.model_type = kwargs.get('model_type', 'LSTM')
        self.model_name = kwargs.get('model_name', 'LSTM')
        self.time_scaled = kwargs.get('time_scaled', 'unscaled')
        self.load_scaler = kwargs.get('load_scaler', True)
        self.scaler = kwargs.get('scaler', None)
        self.df = kwargs.get('df', None)
        
        if self.load_scaler:
            self._load_scalers()
        else:
            self.scaler = MinMaxScaler()
            self.scaler.fit_transform(self.df)

    def _load_scalers(self):
        scaler_file = os.path.join(self.scaler_path, self.model_name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self._load_model_specific_scalers(scaler_file)
        except FileNotFoundError as e:
            print(f"Scaler file not found: {e}", file=sys.stderr)
            sys.exit(1)

    def _load_model_specific_scalers(self, scaler_file):
        """Loads scalers specific to the model type and scaling settings."""
        if self.model_type == 'LSTM':
            self.feature_scaler = joblib.load(os.path.join(scaler_file, 'feature_scaler.gz'))
            if self.time_scaled.lower() != 'unscaled':
                self.time_scaler = joblib.load(os.path.join(scaler_file, 'time_scaler.gz'))
        else:
            self.scaler = joblib.load(os.path.join(scaler_file, 'scaler.gz'))

    def scale_data(self, df):
        """Applies the appropriate scalers to the dataframe."""
        df_temp = deepcopy(df)  # Copy to avoid modifying the original
        if hasattr(self, 'feature_scaler'):
            num_features = self.feature_scaler.n_features_in_
            df_temp.iloc[:, :num_features] = self.feature_scaler.transform(df_temp.iloc[:, :num_features])
        if hasattr(self, 'time_scaler'):
            num_features = self.time_scaler.n_features_in_
            df_temp.iloc[:, -num_features:] = self.time_scaler.transform(df_temp.iloc[:, -num_features:])
        else:
            df_temp = self.scaler.transform(df_temp)

        return df_temp

    def inverse_transform(self, arr):
        """Reverses the scaling of data according to the model configuration."""
        if self.model_type == 'LSTM':
            num_features = self.feature_scaler.n_features_in_
            arr[:, :num_features] = self.feature_scaler.inverse_transform(arr[:, :num_features])
            if self.time_scaled.lower() == 'scaled':
                arr[:, num_features:] = self.time_scaler.inverse_transform(arr[:, num_features:])
            else:
                arr[:, num_features:] = arr[:, num_features:]  # Pass through unscaled
        else:
            arr = self.scaler.inverse_transform(arr)
        
        return arr