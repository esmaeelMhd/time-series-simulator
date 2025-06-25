import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
import joblib
from copy import deepcopy
from typing import Any, Optional

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from dataclasses import dataclass, field, fields

# Local Module Imports
from models import NLinear, Informer, Transformer, Autoformer, DLinear
from models.LSTM import LSTMModel
from exp.exp_main_env import Exp_Main
from exp.exp_lstm_env import ExpLSTM

def load_dataset(data_root_path, data_name):
    """Loads the dataset used in training the model."""
    print('Loading the dataset ...')
    dataset_path = os.path.join(data_root_path, data_name)
    df = pd.read_csv(dataset_path, index_col=["date"], parse_dates=["date"])
    df.sort_index(inplace=True)
    df = df.astype('float32').ffill()
    
    return df

def add_time_specs(df):
    """Embeds cyclical time features into the DataFrame based on its index."""
    def generate_cyclical_features(df, col_name, period, start_num=0):
        sin_col = np.sin(2 * np.pi * (df[col_name] - start_num) / period)
        cos_col = np.cos(2 * np.pi * (df[col_name] - start_num) / period)
        df[f'sin_{col_name}'] = sin_col
        df[f'cos_{col_name}'] = cos_col
        return df.drop(columns=[col_name])
    
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df = generate_cyclical_features(df, 'hour', 24)
    df = generate_cyclical_features(df, 'month', 12, 1)  # Starts at 1
    df = generate_cyclical_features(df, 'day_of_week', 7)
    
    return df
        
def load_args(args_path, folder_name):
    """Loads the args file of the model."""
    file_path = os.path.join(args_path, folder_name)
    try:
        with open(os.path.join(file_path, 'args.pkl'), 'rb') as file:
            args = pickle.load(file)
    except OSError as e:
        print(f"Unable to open {file_path}: {e}", file=sys.stderr)
        args=None
    
    return args

def args_mapping(args, map_dict):
    # Assign variables based on the mapping
    for possible_name, standardized_name in map_dict.items():
        if hasattr(args, possible_name):
            # Assign to a local variable dynamically
            setattr(args, standardized_name, getattr(args, possible_name))
    
    return args

class Args:
    def __init__(self, **kwargs):
        # Loop through each key-value pair in kwargs and set them as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        

@dataclass
class ModelBuilder:
    """
    A class to build models with customizable configurations.
    
    Attributes:
        custom_model (bool): Flag to indicate if a custom model is being used.
        model (Any): The model object, typically a machine learning model instance.
        args_path (str): Filesystem path to where model arguments are stored.
        checkpoint_path (str): Filesystem path to where checkpoints are stored.
        model_type (str): Type of model, e.g., 'LSTM', 'CNN'.
        model_name (str): Name of the model configuration.
        checkpoint (str): The filename of the model training checkpoint.
        use_gpu (bool): Flag to enable GPU usage.
        use_multi_gpu (bool): Flag to enable multiple GPU usage.
        device_ids (str): Comma-separated string of GPU device IDs to use.
        device (str): Computed attribute to set the device based on GPU availability.

    Raises:
        ValueError: If custom_model is True but no model is provided.
    """
    args: Any = None 
    custom_model: bool = False
    model: Optional[Any] = None
    args_path: str = './args'
    checkpoint_path: str = './checkpoints'
    model_type: str = 'LSTM'
    model_name: str = 'LSTM'
    checkpoint: str = 'checkpoint.pth'
    use_gpu: bool = True
    device: str = 'cuda'
    use_multi_gpu: bool = False
    device_ids: str = '0,1'

    def __post_init__(self):           
        # Setup the device based on GPU usage
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.use_gpu else torch.device('cpu')
        if self.device == 'cuda':
            self.device_ids = ','.join(str(i) for i in range(torch.cuda.device_count()))
            self.use_multi_gpu = torch.cuda.device_count() > 1
        else:
            self.device_ids = ''
            self.use_multi_gpu = False
        
        # Validate model configuration
        if self.custom_model and self.model is None:
            raise ValueError("model should be provided if custom_model is True")
 
    @staticmethod
    def from_namespace(args, **kwargs):
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(ModelBuilder))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        
        # Override filtered_args with manually set attributes from kwargs
        filtered_args.update(kwargs)
        
        return ModelBuilder(**filtered_args)
    
    def load_model(self):
        """Loads the model from a checkpoint and sets it up for inference or further training."""
        print('Loading the model ...')
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
            self.model.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
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
        
        return self.exp

@dataclass
class ScalerHandler:
    """
    A class to handle scaler operations for machine learning model preprocessing.
    It supports loading a pre-configured scaler or creating and fitting a new scaler
    based on provided data.

    Attributes:
        scaler_root_path (str): Root path where scalers are stored.
        model_type (str): Type of the model associated with the scaler.
        model_name (str): Name of the model configuration.
        time_scaled (str): Indicates if the time feature is scaled.
        load_scaler (bool): Flag to decide if a scaler should be loaded from disk.
        scaler (MinMaxScaler): The scaler instance, loaded or newly created.
        df (pd.DataFrame): DataFrame to fit the scaler if creating a new one.
    """
    args: Any = None
    scaler_root_path: str = './scalers'
    model_type: str = 'LSTM'
    model_name: str = 'LSTM'
    time_scaled: str = 'unscaled'
    load_scaler: bool = True
    scaler: Optional[MinMaxScaler] = None
    df: Optional[pd.DataFrame] = None

    def __post_init__(self):                
        if self.load_scaler:
            self.scaler = self._load_scalers()
        else:
            self.scaler = MinMaxScaler()
            if self.df is not None:
                self.scaler.fit_transform(self.df)
                
    @staticmethod
    def from_namespace(args, **kwargs):
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(ScalerHandler))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        
        # Override filtered_args with manually set attributes from kwargs
        filtered_args.update(kwargs)
        
        return ScalerHandler(**filtered_args)

    def _load_scalers(self):
        scaler_file = os.path.join(self.scaler_root_path, self.model_name)
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
        if self.scaler is not None:
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