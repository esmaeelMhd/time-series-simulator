"""
Created on April 26 2024
@author: Esmaeel Mohammadi (esm@kruger.dk; esmo@bio.aau.dk; https://github.com/esmaeelMhd)

# =============================================================================
# This script is used to create an DRL environment for Phosphorus
  Models are: DLinear, Transformer, Informer, and Autoformer
    1. Preprocessing of the data
    2. Convert data to tensors for the model
    3. Load the model and parameters
    4. Use the environment for generating the next state
    5. Action and Observation spaces are continous
# =============================================================================
"""
# Standard Library Imports
import os
import sys
import pickle
import joblib
import warnings
import logging
import copy
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Union

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn as nn
import torch
import gym
from gym import spaces

# Local Module Imports
from models import NLinear, Informer, Transformer, Autoformer, DLinear
from models.LSTM import LSTMModel
from exp.exp_main_env import Exp_Main
from exp.exp_lstm_env import ExpLSTM

# Set configurations
pd.options.mode.chained_assignment = None
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.xmargin'] = 0.02
plt.rcParams['axes.ymargin'] = 0.04
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['lines.markersize'] = 0.8
fig_dpi = 1000

# %% Defining the Phosphorus Environment Class
@dataclass
class ModelArgs:
    root_path: str
    data_path: str
    setting: str
    model: str
    scale: bool
    time_scaled: str
    ctrl_vars: Union[List[str], str]
    ind_vars: Union[List[str], str]
    num_time_f: int
    target: str
    simulator: bool = False
    use_multi_gpu: bool = False
    
@dataclass
class AgentArgs:
    experiment: str
    const_el: bool
    min_el: float
    max_el: float
    delay_type: str
    max_delay: int
    title: str
    scaled: bool
    setting: str
    retrain: bool
    retr_chkpt: str
    data_path: str
    root_path: str
    ctrl_vars: List[str]
    ind_vars: List[str]
    target: str
    scale: bool
    time_scaled: str
    model: str
        
class PhosphorusEnvironment(gym.Env):   
    # Define constants used in the reward calculation
    ideal_target_level = 1  # Optimal P-concentration in mg/L
    green_tax = 1.675       # Tax rate per Kg of P in DKK
    JSF_price = 0.20        # Cost per liter of JSF in DKK
    PAX_price = 3.54        # Cost per liter of PAX in DKK
    weight_target = 0.3     # Weighting factor for P-concentration deviations
    weight_tax = 0.4        # Weighting factor for tax penalties
    weight_action = 0.3     # Weighting factor for the cost of actions
    
    def __init__(self, model_args: ModelArgs, agent_args: AgentArgs, num_envs: int, device: str, 
                 fig_folder: str = './agent_results/', mode: str = 'not_live'):
        super().__init__()

        '''
        sequence: The input sequence to the model which it is used for the prediction
        state: The predicted one step in the future that the simulation is in it
        obs: The observation which includes one or a history of system states
        '''        
        super().__init__()
        self.model_args = model_args
        self.agent_args = agent_args
        self.num_envs = num_envs
        self.device = device
        self.fig_folder = fig_folder
        self.mode = mode
        self._setup_visualization()
        self._load_args()
        self._load_dataset()
        self._load_scalers()
        self._scale_dataframe()
        self._initialize_data()
        self._calculate_min_max()        
        self._setup_spaces()
        self._load_and_setup_model()

        """Initializes delay-related settings."""
        self.const_delay = self.agent_args.delay_type == 'constant'
        self.max_delay = self.agent_args.max_delay

        """Initializes visualization-related settings."""
        self.visualization = None
        self.window_size = 3
        self.figure = None
        self.observation_ax = None
        self.action_axes = []

        """Initializes settings related to data scaling."""
        self.scaled = self.agent_args.scaled

        """Initializes settings for model improvement."""
        self.improve_epochs = 20
        self.setting = self.agent_args.setting  # Often used for file paths or model configuration

        """Initializes various data structures used throughout the environment."""
        self.dates = []
        self.scaled_inputs = []
        self.targets = []
        self.real_targets = []
        self.actions = []
        self.real_actions = []
        self.observations = []
        self.rewards = []
        self.real_rewards = []
        self.sequences = []
        
        # Assign colors for plots
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        
    def _setup_visualization(self):
        """Sets up the visualization method in render."""
        if self.mode == 'live':
            plt.switch_backend('qt5agg')
            plt.ion()

    def _load_args(self):
        """Loads the args file of the model."""
        ARGS_PATH = f'./args/{self.agent_args.setting}/'
        try:
            with open(f'{ARGS_PATH}args.pkl', 'rb') as file:
                self.model_args = pickle.load(file)
        except OSError as e:
            print(f"Unable to open {ARGS_PATH}: {e}", file=sys.stderr)
            sys.exit(1)  # Exit if configuration cannot be loaded
        self.model_args.simulator = True
        self.model_args.use_multi_gpu = False
    
    def _load_dataset(self):
        """Loads the dataset used in training the model."""
        dataset_path = self.model_args.root_path + self.model_args.data_path
        self.df_raw = pd.read_csv(dataset_path, index_col=["date"], parse_dates=["date"], infer_datetime_format=True)
        self.df_raw.sort_index(inplace=True)
        self.df_raw = self.df_raw.astype('float32').fillna(method='ffill')
        
        self._check_frequency_uniformity()
        
        self.columns = self.df_raw.columns
        self.num_cols = len(self.columns)
        self.freq = self.df_raw.index.to_series().diff().dropna().mode()[0]
        self.freq_min = self.freq.total_seconds() / 60
    
    def _check_frequency_uniformity(self):
        """Checks if the frequncy of the dataset is uniform."""
        time_diffs = self.df_raw.index.to_series().diff()
        is_uniform = (time_diffs == time_diffs.mode()[0]).all()
        if not is_uniform:
            print('ATTENTION: The frequency is not uniform.')
    
    def _load_scalers(self):
        """Loads the necessary scalers based on model configuration."""
        if not self.model_args.scale:
            return
        
        scaler_path = f'./scalers/{self.model_args.setting}/'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self._load_model_specific_scalers(scaler_path)
        except FileNotFoundError as e:
            print(f"Scaler file not found: {e}", file=sys.stderr)
            sys.exit(1)

    def _load_model_specific_scalers(self, scaler_path):
        """Loads scalers specific to the model type and scaling settings."""
        if self.model_args.model == 'LSTM':
            self.feature_scaler = joblib.load(f'{scaler_path}feature_scaler.gz')
            if self.model_args.time_scaled != 'Unscaled':
                self.time_scaler = joblib.load(f'{scaler_path}time_scaler.gz')
        else:
            self.scaler = joblib.load(f'{scaler_path}scaler.gz')

    def _scale_dataframe(self):
        """Scales the raw dataframe using loaded scalers, if scaling is enabled."""
        if not self.model_args.scale:
            self.df_scaled = self.df_raw
        else:
            self.df_scaled = self._apply_scalers()

    def _apply_scalers(self):
        """Applies the appropriate scalers to the dataframe."""
        df_temp = deepcopy(self.df_raw)  # Copy to avoid modifying the original
        if hasattr(self, 'feature_scaler'):
            df_temp = self.feature_scaler.transform(df_temp)
        if hasattr(self, 'time_scaler'):
            df_temp = self.time_scaler.transform(df_temp)
        return df_temp
        
    def _initialize_data(self):
        """Initializes and prepares the data for the environment."""
        self.target = self.model_args.target
        self.target_idx = self.df_raw.columns.get_loc(self.target)
        self.df_raw = self._add_time_specs(self.df_raw)
        self.df_raw = self.df_raw.astype('float32')
        self.q_tank = copy.deepcopy(self.df_raw['IN_Q'])  # Assuming relevance for some functionality
        self._calculate_reward(data='plant')  # Placeholder for reward calculation
        
        # Setup control and independent variables
        self.ctrl_vars = self._parse_variables(['ctrl_vars', 'control_variable'])
        self.ind_vars = self._parse_variables(['ind_vars', 'independent_vars'])
        
        # Set the number of time features and actions
        self.num_time_f = getattr(self.model_args, 'num_time_f', 6)
        self.n_actions = len(self.ctrl_vars)
        self.n_ind = len(self.ind_vars)
    
    def _parse_variables(self, attribute_names: List[str]) -> List[str]:
        """Parses control or independent variables from model_args using possible attribute names."""
        for attr in attribute_names:
            value = getattr(self.model_args, attr, None)
            if value is not None:
                if isinstance(value, str):
                    return [value]
                elif isinstance(value, list):
                    return value
        return []  # Return an empty list if no attributes match

    def _calculate_min_max(self):
        """Calculates the minimum and maximum values for targets and control variables."""
        if self.scaled:
            df_scaled = pd.DataFrame(self.df_scaled, columns=self.df_raw.columns)
            self.min_target, self.max_target = df_scaled[self.target].agg(['min', 'max']).astype(np.float32)
            self.min_ctrl_vars, self.max_ctrl_vars = self._get_min_max_vars(df_scaled, self.ctrl_vars)
        else:
            self.min_target, self.max_target = self.df_raw[self.target].agg(['min', 'max']).astype(np.float32)
            self.min_ctrl_vars, self.max_ctrl_vars = self._get_min_max_vars(self.df_raw, self.ctrl_vars)
    
        self._set_min_max_vars_by_delay_type(df_scaled if self.scaled else self.df_raw)
    
    def _get_min_max_vars(self, df, variables):
        """Helper function to get min and max values for a list of variables."""
        min_values = df[variables].min().tolist()
        max_values = df[variables].max().tolist()
        return min_values, max_values
    
    def _set_min_max_vars_by_delay_type(self, df):
        """Sets min and max for observation spaces based on delay type."""
        var_indices = df.columns[self.n_actions:] if not self.const_delay else df.columns
        self.min_vars, self.max_vars = self._get_min_max_vars(df, var_indices)
        if self.const_delay:
            self.min_vars = np.tile(self.min_vars, (self.max_delay, 1))
            self.max_vars = np.tile(self.max_vars, (self.max_delay, 1))
    
    def _setup_spaces(self):        
        """Defines the action space based on control variables' min and max values."""
        self.action_space = spaces.Box(
            low=np.array(self.min_ctrl_vars, dtype=np.float32),
            high=np.array(self.max_ctrl_vars, dtype=np.float32),
            dtype=np.float32
        )
    
        """Defines the observation space based on the environment settings."""
        num_columns = len(self.df_raw.columns)
        if self.const_delay:
            # Here the observation space includes a delay dimension
            self.observation_space = spaces.Box(
                low=np.float32(self.min_vars),
                high=np.float32(self.max_vars),
                shape=(self.max_delay, num_columns),
                dtype=np.float32
            )
        else:
            # Standard observation space without considering delay
            self.observation_space = spaces.Box(
                low=np.float32(self.min_vars),
                high=np.float32(self.max_vars),
                dtype=np.float32
            )
    
    def _load_and_setup_model(self):
        """Loads the model from a checkpoint and sets it up for inference or further training."""
        self.model = self._build_model().to(self.device)
        self._load_model_checkpoint()
        self._initialize_model_exp()
        self.model.eval()   
    
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
        if self.model_args.model not in model_dict:
            raise ValueError(f"Unsupported model type: {self.model_args.model}")
    
        # Initialize the model; assuming all models have a similar constructor interface
        if self.model_args.model in model_dict:
            model_class = model_dict[self.model_args.model]
            # Check if model needs special construction like LSTM
            if self.model_args.model == 'LSTM':
                model = model_class(self.model_args, self.device).float()
            else:
                model = model_class(self.model_args).float()
    
        # Apply Data Parallel if using multiple GPUs
        if self.model_args.use_multi_gpu and self.model_args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.model_args.device_ids)
    
        return model.to(self.device)  # Ensure the model is on the correct device

    def _load_model_checkpoint(self):
        """Loads the model state from a checkpoint file."""
        checkpoint_path = f'./checkpoints/{self.setting}'
        checkpoint_file = self.agent_args.retr_chkpt if self.agent_args.retrain else 'checkpoint.pth'
        checkpoint_full_path = os.path.join(checkpoint_path, checkpoint_file)
        try:
            self.model.load_state_dict(torch.load(checkpoint_full_path))
        except FileNotFoundError:
            print(f"Checkpoint file not found: {checkpoint_full_path}", file=sys.stderr)
            raise

    def _initialize_model_exp(self):
        """Initializes the experiment based on the model type."""
        if self.model_args.model == 'LSTM':
            self.exp = ExpLSTM(self.model_args, self.device, self.model)
        else:
            self.exp = Exp_Main(self.model_args, self.model)

    def _set_experiment(self):
        """Configuration mapping for experiments."""
        # Key: experiment number
        # Value: tuple (random_episode_start, random_episode_length)
        experiment_config = {
            1: (False, False),
            2: (False, True),
            3: (True, False),
            4: (True, True)
        }
    
        # Apply the configuration based on the experiment number
        if self.experiment in experiment_config:
            self.random_episode_start, self.random_episode_length = experiment_config[self.experiment]
        else:
            raise ValueError(f"Unsupported experiment number: {self.experiment}")

    def _make_sequence(self):
        """Create and configure the sequence for the current episode."""
        self.real_targets.clear()
        self.real_actions.clear()

        if self.flag == 'eval':
            self.ep_len = self.eval_len
            self.ep_start = self.eval_start
        else:
            self._configure_training_episode()

        self.start_date = pd.to_datetime(self.df_raw.index[self.ep_start])
        self.sequence = copy.deepcopy(self.df_scaled[self.ep_start:self.ep_start + self.model_args.seq_len])
        self.in_features = self.sequence.shape[1]
        self._setup_initial_state_and_data()

    def _configure_training_episode(self):
        """Configures episode length and start for training based on episode settings."""
        self.ep_len = np.random.randint(self.min_el, self.max_el) if self.random_episode_length else self.const_el
        if self.random_episode_start:
            self.ep_start = np.random.randint(0, len(self.df_raw) - self.model_args.seq_len - self.ep_len)
        else:
            ep_locs = range(0, len(self.df_raw) - self.model_args.seq_len - self.ep_len + 1, self.ep_len)
            self.ep_start = ep_locs[self.round]

    def _setup_initial_state_and_data(self):
        """Sets up initial observation state and extracts data for the current episode."""
        self.obs = copy.deepcopy(self.sequence[-self.max_delay:, self.n_actions:] if not self.const_delay else self.sequence[-self.max_delay:, :])

        seq_end = self.ep_start + self.model_args.seq_len
        ep_end = seq_end + self.ep_len

        self.real_targets = list(self.df_raw.iloc[seq_end:ep_end + 1, self.target_idx].values)
        self.real_actions = list(self.df_raw.iloc[seq_end:ep_end + 1, :self.n_actions].values)
        self.q_ep = list(self.q_tank.iloc[seq_end:ep_end + 1])
        self.real_rewards_ep = list(self.real_rewards[seq_end:ep_end + 1])
        self.max_delays_metal_ep = list(self.max_delays_metal[seq_end:ep_end + 1])
        # Uncomment if needed: self.max_delay = np.max(self.max_delays_metal_ep)

    def _scale_data(self, df):
        """Scales data according to the model configuration."""
        arr = np.zeros((df.shape[0], self.model_args.in_features))
        feature_cols = df.iloc[:, :self.num_cols]
        time_cols = df.iloc[:, self.num_cols:]
        
        if self.model_args.model == 'LSTM':
            arr[:, :self.num_cols] = self.feature_scaler.transform(feature_cols)
            if self.model_args.time_scaled == 'Scaled':
                arr[:, self.num_cols:] = self.time_scaler.transform(time_cols)
            else:
                arr[:, self.num_cols:] = time_cols.values  # Pass through time columns unscaled
        else:
            arr = self.scaler.transform(df)
        
        return arr

    def _inverse_transform(self, arr):
        """Reverses the scaling of data according to the model configuration."""
        if self.model_args.model == 'LSTM':
            arr[:, :self.num_cols] = self.feature_scaler.inverse_transform(arr[:, :self.num_cols])
            if self.model_args.time_scaled == 'Scaled':
                arr[:, self.num_cols:] = self.time_scaler.inverse_transform(arr[:, self.num_cols:])
            else:
                arr[:, self.num_cols:] = arr[:, self.num_cols:]  # Pass through unscaled
        else:
            arr = self.scaler.inverse_transform(arr)
        
        return arr

    def _make_df(self, arr, step):
        """Converts array to DataFrame, applies time features if needed."""
        start = self.start_date
        first_date = start + pd.Timedelta(minutes=step * self.freq_min)
        index = pd.date_range(start=first_date, periods=self.model_args.seq_len, freq='T')  # Assuming 'T' for minute frequency
        df = pd.DataFrame(arr, index=index, columns=self.columns)
        
        if self.model_args.model == 'LSTM' and self.model_args.embed == 'timeF':
            df = self._add_time_specs(df)
        
        # Keep track of dates for reference
        self.dates.append([first_date, df.index[-1]])
        return df

    def _add_time_specs(self, df):
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

    def _state_predictor(self):
        """Predicts the next state using the model."""
        df_forecast = copy.deepcopy(self.sequence)  # Deep copy to avoid modifying the original data

        if self.model_args.model == 'LSTM':
            # LSTM model prediction flow
            forecasted = self.exp.predict(self.model_args.setting, df_forecast, save=False)
            forecasted = np.reshape(forecasted[0, :], (1, self.model_args.out_features))
        else:
            # Other models prediction flow
            if self.scaled:
                df_forecast = self._scale_data(df_forecast)
            forecasted = self.exp.predict(df_forecast, self.model_args.setting)
        
        # Inverse transform if data was scaled and model is LSTM or if forecasted is not scaled
        if not self.scaled or self.model_args.model == 'LSTM':
            forecasted = self._inverse_transform(forecasted)

        return forecasted

    def _normalize_predictions(self, state):
        """Normalizes predictions to be within the feature range set by df_info."""
        min_vals = self.df_info.loc['min'].values
        max_vals = self.df_info.loc['max'].values

        # Apply clipping based on min/max values to ensure predictions are within the allowable range
        state = np.clip(state, min_vals + 0.0001, max_vals - 0.0001)

        return state

    def _calculate_reward(self, data='env'):
        """Determine the appropriate reward calculation method based on the data context."""
        if data == 'plant':
            return self._calculate_plant_data_rewards()
        else:
            return self._calculate_env_data_rewards()

    def _calculate_plant_data_rewards(self):
        """Calculate rewards for the whole dataset within a plant environment."""
        # Extract the target column from the dataset
        target_column = self.df_raw.iloc[:, self.target_idx]
        # Calculate the absolute deviations from the ideal target level
        deviations = np.abs(target_column - self.ideal_target_level)
        max_deviation = deviations.max()
        # Normalize deviations to have a relative scale
        norm_deviation = deviations / max_deviation

        # Calculate the tax based on phosphorus levels and flow rates
        target_kg = target_column * self.q_tank * self.freq_min / (60 * 1000)
        taxes = self.green_tax * target_kg
        # Normalize the tax to be between 0 and 1
        norm_tax = (taxes - taxes.min()) / (taxes.max() - taxes.min())

        # Calculate costs for chemicals used based on flow rates
        JSF_L = self.df_raw.iloc[:, 0] * self.freq_min / 60
        PAX_L = self.df_raw.iloc[:, 2] * self.freq_min / 60
        JSF_costs = self.JSF_price * JSF_L
        PAX_costs = self.PAX_price * PAX_L
        # Normalize costs for actions
        norm_JSF = (JSF_costs - JSF_costs.min()) / (JSF_costs.max() - JSF_costs.min())
        norm_PAX = (PAX_costs - PAX_costs.min()) / (PAX_costs.max() - PAX_costs.min())
        norm_action_cost = norm_JSF + norm_PAX

        # Combine the weighted costs and penalties to form the overall reward
        real_rewards = - (self.weight_target * norm_deviation +
                          self.weight_tax * norm_tax +
                          self.weight_action * norm_action_cost)

        return np.array(real_rewards)

    def _calculate_env_data_rewards(self):
        """Calculate rewards based on the latest data in a non-plant environment."""
        # Calculate phosphorus deviation and normalize it
        target_deviation = abs(self.targets[-1] - self.ideal_target_level)
        norm_deviation = target_deviation / self.max_deviation
        
        # Update phosphorus load and calculate tax
        target_kg = self.targets[-1] * self.q_ep[self.round] * self.freq_min / (60 * 1000)
        tax = self.green_tax * target_kg
        tax_min, tax_max = min(self.taxes), max(self.taxes)
        # Normalize the tax to avoid scaling issues
        norm_tax = (tax - tax_min) / (tax_max - tax_min) if tax_max != tax_min else 0

        # Calculate the costs for actions taken and normalize them
        JSF_L = self.actions[-1][0] * self.freq_min / 60
        PAX_L = self.actions[-1][2] * self.freq_min / 60
        JSF_cost = self.JSF_price * JSF_L
        PAX_cost = self.PAX_price * PAX_L
        JSF_min, JSF_max = min(self.JSF_costs), max(self.JSF_costs)
        PAX_min, PAX_max = min(self.PAX_costs), max(self.PAX_costs)
        norm_JSF = (JSF_cost - JSF_min) / (JSF_max - JSF_min) if JSF_max != JSF_min else 0
        norm_PAX = (PAX_cost - PAX_min) / (PAX_max - PAX_min) if PAX_max != PAX_min else 0
        norm_action_cost = norm_JSF + norm_PAX
        
        # Combine all normalized values to calculate the reward
        reward = - (self.weight_target * norm_deviation +
                    self.weight_tax * norm_tax +
                    self.weight_action * norm_action_cost)

        return reward

    # HINT: Step function for each state of the training
    def step(self, action):
        """Executes one time step within the environment based on the given action."""
        self.round += 1
        info = {'round': self.round}

        # Predict the next state using the model, ensure no negative values
        state = self._state_predictor()
        state[state < 0] = 0

        # Prepare new state array and incorporate actions
        new_state = np.zeros((1, self.num_cols))
        new_state[0, self.n_actions:self.n_actions+self.n_ind] = self.df_scaled[
            self.ep_start + self.model_args.seq_len + self.round, self.n_actions:self.n_actions+self.n_ind]
        new_state[0, :self.n_actions] = action

        # Update the sequence of observations in the environment
        self.sequence = np.vstack([self.sequence[1:], new_state])  # Append new state and remove the oldest
        self.sequence = self._make_df(self.sequence, self.round)

        # Update the observation space using the latest sequence data
        self.obs = self.sequence.iloc[-self.max_delay:].values if self.const_delay else \
            self.sequence.iloc[-self.max_delay:, self.n_actions:].values.squeeze()

        # Calculate the reward, update rewards history
        reward = self._calculate_reward()
        self.rewards.append(reward)

        # Check if the episode has ended
        done = self.round >= self.ep_len

        # Render the environment's current state
        self.render()

        return (self.obs, reward, done, info)

    # HINT: Reset function for reseting to the initial state
    def reset(self, flag='train', eval_len=10, eval_start='2022-08-01'):
        """Resets the environment for a new episode, setting up for new training or evaluation."""
        # Clear historical data for the new episode
        self.sequences, self.observations, self.rewards, self.targets, self.actions = [], [], [], [], []
        self.flag = flag
        self.episode_num += 1
        self.round = 0

        # Setup evaluation start time if in eval mode
        if flag == 'eval':
            timezone = self.df_raw.index.tz
            eval_datetime = pd.to_datetime(eval_start).tz_localize(timezone)
            self.eval_start = self.df_raw.index.get_loc(eval_datetime)
            self.eval_len = eval_len

        # Set experiment specifics and make the initial sequence
        self._set_experiment()
        self._make_sequence()

        # Append the initial sequence and observation
        self.sequences.append(self.sequence)
        self.observations.append(self.obs)

        # Transform the last observation and set initial targets and actions
        last_obs = self.obs[-1, :] if self.const_delay else self.obs
        obs_inv = self._inverse_transform(copy.deepcopy(last_obs.reshape(1, -1)))
        self.targets.append(obs_inv[0, -1-self.num_time_f])
        self.actions.append(obs_inv[0, :self.n_actions])
        self.rewards.append(0)  # Initialize the first reward

        # Close any existing plot and setup a new one for the episode
        if self.figure:
            plt.close(self.figure)
        self.figure, axes = plt.subplots(self.n_actions + 2, 1, figsize=(6.5, 6.5), constrained_layout=True, sharex=True)
        self.observation_ax = axes[0]
        self.action_axes = axes[1:-1]
        self.reward_ax = axes[-1]
        self.colors = [color for color in mcolors.TABLEAU_COLORS.values()]
        for i, ax in enumerate(self.action_axes):
            ax.set_title(f'Action {i} History')

        return self.obs

    # HINT: Render function to visualize and print
    def render(self, mode='live'):
        """Handles the rendering of the environment based on the mode specified."""
        if mode == 'live':
            self._render_live()
        elif mode == 'not_live':
            self._render_not_live()
        else:
            self._print_environment_status()

    def _render_live(self):
        """Render the environment live with plots updating in real-time."""
        self._plot_live()
        self.figure.canvas.draw()
        plt.show()
        plt.pause(0.01)  # Small pause to ensure the plot updates

        if self.round == self.ep_len:
            self._finalize_plot('Live_plot')
            self._plot_env_real()
            plt.close('all')
    
    def _initialize_plots(self):
        """Initialize the figure and axes for plotting."""
        fig, axes = plt.subplots(self.n_actions + 2, 1, figsize=(6.5, 6.5),
                                 constrained_layout=True, sharex=True)
        self.observation_ax = axes[0]
        self.action_axes = axes[1:-1]
        self.reward_ax = axes[-1]
        return fig

    def _plot_live(self):
        """Live plot for each episode."""
        if self.figure is None:
            self.figure = self._initialize_plots()  # Initialize plots if not yet created

        self.figure.suptitle(f'Environment Visualization for {self.title}')
        self._update_plot(self.observation_ax, self.targets, 'Observation (P)')

        for i, ax in enumerate(self.action_axes):
            self._update_plot(ax, [arr[i] for arr in self.actions],
                              label=f'{self.df_raw.columns[i]}')

        self._update_plot(self.reward_ax, self.rewards, 'Reward', xlabel='Minutes')

    def _plot_env_real(self):
        """Live plot for each episode with the plant data."""
        if self.figure is None:
            self.figure = self._initialize_plots()

        self.figure.suptitle(f'Environment Visualization for {self.title}')
        self._update_plot(self.observation_ax, self.targets, 'Observation (P)',
                          additional_data=self.real_targets, additional_label='Plant')

        for i, ax in enumerate(self.action_axes):
            self._update_plot(ax, [arr[i] for arr in self.actions],
                              label=f'{self.df_raw.columns[i]}',
                              additional_data=[arr[i] for arr in self.real_actions], additional_label='Plant')

        self._update_plot(self.reward_ax, self.rewards, 'Reward', xlabel='Minutes',
                          additional_data=self.real_rewards_ep, additional_label='Plant')

        if self.flag == 'eval' and self.round == self.ep_len:
            self._finalize_and_save_plot(self.figure, 'Live_plot_with_actual_values')

    def _update_plot(self, ax, data, label, xlabel='', additional_data=None, additional_label=''):
        """Update plot for given axes."""
        ax.clear()
        ax.plot(data, 'x-', color=self.colors[0], label='Agent')
        if additional_data is not None:
            ax.plot(additional_data, '--', color='black', label=additional_label, linewidth=1.2)
        ax.set_ylabel(label, labelpad=10)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.legend()

    def _finalize_and_save_plot(self, fig, fig_name):
        """Finalize plots and save to file."""
        for ax in fig.get_axes():
            ax.grid(visible=True, which='major', color='gray', linewidth=0.0025)
            ax.grid(visible=True, which='minor', color='gray', linewidth=0.0025)
        fig.savefig(f'{self.fig_folder}{fig_name}.pdf', dpi=300)

    def _render_not_live(self):
        """Render the final state of the environment for the episode."""
        fig_name = 'Eval_Live_plot' if self.flag == 'eval' else 'Live_plot'
        self._plot_live()
        self._finalize_and_save_plot(fig_name)
        self._plot_env_real()
        plt.close('all')

    def _print_environment_status(self):
        """Prints current environment status for non-visual modes."""
        print(f'Round: {self.round}, Metal Amount: {self.actions[-1][0]:0.3f}, Phosphorus: {self.targets[-1]:0.3f}')
        print(f'Reward Received: {self.rewards[-1]}, Total Reward: {self.collected_reward}')
        print("==============================================================")
    
    def seed(self, seed=None):
        """Seed the random number generator for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        """Clean up resources used by the environment (if any)."""
        pass
