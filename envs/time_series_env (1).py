"""
Created on April 26 2024
@author: Esmaeel Mohammadi (esm@kruger.dk; esmo@bio.aau.dk; https://github.com/esmaeelMhd)

# =============================================================================
# This script is used to create an DRL environment for Time Series models
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
from dataclasses import dataclass, field
from typing import List, Union, Any, Optional

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.axes as axes
import torch.nn as nn
import torch
import gym
from gym import spaces

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
    """
    A class to hold arguments for configuring model data.

    Attributes:
        data_root_path (str): Root directory for datasets.
        data_name (str): Path to the data file within the root directory.
        model_type (str): Type of the trained model.
        model_name (str): Folder of the saved model.
        checkpoint (str): Checkpoint file name for the saved model.
        scale (bool): Indicates whether the model has been trained on scaled data.
        ctrl_vars (Union[List[str], str]): Control variables for the model, can be either a list of strings or a single string.
        ind_vars (Union[List[str], str]): Independent variables for model analysis, can be either a list of strings or a single string.
        target_vars (Union[List[str], str]): Target variables for the model.
        obs_vars (Union[List[str], str]): Variables considered in the observation space, 
            defaults to ['target_vars']. Combines 'ctrl_vars', 'ind_vars', and 'target_vars'.
        time_f (bool): Indicates whether time features (hour, day, week, etc.) are used in the dataset, defaults to False.
        num_time_f (Optional[int]): Number of time features used; required if time_f is True.
        time_scaled (str): Describes whether time features are scaled, with options 'Scaled' or 'Unscaled', defaults to 'Unscaled'.
        use_multi_gpu (bool): Specifies whether multiple GPUs are used for simulation, defaults to False.

    Raises:
        ValueError: If `time_f` is True and either `num_time_f` or `time_scaled` is not provided.
                    If any of the variables list (`ctrl_vars`, `ind_vars`, `target_vars`, `obs_vars`)
                        contains non-string items when expected to be a list of strings.
    """
    data_root_path: str
    data_name: str
    model_type: str
    model_name: str
    checkpoint: str
    scale: bool
    ctrl_vars: Union[List[str], str]
    ind_vars: Union[List[str], str]
    target_vars: Union[List[str], str]
    obs_vars: Union[List[str], str] = field(default_factory=lambda: ['target_vars'])
    time_f: bool = False
    num_time_f: Optional[int] = None
    time_scaled: str = 'Unscaled'
    use_multi_gpu: bool = False

    def __post_init__(self, args, **kwargs):
        """Post-initialization to validate and process the dataclass fields."""
        # Sets the attributes' values according to the input data provided.
        if args:
           for field_name in self.__dataclass_fields__:
               if hasattr(args, field_name):
                   setattr(self, field_name, getattr(args, field_name))
       
        # Override with kwargs if available
        for key, value in kwargs.items():
            if key in self.__dataclass_fields__:
                setattr(self, key, value)
               
        self._validate_and_convert_vars()
        self._process_observation_variables()
        self._validate_time_features()

    def _validate_and_convert_vars(self):
        """Validates and converts control, independent, and target variables into lists if they are not already."""
        for attr_name in ['ctrl_vars', 'ind_vars', 'target_vars', 'obs_vars']:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str):
                setattr(self, attr_name, [attr_value])
            elif not all(isinstance(x, str) for x in attr_value):
                raise ValueError(f"{attr_name} must be a list of strings or a single string")

    def _process_observation_variables(self):
        """Processes observation variables by expanding placeholders."""
        expanded_obs_vars = []
        for var in self.obs_vars:
            if var == 'ctrl_vars':
                expanded_obs_vars.extend(self.ctrl_vars)
            elif var == 'ind_vars':
                expanded_obs_vars.extend(self.ind_vars)
            elif var == 'target_vars':
                expanded_obs_vars.extend(self.target_vars)
            else:
                expanded_obs_vars.append(var)
        self.obs_vars = list(set(expanded_obs_vars))

    def _validate_time_features(self):
        """Validates the provision of required fields when time features are used."""
        if self.time_f and (self.num_time_f is None or self.time_scaled is None):
            raise ValueError("num_time_f and time_scaled must be provided if time_f is True")
    
@dataclass
class AgentArgs:
    """
    A class to hold configuration parameters for an agent in various experiments.
    
    Parameters:
    ----------
    agent_name : str, required
        The name of the agent which is being trained.
    experiment : int, required
        Specifies the experiment number. The experiments are:
            1: constant episode start, constant episode length
            2: constant episode start, random episode length
            3: random episode start, constant episode length
            4: random episode start, rando, episode length
        Based on the experiment number, additional parameters might be required:
        - Experiment 1 or 3: `const_el` must be provided.
        - Experiment 2 or 4: Both `min_el` and `max_el` must be provided.
    const_el : Optional[int], optional, default=None
        A constant episode length required for experiments 1 and 3.
    min_el : Optional[int], optional, default=None
        The minimum episode length required for experiments 2 and 4.
    max_el : Optional[int], optional, default=None
        The maximum episode length required for experiments 2 and 4.
    delay_type : str, optional, default='none'
        Type of delay applied in the experiment. If 'constant', `const_delay` must be provided.
    const_delay : Optional[int], optional, default=None
        The maximum delay value required if `delay_type` is 'constant'.
    title : str, optional, default='Agent'
        Title for the agent, used primarily for labeling and identification purposes.
    norm_values : bool, optional, default=True
        It determines whether to normalize the observations, actions and rewards or not.
    
    Raises:
    ------
    ValueError
        If required conditions based on `experiment` or `delay_type` are not met.
    """
    agent_name: str
    experiment: int
    const_el: Optional[int] = None
    min_el: Optional[int] = None
    max_el: Optional[int] = None
    delay_type: str = 'none'
    const_delay: Optional[int] = None
    title: str = 'Agent'
    norm_values: bool = True
        
    def __post_init__(self, args=None, **kwargs):
        """Initializes the class attributes when instancing."""
        # Initialize from args if available
        if args:
            for field_name in self.__dataclass_fields__:
                if hasattr(args, field_name):
                    setattr(self, field_name, getattr(args, field_name))
                    
        if (self.experiment==1 or self.experiment==3) and self.const_el is None:
            raise ValueError("const_el must be provided if the experiment is 1 or 3")
        if (self.experiment==2 or self.experiment==4) and (self.min_el is None or self.max_el is None):
            raise ValueError("min_el and max_el must be provided if the experiment is 2 or 4")
        if self.delay_type=='constant' and self.const_delay is None:
            raise ValueError("const_delay must be provided if the delay is constant")

@dataclass(eq=0)    
class TimeSeriesEnv(gym.Env): 
    """
    This environment simulates a time series model within a gym framework.
    
    Attributes:
        model_args (ModelArgs): Contains configuration for the model.
        agent_args (AgentArgs): Contains configuration for the agent.
        reward_function: Reward function for the environment.
        num_envs (int): Number of environments to simulate, default is 1.
        device (str): Device to run the simulation on, e.g., 'cpu' or 'cuda'.
        results_folder (str): Directory path where results and figures will be saved.
        mode (str): Operation mode of the environment, 'live' for real-time plotting or 'not_live' for plotting at the end of each episode.
    
        const_delay (bool): Flag indicating if the delay in the environment is constant.
        const_delay (int): Maximum delay in the environment (only relevant if const_delay is true).
        window_size (int): Size of the window for rolling operations; default is 3.
        figure (plt.Figure): Placeholder for a figure object for plotting.
        observation_ax (axes.Axes): Axis for plotting observations.
        action_axes (List[axes.Axes]): List of axes for plotting actions.
        norm_values (bool): Normalization values for data processing, specific to agent_args.
            It determines whether to normalize the observations, actions and rewards or not.
        improve_epochs (int): Number of epochs for model improvement phase; default is 20.
        dates (List[Any]): List to store date information for indexing.
        scaled_inputs (List[Any]): List of scaled input data.
        targets (List[Any]): List of target values for the model.
        real_targets (List[Any]): List of real target values as observed from the dataset.
        actions (List[Any]): List of actions taken by the agent.
        real_actions (List[Any]): List of real actions performed from the dataset.
        observations (List[Any]): List of observed states.
        rewards (List[Any]): List of rewards obtained.
        real_rewards (List[Any]): List of real rewards observed from the dataset.
        sequences (List[Any]): List of input sequences used for the model.
        colors (List[str]): List of colors for plotting; default colors are taken from matplotlib's TABLEAU_COLORS.
    """
    model_args: ModelArgs
    agent_args: AgentArgs
    df_raw: pd.DataFrame = None
    reward_function: Any
    results_folder: str
    num_envs: int = 1
    device: str = 'cuda'
    mode: str = 'not_live'
    window_size: int = 3
    
    # Initialize the private variables
    figure: plt.Figure = None
    observation_axes: List[axes.Axes] = None
    action_axes: List[axes.Axes] = None
    improve_epochs: int = 20
    dates: List[Any] = None
    scaled_inputs: List[Any] = None
    targets: List[Any] = None
    real_targets: List[Any] = None
    actions: List[Any] = None
    real_actions: List[Any] = None
    observations: List[Any] = None
    rewards: List[Any] = None
    real_rewards: List[Any] = None
    sequences: List[Any] = None
    colors: List[str] = None

    def __post_init__(self):
        '''
        Throughout the environment we have 3 variables getting handles at each step:
            sequence: The input sequence to the model which it is used for the prediction
            state: The predicted one step in the future that the simulation is in it
            obs: The observation which includes one or a history of system states
        '''    
        super().__init__()
        """Initializes visualization-related settings."""
        self.action_axes = []
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        
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
    
        """Initializes methods that setup the environment."""
        self._setup_visualization()
        if self.df_raw is None:
            self._load_dataset()
        self._initialize_data()
        self._load_scalers()
        if not self.scale:
            self.df_scaled = self.df_raw
        else:
            self.df_scaled = self._apply_scalers()
        self._calculate_min_max()
        self._setup_spaces()
        self._load_and_setup_model()
        
    def _setup_visualization(self):
        """Sets up the visualization method in render."""
        if self.mode == 'live':
            plt.switch_backend('qt5agg')
            plt.ion()
    
    def _load_dataset(data_root_path, data_name):
        """Loads the dataset used in training the model."""
        dataset_path = os.path.join(data_root_path, data_name)
        df = pd.read_csv(dataset_path, index_col=["date"], parse_dates=["date"], infer_datetime_format=True)
        df.sort_index(inplace=True)
        df = df.astype('float32').fillna(method='ffill')
    
    def _initialize_data(self):
        """Initializes and prepares the data for the environment."""
            
        self._check_frequency_uniformity()
        
        self.columns = self.df_raw.columns
        self.num_cols = len(self.columns)
        self.freq = self.df_raw.index.to_series().diff().dropna().mode()[0]
        self.freq_min = self.freq.total_seconds() / 60
        self.target_idxs = [self.df_raw.columns.get_loc(col) for col in self.model_args.target_vars]
        self.df_raw = self._add_time_specs(self.df_raw)
        self.df_raw = self.df_raw.astype('float32')
        self.reward_function.data = self.df_raw
        print('### Calculating the reward for dataset.')
        self.reward_function.calculate_reward(source='actual')  # Placeholder for reward calculation
        
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
    
    def _check_frequency_uniformity(self):
        """Checks if the frequncy of the dataset is uniform."""
        time_diffs = self.df_raw.index.to_series().diff()
        is_uniform = (time_diffs == time_diffs.mode()[0]).all()
        if not is_uniform:
            print('ATTENTION: The frequency is not uniform.')
    


    def _calculate_min_max(self):
        """Calculates the minimum and maximum values for targets and control variables."""
        if self.agent_args.norm_values:
            df_scaled = pd.DataFrame(self.df_scaled, columns=self.df_raw.columns)
            self.min_obs_vars, self.max_obs_vars = self._get_min_max_vars(df_scaled, self.model_args.obs_vars)
            self.min_ctrl_vars, self.max_ctrl_vars = self._get_min_max_vars(df_scaled, self.ctrl_vars)
        else:
            self.min_obs_vars, self.max_obs_vars = self._get_min_max_vars(self.df_raw, self.model_args.obs_vars)
            self.min_ctrl_vars, self.max_ctrl_vars = self._get_min_max_vars(self.df_raw, self.ctrl_vars)
    
        if self.agent_args.const_delay:
            self.min_obs_vars = np.tile(self.min_obs_vars, (self.agent_args.const_delay, 1))
            self.max_obs_vars = np.tile(self.max_obs_vars, (self.agent_args.const_delay, 1))
    
    def _get_min_max_vars(self, df, variables):
        """Helper function to get min and max values for a list of variables."""
        min_values = df[variables].min().tolist()
        max_values = df[variables].max().tolist()
        return min_values, max_values
    
    def _setup_spaces(self):        
        """Defines the action space based on control variables' min and max values."""
        self.action_space = spaces.Box(
            low=np.array(self.min_ctrl_vars, dtype=np.float32),
            high=np.array(self.max_ctrl_vars, dtype=np.float32),
            dtype=np.float32
        )
    
        """Defines the observation space based on the environment settings."""
        num_columns = len(self.df_raw.columns)
        if self.agent_args.const_delay:
            # Here the observation space includes a delay dimension
            self.observation_space = spaces.Box(
                low=np.float32(self.min_obs_vars),
                high=np.float32(self.max_obs_vars),
                shape=(self.agent_args.const_delay, len(self.model_args.obs_vars)),
                dtype=np.float32
            )
        else:
            # Standard observation space without considering delay
            self.observation_space = spaces.Box(
                low=np.float32(self.min_obs_vars),
                high=np.float32(self.max_obs_vars),
                dtype=np.float32
            )

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
        self.obs = copy.deepcopy(self.sequence[-self.agent_args.const_delay:, self.n_actions:] if not self.agent_args.const_delay else self.sequence[-self.agent_args.const_delay:, :])

        seq_end = self.ep_start + self.model_args.seq_len
        ep_end = seq_end + self.ep_len

        self.real_targets = list(self.df_raw.iloc[seq_end:ep_end + 1, self.target_idxs].values)
        self.real_actions = list(self.df_raw.iloc[seq_end:ep_end + 1, :self.n_actions].values)
        self.q_ep = list(self.q_tank.iloc[seq_end:ep_end + 1])
        self.real_rewards_ep = list(self.real_rewards[seq_end:ep_end + 1])
        self.agent_args.const_delays_metal_ep = list(self.agent_args.const_delays_metal[seq_end:ep_end + 1])
        # Uncomment if needed: self.agent_args.const_delay = np.max(self.agent_args.const_delays_metal_ep)

    def _make_df(self, arr, step):
        """Converts array to DataFrame, applies time features if needed."""
        start = self.start_date
        first_date = start + pd.Timedelta(minutes=step * self.freq_min)
        index = pd.date_range(start=first_date, periods=self.model_args.seq_len, freq=self.freq)
        df = pd.DataFrame(arr, index=index, columns=self.columns)
        
        if self.model_args.model_type == 'LSTM' and self.model_args.embed == 'timeF':
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

        if self.model_args.model_type == 'LSTM':
            # LSTM model prediction flow
            forecasted = self.exp.predict(self.model_args.model_name, df_forecast, save=False)
            forecasted = np.reshape(forecasted[0, :], (1, self.model_args.out_features))
        else:
            # Other models prediction flow
            if self.agent_args.norm_values:
                df_forecast = self._scale_data(df_forecast)
            forecasted = self.exp.predict(df_forecast, self.model_args.model_name)
        
        # Inverse transform if data was scaled and model is LSTM or if forecasted is not scaled
        if not self.agent_args.norm_values or self.model_args.model_type == 'LSTM':
            forecasted = self._inverse_transform(forecasted)

        return forecasted

    def _normalize_predictions(self, state):
        """Normalizes predictions to be within the feature range set by df_info."""
        min_vals = self.df_info.loc['min'].values
        max_vals = self.df_info.loc['max'].values

        # Apply clipping based on min/max values to ensure predictions are within the allowable range
        state = np.clip(state, min_vals + 0.0001, max_vals - 0.0001)

        return state

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
        self.obs = self.sequence.iloc[-self.agent_args.const_delay:].values if self.agent_args.const_delay else \
            self.sequence.iloc[-self.agent_args.const_delay:, self.n_actions:].values.squeeze()

        # Calculate the reward, update rewards history
        reward = self.reward_function._calculate_reward()
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
        last_obs = self.obs[-1, :] if self.agent_args.const_delay else self.obs
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
        fig.savefig(f'{self.results_folder}{fig_name}.pdf', dpi=300)

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
