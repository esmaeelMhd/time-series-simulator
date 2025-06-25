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
import logging
import copy
from dataclasses import dataclass, field, fields
from typing import List, Union, Any, Optional

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.axes as axes
import gym
from gym import spaces
import torch

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
    ctrl_vars: Union[List[str], str]
    ind_vars: Union[List[str], str]
    target_vars: Union[List[str], str]
    ctrl_names: Union[List[str], str] = None
    target_names: Union[List[str], str] = None
    obs_vars: Union[List[str], str] = field(default_factory=lambda: ['target_vars'])
    in_features: int = None
    seq_len: int = 240
    scale: bool = True
    index_col: str = 'date'
    time_f: bool = None
    has_time_f: bool = None
    num_time_f: Optional[int] = None
    time_scaled: str = None
    use_multi_gpu: bool = False

    def __post_init__(self):
        """Post-initialization to validate and process the dataclass fields."""     
        # Setup control and independent variables
        self.ctrl_vars = self._parse_variables(['ctrl_vars', 'control_variable'])
        self.ind_vars = self._parse_variables(['ind_vars', 'independent_vars'])
               
        self._validate_and_convert_vars()
        self._process_observation_variables()
        self._validate_time_features()
        
        if self.in_features is None:
            self.in_features = len(self.ctrl_vars) + len(self.ind_vars) + len(self.target_vars) + self.num_time_f
    
    @staticmethod
    def from_namespace(args, **kwargs):
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(ModelArgs))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        
        # Override filtered_args with manually set attributes from kwargs
        filtered_args.update(kwargs)
        
        return ModelArgs(**filtered_args)

    def _parse_variables(self, attribute_names: List[str]) -> List[str]:
        """Parses control or independent variables from model_args using possible attribute names."""
        for attr in attribute_names:
            value = getattr(self, attr, None)
            if value is not None:
                if isinstance(value, str):
                    return [value]
                elif isinstance(value, list):
                    return value
        return []  # Return an empty list if no attributes match   
    
    def _validate_and_convert_vars(self):
        """Validates and converts control, independent, and target variables into lists if they are not already."""
        for attr_name in ['ctrl_vars', 'ind_vars', 'target_vars', 'obs_vars']:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str):
                setattr(self, attr_name, [attr_value])
            elif not all(isinstance(x, str) for x in attr_value):
                raise ValueError(f"{attr_name} must be a list of strings or a single string")
        
        if self.ctrl_names is None:
            self.ctrl_names = self.ctrl_vars
            
        if self.target_names is None:
            self.target_names = self.target_vars

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
        """Validates the provision of required fields."""
        if self.time_f and self.has_time_f is None:
            raise ValueError("has_time_f must be provided if time_f is True")
            
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
    agent_name: str = None
    experiment: int = None
    const_el: Optional[int] = None
    min_el: Optional[int] = None
    max_el: Optional[int] = None
    delay_type: str = 'none'
    const_delay: Optional[int] = None
    title: str = 'Agent'
    norm_values: bool = True
    obs_history: int = 1
        
    def __post_init__(self):
        """Perform validation checks based on experiment and delay_type."""                
        if (self.experiment==1 or self.experiment==3) and self.const_el is None:
            raise ValueError("const_el must be provided if the experiment is 1 or 3")
        if (self.experiment==2 or self.experiment==4) and (self.min_el is None or self.max_el is None):
            raise ValueError("min_el and max_el must be provided if the experiment is 2 or 4")
        if self.delay_type=='constant' and self.const_delay is None:
            raise ValueError("const_delay must be provided if the delay is constant")
    
    @staticmethod
    def from_namespace(args, **kwargs):
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(AgentArgs))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        
        # Override filtered_args with manually set attributes from kwargs
        filtered_args.update(kwargs)
        
        return AgentArgs(**filtered_args)

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
    model: Any
    exp: Any
    scaler_handler: Any
    reward_function: Any
    df_raw: pd.DataFrame = None
    results_folder: str = './results'
    num_envs: int = 1
    device: str = 'cuda'
    mode: str = 'not_live'
    window_size: int = 3
    use_gpu: bool = True
    
    # Initialize the private variables
    figure: plt.Figure = None
    target_axes: List[axes.Axes] = None
    action_axes: List[axes.Axes] = None
    episode_num: int = 0
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
        
        # Setup the device based on GPU usage
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.use_gpu else torch.device('cpu')
        if self.device == 'cuda':
            self.device_ids = ','.join(str(i) for i in range(torch.cuda.device_count()))
            self.use_multi_gpu = torch.cuda.device_count() > 1
        else:
            self.device_ids = ''
            self.use_multi_gpu = False  
    
        """Initializes methods that setup the environment."""
        self._setup_visualization()
        if self.df_raw is None:
            self._load_dataset()
        self._initialize_data()
        if not self.model_args.scale:
            self.df_scaled = self.df_raw
        else:
            self.df_scaled = self.scaler_handler.scale_data(self.df_raw)
            
        self._calculate_min_max()
        self._setup_spaces()
        
        """Setup the figure size."""
        self.figsize = (6.5, len(self.model_args.target_vars) + len(self.model_args.ctrl_vars) + 2)
        
    def _setup_visualization(self):
        """Sets up the visualization method in render."""
        if self.mode == 'live':
            plt.switch_backend('qt5agg')
            plt.ion()
    
    def _load_dataset(self):
        """Loads the dataset used in training the model."""
        print('Loading the dataset ...')
        dataset_path = os.path.join(self.model_args.data_root_path, self.model_args.data_name)
        self.df_raw = pd.read_csv(dataset_path, index_col=self.model_args.index_col, 
                                  parse_dates=[self.model_args.index_col])
        self.df_raw.sort_index(inplace=True)
        self.df_raw = self.df_raw.astype('float32').ffill()
    
    def _initialize_data(self):
        """Initializes and prepares the data for the environment."""            
        self._check_frequency_uniformity()
        
        self.num_cols = len(self.df_raw.columns) if not self.model_args.has_time_f else len(self.df_raw.columns) - self.model_args.num_time_f
        self.freq = self.df_raw.index.to_series().diff().dropna().mode()[0]
        self.freq_min = self.freq.total_seconds() / 60
        
        self.target_idxs = [self.df_raw.columns.get_loc(col) for col in self.model_args.target_vars]
        self.obs_idxs = [self.df_raw.columns.get_loc(col) for col in self.model_args.obs_vars]
        self.action_idxs = [self.df_raw.columns.get_loc(col) for col in self.model_args.ctrl_vars]
        self.ind_idxs = [self.df_raw.columns.get_loc(col) for col in self.model_args.ind_vars]
        self.time_idxs = [col for col in range(self.model_args.in_features-self.model_args.num_time_f, self.model_args.in_features)]
        
        # TODO: add diferent combinations of time features
        if not self.model_args.has_time_f:
            self.df_raw = self._add_time_specs(self.df_raw)
            
        self.df_raw = self.df_raw.astype('float32')
        self.columns = self.df_raw.columns
        
        # Set the number of time features and actions
        self.n_actions = len(self.model_args.ctrl_vars)
        self.n_ind = len(self.model_args.ind_vars)
        self.n_targets = len(self.model_args.target_vars)
        self.n_obs = len(self.model_args.obs_vars)
        
        # Calculating the reward for the whole actual dataset
        self.reward_function.data = self.df_raw
        print('Calculating the reward for dataset ...')
        self.real_rewards = self.reward_function.calculate_reward(source='actual')
    
    def _check_frequency_uniformity(self):
        """Checks if the frequncy of the dataset is uniform."""
        time_diffs = self.df_raw.index.to_series().diff()
        is_uniform = (time_diffs.iloc[1:] == time_diffs.mode()[0]).all()
        if not is_uniform:
            print('ATTENTION: The frequency is not uniform.')

    def _calculate_min_max(self):
        """Calculates the minimum and maximum values for targets and control variables."""
        if self.agent_args.norm_values:
            df_scaled = pd.DataFrame(self.df_scaled, columns=self.df_raw.columns)
            self.min_obs_vars, self.max_obs_vars = self._get_min_max_vars(df_scaled, self.model_args.obs_vars)
            self.min_ctrl_vars, self.max_ctrl_vars = self._get_min_max_vars(df_scaled, self.model_args.ctrl_vars)
        else:
            self.min_obs_vars, self.max_obs_vars = self._get_min_max_vars(self.df_raw, self.model_args.obs_vars)
            self.min_ctrl_vars, self.max_ctrl_vars = self._get_min_max_vars(self.df_raw, self.model_args.ctrl_vars)
    
        if self.agent_args.obs_history > 1:
            self.min_obs_vars = np.tile(self.min_obs_vars, (self.agent_args.obs_history, 1))
            self.max_obs_vars = np.tile(self.max_obs_vars, (self.agent_args.obs_history, 1))
    
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
        # 2D array if considering obs history
        if self.agent_args.obs_history > 1:
            # Here the observation space includes a delay dimension
            self.observation_space = spaces.Box(
                low=np.float32(self.min_obs_vars),
                high=np.float32(self.max_obs_vars),
                shape=(self.agent_args.obs_history, len(self.model_args.obs_vars)),
                dtype=np.float32
            )
        # 1D array if not considering history
        else:
            # Standard observation space without considering delay
            self.observation_space = spaces.Box(
                low=np.array(self.min_obs_vars, dtype=np.float32),
                high=np.array(self.max_obs_vars, dtype=np.float32),
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
        if self.agent_args.experiment in experiment_config:
            self.random_episode_start, self.random_episode_length = experiment_config[self.agent_args.experiment]
        else:
            raise ValueError(f"Unsupported experiment number: {self.agent_args.experiment}")

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
        self.sequence = np.array(copy.deepcopy(self.df_scaled[self.ep_start:self.ep_start + self.model_args.seq_len]))
        self._setup_initial_state_and_data()

    def _configure_training_episode(self):
        """Configures episode length and start for training based on episode settings."""
        self.ep_len = np.random.randint(self.agent_args.min_el, self.agent_args.max_el) if self.random_episode_length else self.agent_args.const_el
        if self.random_episode_start:
            self.ep_start = np.random.randint(0, len(self.df_raw) - self.model_args.seq_len - self.ep_len)
        else:
            ep_locs = range(0, len(self.df_raw) - self.model_args.seq_len - self.ep_len + 1, self.ep_len)
            self.ep_start = ep_locs[self.round]

    def _setup_initial_state_and_data(self):
        """Sets up initial observation state and extracts data for the current episode."""
        self.state = copy.deepcopy(self.sequence[-1, :] if not self.agent_args.obs_history > 1 else \
                                   self.sequence[-self.agent_args.obs_history:, :])
            
        self.obs = copy.deepcopy(self.sequence[-1, self.obs_idxs].reshape(-1) if not self.agent_args.obs_history > 1 else \
                                 self.sequence[-self.agent_args.obs_history:, self.obs_idxs])

        seq_end = self.ep_start + self.model_args.seq_len
        ep_end = seq_end + self.ep_len

        self.real_targets = list(self.df_raw.iloc[seq_end:ep_end + 1, self.target_idxs].values)
        self.real_actions = list(self.df_raw.iloc[seq_end:ep_end + 1, :self.n_actions].values)
        self.real_rewards_ep = list(self.real_rewards[seq_end:ep_end + 1])

    def _make_df(self, arr, step):
        """Converts array to DataFrame, applies time features if needed."""
        start = self.start_date
        first_date = start + pd.Timedelta(minutes=step * self.freq_min)
        index = pd.date_range(start=first_date, periods=self.model_args.seq_len, freq=self.freq)
        df = pd.DataFrame(arr, index=index, columns=self.columns)
        
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
        # TODO: modify this for custom models
        """Predicts the next state using the model."""
        df_forecast = copy.deepcopy(self.sequence)  # Deep copy to avoid modifying the original data

        if self.model_args.model_type == 'LSTM':
            # LSTM model prediction flow
            forecasted = self.exp.predict(self.model_args.model_name, df_forecast, save=False)
            forecasted = np.reshape(forecasted[0, :], (1, self.n_targets))
        else:
            # Other models prediction flow
            if self.agent_args.norm_values:
                df_forecast = self.scaler_handler.scale_data(df_forecast)
            forecasted = self.exp.predict(df_forecast, self.model_args.model_name)
            
                    
        # Inverse transform if data was scaled and model is LSTM or if forecasted is not scaled
        if not self.agent_args.norm_values:
            temp = np.zeros((forecasted.shape[0], self.num_cols))
            temp[:, self.target_idxs] = forecasted[:, self.target_idxs]
            forecasted = self.scaler_handler.inverse_transform(forecasted)[:, self.target_idxs]

        return forecasted

    def _normalize_predictions(self, state):
        # TODO: see if this is necessary
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

        # Prepare new state array and incorporate actions
        new_state = np.zeros((1, len(self.columns)))
        new_state[0, self.ind_idxs + self.time_idxs] = self.df_scaled.iloc[
            self.ep_start + self.model_args.seq_len + self.round, self.ind_idxs + self.time_idxs]
        new_state[0, self.action_idxs] = action
        new_state[0, self.target_idxs] = state

        # Update the sequence of observations in the environment
        self.sequence = np.vstack([self.sequence[1:], new_state])  # Append new state and remove the oldest
        self.sequence = self._make_df(self.sequence, self.round)

        # Update the observation space using the latest sequence data
        self.obs = self.sequence.iloc[-self.agent_args.obs_history:, self.obs_idxs].values if \
            self.agent_args.obs_history > 1 else self.sequence.iloc[-1, self.obs_idxs].values
        
        # Update the targets history
        if self.agent_args.norm_values:
            new_state = self.scaler_handler.inverse_transform(new_state)
        self.targets.append(new_state[0, self.target_idxs])
        self.actions.append(new_state[0, self.action_idxs])

        # Calculate the reward, update rewards history
        # TODO: adapt the reward function inputs to different scenarios
        reward = self.reward_function.calculate_reward(state=new_state, actions=self.actions, targets=self.targets)
        self.rewards.append(reward)

        # Check if the episode has ended
        done = self.round >= self.ep_len

        # Render the environment's current state
        self.render()
        
        trunc = False

        return (self.obs, reward, done, trunc, info)

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
        last_state = self.state[-1, :] if self.agent_args.obs_history > 1 else self.state
        state_inv = self.scaler_handler.inverse_transform(copy.deepcopy(last_state.reshape(1, -1)))
        self.observations.append(state_inv[0, self.obs_idxs])
        self.targets.append(state_inv[0, self.target_idxs])
        self.actions.append(state_inv[0, self.action_idxs])
        self.rewards.append(self.real_rewards_ep[0])  # Initialize the first reward

        # Close any existing plot and setup a new one for the episode
        if self.figure:
            plt.close(self.figure)
        
        info = {'round': self.round}

        return self.obs, info

    # HINT: Render function to visualize and print
    def render(self):
        """Handles the rendering of the environment based on the mode specified."""
        if self.mode == 'live':
            self._render_live()
        elif self.mode == 'not_live':
            if self.round == self.ep_len:
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
        fig, axes = plt.subplots(self.n_actions + self.n_targets + 1, 1, figsize=self.figsize, 
                                         constrained_layout=True, sharex=True)
        self.target_axes = axes[0: self.n_targets]
        self.action_axes = axes[self.n_targets: -1]
        self.reward_ax = axes[-1]
        return fig

    def _plot_live(self):
        """Live plot for each episode."""
        if self.figure is None:
            self.figure = self._initialize_plots()  # Initialize plots if not yet created

        self.figure.suptitle(f'Environment Visualization for {self.agent_args.title}')
        
        for i, ax in enumerate(self.target_axes):
            self._update_plot(ax, [arr[i] for arr in self.targets], label='', color_idx=1,
                              ylabel=f'{self.model_args.target_names[i]}')

        for i, ax in enumerate(self.action_axes):
            self._update_plot(ax, [arr[i] for arr in self.actions], label='',  color_idx=2,
                              ylabel=f'{self.model_args.ctrl_names[i]}')

        self._update_plot(self.reward_ax, self.rewards, label='', color_idx=3, ylabel='Reward', xlabel='Steps')

    def _plot_env_real(self):
        """Live plot for each episode with the Data data."""
        if self.figure is None:
            self.figure = self._initialize_plots()

        self.figure.suptitle(f'Environment Visualization for {self.agent_args.title}')
        
        for i, ax in enumerate(self.target_axes):
            self._update_plot(ax, [arr[i] for arr in self.targets],
                              label='Simulation',
                              color_idx=1,
                              ylabel=f'{self.model_args.target_names[i]}',
                              additional_data=[arr[i] for arr in self.real_targets], 
                              additional_label='Data')

        for i, ax in enumerate(self.action_axes):
            self._update_plot(ax, [arr[i] for arr in self.actions],
                              label='Agent',
                              color_idx=2,
                              ylabel=f'{self.model_args.ctrl_names[i]}',
                              additional_data=[arr[i] for arr in self.real_actions], 
                              additional_label='Data')

        self._update_plot(self.reward_ax, self.rewards, 'Agent', color_idx=3, ylabel='Reward', xlabel='Steps',
                          additional_data=self.real_rewards_ep, additional_label='Data')

        if self.flag == 'eval' and self.round == self.ep_len:
            self._finalize_and_save_plot(self.figure, 'Live_plot_with_actual_values')

    def _update_plot(self, ax, data, label, color_idx=0, ylabel='', xlabel='', additional_data=None, additional_label=''):
        """Update plot for given axes."""
        ax.clear()
        ax.plot(data, 'x-', color=self.colors[color_idx], label=label)
        if additional_data is not None:
            ax.plot(additional_data, '--', color='black', label=additional_label, linewidth=1.2)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.yaxis.set_label_coords(-0.08, 0.5)

        if xlabel:
            ax.set_xlabel(xlabel)
            
        handles, labels = ax.get_legend_handles_labels()
        if any(label != '_nolegend_' for label in labels):
            ax.legend()

    def _finalize_and_save_plot(self, fig, fig_name):
        """Finalize plots and save to file."""
        results_path = os.path.join(self.results_folder, self.agent_args.agent_name)
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
        
        for ax in fig.get_axes():
            ax.grid(visible=True, which='major', color='gray', linewidth=0.0025)
            ax.grid(visible=True, which='minor', color='gray', linewidth=0.0025)
            
        fig.savefig(os.path.join(results_path, f'{fig_name}.pdf'), dpi=300)

    def _render_not_live(self):
        """Render the final state of the environment for the episode."""
        fig_name = 'Eval_Live_plot' if self.flag == 'eval' else 'Live_plot'
        self._plot_live()
        self._finalize_and_save_plot(self.figure, fig_name)
        fig_name = 'Eval_Live_plot_with_actual_values' if self.flag == 'eval' else 'Live_plot_with_actual_values'
        self._plot_env_real()
        self._finalize_and_save_plot(self.figure, fig_name)
        plt.close('all')

    def _print_environment_status(self):
        """Prints current environment status for non-visual modes."""
        print(f'Round: {self.round}, Actions: {self.actions[-1]}, Targets: {self.targets[-1]}')
        print(f'Reward Received: {self.rewards[-1]}, Total Reward: {sum(self.rewards)}')
        print("==============================================================")
    
    def seed(self, seed=None):
        """Seed the random number generator for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        """Clean up resources used by the environment (if any)."""
        pass
