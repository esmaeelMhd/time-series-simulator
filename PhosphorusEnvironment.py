"""
Created on Monday July 18 2022
@author: Esmaeel Mohammadi

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

from PhosphorusEnvGraph import PhosphorusEnvGraph
import os
import pickle
import joblib
import warnings

from gym import spaces
import gym

from models import NLinear
from models import Informer
from models import Transformer
from models import Autoformer
from models import DLinear
from models.LSTM import LSTMModel

from exp.exp_main_env import Exp_Main
from exp.exp_lstm_env import ExpLSTM

import torch.nn as nn
import torch
import datetime
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import pdb

# %% Defining the Phosphorus Environment Class
class PhosphorusEnvironment(gym.Env):
    # First we will define some functions for our simulator
    # Scaling the data
    def _scale_data(self, df):
        if self.args.model == 'LSTM':
            arr = np.zeros(shape=(df.shape[0], self.args.in_features))
            if self.args.time_scaled == 'Scaled':
                # Scale the feature columns
                arr[:, :self.num_cols] = self.feature_scaler.transform(df.iloc[:, :self.num_cols])        
                # Scale the time columns
                arr[:, self.num_cols:] = self.time_scaler.transform(df.iloc[:, self.num_cols:])
            else:
                # Scale only the feature columns
                arr[:, :self.num_cols] = self.feature_scaler.transform(df.iloc[:, :self.num_cols])
                arr[:, self.num_cols:] = np.array(df.iloc[:, self.num_cols:])
        else:
            arr = self.scaler.transform(df)
        
        return arr

    # Function for the inverse transform of the scaled data
    def _inverse_transform(self, arr):
        if self.args.model == 'LSTM':
            arr = self.feature_scaler.inverse_transform(arr)
        else:
            arr = self.scaler.inverse_transform(arr)
        return arr

    # Converting the arrays to dataframe, we need to do it because of the scaler
    # and also making Tensor dataset
    def _make_df(self, arr, step):
        start = self.start_date
        first_date = start + (step)*self.freq
        index = pd.date_range(
            start=first_date, freq=self.freq, periods=self.args.seq_len)
        df = pd.DataFrame(arr, columns=self.columns)
        df = df.set_index(index)
        if self.args.model == 'LSTM' and self.args.embed == 'timeF':
            df = self._add_time_specs(df)
        # Keep track of dates
        self.dates.append([first_date, df.index[-1]])
        return df

    # Addition of Time Specifications if we need them
    def _add_time_specs(self, df):
        # Cyclical features generator
        def generate_cyclical_features(df, col_name, period, start_num=0):
            kwargs = {
                f'sin_{col_name}': lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
                f'cos_{col_name}': lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)
            }

            return df.assign(**kwargs).drop(columns=[col_name])

        # Add time specs
        df = (df
              .assign(hour=df.index.hour)
              .assign(month=df.index.month)
              .assign(day_of_week=df.index.dayofweek)
              )

        # Convert time specs to sin and cos
        df = generate_cyclical_features(df, 'hour', 24, 0)
        df = generate_cyclical_features(df, 'day_of_week', 7, 0)
        df = generate_cyclical_features(df, 'month', 12, 1)

        return df

    # Predictor function which predicts every state based on the past data
    def _state_predictor(self):
        df_forecast = self.df.copy()
        if self.args.model == 'LSTM':
            if self.args.scale:
                df_forecast = self._scale_data(df_forecast)
                self.scaled_inputs.append(df_forecast)
            forecasted = self.exp.predict(self.args.setting, df_forecast, save=False)
            forecasted = np.squeeze(forecasted, axis=0)
            forecasted = np.reshape(forecasted[0, :], (1, self.num_cols))
            forecasted = self._inverse_transform(forecasted)
            return forecasted
        else:
            df_forecast = df_forecast.rename_axis('date').reset_index(level=0)
            return self.exp.predict(df_forecast, self.args.setting)

    # Building the model
    def _build_model(self):
        model_dict = {
            'LSTM': LSTMModel,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear
        }
        
        if self.args.model == 'LSTM':
            model = model_dict[self.args.model](self.args, self.device).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _normalize_predictions(self, state):
        for col in range(state.shape[1]):
            min_feature = self.df_info.loc['min', self.df_info.columns[col]]
            max_feature = self.df_info.loc['max', self.df_info.columns[col]]
            if state[0, col] > max_feature:
                state[0, col] = max_feature - 0.0001
            elif state[0, col] < min_feature:
                state[0, col] = min_feature + 0.0001

        return state
    
    def improve_model(self):
        self.improve_x = torch.Tensor(self._scale_data(self.df)).unsqueeze(0).to(self.device)
        df_raw = self._add_time_specs(self.df_raw)
        targets = np.array(self._scale_data(df_raw))
        self.improve_targets = torch.Tensor(targets).unsqueeze(0).to(self.device)
        print('\nImproving the model ...')
        if self.args.model == 'LSTM':
            self.model = self.exp.opt.improve_model_env(self.improve_x, self.improve_targets, retrain_epochs=100,
                                                        use_actual_actions=True, improve_mode='batch_pred')

    def __init__(self, df, setting, episode_length, num_envs, device, 
                 title='Test', retrain=False, retr_chk='retrained_checkpoint.pth',
                 all_actual_values=False):
        super(PhosphorusEnvironment, self).__init__()
        
        self.scaled_inputs = []
        # Get the device
        self.device = device
        self.title = title
        self.visualization = None
        self.mode = 'not_live'
        self.window_size = 3
        self.all_actual_values = all_actual_values
        
        self.retrain_epochs = 20

        # Get the model name
        self.setting = setting

        self.dates = []
        self.targets = []
        self.real_targets = []
        self.control_variables = []

        # Load the args file of the model
        ARGS_PATH = './args/' + self.setting + '/'
        with open(ARGS_PATH + 'args.pkl', 'rb') as file:
            self.args = pickle.load(file)

        # Get some info from the args file
        self.args.simulator = True
        self.args.use_multi_gpu = False

        dataset_path = self.args.root_path + self.args.data_path
        self.df_raw = pd.read_csv(dataset_path)

        self.df_raw = self.df_raw.set_index(["date"])
        self.df_raw.index = pd.to_datetime(self.df_raw.index)
        if not self.df_raw.index.is_monotonic_increasing:
            self.df_raw = self.df_raw.sort_index()

        # Get the specification of the used dataset
        self.df = df    # Dataframe
        self.columns = self.df.columns
        self.num_cols = len(self.columns)
        self.freq = datetime.timedelta(minutes=1)
        # self.freq = pd.to_datetime(self.df.index[-1]) - pd.to_datetime(self.df.index[-2])
        self.start_date = pd.to_datetime(self.df.index[0])
        start_loc = self.df_raw.index.get_loc(self.start_date) + self.args.seq_len
        self.df_raw = self.df_raw.iloc[start_loc:start_loc + episode_length]

        if self.args.model == 'LSTM' and self.args.embed == 'timeF':
            self.df = self._add_time_specs(self.df)

        with open('df_info.pkl', 'rb') as file:
            self.df_info = pickle.load(file)

        for col in self.df_info.columns:
            if col not in self.df.columns:
                self.df_info = self.df_info.drop([col], axis=1)

        # Target and Control Variable
        self.target = self.args.target
        self.control_variable = self.args.control_variable
        self.target_idx = self.df.columns.get_loc(self.target)
        self.control_idx = self.df.columns.get_loc(self.control_variable)

        self.min_target = self.df_info.loc['min', self.target]
        self.min_control_variable = self.df_info.loc['min',
                                                     self.control_variable]
        self.max_target = self.df_info.loc['max', self.target]
        self.max_control_variable = self.df_info.loc['max',
                                                     self.control_variable]

        # Convert the dataset to an array to use in the simulator
        self.init_sequence = self.df.copy(deep=True)
        self.in_features = self.df.shape[1]

        # Define ACTION and OBSERVATION space
        # They must be gym.spaces objects
        # Here we have simultaneous environments for each step
        # They run different policies for the same state at the same time
        # Right NOW we only have 1 control variable or ACTION = Metal Addition
        # The shape of action space will be ONE action in the num of environments
        self.action_space = spaces.Box(low=self.min_control_variable, high=self.min_control_variable,
                                       shape=(num_envs, 1), dtype=np.float32)

        # Right NOW we only have 1 target or OBSERVATION = P amount in Tank 1
        # The shape of observation space will be ONE obs in the num of environments
        self.observation_space = spaces.Box(low=self.min_target, high=self.max_target,
                                            shape=(num_envs, 1), dtype=np.float32)

        # Define the initial state, the last time step of the dataset
        # State in each step has all of the features, but our target (observation)
        # which the controller will decide based on that is only ONE feature
        self.init_state = self.df.iloc[-1, :]
        self.collected_reward = 0.0
        self.round = 0
        self.max_rounds = episode_length
        # Sequences to keep track of all of the states and the simulator
        self.sequences = [self.df]
        # Checkpoint path of the saved model
        checkpoint_path = './checkpoints/' + self.setting

        if self.args.scale:
            # Load the scaler path
            SCALER_PATH = './scalers/' + self.args.setting + '/'
            if self.args.model == 'LSTM':
                if self.args.time_scaled == 'Unscaled':
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        self.feature_scaler = joblib.load(SCALER_PATH + 'feature_scaler.gz') 
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        self.feature_scaler = joblib.load(SCALER_PATH + 'feature_scaler.gz') 
                        self.time_scaler = joblib.load(SCALER_PATH + 'time_scaler.gz') 
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    self.scaler = joblib.load(SCALER_PATH + 'scaler.gz')

        # Loading the model
        self.model = self._build_model().to(self.device)
        if retrain == True:
            self.model.load_state_dict(torch.load(
                os.path.join(checkpoint_path, retr_chk)))
        else:
            self.model.load_state_dict(torch.load(
                os.path.join(checkpoint_path, 'checkpoint.pth')))
            

        if self.args.model == 'LSTM':
            Exp = ExpLSTM
            self.exp = Exp(self.args, self.device, self.model)
        else:
            Exp = Exp_Main    
            self.exp = Exp(self.args, self.model)
        
        self.model.eval()
    
        self.improve_model()   

    # Step function for each state of the training
    def step(self, action):
        # Execute one time step within the environment
        done = False
        rw = 0
        self.round += 1

        info = {'sequences': self.sequences,
                'round': self.round, 'dates': self.dates,
                'scaled_inputs': self.scaled_inputs}
        # Get the ACTION from action generating function
        metal_amount = self._take_action(action).item()
        # Replacing the action from last time step with the one from controller
        self.df.iloc[-1, self.control_idx] = metal_amount
        
        # Predict the next state with the last sequence
        state = self._state_predictor()
        # State has the shape of [pred_len, in_features]

        # state = self._normalize_predictions(state)
        
        # Get the OBSERVATION item from list
        # Phosphorus is the last column of predicted dataset
        obs = state[0, self.target_idx]

        self.targets.append(obs)
        current_date = pd.to_datetime(self.df.index[-1] + self.freq)
        #self.real_targets.append(self.df_raw.loc[current_date, self.target])

        # Add the prediction to the end of dataset and delete the first row
        self.df = np.array(self.df)[:, :self.num_cols]
        self.df = np.append(
            self.df, state[0].reshape(1, self.num_cols), axis=0)
        self.df = np.delete(self.df, (0), axis=0)

        # Append every sequence to keep track of changes
        self.state = state

        # Calculating the reward
        if obs < 1:
            self.collected_reward += 1
            rw = 1
        elif obs < 2:
            self.collected_reward += 0
            rw = 0
        else:
            self.collected_reward -= 1
            rw = -1

        # Checking if the rounds are ended or not
        if self.round == self.max_rounds:
            done = True
        else:
            self.df = self._make_df(self.df, self.round)
            if self.all_actual_values:
                self.df.iloc[-1, 1:self.num_cols-1] = self.df_raw.iloc[self.round-1, 1:self.num_cols-1]
            self.sequences.append(self.df)

        # Sending to the render for printing and graphs
        # self.render(action, rw, obs)

        return state[0], rw, done, info

    def _take_action(self, action):

        metal_amount = action
        self.control_variables.append(metal_amount)

        return metal_amount

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = self.init_state
        obs = self.state[self.target_idx]
        self.sequences.clear()
        self.sequences.append(self.init_sequence)
        self.round = 0
        info = {'sequences': self.sequences, 'round': self.round}
        return obs, info

    def render(self, action, rw, obs):
        if self.mode == 'live':
            fig_title = f'Environment Visualization for {self.title}'
            if self.visualization is None:
                self.visualization = PhosphorusEnvGraph(self.df_raw, fig_title)
            if self.round > self.window_size:
                self.visualization.render(self.round, self.targets, self.real_targets,
                                          self.control_variables, window_size=self.window_size)
        else:
            # Render the environment to the screen
            print(
                f"Round : {self.round}\nMetal Amount : {action:0.3f}\nPhosphorus : {obs.item():0.3f}\nReward Received : {rw}")
            print(f"Total Reward : {self.collected_reward}")
            print(
                "=============================================================================")
