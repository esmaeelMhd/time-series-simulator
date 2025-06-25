"""
Created on Monday October 20 2022
@author: Esmaeel Mohammadi

# =============================================================================
# This class is used to create a dataset for the LSTM model
    1. Preprocessing of the data
    2. Addition of time specifications
    3. Scaling the data
    4. Dividing the dataset to train, test, and validation
    5. Converting data to Tensors for the model
    6. Converting the tensors to DataLoaders
    
# =============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import os
import joblib
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader

#%% The dataset class

class LSTMDataset:   
    def __init__(self, args, device, is_policy=False):
        self.args = args 
        
        # Setup the device based on GPU usage
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.args.use_gpu else torch.device('cpu')
        if self.device == 'cuda':
            self.device_ids = ','.join(str(i) for i in range(torch.cuda.device_count()))
            self.use_multi_gpu = torch.cuda.device_count() > 1
        else:
            self.device_ids = ''
            self.use_multi_gpu = False  
        
        self.is_policy = is_policy
        
        # Load the Dataset
        self.df = pd.read_csv(self.args.root_path + self.args.data_path)
        self.df = self.df.iloc[:int(0.1*len(self.df))]
        self.num_cols = self.df.shape[1] - 1
        
        # Preparing Dataset
        self.df = self.df.set_index(["date"])
        self.df.index = pd.to_datetime(self.df.index)
        if not self.df.index.is_monotonic_increasing:
            self.df = self.df.sort_index()
        
        self.target_idx = self.df.columns.get_loc(self.args.target)
                
        self.ctrl_vars = []
        if hasattr(self.args, 'ctrl_vars'):
            if isinstance(self.args.ctrl_vars, str):
                self.ctrl_vars.append(self.args.ctrl_vars)
            elif isinstance(self.args.ctrl_vars, list):
                self.ctrl_vars = self.args.ctrl_vars
        elif hasattr(self.args, 'control_variable'):
            if isinstance(self.args.control_variable, str):
                self.ctrl_vars.append(self.args.control_variable)
            elif isinstance(self.args.control_variable, list):
                self.ctrl_vars = self.args.control_variable
        
        self.ind_vars = []
        if hasattr(self.args, 'ind_vars'):
            if isinstance(self.args.ind_vars, str):
                self.ind_vars.append(self.args.ind_vars)
            elif isinstance(self.args.ind_vars, list):
                self.ind_vars = self.args.ind_vars
        elif hasattr(self.args, 'independent_vars'):
            if isinstance(self.args.independent_vars, str):
                self.ind_vars.append(self.args.independent_vars)
            elif isinstance(self.args.independent_vars, list):
                self.ind_vars = self.args.independent_vars
            
        self.num_time_f = 6
        if hasattr(self.args, 'num_time_f'):
            self.num_time_f = self.args.num_time_f
    
    def create_dataset(self):       
        if self.args.embed == 'timeF':
            self.add_time_specs()
                    
        self.df_train, self.df_val, self.df_test = self.train_val_test_split(
            self.df, self.args.test_ratio)
        
        print('train: ', len(self.df_train))
        print('validation: ', len(self.df_val))
        print('test: ', len(self.df_test))

        # Saving test dataset
        if self.is_policy:
            RESULTS_PATH = './policy_results/' + self.args.setting + '/'
        else:
            RESULTS_PATH = './results/' + self.args.setting + '/'

        if not os.path.exists(RESULTS_PATH):    
            # if the folder directory is not present then create it
            os.makedirs(RESULTS_PATH)
            
        self.df_test.to_pickle(RESULTS_PATH + "df_test.pkl")
                
        # Scaling the data    
        if self.args.scale:
            self.scale_data()
             
        # Dividing to train, validation, and test
        if self.is_policy:
            self.X_train_arr, self.y_train_arr = self.feature_label_split_policy(self.df_train.astype(np.float32))
            self.X_val_arr, self.y_val_arr = self.feature_label_split_policy(self.df_val.astype(np.float32))
            self.X_test_arr, self.y_test_arr = self.feature_label_split_policy(self.df_test.astype(np.float32))
        else:
            self.X_train_arr, self.y_train_arr = self.feature_label_split(self.df_train.astype(np.float32))
            self.X_val_arr, self.y_val_arr = self.feature_label_split(self.df_val.astype(np.float32))
            self.X_test_arr, self.y_test_arr = self.feature_label_split(self.df_test.astype(np.float32))
            
        self.y_train_arr = np.array(self.y_train_arr)
        self.y_val_arr = np.array(self.y_val_arr)
        self.y_test_arr = np.array(self.y_test_arr)
        
        print('X_train: ', np.array(self.X_train_arr).shape)
        # np.save('./results/' + self.args.setting + '/' + 'X_val.npy', np.array(self.X_val_arr)[:100,:,:])

        print('y_train: ', self.y_train_arr.shape)
        # np.save('./results/' + self.args.setting + '/' + 'y_val.npy', np.array(self.y_val_arr)[:100,:,:])
    
    # Loading the datasets into DataLoaders
    def create_data_loader(self):
        train_features = torch.as_tensor(np.array(self.X_train_arr).astype(np.float32)).to(self.device)
        train_targets = torch.as_tensor(self.y_train_arr.astype(np.float32)).to(self.device)
        val_features = torch.as_tensor(np.array(self.X_val_arr).astype(np.float32)).to(self.device)
        val_targets = torch.as_tensor(self.y_val_arr.astype(np.float32)).to(self.device)
        test_features = torch.as_tensor(np.array(self.X_test_arr).astype(np.float32)).to(self.device)
        test_targets = torch.as_tensor(self.y_test_arr.astype(np.float32)).to(self.device)
                        
        train_loader = DataLoader(TensorDataset(train_features, train_targets),
                                  batch_size=self.args.batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(TensorDataset(val_features, val_targets), 
                                batch_size=self.args.batch_size, shuffle=False, drop_last=True)
        test_loader_one = DataLoader(TensorDataset(test_features, test_targets), 
                                     batch_size=1, shuffle=False, drop_last=True)
                        
        return train_loader, val_loader, test_loader_one
    
    def create_self_supervised_data(self):
        if self.args.embed == 'timeF':
            self.add_time_specs()
                    
        self.df_train, self.df_val, self.df_test = self.train_val_test_split(
            self.df, self.args.test_ratio)
        
        print('train: ', len(self.df_train))
        print('validation: ', len(self.df_val))
        print('test: ', len(self.df_test))

        # Saving test dataset
        if self.is_policy:
            RESULTS_PATH = './policy_results/' + self.args.setting + '/'
        else:
            RESULTS_PATH = './results/' + self.args.setting + '/'
            
        if not os.path.exists(RESULTS_PATH):    
            # if the folder directory is not present then create it
            os.makedirs(RESULTS_PATH)
            
        self.df_test.to_pickle(RESULTS_PATH + "df_test.pkl")
                
        # Scaling the data    
        if self.args.scale:
            self.scale_data()
        
        if self.args.random_episode_length:
            X_self_supervised, y_self_supervised = self.self_supervised_split(self.df_train,
                                                                              self.args.random_episode_length)
        
        return X_self_supervised, y_self_supervised 
        
    
    # Addition of Time Specifications       
    def add_time_specs(self):
        # Cyclical features generator
        def generate_cyclical_features(df, col_name, period, start_num = 0):
            kwargs = {
                f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
                f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period) 
                }
            
            return df.assign(**kwargs).drop(columns = [col_name])
        
        # Add time specs
        self.df = (self.df
              .assign(hour = self.df.index.hour)
              .assign(month = self.df.index.month)
              .assign(day_of_week = self.df.index.dayofweek)
            )
        
        # Convert time specs to sin and cos
        self.df = generate_cyclical_features(self.df, 'hour', 24, 0)
        self.df = generate_cyclical_features(self.df, 'day_of_week', 7, 0)
        self.df = generate_cyclical_features(self.df, 'month', 12, 1)
    
    # Dividing to Train, Test, and Validation   
    def train_val_test_split(self, df, test_ratio):
        val_ratio = test_ratio / (1 - test_ratio)
        df_train, df_test = train_test_split(df, test_size=test_ratio, shuffle=False)
        df_train, df_val = train_test_split(df_train, test_size=val_ratio, shuffle=False)
        return df_train, df_val, df_test
    
    # Function for Splitting Datasets to X, and y 
    def feature_label_split(self, df):
        X = []
        y = []
        
        for i in range(self.args.seq_len, len(df)-self.args.pred_len):
            X.append(df[i-self.args.seq_len : i])
            y.append(df[i : i+self.args.pred_len, 
                         self.args.in_features-self.num_time_f-self.args.out_features: 
                             self.args.in_features-self.num_time_f])

        return (X, y)
    
    # Function for Splitting Datasets to X, and y for policy training 
    def feature_label_split_policy(self, df):
        X = []
        y = []
        
        for i in range(self.args.seq_len, len(df)-self.args.pred_len):
            X.append(df[i-self.args.seq_len : i])
            y.append(df[i : i+self.args.pred_len, : len(self.ctrl_vars)])

        return (X, y)
    
    def self_supervised_split(self, df, random_el=False):
        if random_el:
            #Empty lists to be populated using formatted training data
            X = []
            y = []
            start_idx = 0
            episode_length = np.random.randint(self.args.min_episode_length, self.args.max_episode_length)

            while start_idx + self.args.seq_len + episode_length < len(df):
                end_idx = start_idx + self.args.seq_len
                X.append(df[start_idx : end_idx])
                y.append(df[end_idx : end_idx + episode_length])
                
                start_idx = start_idx + self.args.seq_len + episode_length
                episode_length = np.random.randint(self.args.min_episode_length, self.args.max_episode_length)
                
        return (X, y)

    
    # Switch Scalers
    def get_scaler(self, scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler
        }
        return scalers.get(scaler.lower())()
    
    # Scale the data
    def scale_data(self):
        feature_scaler = self.get_scaler('minmax')
        time_scaler = self.get_scaler('minmax') if self.args.time_scaled == 'Scaled' else None
        
        # Create the scaler path
        if self.is_policy:
            SCALER_PATH = './policy_scalers/' + self.args.setting + '/'
        else:
            SCALER_PATH = './scalers/' + self.args.setting + '/'

        if not os.path.exists(SCALER_PATH):    
            os.makedirs(SCALER_PATH) 
            
        if self.args.time_scaled == 'Scaled':
            # Scale the feature and time columns
            self.df_train = np.concatenate((feature_scaler.fit_transform(self.df_train.iloc[:, :self.num_cols]),
                                            time_scaler.fit_transform(self.df_train.iloc[:, self.num_cols:])), axis=1)
            self.df_val = np.concatenate((feature_scaler.transform(self.df_val.iloc[:, :self.num_cols]),
                                          time_scaler.transform(self.df_val.iloc[:, self.num_cols:])), axis=1)
            self.df_test = np.concatenate((feature_scaler.transform(self.df_test.iloc[:, :self.num_cols]),
                                           time_scaler.transform(self.df_test.iloc[:, self.num_cols:])), axis=1)
            
            # Save the feature scaler and time scaler
            joblib.dump(feature_scaler, SCALER_PATH + 'feature_scaler.gz')
            joblib.dump(time_scaler, SCALER_PATH + 'time_scaler.gz')
            
            # Save with pickle
            with open(SCALER_PATH + 'feature_scaler.pkl', 'wb') as file:          
                pickle.dump(feature_scaler, file)
            with open(SCALER_PATH + 'time_scaler.pkl', 'wb') as file:          
                pickle.dump(time_scaler, file)
        else:
            # Scale only the feature columns
            self.df_train = np.concatenate((feature_scaler.fit_transform(self.df_train.iloc[:, :self.num_cols]),
                                            np.array(self.df_train.iloc[:,self.num_cols:])), axis=1)         
            self.df_val = np.concatenate((feature_scaler.transform(self.df_val.iloc[:, :self.num_cols]),
                                            np.array(self.df_val.iloc[:,self.num_cols:])), axis=1) 
            self.df_test = np.concatenate((feature_scaler.transform(self.df_test.iloc[:, :self.num_cols]),
                                            np.array(self.df_test.iloc[:,self.num_cols:])), axis=1) 
            
        # Save the feature scaler
        joblib.dump(feature_scaler, SCALER_PATH + 'feature_scaler.gz')
        
        # Save with pickle
        with open(SCALER_PATH + 'feature_scaler.pkl', 'wb') as file:          
            pickle.dump(feature_scaler, file)
    
    def inverse_transform(self, data):
        # Load the appropriate scalers
        if self.is_policy:
            SCALER_PATH = './policy_scalers/' + self.args.setting + '/'
        else:
            SCALER_PATH = './scalers/' + self.args.setting + '/'

        with open(SCALER_PATH + 'feature_scaler.pkl', 'rb') as file:
            feature_scaler = pickle.load(file)
        
        num_features = len(feature_scaler.scale_)
        if self.args.out_features < num_features:
            data = np.hstack((np.zeros((len(data), num_features-self.args.out_features)), data))
            data = feature_scaler.inverse_transform(data)
            data = data[:, num_features-self.args.out_features:]
        else:
            data = feature_scaler.inverse_transform(data)
    
        return data
    
    def create_forecast_data(self, arr):
        X_forecast = np.array(arr)
        # y_forecast will not be used and we only need it to create the Tensor Dataset
        y_forecast = np.array(arr)[-1,:]

        # Conveting the input X to: [batch_size=1, sequence_length, in_features]
        X_forecast = X_forecast.reshape((1, X_forecast.shape[0], X_forecast.shape[1]))
        # Converting the y array to: [batch_size=1, in_features] to avoid mismatch
        y_forecast = y_forecast.reshape((1, y_forecast.shape[0]))

        # Creating the Tensor Dataset
        forecast_dataset = TensorDataset(torch.Tensor(np.array(X_forecast)).to(self.device), 
                                         torch.Tensor(np.array(y_forecast)).to(self.device))
        # Creating the DataLoader
        forecast_loader = DataLoader(
            forecast_dataset, batch_size=1, shuffle=False, drop_last=True)
        
        return forecast_loader
        
   