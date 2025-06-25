"""
Created on July 10 2022
@author: Esmaeel Mohammadi

# =============================================================================
# This script is used to train and test the LSTM model
    1. Preprocessing of the data
    2. Convert data to Tensors for the model
    3. Convert Tensors to DataLoaders for the model
    4. Train the model
    5. Test the model
# =============================================================================
"""

import pickle
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.LSTM import LSTMModel, EncoderLSTM, DecoderLSTM, Net_LSTM
from utils.LSTM_Dataset import LSTMDataset
from utils.LSTM_model_optimizer import Optimization
from utils.env_helper import EnvHelper

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except Exception as e:
    pass
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.xmargin'] = 0.02
plt.rcParams['axes.ymargin'] = 0.04
plt.rcParams['axes.labelsize'] = 8

plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=6)

#%%

class ExpLSTM:
    def __init__(self, args, device, is_policy=False):
        self.args = args
        self.device = device
        self.is_policy = is_policy
        
        self.df_raw = pd.read_csv(args.root_path + args.data_path)
        self.num_cols = self.df_raw.shape[1] - 1

        self.df_raw = self.df_raw.set_index(["date"])
        self.df_raw.index = pd.to_datetime(self.df_raw.index)
        if not self.df_raw.index.is_monotonic:
            self.df_raw = self.df_raw.sort_index()
        
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
        
        self.target_idx = self.df_raw.columns.get_loc(self.args.target) - len(self.ctrl_vars) - len(self.ind_vars)
        
        self.lstm_dataset = LSTMDataset(self.args, self.device, self.is_policy)
        
        # Building the model
        if not hasattr(self.args, 'lstm_type') or (hasattr(self.args, 'lstm_type') and self.args.lstm_type != 'EncDec'):
            self.model = LSTMModel(self.args).float()
        else:
            encoder = EncoderLSTM(self.args)
            decoder = DecoderLSTM(self.args)
            self.model = Net_LSTM(encoder, decoder, self.args, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)
        self.model = self.model.to(self.device)
        
        # Building the optimizer to train
        loss_fn = nn.MSELoss(reduction="mean")
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, 
                               weight_decay=self.args.weight_decay)

        self.opt = Optimization(self, model=self.model, loss_fn=loss_fn, optimizer=optimizer, args=self.args, 
                                setting=self.args.setting, device=self.device, target_idx=self.target_idx, is_policy=self.is_policy)
        
    def train(self, setting):
        # Building the dataset and dataloader
        self.lstm_dataset.create_dataset()
        train_loader, val_loader, self.test_loader_one = self.lstm_dataset.create_data_loader()

        self.opt.train(train_loader = train_loader, 
                       val_loader = val_loader)
        
    def self_supervised_train(self, setting):
        # Create a folder for saving the best model
        checkpoints = self.args.checkpoints
        path = os.path.join(checkpoints, self.args.setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        data, targets = self.lstm_dataset.create_self_supervised_data()
        num_batches = len(data)
        
        best_batch_loss = None
        train_losses = []
        for i, (data_item, target_item) in enumerate(zip(data, targets)):
            data = torch.Tensor(np.array(data_item)).unsqueeze(0).to(self.device)
            targets = torch.Tensor(np.array(target_item)).unsqueeze(0).to(self.device)
            
            train_data = TensorDataset(data, targets)
            train_loader = DataLoader(train_data, batch_size=1, 
                                      shuffle=False, drop_last=True)
            
            batch_loss, self.model = self.opt.self_supervised_train(train_loader = train_loader)
            train_losses.append(batch_loss)
            train_loss = np.mean(train_losses)
            
            if best_batch_loss is None or best_batch_loss > batch_loss:
                best_batch_loss = batch_loss
                
            print(f'[Batch {i+1}/{num_batches}] | Episode length: {targets.shape[1]} | Batch loss: {batch_loss:.7f} '+\
                  f'| Best batch loss: {best_batch_loss:.7f} | Train loss : {train_loss:.7f}')

            if i == num_batches - 1:
                print('Saving the model ...')
                torch.save(self.model.state_dict(), path + '/' + 'self_train_checkpoint.pth')
        
    # Testing the trained model
    def test(self, setting):         
        predictions, values = self.opt.evaluate(
            self.test_loader_one,
            setting = self.args.setting,
            batch_size=1,
            n_features=self.args.in_features
        )
        
        control_vars = []
        if isinstance(self.args.ctrl_vars, str):
            control_vars.append(self.args.ctrl_vars)
        elif isinstance(self.args.ctrl_vars, list):
            control_vars = self.args.ctrl_vars
            
        if self.is_policy:
            out_features = len(self.args.ctrl_vars)
        else:
            out_features = self.args.in_features - len(control_vars) - len(self.ind_vars) - self.num_time_f
        
        if self.args.scale:
            predictions = np.array(predictions).reshape(len(predictions), self.args.pred_len, out_features)
            values = np.array(values).reshape(len(values), self.args.pred_len, out_features)
            
            for i in range(len(predictions)):
                predictions[i,:,:] = self.lstm_dataset.inverse_transform(predictions[i,:,:])
                values[i,:,:] = self.lstm_dataset.inverse_transform(values[i,:,:])

        if self.is_policy:
            file = open('./policy_results/' + self.args.setting + '/' + 'df_test.pkl', 'rb')    
        else:
            file = open('./results/' + self.args.setting + '/' + 'df_test.pkl', 'rb')        

        df_test = pickle.load(file)
        
        def format_predictions(preds, vals):
            df_result = pd.DataFrame(data={"value": vals[:, 0, self.target_idx], 
                                           "prediction": preds[:, 0, self.target_idx]}, 
                                            index=df_test.tail(len(vals)).index)
            
            df_result = df_result.sort_index()
            return df_result

        df_result = format_predictions(predictions, values)

        # Calculate Metrics of the Model
        def calculate_metrics(df):
            return {'mae' : mean_absolute_error(df.value, df.prediction),
                    'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
                    'r2' : r2_score(df.value, df.prediction)}

        result_metrics = calculate_metrics(df_result)

        print(f'Mean Absolte Error (MAE) of the Test is: {result_metrics.get("mae")}')
        print(f'Mean Squared Error (RMSE) of the Test is: {result_metrics.get("rmse")}')
        print(f'R2 Score (R2) of the Test is: {result_metrics.get("r2")}')

        # Plotting the test results
        def plot_results(df_result):
            # plt.switch_backend('qt5agg')
            if self.args.target == 'T1_PO4':
                y_label = 'P-amount [$mg/L$]'
                target_name = 'Phosphate concentration'
            elif self.args.target == 'IN_METAL_Q':
                y_label = 'Metal Flow [$m^3/h$]'
                target_name = 'Metal flow'
            elif self.args.target == 'N2O':
                y_label = 'N2O'
                target_name = 'N2O'

            fig = plt.figure(figsize=(6.5,3), dpi=1000)
            ax = plt.axes()
            plt.xlabel('Time Steps', labelpad=5, rotation=0)
            fig.suptitle(f'Prediction of the test dataset for {target_name}')
            
            plt.gca().xaxis.set_major_locator(matplotlib.dates.MonthLocator()) 
            plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
            plt.margins(x=0,y=0)
                
            ax.set_ylabel(y_label)
            ax.plot(df_result.prediction, linewidth=0.75, label='Prediction')
            ax.plot(df_result.value, '.', markersize=2, color='black', label='Ground Truth')
            ax.legend()
            
            plt.grid()
            # plt.ion()
            # plt.show()
            
            if self.is_policy:
                fig_folder_path = './policy_results/' + self.args.setting + '/'
            else:
                fig_folder_path = './results/' + self.args.setting + '/'

            if not os.path.exists(fig_folder_path):
                os.makedirs(fig_folder_path)
            
            fig_name = 'Test_dataset_results'
            plt.savefig(fig_folder_path + fig_name + '.svg')
            plt.savefig(fig_folder_path + fig_name + '.pdf')
            plt.savefig(fig_folder_path + fig_name + '.png')

        plot_results(df_result)
    
    def predict(self, setting, df_forecast, save=False):       
        if save:
            forecast_loader = self.test_loader_one
        else:    
            forecast_loader = self.lstm_dataset.create_forecast_data(df_forecast)
            
        predictions = self.opt.forecast_with_predictors(forecast_loader,
                                                        batch_size=1, 
                                                        n_features=self.args.in_features, 
                                                        n_steps=1)

        

        # Reshape it from [1, pred_len, in_features] to [pred_len, in_features]
        # predictions = np.array(predictions).reshape(np.array(predictions).shape[0], -1)
        predictions = np.array(predictions[0])
        print(f'P raw in predictions: {predictions[-1,self.target_idx]}')
        print('predictions shape: ', predictions.shape)

        if save:
            if self.is_policy:
                folder_path = './policy_results/' + setting + '/'
            else:
                folder_path = './results/' + setting + '/'
            np.save(folder_path + 'real_prediction.npy', predictions)

        return predictions
    
    def predict_simulation(self):
        # freq = datetime.timedelta(minutes=1)
        freq = self.df_raw.index.to_series().diff().dropna().mode()[0]
        episode_length = 360
        helper = EnvHelper()
        pred_points, pred_points_names = helper.make_points(
            test_frequency='Seasons', time_of_the_day='Morning',
            day_of_the_month = 'First', first_date=pd.to_datetime('2021-08'), 
            last_date=pd.to_datetime('2022-07'))
            
        pred_data = []
        pred_targets = []
        for point in pred_points:
            pred_start = self.df_raw.index.get_loc(point)
            pred_end = pred_start + self.args.seq_len
            X = helper.scale_data(self.args, helper.add_time_specs(
                self.df_raw[pred_start:pred_end]))
            y = helper.scale_data(self.args, helper.add_time_specs(
                self.df_raw[pred_end:pred_end + episode_length + self.args.pred_len-1]))
            pred_data.append(X)           
            pred_targets.append(y) 
            
        pred_data = torch.Tensor(np.array(pred_data)).to(self.device)
        pred_targets = torch.Tensor(np.array(pred_targets)).to(self.device)       
        pred_dataset = TensorDataset(pred_data, pred_targets)
        pred_loader = DataLoader(pred_dataset, batch_size=1, 
                                shuffle=False, drop_last=False)
        
        self.pred_results_dict = {key:None for key in pred_points_names}
        
        # Loading the model
        print('loading model')
        if self.is_policy:
            self.model.load_state_dict(torch.load(
                os.path.join('./policy_checkpoints/' + self.args.setting,'checkpoint.pth'))) 
        else:
            self.model.load_state_dict(torch.load(
                os.path.join('./checkpoints/' + self.args.setting,'checkpoint.pth'))) 
        
        pred_results_list, pred_losses = self.opt.test_simulation(self.model, pred_loader)
        
        for i, (key, item) in enumerate(zip(self.pred_results_dict, pred_results_list)):        
            self.pred_results_dict[key] = item
            y_real = self.pred_results_dict[key]['y_real']
            y_pred = self.pred_results_dict[key]['y_pred']
            self.pred_results_dict[key]['y_real'] = helper.inverse_transform(y_real)
            self.pred_results_dict[key]['y_pred'] = helper.inverse_transform(y_pred)
        
        pred_loss = np.mean(pred_losses)
        
        # Plotting
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 0.5
        plt.rcParams['axes.xmargin'] = 0.02
        plt.rcParams['axes.ymargin'] = 0.04
        fig_dpi = 1000
        hspace = 0.5
        wspace = 0.2

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6.5,3), dpi=fig_dpi)
        # fig.tight_layout(pad=4.0)

        for i , (ax, point) in enumerate(zip(axs.ravel(), self.pred_results_dict.keys())):
            start_date = pred_points[i] + freq*self.args.seq_len
            dates = pd.date_range(start=start_date, end=start_date+episode_length*freq, freq=freq)
            if self.args.target == 'N2O':
                times = pd.to_datetime(dates).strftime('%d %b - %H')
            else:
                times = pd.to_datetime(dates).strftime('%H:%M')
            
            target_idx = -1
            test_real = self.pred_results_dict[point].get('y_real')
            test_pred = self.pred_results_dict[point].get('y_pred')
            ax.plot(test_real[:, target_idx], label='Ground Truth', color='black', linewidth=0.5)
            ax.plot(test_pred[:, target_idx], label='Prediction', linewidth=0.25)
            ax.set_title(point, fontweight="bold")
            
            ax.set_xticks(np.arange(len(times)))
            ax.set_xticklabels(times)
            ax.xaxis.set_major_locator(MultipleLocator(60))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.tick_params(which='minor', length=2)
            if self.args.target == 'N2O':
                ax.tick_params(axis='x', labelsize=6)
                    
            ax.grid(visible=True, which='major', color='gray', linewidth=0.025)
            ax.grid(visible=True, which='minor', color='gray', linewidth=0.025)
            ax.legend(fontsize=6)
            handles, labels = ax.get_legend_handles_labels()
        
        if self.args.target == 'T1_PO4':
            y_label = 'P-amount [$mg/L$]'
        elif self.args.target == 'IN_METAL_Q':
            y_label = 'Metal Flow [$m^3/h$]'
        elif self.args.target == 'N2O':
            y_label = 'N2O'
        for ax_col_idx in range(axs.shape[1]): 
            if ax_col_idx == 0:
                axs[-1, ax_col_idx].set_xlabel('Time (24-hour)', fontsize=6, labelpad=5)
                axs[-1, ax_col_idx].set_ylabel(y_label, fontsize=6, labelpad=5)
        
        plt.subplots_adjust(hspace=hspace, wspace=wspace, left=0.08, right=0.95, bottom=0.12, top=0.92)

        if self.is_policy:
            RESULTS_PATH = './policy_results/'
        else:
            RESULTS_PATH = './results/'

        fig_folder_path = RESULTS_PATH + self.args.setting + '/'
        if not os.path.exists(fig_folder_path):    
            os.makedirs(fig_folder_path)
            
        fig_name = 'Env Test - 360Rounds_Seasons_First_Morning'
        
        plt.savefig(fig_folder_path + fig_name + '.svg')
        plt.savefig(fig_folder_path + fig_name + '.png')
        plt.savefig(fig_folder_path + fig_name + '.pdf')
        
        return 
        
        