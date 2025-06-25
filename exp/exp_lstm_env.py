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

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils.LSTM_Dataset import LSTMDataset
from utils.LSTM_model_optimizer import Optimization

#%%

class ExpLSTM:
    def __init__(self, args, model):
        # Initiate variables
        self.args = args
        
        # Setup the device based on GPU usage
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.args.use_gpu else torch.device('cpu')
        if self.device == 'cuda':
            self.device_ids = ','.join(str(i) for i in range(torch.cuda.device_count()))
            self.use_multi_gpu = torch.cuda.device_count() > 1
        else:
            self.device_ids = ''
            self.use_multi_gpu = False  
            
        self.model = model
        self.lstm_dataset = LSTMDataset(self.args, self.device)
        
        # Building the optimizer to predict
        loss_fn = nn.MSELoss(reduction="mean")
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, 
                               weight_decay=self.args.weight_decay)

        self.opt = Optimization(model=self.model, loss_fn=loss_fn, optimizer=optimizer, 
                           args=self.args, setting=self.args.setting, device=self.device)
        
    def predict(self, setting, df_forecast, save=False):       
        # Create forecast Data Loader
        forecast_loader = self.lstm_dataset.create_forecast_data(df_forecast)
        # Forecast
        predictions = self.opt.forecast_with_predictors(forecast_loader,
                                                        batch_size=1, 
                                                        n_features=self.args.in_features, 
                                                        n_steps=1)
        
        predictions = np.array(predictions[0])
        
        if save:
            folder_path = './results/' + setting + '/'
            np.save(folder_path + 'real_prediction.npy', predictions)

        return predictions
        
        