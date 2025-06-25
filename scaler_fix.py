"""
Created on Monday October 18 2022
@author: Esmaeel Mohammadi

# =============================================================================
# This script is used to fix the scaler loading issue:
    1. Fixes the scaler for the specific set of args 
# =============================================================================
"""

import os
import pickle 
import pandas as pd 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import joblib
import numpy as np

from sklearn.metrics import mean_squared_error
import datetime
from dateutil.relativedelta import relativedelta
import pytz
import warnings

import torch
from utils.LSTM_Dataset import LSTMDataset

#%% Initializing

# The raw dataset
dataset_name = 'wastewater.csv'
retrained_checkpoint_name = '30_episodes_2_months_no_gap_act_actions_40Epochs_retrained_checkpoint.pth' 

# Environment and the model specifications    
episode_length = 360
sequence_length = 1

# Points to test
test_frequency = 'Seasons'       # options: Months, Seasons 
day_of_the_month = 'Middle'      # options: First, Middle  
time_of_the_day = 'Morning'     # options: Morning, Noon
first_date = pd.to_datetime('2021-08')
last_date = pd.to_datetime('2022-07')   
points = test_frequency + '_' + day_of_the_month + '_' + time_of_the_day

# Choosing the models to run
retrained = False
single = True
setting = 'LSTM_New_CorrH_2_11F_timeF_Unscaled_1Seq_0Label_8Batch_1e-06LR_256Hidden_2LayerDim'
best_test_results = False
best_env_results = False
dataset_based = False
data_tag = 'New_CorrH'
all_models = False

#%%

def season_of_date(date):
    year = date.year
    seasons = {'Summer':(datetime.datetime(year,6,21), datetime.datetime(year,9,22)),
               'Autumn':(datetime.datetime(year,9,23), datetime.datetime(year,12,20)),
               'Spring':(datetime.datetime(year,3,21), datetime.datetime(year,6,20))}
    for season,(season_start, season_end) in seasons.items():
        season_start = season_start.replace(tzinfo=pytz.UTC)
        season_end = season_end.replace(tzinfo=pytz.UTC)
        if date>=season_start and date<= season_end:
            return season
    else:
        return 'Winter'

def create_points():
    freq_values = {'Months':1, 'Seasons':3}
    day_values = {'First':1, 'Middle':15}
    time_values = {'Morning':0, 'Noon':12}
    
    start_date = datetime.datetime(first_date.year, first_date.month, day_values[day_of_the_month],
                                   time_values[time_of_the_day], 0, 0).replace(tzinfo=pytz.UTC)
    end_date = datetime.datetime(last_date.year, last_date.month, day_values[day_of_the_month],
                                 time_values[time_of_the_day], 0, 0).replace(tzinfo=pytz.UTC)
    
    month_freq = relativedelta(months=freq_values[test_frequency])
    list_points = []
    points_names = []
    date = start_date
    
    while date <= end_date:
        date = date.replace(tzinfo=pytz.UTC)
        list_points.append(date)
        if test_frequency == 'Seasons':
            season = season_of_date(date)
            points_names.append(f'{date.strftime("%b %d %Y")}')
        else:
            points_names.append(f'{date.strftime("%b %d %Y")}')
        
        date += month_freq 
    return list_points, points_names

# Creating the points dataframe and names
list_points, points_names = create_points()

#%%

# Frequency of the dataset
freq = datetime.timedelta(minutes=1)
# Information about the models
result_folders = list([name for name in os.listdir('./results/')])
list_metrics = list(['mae', 'mse', 'rmse', 'rse', 'corr'])
df_metrics = pd.DataFrame(columns = list_metrics)
df_metrics['names'] = result_folders
df_metrics = df_metrics.set_index(['names'])

# Handling the metrics dataframe
for folder in result_folders:
    if 'metrics.npy' in os.listdir('./results/' + folder):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            metrics = np.load('./results/' + folder + '/metrics.npy', allow_pickle=True)
        if len(metrics) == 7:
            # Deleting the unnecessary metrics
            metrics = np.delete(metrics, 3)
            metrics = np.delete(metrics, 4)
        #metrics = metrics.astype(float)
        df_metrics.loc[folder, :] = metrics

#%% Choosing the models to run

best_models = {'LSTM':'',
               'Transformer':'',
               'Informer':'',
               'Autoformer':'',
               'DLinear':'',
               'NLinear':'',   
               }

models_list = []

if single:
    models_list.append(setting)    
    
elif best_test_results:
    if dataset_based:
        folders = list([x for x in result_folders if data_tag in x])
    else:
        folders = result_folders
        
    for model in best_models.keys():
        items = list([x for x in folders if model in x])
        df_metrics['mse'] = pd.to_numeric(df_metrics['mse'])
        list_mse = df_metrics.loc[items, 'mse']
        min_mse_idx = list_mse.idxmin()
        best_models[model] = min_mse_idx
    models_list = best_models.values()

elif best_env_results:
    if dataset_based:
        folders = list([x for x in os.listdir('./env_results/') if data_tag in x])
    else:
        folders = list([x for x in os.listdir('./env_results/')])
        
    p_names = list(['Winter', 'Spring', 'Summer', 'Autumn'])
    df_mse = pd.DataFrame(columns=p_names)
    df_mse['Average'] = None
    df_mse['names'] = folders
    df_mse = df_mse.set_index(['names'])

    for folder in folders:
        for point_name in p_names:
            RESULTS_PATH = './env_results/' + folder + '/'
            states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point_name}_Phosphorous.npy')
            actual_states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point_name}_actual_p.npy')
            
            states = np.array(states).reshape(np.array(states).shape[0], -1)
            actual_states = np.array(actual_states).reshape(np.array(actual_states).shape[0], -1)
            
            mse = mean_squared_error(actual_states, states)
            df_mse.loc[folder, point_name] = mse

    df_mse['Average'] = df_mse.mean(axis=1)
    df_mse = df_mse.sort_values('Average')
    
    for model in best_models.keys():
        items = list([x for x in df_mse.index if model in x])
        list_mse = df_mse.loc[items, 'Average']
        max_mse_idx = df_mse.loc[items, 'Average'].idxmin()
        best_models[model] = max_mse_idx
    
    models_list = best_models.values()
    
elif all_models:
    if dataset_based:
        models_list = list([x for x in result_folders if data_tag in x])
    else:
        models_list = result_folders
    
print(f'The environment will run for {len(models_list)} model(s)')

#%% Switch Scalers

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

def set_type_data(flag):
    # init
    assert flag in ['train', 'test', 'val']
    type_map = {'train': 0, 'val': 1, 'test': 2}
    set_type = type_map[flag]
    return set_type

def get_df(flag):
    set_type = set_type_data(flag)
    border1 = border1s[set_type]
    border2 = border2s[set_type]
    data = df_data[border1:border2]
    return data

#%%

if single:
    names_list = list([setting])
else:
    names_list = models_list    

for name in names_list:
    ARGS_PATH = './args/' + name + '/'
    with open(ARGS_PATH + 'args.pkl', 'rb') as file:
        args = pickle.load(file)
    
    target = args.target
    features = args.features
    scale = args.scale
    
    def acquire_device():
        if args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                args.gpu) if not args.use_multi_gpu else args.devices
            device = torch.device('cuda:{}'.format(args.gpu))
            print('Use GPU: cuda:{}'.format(args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    device = acquire_device()
    
    if args.model == 'LSTM':
        lstm_dataset = LSTMDataset(args, device)
        lstm_dataset.create_dataset()
    
    else:
        size=[args.seq_len, args.label_len, args.pred_len]
        
        if size == None:
            seq_len = 24 * 4 * 4
            label_len = 24 * 4
            pred_len = 24 * 4
        else:
            seq_len = size[0]
            label_len = size[1]
            pred_len = size[2]
        
        # Load the Dataset
        df_raw = pd.read_csv(args.root_path + args.data_path)
        
        cols = list(df_raw.columns)
        cols.remove(target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [target]]
        # print(cols)
        
        num_train = int(len(df_raw) * (1-args.test_ratio-0.1))
        num_test = int(len(df_raw) * args.test_ratio)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        if features == 'M' or features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif features == 'S':
            df_data = df_raw[[target]]
        
        if scale:
            scaler = get_scaler('minmax')
            train_data = get_df(flag='train')
            scaler.fit(train_data.values)
            data = scaler.transform(df_data.values)
            
            # Saving the scaler
            SCALER_PATH = './scalers/' + name + '/'
            if not os.path.exists(SCALER_PATH):    
                os.makedirs(SCALER_PATH)
            joblib.dump(scaler, SCALER_PATH + 'scaler.gz')
            print(f'Scaler is fixed for: {name}')
            
        else:
            data = df_data.values
        
        RESULTS_PATH = './results/' + args.setting + '/'
        if not os.path.exists(RESULTS_PATH):    
            os.makedirs(RESULTS_PATH)

        df_test_original = df_raw[border1s[2]:border2s[2]]
        df_test_original = df_test_original.set_index(['date'])
        df_test_original.index = pd.to_datetime(df_test_original.index)
        if not df_test_original.index.is_monotonic:
            df_test_original = df_test_original.sort_index()

        df_test_original.to_pickle(RESULTS_PATH + 'df_test.pkl')

stop
#%% Testing the scaler
 
if args.scale:         
    df_train = get_df(flag='train')
    df_val = get_df(flag='val')
    df_test = get_df(flag='test')
    num_cols = len(cols) - 1
    
    if args.model == 'LSTM':
        # with open(SCALER_PATH + 'train_scaler.pkl', 'rb') as file:
            # train_scaler = pickle.load(file)
        scaler_load = joblib.load(SCALER_PATH + 'scaler.gz')
        if args.embed == 'timeF':
            if args.time_scaled == 'Unscaled':
                scaler_load.inverse_transform(df_train[:,:num_cols])
                scaler_load.inverse_transform(df_val[:,:num_cols])
                scaler_load.inverse_transform(df_test[:,:num_cols])
    else:
        scaler_load = joblib.load(SCALER_PATH + 'scaler.gz')
        
        df_train = scaler_load.fit_transform(df_train)
        df_val = scaler_load.transform(df_val)
        df_test = scaler_load.transform(df_test)
    
        df_train_inv = scaler_load.inverse_transform(df_train)
        df_val_inv = scaler_load.inverse_transform(df_val)
        df_test_inv = scaler_load.inverse_transform(df_test)