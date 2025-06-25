import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import pickle
import warnings
import os
import datetime
import timeit
from tqdm import tqdm
import joblib

import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

from models import NLinear
from models import Informer
from models import Transformer
from models import Autoformer
from models import DLinear
from models.LSTM import LSTMModel

from exp.exp_main_env import Exp_Main
from exp.exp_lstm_env import ExpLSTM

import torch

from PhosphorusEnvironment import PhosphorusEnvironment as Env
from utils.env_helper import EnvHelper

warnings.filterwarnings('ignore')


#%% Plot and device options

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.xmargin'] = 0.02
plt.rcParams['axes.ymargin'] = 0.04
fig_dpi = 500
# plt.switch_backend('qt5agg')

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"{device}" " is available.")

# Frequency of the dataset
freq = datetime.timedelta(minutes=1)

#%% Initializing

# The raw dataset
dataset_name = 'wastewater.csv'

all_actual_values = False
improve = False

# If retrained
retrained = False
single_retrained = True
experiment_based = False
experiment = 'E3'
number_based = False
number_list = ['N5']
all_retrained = False
retrained_checkpoint_name = 'V2_E2N1 - 20Epochs_ActA_SGap_180MaxEL_Batch_final.pth' 

# Environment and the model specifications    
episode_length = 180
sequence_length = 240

# Create an instance of the helper
helper = EnvHelper()

# Choosing the models to run
single = False
best_test_results = True
best_env_results = False
dataset_based = True
all_models = False
setting = 'LSTM_New_CorrH_11F_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256Hidden_2LayerDim'
data_tag = 'New_CorrH'

models_list, best_models = helper.make_models_list(setting=setting,
                                      single=single,
                                      best_test_results=best_test_results,
                                      best_env_results=best_env_results,
                                      dataset_based=dataset_based,
                                      all_models=all_models,
                                      data_tag=data_tag,
                                      episode_length=episode_length)
    
print(f'The environment will run for {len(models_list)} model(s)') 

#%%

def sensitivity_analysis(df, args, model_name, checkpoint):
    scaler = MinMaxScaler()
    checkpoint_path = checkpoint
    start_date = pd.to_datetime(df.index[0])
    num_cols = len(df.columns)
    seq_len = 240
    
    if args.scale:
        # Load the scaler path
        SCALER_PATH = './scalers/' + args.setting + '/'
        if args.model == 'LSTM':
            if args.time_scaled == 'Unscaled':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    feature_scaler = joblib.load(SCALER_PATH + 'feature_scaler.gz') 
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    feature_scaler = joblib.load(SCALER_PATH + 'feature_scaler.gz') 
                    time_scaler = joblib.load(SCALER_PATH + 'time_scaler.gz') 
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                scaler = joblib.load(SCALER_PATH + 'scaler.gz')
    
    # Building the model
    def build_model(model_name):
        model_dict = {
            'LSTM': LSTMModel,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear
        }
        
        if model_name == 'LSTM':
            model = model_dict[model_name](args, device).float()
        else:
            model = model_dict[model_name].Model(args).float()

        return model
    
    model = build_model(model_name)
    model.load_state_dict(torch.load(os.path.join('./checkpoints/' + checkpoint_path, 'checkpoint.pth')))
    model = model.to(device)

    if model_name == 'LSTM':
        Exp = ExpLSTM
        exp = Exp(args, device, model)
    else:
        Exp = Exp_Main    
        exp = Exp(args, model, df_raw)
    
    model.eval()
    
    # Addition of Time Specifications if we need them
    def add_time_specs(df):
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
    
    def scale_data(df):
        if model_name == 'LSTM':
            arr = np.zeros(shape=(df.shape[0], args.in_features))
            if args.time_scaled == 'Scaled':
                # Scale the feature columns
                arr[:, :num_cols] = feature_scaler.transform(df.iloc[:, :num_cols])        
                # Scale the time columns
                arr[:, num_cols:] = time_scaler.transform(df.iloc[:, num_cols:])
            else:
                # Scale only the feature columns
                arr[:, :num_cols] = feature_scaler.transform(df.iloc[:, :num_cols])
                arr[:, num_cols:] = np.array(df.iloc[:, num_cols:])
        else:
            df = np.array(df)
            arr = scaler.transform(df[:, :num_cols])
        
        return arr
    
    df_columns = df.columns
    df = add_time_specs(df)
    df = scale_data(df)


    # Converting the arrays to dataframe, we need to do it because of the scaler
    # and also making Tensor dataset
    def make_df(arr, step):
        start = start_date
        first_date = start + (step)*freq
        index = pd.date_range(
            start=first_date, freq=freq, periods=args.seq_len)
        df = pd.DataFrame(arr, columns=df_columns)
        df = df.set_index(index)
        if model_name == 'LSTM' and args.embed == 'timeF':
            df = add_time_specs(df)

        return df

    # Predictor function which predicts every state based on the past data
    def state_predictor(input_data):
        df_forecast = input_data.copy()
        if model_name == 'LSTM':
            if args.scale:
                df_forecast = scale_data(df_forecast)
            forecasted = exp.predict(args.setting, df_forecast, save=False)
            forecasted = np.squeeze(forecasted, axis=0)
            forecasted = np.reshape(forecasted[0, :], (1, num_cols))
            # forecasted = _inverse_transform(forecasted)
            return forecasted
        else:
            df_forecast = df_forecast.rename_axis('date').reset_index(level=0)
            return exp.predict(df_forecast, args.setting)
    
    # Define the model evaluation function
    def evaluate_model(input_data, step):
        # Make predictions using the model
        input_data = make_df(input_data, step)
        predictions = state_predictor(input_data)
        # predictions = predictions.reshape(1, num_cols)
    
        return predictions
        
    # Step 4: Compute sensitivity for each feature separately
    sensitivity_indices = []
    n_features = len(df_columns)
    df = df[-20000:, :]
    
    # Create a problem definition for the current feature
    problem = {
        'num_vars': n_features,  # Total number of input features over time steps
        'names': [f'{name}' for name in df_columns],
        'bounds': [(0, 1) for _ in range(n_features)]  # Define bounds based on the data range
    }
    
    # Generate input samples using Saltelli sampling
    param_values = saltelli.sample(problem, N=64)  # Adjust N based on your needs
    
    # Evaluate the model for each input sample
    Y = np.empty((param_values.shape[0]))  # Array to store model predictions
    step = 0
    
    for i in tqdm(range(param_values.shape[0])):
        inputs = np.concatenate((df[i:i+seq_len-1, :n_features], param_values[i].reshape(1, n_features)), axis=0)
        y_pred_modified = evaluate_model(inputs, step)
        Y[i] = y_pred_modified[0, -1]
        step += 1
    
    Si = sobol.analyze(problem, Y, print_to_console=True)
         
    return Si

#%%

def plot_sensitivity(df_raw, sobol_dict):
    columns = df_raw.columns   
    fig, axs = plt.subplots(2,3, figsize=(6.5, 3), dpi=fig_dpi)
    axs_names = ['X1', 'X2', 'X3', 'X4', 'X5']
    for i, (model, ax) in enumerate(zip(sobol_dict.keys(), axs.ravel())):
        ax.set_yscale('log')
        df_result = pd.DataFrame({"variable":axs_names, "Sobol Si":None, "Sobol STi":None})
        df_result = df_result.set_index('variable')
        for j, var in enumerate(columns):
            df_result['Sobol Si'].iloc[j] = abs(sobol_dict[model]['S1'][j])
            df_result['Sobol STi'].iloc[j] = abs(sobol_dict[model]['ST'][j])
        
        df_result.plot.bar(stacked=False, ax=ax, rot=0)
        ax.xaxis.label.set_visible(False)
        ax.set_title(model, fontsize=8, pad=5)
        ax.tick_params(axis='both', labelsize=6)
        # ax.tick_params(axis='x', rotation=90)
        # ax.set_xticks(df_result.index)
        ax.legend(fontsize=6)
        
    plt.subplots_adjust(wspace=0.3, hspace=0.45)
    #fig.subplots_adjust(top=0.92, bottom=0.1, right=0.95, left=0.1)

    for i in range(axs.shape[1]): 
        axs[-1, i].xaxis.label.set_visible(True)
        axs[-1, i].set_xlabel('Features', labelpad=5, fontsize=8)
        
    for j in range(axs.shape[0]):
        axs[j, 0].set_ylabel('Sensitivity Indices (log)', labelpad=5, fontsize=8)

    FIG_PATH = './figures/' 
    if not os.path.exists(FIG_PATH):    
        os.makedirs(FIG_PATH)
    
    fig_name = 'Sobol'
    plt.savefig(FIG_PATH + fig_name + '_' + str(fig.get_dpi()) + 'dpi' + '.svg')
    plt.savefig(FIG_PATH + fig_name + '_' + str(fig.get_dpi()) + 'dpi' +'.pdf') 
    plt.savefig(FIG_PATH + fig_name + '_' + str(fig.get_dpi()) + 'dpi' + '.png')
    
#%%

if __name__ == '__main__': 
    # Loading the dataset
    df_raw = pd.read_csv('./datasets/' + dataset_name)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.set_index(["date"])
    if not df_raw.index.is_monotonic_increasing:
        df_raw = df_raw.sort_index()
        
    sobol_dict = {key:None for key in best_models.keys()}
    
    for model in best_models.values(): 
        for name in best_models.keys():
            if name in model:
                model_name = name
                
        torch.cuda.empty_cache() 
        start = timeit.default_timer() 
        '''
        checkpoints_list, checkpoints_names = helper.make_retrained_list(setting=model,
                                                                         checkpt_name=retrained_checkpoint_name,
                                                                         single_retr=single_retrained,
                                                                         experiment_based=experiment_based,
                                                                         experiment=experiment,
                                                                         number_based=number_based,
                                                                         number_list=number_list,
                                                                         all_retrained=all_retrained)
        
        
        checkpoints_names = [string.replace('.pth', '') for string in checkpoints_list]
        '''
        # Load the args file of the model
        ARGS_PATH = './args/' + model + '/'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with open(ARGS_PATH + 'args.pkl', 'rb') as file:
                args = pickle.load(file)
                
        if all_models:
            print(f'{args.setting}:')
        else:
            print(f'{args.model}')
            
        embed = args.embed    
        target = args.target
        control_variable = args.control_variable
        '''
        indices = sensitivity_analysis(df_raw, args, args.model, model)
        sobol_dict[model_name] = indices
               
        # Getting the results dictionary
        if retrained:
            pass
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with open('sobol_dict.pkl', 'wb') as file:
            pickle.dump(sobol_dict, file)
       ''' 
    #%%
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with open('sobol_dict.pkl', 'rb') as file:
            sobol_dict = pickle.load(file)
    
    plot_sensitivity(df_raw, sobol_dict)
    

        
