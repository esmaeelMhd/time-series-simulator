"""
Created on August 2023
@author: Esmaeel Mohammadi

# =============================================================================
# This script is used to test the simulation environment for improved models
    1. Making the models list
    2. Making the dataset and build models
    3. Running the simulation test for model checkpoints
    4. Creating the simulation figures and tables
# =============================================================================
"""
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
from exp.exp_lstm_env import ExpLSTM
from utils.env_helper import EnvHelper
from models.LSTM import LSTMModel, EncoderLSTM, DecoderLSTM, Net_LSTM
from models import DLinear
from models import Autoformer
from models import Transformer
from models import Informer
from models import NLinear
import numpy as np
import pandas as pd
import pickle
import warnings
import os
import datetime
import timeit
import seaborn as sns
import matplotlib.pyplot as plt
import re

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import copy

# from exp.exp_main_env import Exp_Main

# import torch.utils.data as data

# warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore')

# %% Plot and device options

sns.set_style("white")
# plt.switch_backend('qt5agg')
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except Exception as e:
    pass

# plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.xmargin'] = 0.02
plt.rcParams['axes.ymargin'] = 0.04
plt.rcParams['axes.labelsize'] = 8

plt.rc('axes', titlesize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)

# Figures quality
fig_dpi = 1500
full_hd_w = 1920
full_hd_h = 1080

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"{device}" " is available.")

# %% Initializing

# The raw dataset
dataset_name = 'wastewater.csv'

# Environment and the model specifications
start_date = pd.to_datetime('2021-09-01 00:00:00+00:00')
end_date = pd.to_datetime('2022-09-01 00:00:00+00:00')
ep_len = 1440
sequence_length = 240
sim_plot_el = 720

# Run the tests for not found env results dics
run_test = True

# Retrain experiments
ex_list = ['E1', 'E2', 'E3', 'E4']

# The path to TEX file
LATEX_PATH = 'C:/Users/esmaeel.mohammadi/Dropbox/Apps/Overleaf/Model Improve 2/'

# Create an instance of the helper
helper = EnvHelper()

# Choose the helper legend size for plots
helper.legend_size = 6

# Type of the plots
best_by_ex = True
best_by_model = False

# Choosing the models to run
# custom_model_list = ['LSTM_IOPP_2min_10F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L',
#                      'LSTM_IOPTQCfFiFoP_2min_15F_6Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L',
#                      'LSTM_IOPTQCfFiFoP_2min_15F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L',
#                      ]

custom_model_list = []
custom_model_names = {'LSTM_IOPP_2min_10F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L': '$f_0$',
                     'LSTM_IOPTQCfFiFoP_2min_15F_6Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L': '$f_A$',
                     'LSTM_IOPTQCfFiFoP_2min_15F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L': '$f_B$'}

do_custom_names = False

single = True
best_test_results = False
best_env_results = False
model_based = True
dataset_based = True
all_models = False
setting = 'LSTM_IOPTQCfFiFoP_2min_15F_1Out_timeF_Unscaled_1Seq_0Label_16Batch_1e-06LR_256H_2L'
model_name = 'LSTM'
data_tag = ['IOPQCfFiFoP_2min', 'IOPTQCfFiFoP_2min']

data_tag_name = 'exogenous_sepp_1seq'   # 'exogenous'
# ['IOPQCfFiFoP_2min', 'IOPTQCfFiFoP_2min']
# ['IOTNNSP', 'IOTNNSP', 'IOTNNSP', 'IOTNNSP', 'IOPQCfFiFoP_2min', 'IOPTQCfFiFoP_2min']

all_best = False if single else True

if len(custom_model_list) == 0:
    models_list, best_models, _ = helper.make_models_list(
        setting=setting,
        single=single,
        best_test_results=best_test_results,
        best_env_results=best_env_results,
        model_based=model_based,
        dataset_based=dataset_based,
        all_models=all_models,
        model_name=model_name,
        data_tag=data_tag,
        episode_length=ep_len)
else:
    models_list=custom_model_list

# Set the name for doing all models
all_best_name = 'All'
if model_based:
    all_best_name += '_' + model_name
if dataset_based:
    all_best_name += '_' + data_tag_name

# If retrained
retrained = True
single_retrained = False
experiment_based = False
experiments = ['E3', 'E4']
number_based = False
number_list = ['N5']
all_retrained = True
retrained_chkpt_name = 'V2_E2N1 - 20Epochs_ActA_SGap_180MaxEL_Batch_final.pth'

# Points to test
test_frequency = 'Seasons'      # options: Months, Seasons
day_of_the_month = 'Middle'     # options: First, Middle
time_of_the_day = 'Morning'     # options: Morning, Noon
first_date = pd.to_datetime('2021-10')
last_date = pd.to_datetime('2022-09')
points = test_frequency + '_' + day_of_the_month + '_' + time_of_the_day

list_points, points_names = helper.make_points(
    test_frequency=test_frequency,
    day_of_the_month=day_of_the_month,
    time_of_the_day=time_of_the_day,
    first_date=first_date,
    last_date=last_date)

print(f'The environment will run for {len(models_list)} model(s).')

# %% Helper functions

def make_df(args, df_raw):
    df = helper.add_time_specs(df_raw)
    df = helper.scale_data(args, df)
    return df


def data_split(df, seq_len, ep_len, start_date, end_date, freq_min):
    X = []
    y = []
    list_points = []
    points_names = []
    start_idx = df_raw.index.get_loc(start_date) - seq_len
    end_idx = df_raw.index.get_loc(end_date)
    
    ep_len = int(ep_len / freq_min)

    while start_idx + seq_len + ep_len <= end_idx:
        end = start_idx + seq_len
        X.append(df[start_idx: end])
        y.append(df[end: end + ep_len])
        point = df_raw.index[end]
        list_points.append(point)
        points_names.append(f'{point.strftime("%b %d %Y")}')
        start_idx = start_idx + ep_len

    return (X, y, list_points, points_names)


def build_model(args):
    model_dict = {
        'LSTM': LSTMModel,
        'Autoformer': Autoformer,
        'Transformer': Transformer,
        'Informer': Informer,
        'DLinear': DLinear,
        'NLinear': NLinear
    }

    if args.model == 'LSTM':
        if not hasattr(args, 'lstm_type') or (hasattr(args, 'lstm_type') and args.lstm_type != 'EncDec'):
            model = LSTMModel(args).float()
        else:
            encoder = EncoderLSTM(args)
            decoder = DecoderLSTM(args)
            model = Net_LSTM(encoder, decoder, args, device).float()
    else:
        model = model_dict[args.model].Model(args).float()

    if args.use_multi_gpu and args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    return model


def prepare_vars(args):
    ctrl_vars = []
    if hasattr(args, 'ctrl_vars'):
        if isinstance(args.ctrl_vars, str):
            ctrl_vars.append(args.ctrl_vars)
        elif isinstance(args.ctrl_vars, list):
            ctrl_vars = args.ctrl_vars
    elif hasattr(args, 'control_variable'):
        if isinstance(args.control_variable, str):
            ctrl_vars.append(args.control_variable)
        elif isinstance(args.control_variable, list):
            ctrl_vars = args.control_variable

    ind_vars = []
    if hasattr(args, 'ind_vars'):
        if isinstance(args.ind_vars, str):
            ind_vars.append(args.ind_vars)
        elif isinstance(args.ind_vars, list):
            ind_vars = args.ind_vars
    elif hasattr(args, 'independent_vars'):
        if isinstance(args.independent_vars, str):
            ind_vars.append(args.independent_vars)
        elif isinstance(args.independent_vars, list):
            ind_vars = args.independent_vars

    num_time_f = 6
    if hasattr(args, 'num_time_f'):
        num_time_f = args.num_time_f

    return ctrl_vars, ind_vars, num_time_f

# %% Simulation Plotting
def plot_sim(args, plot_dict, df_plot, ep_len, start_date, end_date,
             plot_type='Seasons', plot_el=360, plot_base=True, normalize=False, 
             all_best=False, all_best_name='All', plot_type_best='best_by_ex'):

    # Plot the worst and best points
    if 'best' in plot_type.lower() or 'worst' in plot_type.lower():
        plot_dict['Base Model'] = {'name': 'Base Model', 'mse': None}
        max_mse_idx = df_plot['point_avg'].idxmax()
        min_loss_idx = df_plot['point_avg'].idxmin()
        plot_points = {'Best MSE': min_loss_idx, 'Worst MSE': max_mse_idx}

        f_name = f'{plot_el}_rounds_Best_and_Worst_retrained'
        helper.plot_retrained(plot_dict, plot_points, args=args, plot_base=plot_base,
                              start_date=start_date, end_date=end_date, el=ep_len, plot_el=plot_el,
                              normalize=normalize, fig_name=f_name, plot_type_best=plot_type_best, latex_path=LATEX_PATH, 
                              seq_len=sequence_length, do_custom_names=do_custom_names, custom_model_names=custom_model_names)

        helper.plot_sim_with_cv(plot_dict, plot_points, args=args, plot_base=plot_base,
                                start_date=start_date, end_date=end_date, el=ep_len, plot_el=plot_el,
                                normalize=normalize, fig_name=f_name + '_with_metal')

    # Plot all true values
    if 'true' in plot_type.lower():
        l_points = [pd.to_datetime('2021-09-15 00:00:00')]
        p_names = ['Sep 15 2021']
        plot_points = {}
        for point, name in zip(l_points, p_names):
            plot_points[name] = point
        f_name = f'{plot_el}_rounds_{p_names[0]}'
        helper.plot_sim_with_cv(plot_dict, plot_points, args=args, plot_base=plot_base, plot_all=True,
                                start_date=start_date, end_date=end_date, el=ep_len, plot_el=plot_el,
                                normalize=normalize, fig_name=f_name + '_with_metal')

    # Plot seasons
    if 'seasons' in plot_type.lower():
        tf = 'Seasons'
        day_of_m = 'Middle'
        t_of_day = 'Morning'
        f_date = pd.to_datetime('2021-09')
        l_date = pd.to_datetime('2022-08')
        l_points, p_names = list_points, points_names = helper.make_points(
            test_frequency=tf, day_of_the_month=day_of_m, time_of_the_day=t_of_day,
            first_date=f_date, last_date=l_date)

        plot_points = {}
        for point, name in zip(l_points, p_names):
            plot_points[name] = point

        if all_best:
            f_name = f'{all_best_name}_{plot_el}_rounds_' + tf + '_' + day_of_m + '_' + t_of_day +\
                f'_{f_date.strftime("%b %d %Y")}_to_{l_date.strftime("%b %d %Y")}'

            helper.plot_retrained(plot_dict, plot_points, args=args, plot_base=plot_base,
                                  start_date=start_date, end_date=end_date, el=ep_len, plot_el=plot_el,
                                  normalize=normalize, fig_name=f_name, all_best=all_best, all_best_name=all_best_name,
                                  plot_type_best=plot_type_best, latex_path=LATEX_PATH, seq_len=sequence_length,
                                  do_custom_names=do_custom_names, custom_model_names=custom_model_names)

            # helper.plot_sim_with_cv(plot_dict, plot_points, args=args, plot_base=plot_base,
            #                       start_date=start_date, end_date=end_date, el=ep_len, plot_el=plot_el,
            #                      normalize=normalize, fig_name=f_name + '_with_metal')
        else:
            f_name = f'{plot_el}_rounds_' + tf + '_' + day_of_m + '_' + t_of_day +\
                f'_{f_date.strftime("%b %d %Y")}_to_{l_date.strftime("%b %d %Y")}'
            
            helper.plot_retrained(plot_dict=plot_dict, plot_points=plot_points, args=args, plot_base=plot_base,
                                  start_date=start_date, end_date=end_date, el=ep_len, plot_el=plot_el,
                                  normalize=normalize, fig_name=f_name, plot_type_best=plot_type_best, latex_path=LATEX_PATH,
                                  seq_len=sequence_length, do_custom_names=do_custom_names, custom_model_names=custom_model_names)

            # helper.plot_sim_with_cv(plot_dict, plot_points, args=args, plot_base=plot_base,
                                    #start_date=start_date, end_date=end_date, el=ep_len, plot_el=plot_el,
                                    #normalize=normalize, fig_name=f_name + '_with_metal')

# %% Find best improvement results
def export_best_results(args, chkpt_list, chkpt_names, ex_list, points_names, ep_len=1440):
    loss_type = 'dtw'
    alpha = 0.5  # alpha*mse + (1-alpha)dtw

    best_retrained, df_monthly, df_loss, all_loss_dict, df_data = helper.find_best_results(
        args, chkpt_list, chkpt_names, ex_list, points_names, start_date,
        end_date, loss_type, alpha)

    def create_df_plot():
        for experiment in best_retrained.keys():
            for col in df_data.columns:
                if experiment in col and best_retrained[experiment]['name'] not in col:
                    df_data.drop(col, axis=1, inplace=True)
                elif experiment in col and '_mse' in col:
                    df_data.rename(
                        columns={col: experiment+'_mse'}, inplace=True)
                elif experiment in col and '_dtw' in col:
                    df_data.rename(
                        columns={col: experiment+'_dtw'}, inplace=True)

        plot_names = [key for key in best_retrained.keys()]
        names = []
        for experiment in plot_names:
            names.append(best_retrained[experiment]['name'])

        names = [name for name in names if name != None]
        df_plot = pd.DataFrame(columns=names)
        df_plot['Datetime'] = df_loss.columns
        df_plot.set_index('Datetime', inplace=True)
        for col in df_plot.columns:
            df_plot[col] = df_loss.loc[col]

        df_plot = df_plot.astype(float)

        return df_plot

    df_plot = create_df_plot()

    best_chkpt_names = []
    for key in best_retrained.keys():
        best_chkpt_names.append(best_retrained[key]['name'])
    print(f'Best Checkpoints: {best_chkpt_names}')

    if not df_plot.isna().all().all():
        helper.plot_box_heat(args, ex_list, df_plot, ep_len, start_date, end_date,
                             plot_base=False, loss_type=loss_type, alpha=alpha)
        print('Heatmap and Box plots are done.')

        helper.plot_mse_together(args, best_retrained, df_plot, plot_base=True,
                                 points_names=points_names, start_date=start_date,
                                 end_date=end_date, loss_type=loss_type, alpha=alpha,
                                 do_custom_names=do_custom_names, custom_model_names=custom_model_names)
        print('Loss together plot is done.')

        # Plot simulations
        plot_sim(args, best_retrained, df_plot, ep_len, start_date, end_date,
                 plot_type='Seasons', plot_el=sim_plot_el, plot_base=True, normalize=False)
        print('Seasons simulation plot is done.')

    # Save df_data to csv
    df_data = df_data.astype(float)
    df_data.index = df_data.index.strftime('%b')
    for col in df_data.columns:
        df_data.loc['Average', col] = df_data[col].mean()
    df_data = df_data.round(4)

    def save_csv(df):
        csv_name = f'monthly_loss_{ep_len}EL_' + f'{start_date.strftime("%b %d %Y")}' +\
            '_to_' + f'{end_date.strftime("%b %d %Y")}'

        if not os.path.exists('./env_results/' + args.setting):
            os.makedirs('./env_results/' + args.setting)

        df.to_csv('./env_results/' + args.setting + '/' + csv_name + '.csv')

    save_csv(df_data)

    # Create df_params and write to table
    def create_df_params():
        df_params = pd.DataFrame(columns=['Models', 'params_Epochs', 'params_Min EL', 'params_Max EL',
                                          'params_Loss Function', 'params_Alpha DILATE'])
        df_params['Models'] = ex_list.remove('Base Model')
        df_params.set_index('Models', inplace=True)
        for experiment in best_retrained.keys():
            if experiment == 'Base Model':
                continue

            if best_retrained[experiment]['name'] != None:
                params_list = best_retrained[experiment]['name'].split('_')
                for item in params_list:
                    if 'Ep' in item:
                        df_params.loc[experiment, 'params_Epochs'] = item.replace(
                            'Ep', '').split(' ')[-1]
                    elif 'MaxEL' in item:
                        df_params.loc[experiment, 'params_Min EL'] = str(10)
                        df_params.loc[experiment, 'params_Max EL'] = item.replace(
                            'MaxEL', '')
                    elif 'EL' in item:
                        df_params.loc[experiment,
                                      'params_Min EL'] = item.replace('EL', '')
                        df_params.loc[experiment,
                                      'params_Max EL'] = item.replace('EL', '')
                    elif 'dilate' in item:
                        df_params.loc[experiment,
                                      'params_Loss Function'] = 'Dilate'
                        df_params.loc[experiment, 'params_Alpha DILATE'] = [
                            str(x.replace('A', '')) for x in params_list if re.search(r'\d+A', x)][0]

                if 'dilate' not in params_list:
                    df_params.loc[experiment, 'params_Loss Function'] = 'MSE'
                    df_params.loc[experiment, 'params_Alpha DILATE'] = '-'
            else:
                df_params.loc[experiment, :] = None

        return df_params

    df_params = create_df_params()

    # Create the parameters table in TEX
    name = f'params_table_{args.data_tag}'
    caption = 'The parameters of the best improved checkpoints for each experiment'
    label = 'tab:params_retrain'
    index_name = 'Experiment'
    helper.write_params_to_tex(
        df_params, LATEX_PATH, name, caption, label, index_name)
    print('Parameters table is generated.')

    # Create the metrics table in TEX
    name = f'metrics_table_{args.data_tag}'
    caption = 'The average Mean Squared Error and Dynamic Time Warping data for the base model and improved versions' +\
        ' during different months of the year. The best values of MSE and DTW for each month are highlighted in bold.'
    label = 'tab:metrics_monthly'
    index_name = 'Month'
    helper.write_to_tex(df_data, LATEX_PATH, name, caption,
                        label, index_name, highlight=True)
    print('Metrics table is generated.')

    return best_retrained, df_monthly, all_loss_dict, df_data


if __name__ == '__main__':
    data_tags = []
    # Test the simulation for each model
    for model in models_list:
        torch.cuda.empty_cache()
        # Timer start
        # start = timeit.default_timer()

        # List of checkpoints for the model
        chkpt_list, chkpt_names = helper.make_retrained_list(
            setting=model,
            chkpt_name=retrained_chkpt_name,
            single_retr=single_retrained,
            experiment_based=experiment_based,
            experiments=experiments,
            number_based=number_based,
            number_list=number_list,
            all_retrained=all_retrained)

        chkpt_names = [string.replace('.pth', '') for string in chkpt_list]
        # Add the base model to the checkpoints list and names
        chkpt_list.insert(0, 'checkpoint.pth')
        chkpt_names.insert(0, 'Base')

        # Load the args file of the model
        ARGS_PATH = './args/' + model + '/'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with open(ARGS_PATH + 'args.pkl', 'rb') as file:
                args = pickle.load(file)

        # Loading the dataset
        df_raw = pd.read_csv('./datasets/' + args.data_path)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.set_index(["date"])
        if not df_raw.index.is_monotonic_increasing:
            df_raw = df_raw.sort_index()

        # Add data_tag to the list
        data_tags.append(args.data_tag if hasattr(args, 'data_tag') else None)

        # Build the model
        model = build_model(args).to(device)
        print(f'{args.model}') if single else print(f'{args.setting}:')

        # The missing test results of checkpoints
        missing_list = helper.check_missing_tests(
            args, chkpt_names, start_date, end_date, ep_len)

        # Preparing the test parameters
        embed = args.embed
        target = args.target
        seq_len = args.seq_len
        ctrl_vars, ind_vars, num_time_f = prepare_vars(args)

        df_test = make_df(args, df_raw)
        freq = df_raw.index.to_series().diff().dropna().mode()[0]
        freq_min = freq.total_seconds() / 60
        helper.freq = freq
        X_test, y_test, list_points, points_names = data_split(
            df_test, seq_len, ep_len=ep_len, start_date=start_date, end_date=end_date, freq_min=freq_min)

        # Creating the test data
        X_test = torch.Tensor(np.array(X_test)).to(device)
        y_test = torch.Tensor(np.array(y_test)).to(device)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        def environment_test(rounds, checkpoint, chkpt_name=''):
            dict_name = f'{ep_len}EL_' + chkpt_name + '_' + f'{start_date.strftime("%b %d %Y")}' +\
                        '_to_' + f'{end_date.strftime("%b %d %Y")}'
            file_path = './env_results/' + args.setting + '/' + dict_name + '.pkl'

            if os.path.exists(file_path):
                # print(f'The file for {chkpt_name} already exists')
                return None, None

            else:
                print(chkpt_name)
                results_dict = {key: None for key in points_names}
                checkpoint_path = './checkpoints/' + args.setting
                model.load_state_dict(torch.load(
                    os.path.join(checkpoint_path, checkpoint)))
                if args.model == 'LSTM':
                    exp = ExpLSTM(args, model)
                    optimizer = exp.opt
                else:
                    # exp = Exp_Main(args, model, df_raw, df_stamp)
                    pass

                # Get the test results
                results_list, losses = optimizer.test_simulation(
                    model, test_loader)
                for idx, (key, item) in enumerate(zip(results_dict, results_list)):
                    results_dict[key] = item
                    y_real = results_dict[key]['y_real']
                    y_pred = results_dict[key]['y_pred']
                    results_dict[key]['y_real'] = helper.inverse_transform(
                        y_real)
                    results_dict[key]['y_pred'] = helper.inverse_transform(
                        y_pred)

                    if 'loss_dtw' not in item.keys():
                        item = helper.add_dtw(item)

                if not os.path.exists('./env_results/' + args.setting):
                    os.makedirs('./env_results/' + args.setting)

                # Save the results to pickle file
                with open('./env_results/' + args.setting + '/' + dict_name + '.pkl', 'wb') as file:
                    pickle.dump(results_dict, file)

                return results_dict, losses

        # Getting the results dictionary
        for i, checkpoint in enumerate(chkpt_list):
            # Testing the Environment
            chkpt_name = chkpt_names[i]
            if run_test:
                results_dict, losses = environment_test(ep_len, checkpoint, chkpt_name)
                #stop = timeit.default_timer()
                #print(f'Time: {round((stop - start)/60)} minutes')
                #print(f'Time for each step: {round((stop - start)/(len(list_points)*ep_len))} seconds')

        helper.feature_scaler = None
        helper.time_of_the_day = None
        helper.scaler = None
        # best_results, df_loss, all_loss, df_data = export_best_results(
        # args, chkpt_list, chkpt_names, ex_list, points_names, ep_len)

# %%
def export_best_results_all(models_list, ex_list, points_names, data_tags, ep_len=1440):
    loss_type = 'dtw'
    alpha = 0.5  # alpha*mse + (1-alpha)*dtw

    models_folders = []
    for model in models_list:
        models_folders.append('./env_results/' + model + '/')
    # Function to make a list of all checkpoints in all models folders

    def list_all_checkpoints(models_folders):
        all_files = []
        for folder in models_folders:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    all_files.append(os.path.join(root, file))
        return all_files

    chkpt_files_dir = list_all_checkpoints(models_folders)
    experiments = ex_list
    experiments.insert(0, 'Base Model')

    global all_dict, all_best_dict, tags_dict
    if best_by_ex:
        all_dict = {key: None for key in experiments}
        all_best_dict = {key: None for key in experiments}
    elif best_by_model:
        all_dict = {key: None for key in models_list}
        all_best_dict = {key: None for key in models_list}
        
    tags_dict = {key: None for key in data_tags}
        
    for tag in data_tags:
        tag_list = [item for item in models_list if tag in item]
        tag_dict = {key: None for key in tag_list}
        for model in tag_list:
            # Load the args file of the model
            ARGS_PATH = './args/' + model + '/'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                with open(ARGS_PATH + 'args.pkl', 'rb') as file:
                    args = pickle.load(file)

            model_dict = {key: None for key in [
                'name', 'best_retrained', 'df_data']}
            model_dict['name'] = args.model + '_' + \
                args.lstm_type if args.lstm_type != '' else args.model

            # List of checkpoints for the model
            chkpt_list, chkpt_names = helper.make_retrained_list(setting=model)
            chkpt_names = [string.replace('.pth', '') for string in chkpt_list]
            # Add the base model to the checkpoints list and names
            chkpt_list.insert(0, 'checkpoint.pth')
            chkpt_names.insert(0, 'Base')

            model_dict['best_retrained'], _, _, _, model_dict['df_data'] = helper.find_best_results(
                args, chkpt_list, chkpt_names, ex_list, points_names, start_date,
                end_date, loss_type, alpha)

            tag_dict[model] = model_dict

        tags_dict[tag] = tag_dict

    global all_df
    if best_by_ex:
        all_df = pd.DataFrame(columns=['Data'])
        all_df['Data'] = data_tags
        all_df = all_df.set_index('Data')
    elif best_by_model:
        all_df = pd.DataFrame(columns=['Data'])
        all_df['Data'] = models_list
        all_df = all_df.set_index('Data')
    
    # Best for the model if only one model
    df = model_dict['df_data']
    df = pd.concat([df, pd.DataFrame(df.mean().rename('Average')).T])
    
    for column in df.columns:
        if 'Base' in column:
            pass
        else:
            if '80Ep' not in column:
                df.drop(columns=[column], inplace=True)
            else:
                items = column.split(' - ')
                name = items[0] + '_' + items[-1].split('_')[-1]
                df.rename(columns={column: name}, inplace=True)
    
    df = df.iloc[[-1]].T
    
    def best_results_by_ex():
        for experiment in experiments:
            ex_dict = {key: None for key in data_tags}
            for tag in data_tags:
                tag_dict = tags_dict[tag]
                best_dict = {}
                best_ex = best_ex_mse = best_ex_dtw = None
                best_ex_name = best_model_name = None
                for model in tag_dict.keys():
                    ex_loss = tag_dict[model]['best_retrained'][experiment]['loss']
                    if ex_loss != None and (best_ex == None or ex_loss < best_ex):
                        best_ex = ex_loss
                        best_model_name = model
                        best_ex_mse = tag_dict[model]['best_retrained'][experiment]['mse']
                        best_ex_dtw = tag_dict[model]['best_retrained'][experiment]['dtw']
                        best_ex_name = tag_dict[model]['best_retrained'][experiment]['name']
    
                best_dict['loss'] = best_ex
                best_dict['model_name'] = best_model_name
                best_dict['mse'] = best_ex_mse
                best_dict['dtw'] = best_ex_dtw
                best_dict['ex_name'] = best_ex_name
    
                ex_dict[tag] = best_dict
                all_df.loc[tag, experiment + '_mse'] = best_dict['mse']
                all_df.loc[tag, experiment + '_dtw'] = best_dict['dtw']
    
            all_dict[experiment] = ex_dict
            best_loss = best_loss_dict = best_tag_name = None
            best_tag_dict = {key: None for key in ['name', 'info']}
            for tag in data_tags:
                tag_loss = all_dict[experiment][tag]['loss']
                if tag_loss != None and (best_loss == None or tag_loss < best_loss):
                    best_loss = tag_loss
                    best_tag_name = tag
                    best_loss_dict = all_dict[experiment][tag]
    
            best_tag_dict['name'] = best_tag_name
            best_tag_dict['info'] = best_loss_dict
            all_best_dict[experiment] = best_tag_dict
            
        return all_best_dict

    def best_results_by_model():
        for model in models_list:
            for t in data_tags:
                if t in model:
                    tag = t
            model_dict = tags_dict[tag][model]
            best_dict = {}
            best_ex = best_ex_mse = best_ex_dtw = None
            best_ex_name = None
            for experiment in experiments:
                ex_loss = model_dict['best_retrained'][experiment]['loss']
                if ex_loss != None and (best_ex == None or ex_loss < best_ex):
                    best_ex = ex_loss
                    best_ex_mse = model_dict['best_retrained'][experiment]['mse']
                    best_ex_dtw = model_dict['best_retrained'][experiment]['dtw']
                    best_ex_name = model_dict['best_retrained'][experiment]['name']
                
                all_df.loc[model, experiment + '_mse'] = model_dict['best_retrained'][experiment]['mse']
                all_df.loc[model, experiment + '_dtw'] = model_dict['best_retrained'][experiment]['dtw']

            best_dict['loss'] = best_ex
            best_dict['mse'] = best_ex_mse
            best_dict['dtw'] = best_ex_dtw
            best_dict['ex_name'] = best_ex_name
    
            all_best_dict[model] = best_dict
            
        return all_best_dict
            
    if best_by_ex:
        all_best_dict = best_results_by_ex()
    elif best_by_model:
        all_best_dict = best_results_by_model()
        
    def create_df_plot_by_ex():
        plot_names = [key for key in all_best_dict.keys()]
        names = []
        for experiment in plot_names:
            if all_best_dict[experiment]['name'] != None:
                names.append(experiment + ' - ' +
                             all_best_dict[experiment]['name'])
            else:
                names.append(None)

        global df_plot
        df_plot = pd.DataFrame(columns=['Datetime'] + names)
        for experiment in plot_names:
            for col in df_plot.columns:
                if col != None and experiment in col:
                    model = all_best_dict[experiment]['info']['model_name']
                    # Load the args file of the model
                    ARGS_PATH = './args/' + model + '/'
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        with open(ARGS_PATH + 'args.pkl', 'rb') as file:
                            args = pickle.load(file)

                    # List of checkpoints for the model
                    chkpt_list, chkpt_names = helper.make_retrained_list(
                        setting=model)
                    chkpt_names = [string.replace(
                        '.pth', '') for string in chkpt_list]
                    # Add the base model to the checkpoints list and names
                    chkpt_list.insert(0, 'checkpoint.pth')
                    chkpt_names.insert(0, 'Base')

                    _, _, df_loss, _, _ = helper.find_best_results(
                        args, chkpt_list, chkpt_names, ex_list, points_names, start_date,
                        end_date, loss_type, alpha)

                    df_plot['Datetime'] = df_loss.columns
                    df_plot[col] = df_loss.loc[all_best_dict[experiment]
                                               ['info']['ex_name']].values

        df_plot.set_index('Datetime', inplace=True)
        df_plot = df_plot.astype(float)

        return df_plot

    def create_df_plot_by_model():
        plot_names = [key for key in all_best_dict.keys()]
        names = []
        for model in plot_names:
            if all_best_dict[model]['ex_name'] != None:
                names.append(model)
            else:
                names.append(None)

        global df_plot
        df_plot = pd.DataFrame(columns=['Datetime'] + names)
        for model in plot_names:
            for col in df_plot.columns:
                if col != None and model in col:
                    experiment = all_best_dict[model]['ex_name']
                    # Load the args file of the model
                    ARGS_PATH = './args/' + model + '/'
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        with open(ARGS_PATH + 'args.pkl', 'rb') as file:
                            args = pickle.load(file)

                    # List of checkpoints for the model
                    chkpt_list, chkpt_names = helper.make_retrained_list(
                        setting=model)
                    chkpt_names = [string.replace(
                        '.pth', '') for string in chkpt_list]
                    # Add the base model to the checkpoints list and names
                    chkpt_list.insert(0, 'checkpoint.pth')
                    chkpt_names.insert(0, 'Base')

                    _, _, df_loss, _, _ = helper.find_best_results(
                        args, chkpt_list, chkpt_names, ex_list, points_names, start_date,
                        end_date, loss_type, alpha)

                    df_plot['Datetime'] = df_loss.columns
                    df_plot[col] = df_loss.loc[all_best_dict[model]['ex_name']].values

        df_plot.set_index('Datetime', inplace=True)
        df_plot = df_plot.astype(float)

        return df_plot
    
    if best_by_ex:
        df_plot = create_df_plot_by_ex()
    elif best_by_model:
        df_plot = create_df_plot_by_model()

    best_chkpt_names = []
    if best_by_ex:
        for key in all_best_dict.keys():
            if all_best_dict[key]['name'] != None:
                best_chkpt_names.append(key + ' - ' + all_best_dict[key]['name'])
            else:
                best_chkpt_names.append(None)
        print(f'Best Checkpoints: {best_chkpt_names}')
    
    df_base = copy.deepcopy(df_plot[[col for col in df_plot.columns if 'Base Model' in col]])
    
    if not df_plot.isna().all().all():
        plot_type = 'best_by_ex' if best_by_ex else 'best_by_model'
        if best_by_ex:
            helper.plot_box_heat(args, ex_list, df_plot, ep_len, start_date, end_date,
                                 plot_base=False, loss_type=loss_type, alpha=alpha, all_best=all_best, all_best_name=all_best_name)
            
            helper.plot_box_heat(args, ex_list, df_base, ep_len, start_date, end_date,
                                 plot_base=True, loss_type=loss_type, alpha=alpha, all_best=all_best, all_best_name=all_best_name)
            print('Heatmap and Box plots are done.')

        helper.plot_mse_together(args, all_best_dict, df_plot, plot_base=True,
                                 points_names=points_names, start_date=start_date,
                                 end_date=end_date, loss_type=loss_type, alpha=alpha, all_best=all_best, 
                                 all_best_name=all_best_name, plot_type=plot_type,
                                 do_custom_names=do_custom_names, custom_model_names=custom_model_names)
        print('Loss together plot is done.')

        # Plot simulations
        plot_sim(args, all_best_dict, df_plot, ep_len, start_date, end_date,
                 plot_type='Seasons', plot_el=sim_plot_el, plot_base=True, normalize=False, 
                 all_best=all_best, all_best_name=all_best_name, plot_type_best=plot_type)
        print('Seasons simulation plot is done.')

    # Save df_data to csv
    all_df = all_df.astype(float)
    if best_by_ex:
        all_df = all_df.sort_index()
    elif best_by_model:
        all_df = all_df.reindex(models_list)
        
    for col in all_df.columns:
        all_df.loc['Average', col] = all_df[col].mean()
    all_df = all_df.round(4)
    
    if best_by_model:
        new_index = []
        for idx in all_df.index:
            if idx != 'Average':
                if do_custom_names:
                    new_name = custom_model_names[idx]
                else:
                    data_tag, in_features, out_features = helper.extract_model_info(idx)
                    new_name = f'{data_tag}-{in_features}i{out_features}o'
                new_index.append(new_name)
        new_index.append(all_df.index[-1])
        all_df.index = new_index
        
    def save_csv(df):
        csv_name = f'{all_best_name}_monthly_loss_{ep_len}EL_{start_date.strftime("%b %d %Y")}' +\
            '_to_' + f'{end_date.strftime("%b %d %Y")}'
        df.to_csv('./env_results/All Best/' + csv_name + '.csv')

    save_csv(all_df)

    # Create df_params and write to table
    def create_df_params_by_ex():
        df_params = pd.DataFrame(columns=['Models', 'params_Epochs', 'params_Min EL', 'params_Max EL',
                                          'params_Loss Function', 'params_Alpha DILATE'])

        col_names = [key for key in all_best_dict.keys() if key !=
                     'Base Model']
        names = []
        for experiment in col_names:
            if all_best_dict[experiment]['name'] != None:
                names.append(experiment + ' - ' +
                             all_best_dict[experiment]['name'])
            else:
                names.append(None)

        df_params['Models'] = names
        df_params.set_index('Models', inplace=True)
        for experiment, name in zip(col_names, names):
            if experiment == 'Base Model':
                continue
            if name != None:
                params_list = all_best_dict[experiment]['info']['ex_name'].split(
                    '_')
                for item in params_list:
                    if 'Ep' in item:
                        df_params.loc[name, 'params_Epochs'] = item.replace(
                            'Ep', '').split(' ')[-1]
                    elif 'MaxEL' in item:
                        df_params.loc[name, 'params_Min EL'] = str(10)
                        df_params.loc[name, 'params_Max EL'] = item.replace(
                            'MaxEL', '')
                    elif 'EL' in item:
                        df_params.loc[name, 'params_Min EL'] = item.replace(
                            'EL', '')
                        df_params.loc[name, 'params_Max EL'] = item.replace(
                            'EL', '')
                    elif 'dilate' in item:
                        df_params.loc[name, 'params_Loss Function'] = 'Dilate'
                        df_params.loc[name, 'params_Alpha DILATE'] = [
                            str(x.replace('A', '')) for x in params_list if re.search(r'\d+A', x)][0]

                if 'dilate' not in params_list:
                    df_params.loc[name, 'params_Loss Function'] = 'MSE'
                    df_params.loc[name, 'params_Alpha DILATE'] = '-'
            else:
                df_params.loc[name, :] = None

        return df_params
    
    # Create df_params and write to table
    def create_df_params_by_model():
        df_params = pd.DataFrame(columns=['Models', 'params_Ex.', 'params_Ep.', 'params_Min EL', 'params_Max EL',
                                          'params_Loss F.', 'params_Alpha'])

        col_names = [key for key in all_best_dict.keys()]
        names = []
        for col in col_names:
            if do_custom_names:
                new_name = custom_model_names[col]
            else:
                data_tag, in_features, out_features = helper.extract_model_info(col)
                new_name = f'{data_tag}-{in_features}i{out_features}o'
            names.append(new_name)



        df_params['Models'] = names
        df_params.set_index('Models', inplace=True)
        for model, name in zip(col_names, names):
            if name != None:
                params_list = all_best_dict[model]['ex_name'].split('_')
                
                df_params.loc[name, 'params_Ex.'] = params_list[0].split('N')[0]

                for item in params_list:
                    if 'Ep' in item:
                        df_params.loc[name, 'params_Ep.'] = item.replace(
                            'Ep', '').split(' ')[-1]
                    elif 'MaxEL' in item:
                        df_params.loc[name, 'params_Min EL'] = str(10)
                        df_params.loc[name, 'params_Max EL'] = item.replace(
                            'MaxEL', '')
                    elif 'EL' in item:
                        df_params.loc[name, 'params_Min EL'] = item.replace(
                            'EL', '')
                        df_params.loc[name, 'params_Max EL'] = item.replace(
                            'EL', '')
                    elif 'dilate' in item:
                        df_params.loc[name, 'params_Loss F.'] = 'Dilate'
                        df_params.loc[name, 'params_Alpha'] = [
                            str(x.replace('A', '')) for x in params_list if re.search(r'\d+A', x)][0]

                if 'dilate' not in params_list:
                    df_params.loc[name, 'params_Loss F.'] = 'MSE'
                    df_params.loc[name, 'params_Alpha'] = '-'
            else:
                df_params.loc[name, :] = None

        return df_params
    
    if best_by_ex:
        df_params = create_df_params_by_ex()
    elif best_by_model:
        df_params = create_df_params_by_model()

    # Create the parameters table in TEX
    name = f'params_table_{all_best_name}'
    caption = 'The parameters of the best improved checkpoints for each experiment. Ex.: Experiment number, ' +\
        'Ep.: Improvement epochs, Min EL and Max EL: Minimum and Maximum episode length during the improvement, '+\
            'Loss F.: The improvement loss function, and Alpha: Alpha in DILATE loss function.'
    label = 'tab:params_retrain'
    index_name = 'Models' if best_by_model else 'Experiments'
    helper.write_params_to_tex(
        df_params, LATEX_PATH, name, caption, label, index_name)
    print('Parameters table is generated.')

    # Create the metrics table in TEX
    name = f'metrics_table_{all_best_name}'
    if best_by_model:
        caption = 'The average Mean Squared Error and Dynamic Time Warping data for the base model and improved versions'+\
            ' during different months of the year. The best MSE and DTW values for each model and experiment are highlighted in bold and underlined, respectively.'
    else:
        caption = 'The average Mean Squared Error and Dynamic Time Warping data for the base model and improved versions' +\
        ' during different months of the year. The best values of MSE and DTW for each month are highlightd in bold.'
    label = 'tab:metrics_monthly'
    index_name = 'Models' if best_by_model else 'Experiments'
    helper.write_to_tex(all_df, LATEX_PATH, name, caption,
                        label, index_name, highlight=True)
    print('Metrics table is generated.')

    return all_best_dict, all_dict, all_df


data_tags = list(set(data_tags))
data_tags = [item for item in data_tags if item is not None]
all_best_dict, all_dict, all_df = export_best_results_all(
    models_list, ex_list, points_names, data_tags)
