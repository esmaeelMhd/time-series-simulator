import pandas as pd
import numpy as np
import pickle
import datetime
import itertools
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from pylab import rcParams
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from dateutil.relativedelta import relativedelta
import warnings
import pytz
from sklearn.preprocessing import MinMaxScaler
from utils.env_helper import EnvHelper

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import calendar
import os
import torch
from tslearn.metrics import dtw, dtw_path


#%% Plot and device options

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.xmargin'] = 0.02
plt.rcParams['axes.ymargin'] = 0.04
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=5)
fig_dpi = 1000
plt.switch_backend('agg')
# plt.switch_backend('qt5agg')

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"{device}" " is available.")

# Frequency of the dataset
freq = datetime.timedelta(minutes=1)

#%% Initializing

# The raw dataset
dataset_name = 'wastewater.csv'
plot_mode = 'Together'
mse_all = False
mse_p = True
model_mse_change = 'LSTM'

# If retrained
retrained = False
retrained_checkpoint_name = '120_episodes_12_months_with_gap_act_actions_10Epochs_retrained_checkpoint.pth' 

# Environment and the model specifications    
episode_length = 180
sequence_length = 240

# Create an instance of the helper
helper = EnvHelper()

# Points to test
test_frequency = 'Seasons'      # options: Months, Seasons 
day_of_the_month = 'Middle'     # options: First, Middle  
time_of_the_day = 'Morning'     # options: Morning, Noon
first_date = pd.to_datetime('2021-10')
last_date = pd.to_datetime('2022-07')  
points = test_frequency + '_' + day_of_the_month + '_' + time_of_the_day

list_points, points_names = helper.make_points(test_frequency=test_frequency,
                                               day_of_the_month=day_of_the_month,
                                               time_of_the_day=time_of_the_day,
                                               first_date=first_date,
                                               last_date=last_date)

# Choosing the models to run
single = False
best_test_results = False
best_env_results = True
dataset_based = False
all_models = False
setting = 'LSTM_New_CorrH_11F_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256Hidden_2LayerDim'
data_tag = 'New_CorrH'

models_list, best_models, df_metrics = helper.make_models_list(setting=setting,
                                      single=single,
                                      best_test_results=best_test_results,
                                      best_env_results=best_env_results,
                                      dataset_based=dataset_based,
                                      all_models=all_models,
                                      data_tag=data_tag,
                                      episode_length=episode_length)

df_raw = pd.read_csv('./datasets/' + dataset_name)
df_raw = df_raw.set_index(["date"])
df_raw.index = pd.to_datetime(df_raw.index)
if not df_raw.index.is_monotonic:
    df_raw = df_raw.sort_index()
    
print(models_list)
df_metrics.to_csv('./figures/' + 'test_metrics.csv', index=True)

#%%
'''
def test_metrics():
    df_metrics = pd.DataFrame(columns=['models', 'mse', 'mae', 'rse', 'rmse', 'r2', 'corr'])
    df_metrics['models'] = [model for model in best_models.keys()]
    df_metrics = df_metrics.set_index(df_metrics['models'])
    for i , model in enumerate(best_models.values()):
        model_name = list(best_models.keys())[i]
        pred = np.load('./results/' + model + '/pred.npy')
        true = np.load('./results/' + model + '/true.npy')
        metrics = np.load('./results/' + model + '/metrics.npy')
        for col in df_metrics.columns:
            if col == 'mse':
                df_metrics.loc[model_name, col] = mean_squared_error(true, pred)
            elif col == 'mae':
                df_metrics.loc[model_name, col] = mean_absolute_error(true, pred)
            elif col == 'rse':
                df_metrics.loc[model_name, col] = metrics[3]
            elif col == 'rmse':
                df_metrics.loc[model_name, col] = mean_squared_error(true, pred) ** 0.5
            elif col == 'r2':
                df_metrics.loc[model_name, col] = r2_score(true, pred)
            elif col == 'corr':
                df_metrics.loc[model_name, col] = metrics[5]
    
    df_metrics = round(df_metrics, 3)
    return df_metrics

# df_metrics = test_metrics()               
'''
#%% Seperate plots for each point

states_dict = {key: [] for key in best_models}
actual_states_dict = {key: [] for key in best_models}
mse_dict = {key: [] for key in best_models}
    
def plot_separate():
    fig_folder_path = './figures/'
    for point_idx, point_name in enumerate(points_names):
        fig, axs = plt.subplots(3,2, figsize=(16,9), dpi=500)
        fig.tight_layout(pad=4.0)
        fig.suptitle(point_name, fontweight="bold")
        start_date = list_points[point_idx]+freq*sequence_length
        dates = pd.date_range(start=start_date, end=start_date + episode_length*freq, freq='min')
        dates = pd.to_datetime(dates)
        times = dates.strftime('%H:%M')

        colors_dict = {'LSTM': 'blue', 'Transformer': 'orange', 'Informer': 'green',
                      'Autoformer': 'red', 'DLinear': 'purple', 'NLinear': 'brown'}

        for k, (i, j) in enumerate(itertools.product(range(3), range(2))):
            model = list(best_models.keys())[k]
            RESULTS_PATH = './env_results/' + best_models.get(model) + '/'

            states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point_name}_states.npy')
            real_states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point_name}_actual_states.npy')

            states = np.array(states).reshape(np.array(states).shape[0], -1)
            real_states = np.array(real_states).reshape(np.array(real_states).shape[0], -1)
            states[:,0] = real_states[:,0]

            mse_list = []
            for step in range(len(states)):
                mse = mean_squared_error(real_states[step,:], states[step,:])
                mse_list.append(mse)

            mse_average = np.mean(mse_list)

            axs[i,j].title.set_text(model + ' | ' + f'mse: {mse_average}')
            axs[i,j].set_ylabel('P-amount [mg/L]', labelpad=10)
            axs[i,j].plot(states[:,-1],  'x-', color=colors_dict[model], label='Simulated-P', linewidth=0.8, markersize=2)
            axs[i,j].plot(real_states[:,-1], color='black', label='Actual-P', linewidth=1.5)
            axs[i,j].grid(visible=True, which='major', color='silver', alpha=0.25, linewidth=0.075)
            axs[i,j].grid(visible=True, which='minor', color='silver', alpha=0.25, linewidth=0.075)
            axs[i,j].margins(tight=True)
            axs[i,j].legend()

            axs[i,j].set_xticks(np.arange(len(times)))
            axs[i,j].set_xticklabels(times)
            axs[i,j].xaxis.set_major_locator(MultipleLocator(20))
            axs[i,j].xaxis.set_minor_locator(MultipleLocator(5))
            axs[i,j].tick_params(which='minor', length=2)

            states_dict.setdefault(model, []).append(states)
            actual_states_dict.setdefault(model, []).append(real_states)
            mse_dict.setdefault(model, []).append(mse_list)

        axs[-1,0].set_xlabel('Time', labelpad=10)
        axs[-1,1].set_xlabel('Time', labelpad=10)

        if not os.path.exists(fig_folder_path):    
            os.makedirs(fig_folder_path)

        fig_name = f'{point_name}_{episode_length}_rounds'
        plt.savefig(fig_folder_path + fig_name + '.svg')
        plt.savefig(fig_folder_path + fig_name + '.pdf')
        plt.savefig(fig_folder_path + fig_name + '.png')

        plt.ion()
        plt.show()

#%%

mse_dict = {key: [] for key in best_models.keys()}
dtw_dict = {key: {} for key in best_models.keys()}
     
def plot_together():
    if 'Seasons' in points:
        fig, axs = plt.subplots(2,2, figsize=(7, 3.6), dpi=fig_dpi)
    elif 'Months' in points:
        fig, axs = plt.subplots(len(list_points)//2,2, figsize=(20,12), dpi=fig_dpi)

    #fig.tight_layout(pad=4.0)
    for idx, (point, ax) in enumerate(zip(points_names, axs.ravel())):
        for k, model in enumerate(best_models.keys()):
            RESULTS_PATH = './env_results/' + best_models.get(model) + '/'

            if 'Seasons' in points:
                # file_name = (point.split('|')[1]).lstrip()
                states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point}_states.npy')
                real_states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point}_actual_states.npy')
            else:
                states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point}_states.npy')
                real_states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point}_actual_states.npy')

            states = np.array(states).reshape(np.array(states).shape[0], -1)
            real_states = np.array(real_states).reshape(np.array(real_states).shape[0], -1)
            states[:,0] = real_states[:,0]

            mse_list = []
            for step in range(len(states)):
                mse = mean_squared_error(real_states[step,:], states[step,:])
                mse_list.append(mse)

            # mse_list = (mse_list-np.min(mse_list))/(np.max(mse_list)-np.min(mse_list))
            mse_dict.setdefault(model, []).append(mse_list)
            
            # DTW and TDI
            loss_dtw, loss_tdi = 0, 0
            for k in range(1):   
                batch_size, N_output = states[:, -1].reshape([1, states.shape[0], 1]).shape[0:2]
                target = real_states[:, -1].reshape([1, len(real_states), 1])
                outputs = states[:, -1].reshape([1, len(states), 1])
                target = torch.Tensor(target)
                outputs = torch.Tensor(outputs)
                target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
                output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()

                path, sim = dtw_path(target_k_cpu, output_k_cpu)   
                loss_dtw += sim
                           
                Dist = 0
                for i,j in path:
                        Dist += (i-j)*(i-j)
                loss_tdi += Dist / (N_output*N_output)            
                            
            loss_dtw = loss_dtw / batch_size
            loss_tdi = loss_tdi / batch_size
            dtw_dict.setdefault(model, {})[point] = loss_dtw

            start_date = list_points[idx]+freq*sequence_length
            dates = pd.date_range(start=start_date, end=start_date + episode_length*freq, freq='min')
            dates = pd.to_datetime(dates)
            times = dates.strftime('%H:%M')

            plot_label = f'{model}'
            
            season_dict = {'Oct': 'Autumn', 'Jan': 'Winter', 'Apr': 'Spring', 'Jul': 'Summer'}
            for key in season_dict.keys():
                if key in point:
                    season = season_dict.get(key)

            ax.set_title(f'{season} ({point})', fontweight="bold", fontsize=8)
            if k == 0:
                ax.plot(real_states[:,-1], color='black', label='Actual P', linewidth=1)

            ax.plot(states[:,-1], 'x-', label=plot_label, linewidth=0.1, markersize=0.5)
            ax.set_xticks(np.arange(len(times)))
            ax.set_xticklabels(times)
            ax.xaxis.set_major_locator(MultipleLocator(30))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.tick_params(which='minor', length=2)


        #ax.set_ylim([-2,7])
        ax.grid(visible=True, which='major', color='silver', alpha=0.25, linewidth=0.075)
        ax.grid(visible=True, which='minor', color='silver', alpha=0.25, linewidth=0.075)
        
        for i in range(axs.shape[1]):   
            axs[-1, i].set_xlabel('Time (24-hour)', labelpad=5, fontsize=8)
            
        for j in range(axs.shape[0]):
            axs[j, 0].set_ylabel('P-amount [mg/L]', labelpad=10, fontsize=8)
        
    handles, labels = axs[-1,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=7, labelspacing=0., fontsize=7)
    
    plt.subplots_adjust(wspace=0.2, hspace=0.4)


    fig_folder_path = './figures/'
    if not os.path.exists(fig_folder_path):    
        os.makedirs(fig_folder_path)

    fig_name = f'{points}_Together'
    plt.savefig(fig_folder_path + fig_name + '.svg')
    plt.savefig(fig_folder_path + fig_name + '.pdf')
    plt.savefig(fig_folder_path + fig_name + '.png')

    # plt.ion()
    # plt.show() 
    plt.close()
    
def plot_mse():
    if 'Seasons' in points:
        fig_mse, axs_mse = plt.subplots(2,2, figsize=(7, 3.6), dpi=fig_dpi)
    elif 'Months' in points:
        fig_mse, axs_mse = plt.subplots(len(list_points)//2,2, figsize=(20,12), dpi=fig_dpi)

    fig_mse.tight_layout(pad=4.0)
    for point_idx, (point, ax_mse) in enumerate(zip(points_names, axs_mse.ravel())):
        for model in best_models.keys():
            mse_list = mse_dict.get(model)[point_idx]

            mse_list = np.array(mse_list).reshape(-1,1)
            scaler = MinMaxScaler()
            mse_list = scaler.fit_transform(mse_list)

            start_date = list_points[point_idx]+freq*sequence_length
            dates = pd.date_range(start=start_date, end=start_date + episode_length*freq, freq='min')
            dates = pd.to_datetime(dates)
            times = dates.strftime('%H:%M')

            plot_label = f'{model}'
            season_dict = {'Oct': 'Autumn', 'Jan': 'Winter', 'Apr': 'Spring', 'Jul': 'Summer'}
            for key in season_dict.keys():
                if key in point:
                    season = season_dict.get(key)
                    
            ax_mse.set_title(f'{season} ({point})', fontweight="bold", fontsize=8)
            ax_mse.plot(mse_list, 'x-', label=plot_label, linewidth=0.1, markersize=0.5)
            ax_mse.set_xticks(np.arange(len(times)))
            ax_mse.set_xticklabels(times)
            ax_mse.xaxis.set_major_locator(MultipleLocator(30))
            ax_mse.xaxis.set_minor_locator(MultipleLocator(5))
            ax_mse.tick_params(which='minor', length=2)
        ax_mse.grid(visible=True, which='major', color='silver', alpha=0.25, linewidth=0.075)
        ax_mse.grid(visible=True, which='minor', color='silver', alpha=0.25, linewidth=0.075)

    for i in range(axs_mse.shape[1]):   
        axs_mse[-1, i].set_xlabel('Time (24-hour)', labelpad=10, fontsize=8)
        
    for j in range(axs_mse.shape[0]):
        axs_mse[j, 0].set_ylabel('Mean Squared Error', labelpad=10, fontsize=8)

    handles_mse, labels_mse = axs_mse[-1,-1].get_legend_handles_labels()
    fig_mse.legend(handles_mse, labels_mse, loc='upper center', ncol=7, labelspacing=0., fontsize=7)
    plt.subplots_adjust(wspace=0.2, hspace=0.55)


    fig_folder_path = './figures/'
    if not os.path.exists(fig_folder_path):    
        os.makedirs(fig_folder_path)

    fig_mse_name = f'mse_{points}_Together'
    plt.savefig(fig_folder_path + fig_mse_name + '.svg')
    plt.savefig(fig_folder_path + fig_mse_name + '.pdf')
    plt.savefig(fig_folder_path + fig_mse_name + '.png')
    plt.close()

list_change = []
def plot_mse_change():
    if 'Seasons' in points:
        fig_mse_change, axs_mse = plt.subplots(2,2, figsize=(16,9), dpi=500)
    elif 'Months' in points:
        fig_mse_change, axs_mse = plt.subplots(len(list_points)//2,2, figsize=(20,12), dpi=500)

    fig_mse_change.tight_layout(pad=4.0)
    bar_width = 0.4
    for point_idx, (point, ax_mse) in enumerate(zip(points_names, axs_mse.ravel())):
        model = model_mse_change
        mse_list = mse_dict.get(model)[point_idx]

        mse_list = np.array(mse_list).reshape(-1,1)
        scaler = MinMaxScaler()
        mse_list = scaler.fit_transform(mse_list)

        mse_change_df = pd.DataFrame(columns = ['mse', 'mse_change', 'mse_change_first'])
        mse_change_df['mse'] = np.array(mse_list[:,0])
        mse_change_df['mse_change'] = mse_change_df['mse'].pct_change().mul(100)
        list_change.append(mse_change_df['mse_change'])

        mse_change_df['mse_change_first'] = 100 * ((mse_change_df.mse - mse_change_df.iloc[0].mse) / mse_change_df.iloc[0].mse)

        mse_change_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # print(mse_change_df.head())

        start_date = list_points[point_idx]+freq*sequence_length
        dates = pd.date_range(start=start_date, end=start_date + episode_length*freq, freq='min')
        dates = pd.to_datetime(dates)
        times = dates.strftime('%H:%M')

        x = np.arange(episode_length)
        mse_changes = {
            'Change per Step': mse_change_df['mse_change'],
            'Change from the Begining': mse_change_df['mse_change_first']
            }

        for multiplier, (attribute, measurement) in enumerate(mse_changes.items()):
            offset = bar_width * multiplier
            rects = ax_mse.bar(x + offset, measurement, bar_width, label=attribute)
        ax_mse.set_title(point, fontweight="bold")
        ax_mse.set_ylabel('Percentage', labelpad=10)
        ax_mse.set_xticks(np.arange(len(times)) + bar_width)
        ax_mse.set_xticklabels(times)
        ax_mse.xaxis.set_major_locator(MultipleLocator(20))
        ax_mse.xaxis.set_minor_locator(MultipleLocator(5))
        ax_mse.tick_params(which='minor', length=2)

        ax_mse.grid(visible=True, which='major', color='silver', alpha=0.25, linewidth=0.075)
        ax_mse.grid(visible=True, which='minor', color='silver', alpha=0.25, linewidth=0.075)

    for a in range(axs_mse.shape[1]):   
        axs_mse[-1,a].set_xlabel('Time', labelpad=10)

    handles_mse, labels_mse = axs_mse[-1,-1].get_legend_handles_labels()
    fig_mse_change.legend(handles_mse, labels_mse, loc='upper center', ncol=7, labelspacing=0.)

    fig_folder_path = './figures/'
    if not os.path.exists(fig_folder_path):    
        os.makedirs(fig_folder_path)

    fig_mse_change_name = f'mse_change_{points}_Together'
    plt.savefig(fig_folder_path + fig_mse_change_name + '.svg')
    plt.savefig(fig_folder_path + fig_mse_change_name + '.pdf')
    plt.savefig(fig_folder_path + fig_mse_change_name + '.png')
    plt.close()
    
#%%
def write_metrics(mse_dict, dtw_dict):
    df_loss = pd.DataFrame(columns=['models'])
    df_loss['models'] = [model for model in best_models.keys()]
    df_loss = df_loss.set_index(['models'])
    season_dict = {'Oct': 'Autumn', 'Jan': 'Winter', 'Apr': 'Spring', 'Jul': 'Summer'}

    for model in best_models.keys():
        for point_idx, (point) in enumerate(points_names):
            dtw = dtw_dict[model][point]
            mse_list = mse_dict.get(model)[point_idx]
            mse = np.mean(mse_list)
            
            for key in season_dict.keys():
                if key in point:
                    season = season_dict.get(key)
                    
            df_loss.loc[model, f'{season}_mse'] = round(mse, 3)
            df_loss.loc[model, f'{season}_dtw'] = round(dtw, 3)
        
        df_loss['avergae_mse'] = round(df_loss[[col for col in df_loss.columns if 'mse' in col]].mean(axis=1), 3)
        df_loss['avergae_dtw'] = round(df_loss[[col for col in df_loss.columns if 'dtw' in col]].mean(axis=1), 3)
    
    return df_loss
            
if plot_mode == 'Separate':
    plot_separate()
elif plot_mode == 'Together':
    plot_together()
    #plot_mse()
    # plot_mse_change()

# plot_separate()

df_loss = write_metrics(mse_dict, dtw_dict)
df_loss.to_csv('./figures/' + f'Loss_{points}.csv', index=True)
